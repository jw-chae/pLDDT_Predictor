import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils import to_undirected
import esm
import json
from tqdm import tqdm
from torch_geometric.data import Data
import torch.distributed as dist

class ProteinData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'sequence':
            return None  # 'sequence' 속성을 리스트로 유지
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

class ProteinDataset(Dataset):
    def __init__(self, file_path, max_proteins=None):
        super(ProteinDataset, self).__init__()
        self.data_list = []
        with open(file_path, 'r') as file:
            for i, line in enumerate(tqdm(file, total=max_proteins, desc="Loading data")):
                if max_proteins and i >= max_proteins:
                    break
                data = json.loads(line)
                sequence = data['uniprotSequence']
                plddt = data['plddt']
                graph = self.sequence_to_graph(sequence, plddt)
                self.data_list.append(graph)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    @staticmethod
    def sequence_to_graph(sequence, plddt, max_distance=10):
        aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_dict['X'] = 20

        # 아미노산 시퀀스를 인덱스로 변환하여 node_features와 sequence_indices를 생성
        sequence_indices = [aa_dict.get(aa, 20) for aa in sequence]
        node_features = sequence_indices  # node_features와 sequence_indices는 동일한 값입니다.
        x = torch.tensor(node_features, dtype=torch.long).unsqueeze(1)

        edge_index = []
        for i in range(len(sequence)):
            for j in range(i - max_distance, i + max_distance + 1):
                if j >= 0 and j < len(sequence) and i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

        y = torch.tensor([plddt/100.0], dtype=torch.float)  # 스칼라 대신 1차원 텐서로 변경
        # sequence_indices를 텐서로 변환
        sequence_indices_tensor = torch.tensor(sequence_indices, dtype=torch.long)

        # ProteinData 객체에 sequence_indices를 포함하여 반환
        return ProteinData(x=x, edge_index=edge_index, y=y, sequence=sequence_indices_tensor)

        
# In network.py, modify the ProteinGATTransformer class as follows:

class ProteinGATTransformer(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(ProteinGATTransformer, self).__init__()
        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False

        encoder_layer = nn.TransformerEncoderLayer(d_model=320, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(320, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.idx_to_aa = "ACDEFGHIKLMNPQRSTVWYX"
        self.device = torch.device("cuda")

    def to(self, device):
        self.device = device
        return super().to(device)

    def index_to_sequence(self, index_list):
        return ''.join(self.idx_to_aa[i] for i in index_list if i < len(self.idx_to_aa))

    def forward(self, batch):
        sequences = [self.index_to_sequence(data.sequence.tolist()) for data in batch]
        batch_labels, batch_strs, batch_tokens = self.batch_converter([(None, seq) for seq in sequences])
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[self.esm_model.num_layers], return_contacts=False)
        esm_embeddings = results["representations"][self.esm_model.num_layers]

        x = self.transformer_encoder(esm_embeddings)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)

        # Aggregate per-residue outputs to get per-protein output
        x = x.mean(dim=1)

        return x


class MyDataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(MyDataParallel, self).__init__(module, device_ids, output_device, dim)

    def forward(self, batch, sequences):
        if not self.device_ids:
            return self.module(batch, sequences)

        inputs = (batch, sequences)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)



# 데이터 로더 설정 (main.py 또는 해당하는 파일에서)
# def collate_fn(batch):
#     sequences = [data.sequence for data in batch]
#     batch = Batch.from_data_list(batch)
#     return batch, sequences
def collate_fn(batch):
    return Batch.from_data_list(batch)


# train_model 함수 수정
def train_model(model, train_loader, optimizer, criterion, device, scaler, rank, writer, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Rank {rank}", mininterval=10, smoothing=0)):
        batch = [data.to(device) for data in batch]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # 혼합 정밀도 사용
            output = model(batch)

            valid_y = [data.y for data in batch if hasattr(data, 'y') and data.y is not None]
            if not valid_y:
                continue  # 유효한 y 값이 없으면 이 배치를 건너뜁니다

            target = torch.cat(valid_y).to(device)
            output = output[:target.size(0)]

            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 배치마다 손실 출력 (예: 100배치마다)
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {avg_loss:.4f}")

        # TensorBoard에 손실 기록 (rank 0에서만)
        if rank == 0 and writer is not None:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)

    # 모든 프로세스에서 평균 손실 계산
    total_loss_tensor = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    total_loss = total_loss_tensor.item() / dist.get_world_size()

    return total_loss / num_batches



# evaluate_model 함수도 비슷하게 수정
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = [data.to(device) for data in batch]
            if isinstance(model, nn.DataParallel):
                output = model.module(batch)
            else:
                output = model(batch)
            target = torch.cat([data.y for data in batch]).to(device)
            loss = criterion(output, target)
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return total_loss / len(loader), all_preds, all_labels




