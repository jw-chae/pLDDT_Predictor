import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import json
import os
import gzip
import numpy as np
import torch.multiprocessing as mp
from network import ProteinDataset, ProteinGATTransformer, collate_fn, train_model, evaluate_model
from torch_geometric.loader import DataListLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data
import io
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter
import time

MAX_SEQ_LEN = 2048
class OptimizedProteinDataset(ProteinDataset):
    def __init__(self, data_dir, max_proteins=None, cache_dir='./cache'):
        super(ProteinDataset, self).__init__()
        self.data_dir = data_dir
        self.max_proteins = max_proteins
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.json.gz')]
        self.data_index = self._build_index()

    def _build_index(self):
        index = []
        for file_name in tqdm(self.file_list, desc="Indexing files"):
            cache_file = os.path.join(self.cache_dir, file_name + '.idx')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    index.extend(json.load(f))
            else:
                file_index = []
                with gzip.open(os.path.join(self.data_dir, file_name), 'rt') as f:
                    offset = 0
                    for line in f:
                        file_index.append((file_name, offset))
                        offset += len(line.encode('utf-8'))
                        if self.max_proteins and len(index) + len(file_index) >= self.max_proteins:
                            break
                with open(cache_file, 'w') as f:
                    json.dump(file_index, f)
                index.extend(file_index)
            if self.max_proteins and len(index) >= self.max_proteins:
                break
        return index[:self.max_proteins] if self.max_proteins else index

    def __len__(self):
        return len(self.data_index)
    

    def __getitem__(self, idx):
        file_name, offset = self.data_index[idx]
        with gzip.open(os.path.join(self.data_dir, file_name), 'rt') as f:
            f.seek(offset)
            data = json.loads(f.readline())
        
        sequence = data['uniprotSequence']
        sequence = sequence[:MAX_SEQ_LEN]
        plddt = data['globalMetricValue']
        return self.sequence_to_graph(sequence, plddt)


# DDP 초기화 함수
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# 모델 설정 및 DDP 래핑
def setup_model(rank, device, hidden_dim, num_heads, num_layers, dropout):
    model = ProteinGATTransformer(num_features=21, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    model = model.to(device)

    # DDP 래핑
    model = DDP(model, device_ids=[device], output_device=device)
    print(f"Model initialized on GPU {device}")
    return model

# 데이터 로더 설정
def get_data_loaders(rank, world_size, batch_size, data_dir):
    dataset = OptimizedProteinDataset(data_dir, max_proteins=5000000)

    # 데이터셋 분할
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # 분산 샘플러 생성
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # 데이터로더 생성
    train_loader = DataListLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,  # 시스템에 맞게 조정
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataListLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataListLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader

# 훈련 함수 수정
def train_model(model, train_loader, optimizer, criterion, device, scaler, rank, writer, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Rank {rank}", mininterval=10, smoothing=0)):
        batch = [data.to(device) for data in batch]
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):  # 혼합 정밀도 사용
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

# 평가 함수 수정
def evaluate_model(model, loader, criterion, device, scaler):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0, device=device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", mininterval=10, smoothing=0):
            batch = [data.to(device) for data in batch]
            with torch.amp.autocast(device_type='cuda'):
                output = model(batch)
            valid_y = [data.y for data in batch if hasattr(data, 'y') and data.y is not None]
            if not valid_y:
                continue  # 유효한 y 값이 없으면 이 배치를 건너뜁니다
            target = torch.cat(valid_y).to(device)
            output = output[:target.size(0)]
            loss = criterion(output, target) * target.size(0)  # 샘플 수만큼 손실 곱하기
            total_loss += loss
            total_samples += target.size(0)

            # 예측값과 라벨 수집 (CPU로 이동)
            all_preds.append(output.detach().cpu())
            all_labels.append(target.detach().cpu())

    # 모든 프로세스에서 손실과 샘플 수 합산
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    avg_loss = total_loss / total_samples

    return avg_loss.item(), all_preds, all_labels

# 메인 함수 수정
def main_ddp(rank, world_size, available_gpus):
    # Hyperparameters
    hidden_dim = 1024
    num_heads = 16
    num_layers = 12
    batch_size = 32  # 배치 사이즈 감소
    learning_rate = 0.0001  # 학습률 조정
    num_epochs = 1
    weight_decay = 1e-5
    dropout = 0.1

    device = torch.device(f'cuda:{available_gpus[rank]}')
    torch.cuda.set_device(device)  # 각 프로세스에 GPU 할당

    # DDP 초기화
    setup_ddp(rank, world_size)

    # 모델 설정 및 DDP 래핑
    model = setup_model(rank, device, hidden_dim, num_heads, num_layers, dropout)

    # 데이터 로더 설정
    data_dir = '/home/chae/alphafold_results'
    train_loader, val_loader, test_loader = get_data_loaders(rank, world_size, batch_size, data_dir)

    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    # TensorBoard 설정 (rank 0에서만)
    if rank == 0:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f"runs/{current_time}"
        writer = SummaryWriter(log_dir)
    else:
        writer = None  # 나머지 프로세스는 writer를 None으로 설정

    scaler = torch.amp.GradScaler(device='cuda')  # 혼합 정밀도 스케일러

    best_val_loss = float('inf')

    # 학습 루프
    for epoch in range(num_epochs):
        print(f"Rank {rank} starting epoch {epoch+1}/{num_epochs}")
        train_loader.sampler.set_epoch(epoch)  # 에폭마다 데이터 섞기
        train_loss = train_model(model, train_loader, optimizer, criterion, device, scaler, rank, writer, epoch)
        print(f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # 검증 세트 평가
        val_loss, _, _ = evaluate_model(model, val_loader, criterion, device, scaler)
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/validation', val_loss, epoch)

        # 최적의 모델 저장 (rank 0에서만)
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict(), 'protein_gat_transformer_best_model.pth')
            print("New best model saved!")

        # 모델 저장 후 모든 프로세스 동기화
        dist.barrier()

        # 모든 프로세스에서 최적 모델 로드
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.module.load_state_dict(torch.load('protein_gat_transformer_best_model.pth', map_location=map_location, weights_only=True))

        scheduler.step()

    # 테스트 세트 평가
    test_loss, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device, scaler)
    if rank == 0:
        print(f"Test Loss: {test_loss:.4f}")
        writer.add_scalar('Loss/test', test_loss, num_epochs)
        writer.close()  # TensorBoard 작성기 닫기

    # 프로세스 그룹 정리
    cleanup()

if __name__ == "__main__":
    world_size = 8  # 6 GPUs 사용

    # 사용할 GPU 설정
    available_gpus = [0, 1, 2, 3, 4, 5,6,7]

    # 각 GPU에 프로세스 스폰
    mp.spawn(main_ddp, args=(world_size, available_gpus), nprocs=world_size, join=True)
