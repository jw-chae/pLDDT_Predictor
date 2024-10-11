import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import esm

# Assuming network.py defines ProteinGATTransformer
# from network import ProteinGATTransformer

class ProteinData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'sequence':
            return None  # Keep 'sequence' as a list
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

def sequence_to_graph(sequence, max_distance=10):
    aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    aa_dict['X'] = 20  # Unknown amino acid

    # Convert amino acid sequence to indices
    sequence_indices = [aa_dict.get(aa, 20) for aa in sequence]
    node_features = sequence_indices
    x = torch.tensor(node_features, dtype=torch.long).unsqueeze(1)

    edge_index = []
    for i in range(len(sequence)):
        for j in range(i - max_distance, i + max_distance + 1):
            if j >= 0 and j < len(sequence) and i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    # Since we don't have a target value during testing, y is not needed
    # Convert sequence indices to tensor
    sequence_indices_tensor = torch.tensor(sequence_indices, dtype=torch.long)

    # Return ProteinData object including sequence indices
    return ProteinData(x=x, edge_index=edge_index, sequence=sequence_indices_tensor)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == "__main__":
    # Load the trained model
    hidden_dim = 1024  # Must match the trained model's settings
    num_heads = 16
    num_layers = 8
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProteinGATTransformer(num_features=21, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    model = model.to(device)

    # Load the trained weights
    model_path = 'plddt_predictor/saved_models/protein_gat_transformer1024_best_model.pth'
    state_dict = torch.load(model_path, map_location=device)

    # If the model was trained using DDP, the state_dict keys might be prefixed with 'module.'
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # remove 'module.' prefix
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    while True:
        input_sequence = input("Enter protein sequence: ").strip().upper()
        
        if input_sequence.lower() == 'quit':
            print("Exiting the program.")
            break
        
        if not all(aa in "ACDEFGHIKLMNPQRSTVWYX" for aa in input_sequence):
            print("Invalid sequence. Please use only valid amino acid letters.")
            continue

        data = sequence_to_graph(input_sequence)
        data = data.to(device)
        batch = [data]  # Model expects a list of Data objects

        # Make a prediction
        with torch.no_grad():
            output = model(batch)
        prediction = output.item()

        print(f"Prediction for the input sequence: {prediction}")
        print("Enter another sequence or 'quit' to exit.")