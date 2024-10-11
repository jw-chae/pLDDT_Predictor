import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import esm
import json
import os
from collections import OrderedDict
from tqdm import tqdm
import argparse

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
    # Argument parser to accept input, output paths and model hyperparameters
    parser = argparse.ArgumentParser(description="Run pLDDT prediction with specified model and parameters.")
    
    # File paths with default values
    parser.add_argument('--input_json_path', type=str, default='plddt_predictor/sample_data/sampled_protein_data.json', 
                        help="Path to the input JSON file (default: plddt_predictor/sample_data/sampled_protein_data.json)")
    parser.add_argument('--output_json_path', type=str, default='plddt_predictor/sample_data/test_output.json', 
                        help="Path to save the output JSON file (default: plddt_predictor/sample_data/test_output.json)")
    parser.add_argument('--model_path', type=str, default='plddt_predictor/saved_models/protein_gat_transformer1024_best_model.pth', 
                        help="Path to the saved model file (default: plddt_predictor/saved_models/protein_gat_transformer1024_best_model.pth)")
    
    # Hyperparameters with default values
    parser.add_argument('--hidden_dim', type=int, default=1024, help="Hidden dimension size of the model (default: 1024)")
    parser.add_argument('--num_heads', type=int, default=16, help="Number of attention heads (default: 16)")
    parser.add_argument('--num_layers', type=int, default=8, help="Number of layers in the model (default: 8)")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate (default: 0.1)")
    
    # Parse the arguments
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    model = ProteinGATTransformer(
        num_features=21, 
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    )
    model = model.to(device)

    # Load trained weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model doesn't exist at path {args.model_path}")

    state_dict = torch.load(args.model_path, map_location=device)

    # If model was trained with DDP (Distributed Data Parallel), remove 'module.' prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # Remove 'module.' prefix
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Check input JSON file existence
    if not os.path.exists(args.input_json_path):
        raise FileNotFoundError(f"Input JSON file doesn't exist at {args.input_json_path}")

    output_data = []

    # Read input JSON file line by line and process
    with open(args.input_json_path, 'r') as f:
        for line in tqdm(f, desc="Sequence processing"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e} - line: {line.strip()}")
                continue

            sequence = entry.get('uniprotSequence', None)
            metric_value = entry.get('globalMetricValue', None)

            if sequence is None or metric_value is None:
                print("Warning: Skipping items that do not have 'uniprotSequence' or 'globalMetricValue'")
                continue

            # Convert sequence to graph
            data = sequence_to_graph(sequence)
            data = data.to(device)
            batch = [data]  # Model expects a list of Data objects

            # Make prediction
            with torch.no_grad():
                output = model(batch)
            prediction = output.item()

            # Store the results
            output_entry = {
                "sequence": sequence,
                "metric_value": metric_value,
                "predicted_value": prediction
            }
            output_data.append(output_entry)

    # Save the results to output JSON file
    with open(args.output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Predicted results saved at {args.output_json_path}")