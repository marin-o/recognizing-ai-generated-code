import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, AttentionalAggregation
from torch_geometric.data import Data


class GCN(nn.Module):
    def __init__(self, num_node_features, embedding_dim=256, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_node_features, embedding_dim=embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        self.classifier = nn.Linear(hidden_dim, 1)

        self.embedding_dim = embedding_dim
        self.num_node_features = num_node_features

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.attention_pool(x, batch)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    print("Testing GCN model...")
    
    # Model parameters
    vocab_size = 1000 
    model = GCN(num_node_features=vocab_size, embedding_dim=128, hidden_dim=64)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dummy graph data
    num_nodes = 50
    x = torch.randint(0, vocab_size, (num_nodes,))  # Random node types
    edge_index = torch.randint(0, num_nodes, (2, 100))  # Random edges
    batch = torch.zeros(num_nodes, dtype=torch.long)  # Single graph
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        output = model(x=x, edge_index=edge_index, batch=batch)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output.squeeze().item():.4f}")
    
    # Test with batch of graphs
    batch_size = 3
    batch_tensor = torch.cat([torch.full((num_nodes,), i) for i in range(batch_size)])
    x_batch = torch.randint(0, vocab_size, (num_nodes * batch_size,))
    edge_index_batch = torch.randint(0, num_nodes * batch_size, (2, 300))
    
    data_batch = Data(x=x_batch, edge_index=edge_index_batch, batch=batch_tensor)
    
    with torch.no_grad():
        x = data_batch.x
        edge_index = data_batch.edge_index
        batch = data_batch.batch
        output_batch = model(x=x, edge_index=edge_index, batch=batch)
        print(f"Batch input: {x_batch.shape} nodes, {batch_size} graphs")
        print(f"Batch output: {output_batch.shape}")
    
    print("Model test completed successfully!")