import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data


class GAT(nn.Module):
    def __init__(self, num_node_features, embedding_dim=256, hidden_dim_1=128, hidden_dim_2=64, heads=4, dropout=0.6, use_two_layer_classifier=False):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_node_features, embedding_dim=embedding_dim)
        
        # First GAT layer with multi-head attention
        self.conv1 = GATConv(
            in_channels=embedding_dim, 
            out_channels=hidden_dim_1, 
            heads=heads, 
            dropout=dropout,
            concat=True  # Concatenate attention heads
        )
        
        # Second GAT layer - typically uses fewer heads and averages them
        self.conv2 = GATConv(
            in_channels=hidden_dim_1 * heads,  # Input size is hidden_dim_1 * heads due to concatenation
            out_channels=hidden_dim_2, 
            heads=1,  # Single head for final layer
            dropout=dropout,
            concat=False  # Average attention heads
        )

        self.pooling = global_mean_pool
        
        # Classifier layers - single or double based on use_two_layer_classifier
        self.use_two_layer_classifier = use_two_layer_classifier
        if use_two_layer_classifier:
            # First classifier layer: input_dim -> hidden_dim_2 // 2
            classifier_hidden_dim = hidden_dim_2 // 2
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim_2, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(classifier_hidden_dim, 1)
            )
        else:
            # Single classifier layer (backward compatible)
            self.classifier = nn.Linear(hidden_dim_2, 1)
        
        # Store configuration for saving/loading
        self.embedding_dim = embedding_dim
        self.num_node_features = num_node_features
        self.heads = heads
        self.dropout_rate = dropout

    def forward(self, x, edge_index, batch):
        # Node embedding
        x = self.emb(x)
        
        # First GAT layer with multi-head attention
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU activation is commonly used with GAT
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Graph-level pooling
        x = self.pooling(x, batch)
        
        # Final classification
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    print("Testing GAT model...")
    
    # Model parameters
    vocab_size = 1000 
    model = GAT(
        num_node_features=vocab_size, 
        embedding_dim=128, 
        hidden_dim_1=64, 
        hidden_dim_2=32,
        heads=4,
        dropout=0.6
    )
    
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
        output = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        print(f"Input shape: {data.x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output.squeeze().item():.4f}")
    
    # Test with batch of graphs
    batch_size = 3
    batch_tensor = torch.cat([torch.full((num_nodes,), i) for i in range(batch_size)])
    x_batch = torch.randint(0, vocab_size, (num_nodes * batch_size,))
    edge_index_batch = torch.randint(0, num_nodes * batch_size, (2, 300))
    
    data_batch = Data(x=x_batch, edge_index=edge_index_batch, batch=batch_tensor)
    
    with torch.no_grad():
        output_batch = model(x=data_batch.x, edge_index=data_batch.edge_index, batch=data_batch.batch)
        print(f"Batch input: {x_batch.shape} nodes, {batch_size} graphs")
        print(f"Batch output: {output_batch.shape}")
    
    print("GAT model test completed successfully!")

    # Test the new two-layer classifier option
    print("\nTesting GAT two-layer classifier...")
    model_2layer = GAT(
        num_node_features=vocab_size, 
        embedding_dim=128, 
        hidden_dim_1=64, 
        hidden_dim_2=32,
        heads=4,
        dropout=0.6,
        use_two_layer_classifier=True
    )
    
    total_params_2layer = sum(p.numel() for p in model_2layer.parameters())
    print(f"Two-layer classifier parameters: {total_params_2layer:,}")
    
    # Test forward pass with two-layer classifier
    model_2layer.eval()
    with torch.no_grad():
        output_2layer = model_2layer(x=data.x, edge_index=data.edge_index, batch=data.batch)
        print(f"Two-layer classifier output shape: {output_2layer.shape}")
        print(f"Two-layer classifier output sample: {output_2layer.squeeze().item():.4f}")
    
    print("GAT two-layer classifier test completed successfully!")
