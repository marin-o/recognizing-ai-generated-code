import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import math


class GraphTransformer(nn.Module):
    def __init__(self, num_node_features, embedding_dim=256, hidden_dim=128, num_heads=8, num_layers=2, 
                 dropout=0.1, pooling_method='mean', use_edge_attr=False):
        super().__init__()
        
        # Node embedding layer
        self.emb = nn.Embedding(num_embeddings=num_node_features, embedding_dim=embedding_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=embedding_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=None if not use_edge_attr else 1,  # Can be extended for edge features
                beta=True,  # Use beta parameter for skip connections
                concat=False  # Average heads instead of concatenating
            ) for i in range(num_layers)
        ])
        
        # Layer normalization for each transformer layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Pooling method
        self.pooling_method = pooling_method
        if pooling_method == 'mean':
            self.pooling = global_mean_pool
        elif pooling_method == 'max':
            self.pooling = global_max_pool
        elif pooling_method == 'add':
            self.pooling = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Store configuration for checkpoints
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_node_features = num_node_features
        self.pooling_method = pooling_method
        self.use_edge_attr = use_edge_attr

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Node embedding
        x = self.emb(x)  # [num_nodes, embedding_dim]
        
        # Apply transformer layers with residual connections
        for i, (transformer, layer_norm) in enumerate(zip(self.transformer_layers, self.layer_norms)):
            # Store input for residual connection
            residual = x
            
            # Apply transformer layer
            x = transformer(x, edge_index, edge_attr=edge_attr)
            
            # Add residual connection and apply layer norm
            if i > 0:  # No residual for first layer due to dimension mismatch
                x = x + residual
            x = layer_norm(x)
            
            # Apply dropout
            x = self.dropout(x)
            x = F.relu(x)
        
        # Global pooling
        x = self.pooling(x, batch)  # [batch_size, hidden_dim]
        
        # Final classification
        x = self.classifier(x)  # [batch_size, 1]
        
        return x


if __name__ == "__main__":
    print("Testing GraphTransformer model...")
    
    # Model parameters
    vocab_size = 1000 
    model = GraphTransformer(
        num_node_features=vocab_size, 
        embedding_dim=128, 
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1
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
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(x=x, edge_index=edge_index, batch=batch)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output.squeeze().item():.4f}")
    
    # Test with batch of graphs
    batch_size = 3
    batch_tensor = torch.cat([torch.full((num_nodes,), i) for i in range(batch_size)])
    x_batch = torch.randint(0, vocab_size, (num_nodes * batch_size,))
    edge_index_batch = torch.randint(0, num_nodes * batch_size, (2, 300))
    
    with torch.no_grad():
        output_batch = model(x=x_batch, edge_index=edge_index_batch, batch=batch_tensor)
        print(f"Batch input: {x_batch.shape} nodes, {batch_size} graphs")
        print(f"Batch output: {output_batch.shape}")
    
    # Test different configurations
    print("\nTesting different configurations...")
    
    # Test with more layers and heads
    model_large = GraphTransformer(
        num_node_features=vocab_size,
        embedding_dim=256,
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        pooling_method='max'
    )
    
    with torch.no_grad():
        output_large = model_large(x=x, edge_index=edge_index, batch=batch)
        print(f"Large model output: {output_large.squeeze().item():.4f}")
    
    print("GraphTransformer test completed successfully!")

# Graph Transformer for AST classification with attention mechanisms
