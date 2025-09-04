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


class GraphTransformerWithPositionalEncoding(nn.Module):
    def __init__(self, num_node_features, max_depth=50, max_child_index=20, 
                 embedding_dim=256, hidden_dim=128, num_heads=8, num_layers=2, 
                 dropout=0.1, pooling_method='mean', use_edge_attr=False,
                 depth_embedding_dim=32, child_embedding_dim=32):
        super().__init__()
        
        # Node type embedding layer
        self.node_emb = nn.Embedding(num_embeddings=num_node_features, embedding_dim=embedding_dim)
        
        # Positional encodings
        self.depth_emb = nn.Embedding(num_embeddings=max_depth + 1, embedding_dim=depth_embedding_dim)
        self.child_emb = nn.Embedding(num_embeddings=max_child_index + 1, embedding_dim=child_embedding_dim)
        
        # Calculate total input dimension after concatenation
        total_embedding_dim = embedding_dim + depth_embedding_dim + child_embedding_dim
        
        # Projection layer to reduce concatenated embeddings to desired dimension
        self.embedding_projection = nn.Linear(total_embedding_dim, embedding_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=embedding_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=None if not use_edge_attr else 1,
                beta=True,
                concat=False
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
        self.max_depth = max_depth
        self.max_child_index = max_child_index
        self.depth_embedding_dim = depth_embedding_dim
        self.child_embedding_dim = child_embedding_dim
        self.pooling_method = pooling_method
        self.use_edge_attr = use_edge_attr

    def forward(self, x, edge_index, batch, node_depth=None, child_index=None, edge_attr=None):
        # Node type embedding
        node_features = self.node_emb(x)  # [num_nodes, embedding_dim]
        
        # Add positional encodings if available
        if node_depth is not None:
            # Clamp depths to valid range
            depth_clamped = torch.clamp(node_depth, 0, self.max_depth)
            depth_features = self.depth_emb(depth_clamped)  # [num_nodes, depth_embedding_dim]
            node_features = torch.cat([node_features, depth_features], dim=1)
        else:
            # Add zero depth embeddings if not provided
            depth_features = torch.zeros(x.size(0), self.depth_embedding_dim, device=x.device)
            node_features = torch.cat([node_features, depth_features], dim=1)
        
        if child_index is not None:
            # Clamp child indices to valid range
            child_clamped = torch.clamp(child_index, 0, self.max_child_index)
            child_features = self.child_emb(child_clamped)  # [num_nodes, child_embedding_dim]
            node_features = torch.cat([node_features, child_features], dim=1)
        else:
            # Add zero child embeddings if not provided
            child_features = torch.zeros(x.size(0), self.child_embedding_dim, device=x.device)
            node_features = torch.cat([node_features, child_features], dim=1)
        
        # Project combined features to target dimension
        x = self.embedding_projection(node_features)  # [num_nodes, embedding_dim]
        x = F.relu(x)
        
        # Apply transformer layers with residual connections
        for i, (transformer, layer_norm) in enumerate(zip(self.transformer_layers, self.layer_norms)):
            # Store input for residual connection
            residual = x
            
            # Apply transformer layer
            x = transformer(x, edge_index, edge_attr=edge_attr)
            
            # Add residual connection and apply layer norm
            if i > 0:
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
    print("Testing GraphTransformer models...")
    
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
    print(f"Original GraphTransformer:")
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
    
    print("\n" + "="*60)
    print("Testing GraphTransformerWithPositionalEncoding...")
    
    # Create enhanced model with positional encodings
    model_enhanced = GraphTransformerWithPositionalEncoding(
        num_node_features=vocab_size,
        max_depth=20,  # Based on typical AST depths
        max_child_index=10,  # Based on typical child counts
        embedding_dim=128,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        depth_embedding_dim=32,
        child_embedding_dim=32
    )
    
    # Print enhanced model info
    total_params_enhanced = sum(p.numel() for p in model_enhanced.parameters())
    trainable_params_enhanced = sum(p.numel() for p in model_enhanced.parameters() if p.requires_grad)
    print(f"Enhanced GraphTransformer:")
    print(f"Total parameters: {total_params_enhanced:,}")
    print(f"Trainable parameters: {trainable_params_enhanced:,}")
    print(f"Parameter increase: {total_params_enhanced - total_params:,} ({((total_params_enhanced - total_params) / total_params * 100):.1f}%)")
    
    # Create dummy positional data
    node_depth = torch.randint(0, 20, (num_nodes,))  # Random depths
    child_index = torch.randint(0, 10, (num_nodes,))  # Random child indices
    
    # Test enhanced model
    model_enhanced.eval()
    with torch.no_grad():
        # Test with positional encodings
        output_enhanced = model_enhanced(x=x, edge_index=edge_index, batch=batch,
                                       node_depth=node_depth, child_index=child_index)
        print(f"Enhanced output with positional encodings: {output_enhanced.squeeze().item():.4f}")
        
        # Test without positional encodings (should still work)
        output_no_pos = model_enhanced(x=x, edge_index=edge_index, batch=batch)
        print(f"Enhanced output without positional encodings: {output_no_pos.squeeze().item():.4f}")
    
    # Test with batch of graphs
    print("\nTesting batch processing...")
    batch_size = 3
    batch_tensor = torch.cat([torch.full((num_nodes,), i) for i in range(batch_size)])
    x_batch = torch.randint(0, vocab_size, (num_nodes * batch_size,))
    edge_index_batch = torch.randint(0, num_nodes * batch_size, (2, 300))
    depth_batch = torch.randint(0, 20, (num_nodes * batch_size,))
    child_batch = torch.randint(0, 10, (num_nodes * batch_size,))
    
    with torch.no_grad():
        output_batch = model_enhanced(x=x_batch, edge_index=edge_index_batch, 
                                    batch=batch_tensor, node_depth=depth_batch, 
                                    child_index=child_batch)
        print(f"Batch input: {x_batch.shape} nodes, {batch_size} graphs")
        print(f"Batch output: {output_batch.shape}")
    
    # Test different configurations
    print("\nTesting different configurations...")
    
    # Test with larger embeddings
    model_large = GraphTransformerWithPositionalEncoding(
        num_node_features=vocab_size,
        max_depth=50,  # Larger depth range
        max_child_index=25,  # Larger child range
        embedding_dim=256,
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        pooling_method='max',
        depth_embedding_dim=64,
        child_embedding_dim=64
    )
    
    large_depth = torch.randint(0, 50, (num_nodes,))
    large_child = torch.randint(0, 25, (num_nodes,))
    
    with torch.no_grad():
        output_large = model_large(x=x, edge_index=edge_index, batch=batch,
                                 node_depth=large_depth, child_index=large_child)
        print(f"Large model output: {output_large.squeeze().item():.4f}")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Test with maximum depth and child indices
    max_depth_tensor = torch.full((num_nodes,), 19)  # max_depth - 1
    max_child_tensor = torch.full((num_nodes,), 9)   # max_child_index - 1
    
    with torch.no_grad():
        output_max = model_enhanced(x=x, edge_index=edge_index, batch=batch,
                                  node_depth=max_depth_tensor, child_index=max_child_tensor)
        print(f"Output with max positional values: {output_max.squeeze().item():.4f}")
    
    # Test with out-of-range values (should be clamped)
    extreme_depth = torch.randint(0, 100, (num_nodes,))  # Some values > max_depth
    extreme_child = torch.randint(0, 50, (num_nodes,))   # Some values > max_child_index
    
    with torch.no_grad():
        output_extreme = model_enhanced(x=x, edge_index=edge_index, batch=batch,
                                      node_depth=extreme_depth, child_index=extreme_child)
        print(f"Output with extreme positional values (clamped): {output_extreme.squeeze().item():.4f}")
    