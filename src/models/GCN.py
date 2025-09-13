import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, AttentionalAggregation, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data


class GCN(nn.Module):
    def __init__(self, num_node_features, embedding_dim=256, hidden_dim_1=128, hidden_dim_2=128, sage=False, use_two_layer_classifier=False):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_node_features, embedding_dim=embedding_dim)
        if sage:
            self.conv1 = SAGEConv(embedding_dim, hidden_dim_1)
            self.conv2 = SAGEConv(hidden_dim_1, hidden_dim_2)
        else:
            self.conv1 = GCNConv(embedding_dim, hidden_dim_1)
            self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)

        # self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
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

        self.embedding_dim = embedding_dim
        self.num_node_features = num_node_features

    def forward(self, x, edge_index, batch):
        x = self.emb(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.pooling(x, batch)
        x = self.classifier(x)

        return x


class GCNWithPositionalEncoding(nn.Module):
    def __init__(self, num_node_features, max_depth=50, max_child_index=20, 
                 embedding_dim=256, hidden_dim_1=128, hidden_dim_2=128, 
                 sage=False, use_two_layer_classifier=False, dropout=0.1, 
                 pooling_method='mean', depth_embedding_dim=32, child_embedding_dim=32):
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
        
        # GCN layers
        if sage:
            self.conv1 = SAGEConv(embedding_dim, hidden_dim_1)
            self.conv2 = SAGEConv(hidden_dim_1, hidden_dim_2)
        else:
            self.conv1 = GCNConv(embedding_dim, hidden_dim_1)
            self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)

        # Dropout
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
        
        # Classifier layers - single or double based on use_two_layer_classifier
        self.use_two_layer_classifier = use_two_layer_classifier
        if use_two_layer_classifier:
            # First classifier layer: input_dim -> hidden_dim_2 // 2
            classifier_hidden_dim = hidden_dim_2 // 2
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim_2, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 1)
            )
        else:
            # Single classifier layer (backward compatible)
            self.classifier = nn.Linear(hidden_dim_2, 1)

        # Store configuration for checkpoints
        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_node_features = num_node_features
        self.max_depth = max_depth
        self.max_child_index = max_child_index
        self.depth_embedding_dim = depth_embedding_dim
        self.child_embedding_dim = child_embedding_dim
        self.sage = sage
        self.pooling_method = pooling_method
        self.dropout_rate = dropout

    def forward(self, x, edge_index, batch, node_depth=None, child_index=None):
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
        
        # Apply GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling
        x = self.pooling(x, batch)
        
        # Final classification
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    print("Testing GCN models...")
    
    # Model parameters
    vocab_size = 1000 
    model = GCN(num_node_features=vocab_size, embedding_dim=128, hidden_dim_1=64)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Original GCN:")
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
        print(f"Batch input: {data_batch.x.shape} nodes, {batch_size} graphs")
        print(f"Batch output: {output_batch.shape}")
    
    print("Original GCN test completed successfully!")
    
    # Test the new two-layer classifier option
    print("\nTesting two-layer classifier...")
    model_2layer = GCN(num_node_features=vocab_size, embedding_dim=128, hidden_dim_1=64, use_two_layer_classifier=True)
    
    total_params_2layer = sum(p.numel() for p in model_2layer.parameters())
    print(f"Two-layer classifier parameters: {total_params_2layer:,}")
    
    # Test forward pass with two-layer classifier
    model_2layer.eval()
    with torch.no_grad():
        output_2layer = model_2layer(x=data.x, edge_index=data.edge_index, batch=data.batch)
        print(f"Two-layer classifier output shape: {output_2layer.shape}")
        print(f"Two-layer classifier output sample: {output_2layer.squeeze().item():.4f}")
    
    print("Two-layer classifier test completed successfully!")
    
    print("\n" + "="*60)
    print("Testing GCNWithPositionalEncoding...")
    
    # Create enhanced model with positional encodings
    model_enhanced = GCNWithPositionalEncoding(
        num_node_features=vocab_size,
        max_depth=20,
        max_child_index=10,
        embedding_dim=128,
        hidden_dim_1=64,
        hidden_dim_2=64,
        sage=False,
        use_two_layer_classifier=True,
        depth_embedding_dim=32,
        child_embedding_dim=32
    )
    
    # Print enhanced model info
    total_params_enhanced = sum(p.numel() for p in model_enhanced.parameters())
    trainable_params_enhanced = sum(p.numel() for p in model_enhanced.parameters() if p.requires_grad)
    print(f"Enhanced GCN:")
    print(f"Total parameters: {total_params_enhanced:,}")
    print(f"Trainable parameters: {trainable_params_enhanced:,}")
    print(f"Parameter increase: {total_params_enhanced - total_params:,} ({((total_params_enhanced - total_params) / total_params * 100):.1f}%)")
    
    # Create dummy positional data
    node_depth = torch.randint(0, 20, (num_nodes,))  # Random depths
    child_index = torch.randint(0, 10, (num_nodes,))  # Random child indices
    
    # Test enhanced model
    model_enhanced.eval()
    with torch.no_grad():
        output_enhanced = model_enhanced(
            x=data.x, edge_index=data.edge_index, batch=data.batch,
            node_depth=node_depth, child_index=child_index
        )
        print(f"Enhanced output shape: {output_enhanced.shape}")
        print(f"Enhanced output sample: {output_enhanced.squeeze().item():.4f}")
    
    # Test enhanced model without positional encodings
    with torch.no_grad():
        output_no_pos = model_enhanced(
            x=data.x, edge_index=data.edge_index, batch=data.batch
        )
        print(f"Enhanced output (no pos encodings) shape: {output_no_pos.shape}")
        print(f"Enhanced output (no pos encodings) sample: {output_no_pos.squeeze().item():.4f}")
    
    # Test with batch of graphs
    print("\nTesting enhanced model batch processing...")
    depth_batch = torch.randint(0, 20, (num_nodes * batch_size,))
    child_batch = torch.randint(0, 10, (num_nodes * batch_size,))
    
    with torch.no_grad():
        output_batch_enhanced = model_enhanced(
            x=data_batch.x, edge_index=data_batch.edge_index, batch=data_batch.batch,
            node_depth=depth_batch, child_index=child_batch
        )
        print(f"Enhanced batch output shape: {output_batch_enhanced.shape}")
    
    # Test different configurations
    print("\nTesting different enhanced configurations...")
    
    # Test with SAGE convolution
    model_sage = GCNWithPositionalEncoding(
        num_node_features=vocab_size,
        max_depth=50,
        max_child_index=25,
        embedding_dim=256,
        hidden_dim_1=128,
        hidden_dim_2=128,
        sage=True,
        pooling_method='max',
        depth_embedding_dim=64,
        child_embedding_dim=64
    )
    
    large_depth = torch.randint(0, 50, (num_nodes,))
    large_child = torch.randint(0, 25, (num_nodes,))
    
    with torch.no_grad():
        output_sage = model_sage(
            x=data.x, edge_index=data.edge_index, batch=data.batch,
            node_depth=large_depth, child_index=large_child
        )
        print(f"SAGE enhanced output shape: {output_sage.shape}")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Test with maximum depth and child indices
    max_depth_tensor = torch.full((num_nodes,), 19)  # max_depth - 1
    max_child_tensor = torch.full((num_nodes,), 9)   # max_child_index - 1
    
    with torch.no_grad():
        output_max = model_enhanced(
            x=data.x, edge_index=data.edge_index, batch=data.batch,
            node_depth=max_depth_tensor, child_index=max_child_tensor
        )
        print(f"Max values output shape: {output_max.shape}")
    
    # Test with out-of-range values (should be clamped)
    extreme_depth = torch.randint(0, 100, (num_nodes,))  # Some values > max_depth
    extreme_child = torch.randint(0, 50, (num_nodes,))   # Some values > max_child_index
    
    with torch.no_grad():
        output_extreme = model_enhanced(
            x=data.x, edge_index=data.edge_index, batch=data.batch,
            node_depth=extreme_depth, child_index=extreme_child
        )
        print(f"Extreme values output shape: {output_extreme.shape}")
    
    print("Enhanced GCN test completed successfully!")

# posebni grafovi, dataflow, comments next steps
