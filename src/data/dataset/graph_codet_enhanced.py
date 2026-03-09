from torch_geometric.data import Data, InMemoryDataset
from torch import load
import os
import numpy as np
import time
import json
from typing import List, Dict, Any, Optional

class GraphCoDeTM4Enhanced(InMemoryDataset):
    """Enhanced GraphCoDeTM4 dataset that supports positional encodings (depth and child index)."""
    
    def __init__(self, data_dir: str, split: str='train', suffix: str=''):
        self.data_path = data_dir
        self.split = split
        self.suffix = suffix
        self.graphs: Optional[List[Data]] = None
        self.type_to_ind: Optional[Dict[str, int]] = None
        self.stats: Optional[Dict[str, Any]] = None
        super(GraphCoDeTM4Enhanced, self).__init__(root=None)
        
        self.graphs = self._load_graphs()
        self.type_to_ind = self._load_type_mapping()
        self.stats = self._load_statistics()
        
    def _load_graphs(self):
        """Load graph data with enhanced features including depth and child index."""
        # Construct filename with optional suffix
        if self.suffix:
            graph_file = os.path.join(self.data_path, f'{self.split}_graphs_{self.suffix}.pt')
        else:
            graph_file = os.path.join(self.data_path, f'{self.split}_graphs.pt')
        
        print(f"Loading {self.split} graphs from {graph_file}...")
        file_size = os.path.getsize(graph_file) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.1f} MB")
        start_time = time.time()
        graphs = load(graph_file, weights_only=False)
        load_time = time.time() - start_time
        print(f"Loaded {len(graphs)} graphs in {load_time:.2f} seconds")
        
        # Validate that graphs have positional encoding data
        if len(graphs) > 0:
            sample_graph = graphs[0]
            has_depth = hasattr(sample_graph, 'node_depth') and sample_graph.node_depth is not None
            has_child_index = hasattr(sample_graph, 'child_index') and sample_graph.child_index is not None
            
            if has_depth and has_child_index:
                print(f"✓ Graphs contain positional encodings (depth and child index)")
            elif has_depth:
                print(f"⚠ Graphs contain only depth information (no child index)")
            elif has_child_index:
                print(f"⚠ Graphs contain only child index information (no depth)")
            else:
                print(f"⚠ Graphs do not contain positional encodings")
                print(f"Available attributes: {list(sample_graph.keys())}")
        
        return graphs

    def _load_type_mapping(self):
        """Load node type to index mapping."""
        if self.suffix:
            type_file = os.path.join(self.data_path, f'type_to_ind_{self.suffix}.pt')
        else:
            type_file = os.path.join(self.data_path, 'type_to_ind.pt')
        
        if os.path.exists(type_file):
            print("Loading type mapping...")
            type_to_ind = load(type_file, weights_only=False)
            print(f"Loaded {len(type_to_ind)} type mappings")
            return type_to_ind
        return None
    
    def _load_statistics(self):
        """Load depth and child index statistics."""
        if self.suffix:
            stats_file = os.path.join(self.data_path, f'depth_child_index_stats_{self.suffix}.json')
        else:
            stats_file = os.path.join(self.data_path, 'depth_child_index_stats.json')
        
        if os.path.exists(stats_file):
            print("Loading positional encoding statistics...")
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"Statistics summary:")
            print(f"  Max depth: {stats['max_depth_global']}")
            print(f"  Max child index: {stats['max_child_index_global']}")
            print(f"  Depth 95th percentile: {stats['depth_statistics']['percentile_95']}")
            print(f"  Child index 95th percentile: {stats['child_index_statistics']['percentile_95']}")
            
            return stats
        else:
            print(f"Warning: Statistics file not found: {stats_file}")
            return None

    @property
    def processed_file_names(self):
        if self.suffix:
            return [
                f'{self.split}_graphs_{self.suffix}.pt', 
                f'type_to_ind_{self.suffix}.pt',
                f'depth_child_index_stats_{self.suffix}.json'
            ]
        else:
            return [
                f'{self.split}_graphs.pt', 
                'type_to_ind.pt',
                'depth_child_index_stats.json'
            ]
    
    @property
    def processed_dir(self):
        return self.data_path  
    
    @property
    def num_node_features(self):
        """Number of unique node types."""
        return len(self.type_to_ind) if self.type_to_ind else 0

    @property
    def max_depth(self):
        """Maximum depth found in the dataset."""
        return self.stats['max_depth_global'] if self.stats else None
    
    @property
    def max_child_index(self):
        """Maximum child index found in the dataset."""
        return self.stats['max_child_index_global'] if self.stats else None
    
    @property
    def recommended_max_depth(self):
        """Recommended max depth based on 99th percentile."""
        if self.stats:
            return int(self.stats['depth_statistics']['percentile_99']) + 1
        return None
    
    @property
    def recommended_max_child_index(self):
        """Recommended max child index based on 99th percentile."""
        if self.stats:
            return int(self.stats['child_index_statistics']['percentile_99']) + 1
        return None

    def get_positional_encoding_config(self, use_percentile=True):
        """Get recommended configuration for positional encodings."""
        if not self.stats:
            print("Warning: No statistics available for positional encoding configuration")
            return None
        
        if use_percentile:
            # Use 99th percentile to handle most cases while avoiding extreme outliers
            max_depth = self.recommended_max_depth
            max_child_index = self.recommended_max_child_index
        else:
            # Use absolute maximum
            max_depth = self.max_depth
            max_child_index = self.max_child_index
        
        # Calculate embedding dimensions safely
        depth_embed_dim = min(64, max(16, max_depth // 4)) if max_depth else 32
        child_embed_dim = min(64, max(16, max_child_index // 8)) if max_child_index else 32
        
        config = {
            'max_depth': max_depth,
            'max_child_index': max_child_index,
            'depth_embedding_dim': depth_embed_dim,
            'child_embedding_dim': child_embed_dim,
            'statistics': self.stats
        }
        
        return config

    def len(self):
        return len(self.graphs) if self.graphs is not None else 0
    
    def get(self, idx):
        if self.graphs is None:
            raise RuntimeError("Graphs not loaded")
        return self.graphs[idx]
    
    def has_positional_encodings(self):
        """Check if the dataset contains positional encoding data."""
        if self.graphs is None or len(self.graphs) == 0:
            return False
        
        sample = self.graphs[0]
        has_depth = hasattr(sample, 'node_depth') and sample.node_depth is not None
        has_child = hasattr(sample, 'child_index') and sample.child_index is not None
        
        return has_depth and has_child

    def print_dataset_info(self):
        """Print comprehensive dataset information."""
        print(f"\n{'='*50}")
        print(f"Enhanced GraphCoDeTM4 Dataset Info - {self.split.upper()}")
        print(f"{'='*50}")
        print(f"Dataset size: {len(self)} graphs")
        print(f"Node types: {self.num_node_features}")
        print(f"Data suffix: '{self.suffix}' " + ("(default)" if not self.suffix else ""))
        
        if self.stats:
            print(f"\nPositional Encoding Statistics:")
            print(f"  Depth range: 0 to {self.max_depth}")
            print(f"  Child index range: 0 to {self.max_child_index}")
            print(f"  Recommended depth limit (99th percentile): {self.recommended_max_depth}")
            print(f"  Recommended child index limit (99th percentile): {self.recommended_max_child_index}")
        
        if self.graphs is not None and len(self.graphs) > 0:
            # Analyze graph sizes
            try:
                graph_sizes = []
                for graph in self.graphs[:1000]:  # Sample first 1000
                    if hasattr(graph, 'x') and graph.x is not None:
                        if hasattr(graph.x, 'shape'):
                            graph_sizes.append(graph.x.shape[0])
                        elif hasattr(graph.x, '__len__'):
                            graph_sizes.append(len(graph.x))
                
                if graph_sizes:
                    print(f"\nGraph Size Statistics (sample):")
                    print(f"  Nodes per graph - Min: {min(graph_sizes)}, Max: {max(graph_sizes)}")
                    print(f"  Mean: {np.mean(graph_sizes):.1f}, Std: {np.std(graph_sizes):.1f}")
            except Exception as e:
                print(f"\nCould not analyze graph sizes: {e}")
            
            # Check for positional encodings
            print(f"\nPositional Encodings: {'✓ Available' if self.has_positional_encodings() else '✗ Not available'}")
            
            # Sample data inspection
            try:
                sample = self.graphs[0]
                print(f"\nSample Graph Attributes:")
                for key in sample.keys():
                    try:
                        value = getattr(sample, key)
                        if hasattr(value, 'shape'):
                            print(f"  {key}: {value.shape}")
                        else:
                            print(f"  {key}: {type(value)}")
                    except Exception:
                        print(f"  {key}: <unavailable>")
            except Exception as e:
                print(f"\nCould not inspect sample graph: {e}")
        
        print(f"{'='*50}\n")


if __name__ == '__main__':
    # Example usage
    print("Testing Enhanced GraphCoDeTM4 Dataset...")
    
    # Load dataset with positional encodings
    train_dataset = GraphCoDeTM4Enhanced(
        data_dir='/home/bosa/diplomska/data/codet_graphs/', 
        split='train', 
        suffix='cleaned_comments_depth'
    )
    
    # Print comprehensive info
    train_dataset.print_dataset_info()
    
    # Get positional encoding configuration
    pos_config = train_dataset.get_positional_encoding_config(use_percentile=True)
    if pos_config:
        print("Recommended Positional Encoding Configuration:")
        print(f"  max_depth: {pos_config['max_depth']}")
        print(f"  max_child_index: {pos_config['max_child_index']}")
        print(f"  depth_embedding_dim: {pos_config['depth_embedding_dim']}")
        print(f"  child_embedding_dim: {pos_config['child_embedding_dim']}")
    
    # Test data loading
    print(f"\nSample data object: {train_dataset[0]}")
    sample = train_dataset[0]
    
    # Check positional encoding data safely
    try:
        if hasattr(sample, 'node_depth') and getattr(sample, 'node_depth', None) is not None:
            depth_tensor = getattr(sample, 'node_depth')
            print(f"Node depth range: {depth_tensor.min().item()} to {depth_tensor.max().item()}")
        if hasattr(sample, 'child_index') and getattr(sample, 'child_index', None) is not None:
            child_tensor = getattr(sample, 'child_index')
            print(f"Child index range: {child_tensor.min().item()} to {child_tensor.max().item()}")
    except Exception as e:
        print(f"Could not access positional encoding info: {e}")
