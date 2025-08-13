from torch_geometric.data import Data, InMemoryDataset
from torch import load
import os
import numpy as np
import time

class GraphAIGCodeSet(InMemoryDataset):
    """PyTorch Geometric dataset class for loading AIGCodeSet graph data"""
    
    def __init__(self, data_dir: str, split: str='train', suffix: str=''):
        """
        Initialize the GraphAIGCodeSet dataset.
        
        Args:
            data_dir (str): Directory containing the graph files
            split (str): Dataset split ('train', 'val', or 'test')
            suffix (str): Optional suffix for graph files (e.g., 'comments')
        """
        self.data_path = data_dir
        self.split = split
        self.suffix = suffix
        self.graphs = None
        self.type_to_ind = None
        super(GraphAIGCodeSet, self).__init__(root=None)
        
        self.graphs = self._load_graphs()
        self.type_to_ind = self._load_type_mapping()
        
    def _load_graphs(self):
        """Load graph data from disk"""
        # Construct filename with optional suffix
        if self.suffix:
            graph_file = os.path.join(self.data_path, f'{self.split}_graphs_{self.suffix}.pt')
        else:
            graph_file = os.path.join(self.data_path, f'{self.split}_graphs.pt')
        
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        print(f"Loading {self.split} graphs from {graph_file}...")
        file_size = os.path.getsize(graph_file) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.1f} MB")
        start_time = time.time()
        graphs = load(graph_file, weights_only=False)
        load_time = time.time() - start_time
        print(f"Loaded {len(graphs)} graphs in {load_time:.2f} seconds")
        return graphs

    def _load_type_mapping(self):
        """Load node type to index mapping"""
        if self.suffix:
            type_file = os.path.join(self.data_path, f'type_to_ind_{self.suffix}.pt')
        else:
            type_file = os.path.join(self.data_path, 'type_to_ind.pt')
            
        if os.path.exists(type_file):
            print("Loading type mapping...")
            type_to_ind = load(type_file, weights_only=False)
            print(f"Loaded {len(type_to_ind)} type mappings")
            return type_to_ind
        else:
            print(f"Warning: Type mapping file not found: {type_file}")
            return None

    @property
    def processed_file_names(self):
        """List of files that should be present after processing"""
        if self.suffix:
            return [f'{self.split}_graphs_{self.suffix}.pt', f'type_to_ind_{self.suffix}.pt']
        else:
            return [f'{self.split}_graphs.pt', 'type_to_ind.pt']
    
    @property
    def processed_dir(self):
        """Directory where processed files are stored"""
        return self.data_path  
    
    @property
    def num_node_features(self):
        """Number of unique node types (used as node features)"""
        if self.type_to_ind is None:
            return 0
        return len(self.type_to_ind)
    
    @property
    def num_classes(self):
        """Number of classes for classification (2 for human vs AI)"""
        return 2

    def len(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)
    
    def get(self, idx):
        """Get a single graph by index"""
        return self.graphs[idx]
    
    def get_statistics(self):
        """Get dataset statistics"""
        if not self.graphs:
            return {}
            
        graph_sizes = [len(graph.x) for graph in self.graphs]
        edge_counts = [graph.edge_index.size(1) for graph in self.graphs]
        targets = [graph.y.item() for graph in self.graphs]
        
        # Language distribution (if available in metadata)
        languages = []
        for graph in self.graphs:
            if hasattr(graph, 'metadata') and 'language' in graph.metadata:
                languages.append(graph.metadata['language'])
        
        stats = {
            'num_graphs': len(self.graphs),
            'num_node_types': self.num_node_features,
            'graph_sizes': {
                'min': min(graph_sizes),
                'max': max(graph_sizes),
                'mean': np.mean(graph_sizes),
                'std': np.std(graph_sizes)
            },
            'edge_counts': {
                'min': min(edge_counts),
                'max': max(edge_counts),
                'mean': np.mean(edge_counts),
                'std': np.std(edge_counts)
            },
            'target_distribution': {
                'human': targets.count(0),
                'ai': targets.count(1)
            }
        }
        
        if languages:
            from collections import Counter
            stats['language_distribution'] = dict(Counter(languages))
            
        return stats
    
    def print_statistics(self):
        """Print formatted dataset statistics"""
        stats = self.get_statistics()
        
        print(f"\n=== AIGCodeSet {self.split.upper()} Dataset Statistics ===")
        print(f"Number of graphs: {stats['num_graphs']}")
        print(f"Number of node types: {stats['num_node_types']}")
        
        print(f"\nGraph sizes:")
        print(f"  Min: {stats['graph_sizes']['min']}")
        print(f"  Max: {stats['graph_sizes']['max']}")
        print(f"  Mean: {stats['graph_sizes']['mean']:.1f}")
        print(f"  Std: {stats['graph_sizes']['std']:.1f}")
        
        print(f"\nEdge counts:")
        print(f"  Min: {stats['edge_counts']['min']}")
        print(f"  Max: {stats['edge_counts']['max']}")
        print(f"  Mean: {stats['edge_counts']['mean']:.1f}")
        print(f"  Std: {stats['edge_counts']['std']:.1f}")
        
        print(f"\nTarget distribution:")
        print(f"  Human-generated: {stats['target_distribution']['human']}")
        print(f"  AI-generated: {stats['target_distribution']['ai']}")
        
        if 'language_distribution' in stats:
            print(f"\nLanguage distribution:")
            for lang, count in stats['language_distribution'].items():
                print(f"  {lang.capitalize()}: {count}")


if __name__ == '__main__':
    # Example usage
    try:
        # Load standard graphs
        train_dataset = GraphAIGCodeSet('../../data/aigcodeset_graphs/', split='train')
        train_dataset.print_statistics()
        
        print(f"\nSample data object: {train_dataset[0]}")
        if len(train_dataset) > 0:
            sample_graph = train_dataset[0]
            print(f"Sample graph node features shape: {sample_graph.x.shape}")
            print(f"Sample graph edge index shape: {sample_graph.edge_index.shape}")
            print(f"Sample graph target: {sample_graph.y.item()} ({'AI' if sample_graph.y.item() == 1 else 'Human'})")
            
            if hasattr(sample_graph, 'metadata'):
                print(f"Sample metadata: {sample_graph.metadata}")
        
        # Load validation and test sets
        val_dataset = GraphAIGCodeSet('../../data/aigcodeset_graphs/', split='val')
        test_dataset = GraphAIGCodeSet('../../data/aigcodeset_graphs/', split='test')
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Validation: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the graph creation notebook first to generate the graph files.")
    except Exception as e:
        print(f"Unexpected error: {e}")
