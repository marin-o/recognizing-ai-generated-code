from torch_geometric.data import Data, InMemoryDataset
from torch import load
import os
import numpy as np

class GraphCoDeTM4(InMemoryDataset):
    def __init__(self, data_dir: str, split: str='train'):
        self.data_path = data_dir
        self.split = split
        self.graphs = None
        self.type_to_ind = None
        super(GraphCoDeTM4, self).__init__(root=None)
        
        graph_file = os.path.join(data_dir, f'{split}_graphs.pt')
        self.graphs = load(graph_file, weights_only=False)

        type_file = os.path.join(data_dir, 'type_to_ind.pt')
        if os.path.exists(type_file):
            self.type_to_ind = load(type_file, weights_only=False)
        
    @property
    def processed_file_names(self):
        return [f'{self.split}_graphs.pt', 'type_to_ind.pt']
    
    @property
    def processed_dir(self):
        return self.data_path  

    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx]
    
if __name__ == '__main__':
    train_dataset = GraphCoDeTM4('data/codet_graphs/', split='train')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Sample data object: {train_dataset[0]}")
    print(f"Sample data object.x shape: {train_dataset[0].x.shape}")

    graph_sizes = [len(graph.x) for graph in train_dataset]
    print(f"Min: {min(graph_sizes)}, Max: {max(graph_sizes)}")
    print(f"Mean: {np.mean(graph_sizes):.1f}, Std: {np.std(graph_sizes):.1f}")

