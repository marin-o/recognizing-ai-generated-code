import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.GCN import GCN
from data.dataset import GraphCoDeTM4
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')



if __name__ == '__main__':
    data = GraphCoDeTM4('data/codet_graphs', split='train')
    train_loader = DataLoader(data, batch_size=32, shuffle=True)
    model = GCN(data.num_node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in tqdm(range(10), desc="Training model...", unit='epoch'):
        running_loss = 0.0
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for i, data in enumerate(epoch_pbar):
            x = data.x.to(DEVICE)
            edge_index = data.edge_index.to(DEVICE)
            batch = data.batch.to(DEVICE)
            y = data.y.to(DEVICE).float()
            out = model(x, edge_index, batch)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            epoch_pbar.set_postfix({
                'Loss':f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
            })
                                  
