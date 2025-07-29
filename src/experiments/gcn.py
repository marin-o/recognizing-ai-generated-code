import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import numpy as np
from models.GCN import GCN
from data.dataset import GraphCoDeTM4
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC

DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def save_model(model, optimizer, epoch, best_vloss, best_vacc, save_path='models/gnn'):
    """Save model and optimizer state dictionaries"""
    os.makedirs(save_path, exist_ok=True)
    
    model_name = getattr(model, 'name', 'GCN')
    model_filename = f"{model_name}_best.pth"
    model_filepath = os.path.join(save_path, model_filename)
    optimizer_filename = f"{model_name}_optimizer_best.pth"
    optimizer_filepath = os.path.join(save_path, optimizer_filename)
    
    torch.save(model.state_dict(), model_filepath)
    
    # Save optimizer state and training metadata
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_vloss': best_vloss,
        'best_vacc': best_vacc
    }, optimizer_filepath)
    
    print(f"\nNew best model saved! Val Loss: {best_vloss:.4f}, Val Acc: {best_vacc:.4f}")
    return model_filepath, optimizer_filepath

def load_model(model, optimizer, save_path='models/gnn', model_name=None):
    """Load model and optimizer state dictionaries"""
    if model_name is None:
        model_name = getattr(model, 'name', 'GCN')
    
    model_filename = f"{model_name}_best.pth"
    model_filepath = os.path.join(save_path, model_filename)
    optimizer_filename = f"{model_name}_optimizer_best.pth"
    optimizer_filepath = os.path.join(save_path, optimizer_filename)
    
    # Load model state
    model.load_state_dict(torch.load(model_filepath, map_location=DEVICE))
    
    # Load optimizer state and metadata
    checkpoint = torch.load(optimizer_filepath, map_location=DEVICE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {model_filepath}")
    print(f"Best validation loss: {checkpoint['best_vloss']:.4f}")
    print(f"Best validation accuracy: {checkpoint['best_vacc']:.4f}")
    print(f"Saved at epoch: {checkpoint['epoch']}")
    
    return checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_batch(model, data, criterion, metrics):
    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    batch = data.batch.to(DEVICE)
    y = data.y.to(DEVICE).float()

    out = model(x, edge_index, batch)
    loss = criterion(out.squeeze(), y)

    results = {}
    for name, metric in metrics.items():
        metric.update(out.squeeze(), y.int())
        results[name] = metric.compute()
    return loss, results, out

def evaluate(model, dataloader, criterion, metrics):
    model.eval()
    eval_loss = 0.0

    for metric in metrics.values():
        metric.reset()

    eval_pbar = tqdm(dataloader, desc="Evaluating...", unit=' batch', leave=False)

    with torch.no_grad():
        for i, data in enumerate(eval_pbar):
            loss, results, out = compute_batch(model, data, criterion, metrics)

            eval_loss += loss.item()
            avg_loss = eval_loss / (i + 1)

            if i % 2 == 0:    
                postfix = {
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{results.get("Acc", 0):.3f}'
                }

                eval_pbar.set_postfix(postfix)

    final_metrics = {name: metric.compute().item() for name, metric in metrics.items()}
    return avg_loss, final_metrics


def validate(model, dataloader, criterion, metrics):
    model.eval()
    valid_loss = 0.0

    for metric in metrics.values():
        metric.reset()

    val_pbar = tqdm(dataloader, desc="Validating model...", unit=' batch', leave=False)
    with torch.no_grad():
        for i, data in enumerate(val_pbar):
            loss, results, out = compute_batch(model, data, criterion, metrics)

            valid_loss += loss.item()
            avg_loss = valid_loss / (i + 1)

            if i % 2 == 0:
                postfix = {
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{results.get("Acc", 0):.3f}'
                }

                val_pbar.set_postfix(postfix)

    final_metrics = {name: metric.compute().item() for name, metric in metrics.items()}

    return avg_loss, final_metrics

def train_epoch(model, optimizer, criterion, dataloader, metrics, epoch=None):
    model.train()
    running_loss = 0.0

    for metric in metrics.values():
        metric.reset()

    desc = f"Epoch {epoch + 1}" if epoch else "Training..."
    batch_pbar = tqdm(dataloader, desc=desc, unit=' batch', leave=False)

    for i, data in enumerate(batch_pbar):
        loss, results, out = compute_batch(model, data, criterion, metrics)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)

        if i % 2 == 0:
            postfix = {
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{results.get("Acc", 0):.3f}'
            }

            batch_pbar.set_postfix(postfix)

    final_metrics = {name: metric.compute().item() for name, metric in metrics.items()}
    return avg_loss, final_metrics

def train(model, optimizer, criterion, train_dataloader, val_dataloader=None, num_epochs=10, metrics={'Accuracy': Accuracy(task='binary')}, save_path='models/gnn'):
    model.train()
    best_vloss = float('inf')
    best_vacc = 0.0
    epoch_pbar = tqdm(range(num_epochs), desc="Training model...", unit=' epoch')
    for epoch in epoch_pbar:
        
        avg_train_loss, train_results = train_epoch(
            model, optimizer, criterion, train_dataloader, metrics, epoch
        )

        postfix = {
            'Train Loss': f'{avg_train_loss:.4f}',
            'Train Acc': f'{train_results.get("Acc", 0):.3f}'
        }

        if val_dataloader:
            avg_val_loss, results = validate(model, val_dataloader, criterion, metrics)
            postfix.update({
                'Val Loss': f'{avg_val_loss:.4f}',
                'Val Acc': f'{results.get("Acc", 0):.3f}',
                'Val AUROC': f'{results.get("AUROC", 0):.3f}'
            })

            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                best_vacc = results['Acc']
                
                save_model(model, optimizer, epoch, best_vloss, best_vacc, save_path)
                

        epoch_pbar.set_postfix(postfix)

MODEL_NAME = 'baseline_gcn'

if __name__ == '__main__':
    set_seed(872002)

    train_data = GraphCoDeTM4('data/codet_graphs', split='train')
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_data = GraphCoDeTM4('data/codet_graphs', split='val')

    # Shuffling because classes are aligned as [class_0, class_0, ..., class_0, class_1, class_1,...]
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True) 
    
    model = GCN(train_data.num_node_features, hidden_dim=256, sage=False).to(DEVICE)
    model.name = 'baseline_gcn'

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    metrics = {
        'Acc': Accuracy(task='binary').to(DEVICE),
        'Prec': Precision(task='binary').to(DEVICE),
        'Rec': Recall(task='binary').to(DEVICE),
        'Spec': Specificity(task='binary').to(DEVICE),
        'AUROC': AUROC(task='binary').to(DEVICE)
    }

    train(model=model, optimizer=optimizer, criterion=criterion, train_dataloader=train_loader, val_dataloader=val_loader, metrics=metrics, num_epochs=30)
    
    # Clean up RAM to make room for the evaluation data
    # Not necessary if you have about 14GB to dedicate just to the training environment
    del train_data, train_loader, val_data, val_loader
    import gc
    gc.collect()

    test_data = GraphCoDeTM4('data/codet_graphs', split='test')
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)
    
    epoch, best_vloss, best_vacc = load_model(model, optimizer, save_path='models/gnn')
    test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS:")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    for metric_name, metric_value in test_metrics.items():
        print(f"Test {metric_name}: {metric_value:.4f}")
    print("="*50)
                                  
