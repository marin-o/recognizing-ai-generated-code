import os
import random
import numpy as np
import optuna
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC
from data.dataset import GraphCoDeTM4
from models.GCN import GCN

DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def save_model(model, optimizer, epoch, best_vloss, best_vacc, save_path='models/gnn'):
    """Save model state dict, optimizer, and architecture information"""
    model_name = getattr(model, 'name', 'GCN')
    model_save_path = os.path.join(save_path, model_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    checkpoint_filename = f"{model_name}_best.pth"
    checkpoint_filepath = os.path.join(model_save_path, checkpoint_filename)
    
    # Extract model architecture parameters
    model_config = {
        'num_node_features': model.num_node_features,
        'embedding_dim': model.embedding_dim,
        'hidden_dim_1': model.conv1.out_channels,
        'hidden_dim_2': model.conv2.out_channels,
        'sage': isinstance(model.conv1, SAGEConv)
    }
    
    # Save everything in one checkpoint file
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'epoch': epoch,
        'best_vloss': best_vloss,
        'best_vacc': best_vacc
    }, checkpoint_filepath)
    
    print(f"\nNew best model saved! Val Loss: {best_vloss:.4f}, Val Acc: {best_vacc:.4f}")
    print(f"Model architecture: {model_config}")
    return checkpoint_filepath

def load_model(model, optimizer, save_path='models/gnn', model_name=None):
    """Load model and optimizer state dictionaries with architecture verification"""
    if model_name is None:
        model_name = getattr(model, 'name', 'GCN')
    
    model_save_path = os.path.join(save_path, model_name)
    checkpoint_filename = f"{model_name}_best.pth"
    checkpoint_filepath = os.path.join(model_save_path, checkpoint_filename)
    
    checkpoint = torch.load(checkpoint_filepath, map_location=DEVICE)
    
    # Verify architecture compatibility if model_config is available
    if 'model_config' in checkpoint:
        saved_config = checkpoint['model_config']
        print(f"Loaded model architecture: {saved_config}")
        
        # Basic compatibility checks
        if hasattr(model, 'num_node_features') and model.num_node_features != saved_config['num_node_features']:
            raise RuntimeError(f"Model architecture mismatch: expected {model.num_node_features} node features, "
                             f"but saved model has {saved_config['num_node_features']}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {checkpoint_filepath}")
    print(f"Best validation loss: {checkpoint['best_vloss']:.4f}")
    print(f"Best validation accuracy: {checkpoint['best_vacc']:.4f}")
    print(f"Saved at epoch: {checkpoint['epoch']}")
    
    return checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']

def create_model_with_optuna_params(num_node_features, storage_url, study_name, model_name, use_default_on_failure=True):
    """
    Create a GCN model and optimizer using best hyperparameters from Optuna study.
    
    Args:
        num_node_features: Number of input node features
        storage_url: Optuna storage URL
        study_name: Name of the Optuna study
        model_name: Name for the model (used for model.name attribute)
        use_default_on_failure: If True, create model with default params if Optuna loading fails
        
    Returns:
        tuple: (model, optimizer, success_flag)
        success_flag: True if loaded from Optuna, False if using defaults
    """
    try:
        study = optuna.load_study(storage=storage_url, study_name=study_name)
        best_params = study.best_trial.params
        
        print("Loading model with best hyperparameters from Optuna:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Create model with best hyperparameters
        model = GCN(
            num_node_features,
            hidden_dim_1=best_params["hidden_dim_1"],
            hidden_dim_2=best_params["hidden_dim_2"],
            embedding_dim=best_params["embedding_dim"],
            sage=best_params["sage"],
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        model.name = model_name
        
        return model, optimizer, True
        
    except Exception as e:
        if use_default_on_failure:
            print(f"Warning: Could not load Optuna study ({e})")
            print("Using default model parameters...")
            
            # Fallback to default parameters
            model = GCN(num_node_features).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.name = model_name
            
            return model, optimizer, False
        else:
            raise e

def create_model_from_checkpoint(checkpoint_path, model_name=None):
    """
    Create a GCN model and optimizer from a saved checkpoint file.
    This function reconstructs the exact model architecture from saved configuration.
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        model_name: Optional model name override
        
    Returns:
        tuple: (model, optimizer, epoch, best_vloss, best_vacc)
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration. "
                        "This checkpoint was saved with an older version of the save function.")
    
    config = checkpoint['model_config']
    print(f"Creating model from saved configuration: {config}")
    
    # Create model with exact same architecture
    model = GCN(
        num_node_features=config['num_node_features'],
        embedding_dim=config['embedding_dim'],
        hidden_dim_1=config['hidden_dim_1'],
        hidden_dim_2=config['hidden_dim_2'],
        sage=config['sage']
    ).to(DEVICE)
    
    # Create optimizer (we don't save lr in config, so use a default)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Set model name
    if model_name:
        model.name = model_name
    elif hasattr(model, 'name'):
        pass  # Keep existing name
    else:
        model.name = 'GCN'
    
    print(f"Model created and loaded from {checkpoint_path}")
    print(f"Best validation loss: {checkpoint['best_vloss']:.4f}")
    print(f"Best validation accuracy: {checkpoint['best_vacc']:.4f}")
    print(f"Saved at epoch: {checkpoint['epoch']}")
    
    return model, optimizer, checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']

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

def train(model, optimizer, criterion, train_dataloader, val_dataloader=None, scheduler=None, num_epochs=10, metrics={'Accuracy': Accuracy(task='binary')}, save_path='models/gnn'):
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
            avg_val_loss, val_metrics = validate(model, val_dataloader, criterion, metrics)
            val_acc = val_metrics.get("Acc", 0)
            val_auroc = val_metrics.get("AUROC", 0)
            postfix.update({
                'Val Loss': f'{avg_val_loss:.4f}',
                'Val Acc': f'{val_acc:.3f}',
                'Val AUROC': f'{val_auroc:.3f}'
            })

            if scheduler:
                scheduler.step(avg_val_loss)

            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                best_vacc = val_acc
                
                save_model(model, optimizer, epoch, best_vloss, best_vacc, save_path)

                

        epoch_pbar.set_postfix(postfix)

def load_single_data(data_dir='data/codet_graphs', split='train', shuffle=True, batch_size=128):
    """Load a single data split and return a DataLoader."""
    data = GraphCoDeTM4(data_dir, split=split)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    loader.num_node_features = data.num_node_features
    del data
    return loader

def load_multiple_data(data_dir='data/codet_graphs', splits=['train', 'val'], shuffles=None, batch_size=128):
    """Load multiple data splits and return a list of DataLoaders."""
    if shuffles is None:
        shuffles = [True] * len(splits)
    
    # Ensure shuffles matches splits length
    if len(shuffles) != len(splits):
        if len(shuffles) < len(splits):
            shuffles = shuffles + [shuffles[-1]] * (len(splits) - len(shuffles))
        else:
            shuffles = shuffles[:len(splits)]
    
    loaders = []
    for split, shuffle in zip(splits, shuffles):
        loader = load_single_data(data_dir, split, shuffle, batch_size)
        loaders.append(loader)
    
    return loaders

def load_data(data_dir='data/codet_graphs', splits=['train', 'val'], shuffles=None, batch_size=128):
    """
    Convenience function for backward compatibility.
    Consider using load_single_data() or load_multiple_data() directly.
    """
    if isinstance(splits, str):
        shuffle = shuffles if isinstance(shuffles, bool) else True
        return load_single_data(data_dir, splits, shuffle, batch_size)
    else:
        return load_multiple_data(data_dir, splits, shuffles, batch_size)

def get_metrics():
    metrics = {
        'Acc': Accuracy(task='binary').to(DEVICE),
        'Prec': Precision(task='binary').to(DEVICE),
        'Rec': Recall(task='binary').to(DEVICE),
        'Spec': Specificity(task='binary').to(DEVICE),
        'AUROC': AUROC(task='binary').to(DEVICE)
    }
    return metrics

def create_objective(train_dataloader, val_dataloader, num_epochs):
    def objective(trial: optuna.Trial):
        hidden_dim_1 = trial.suggest_int("hidden_dim_1", 128, 512, step=16)
        hidden_dim_2 = trial.suggest_int("hidden_dim_2", 128, 512, step=16)
        embedding_dim = trial.suggest_int("embedding_dim", 64, 512, step=16)
        sage = trial.suggest_categorical("sage", [True, False])
        lr = trial.suggest_float('lr', low=0.0001, high=0.01, log=True)
        
        model = GCN(
            train_dataloader.num_node_features,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            embedding_dim=embedding_dim,
            sage=sage,
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        metrics = get_metrics()

        epoch_pbar = tqdm(range(num_epochs), desc=f"Training trial #{trial.number}", unit=' epoch')
        for epoch in epoch_pbar:
            train_loss, train_metrics = train_epoch(
                model, optimizer, criterion, train_dataloader, metrics, epoch
            )

            postfix = {
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_metrics.get("Acc", 0):.3f}'
            }

            val_loss, val_metrics = validate(model, val_dataloader, criterion, metrics)
            val_acc = val_metrics.get("Acc", 0)
            val_auroc = val_metrics.get("AUROC", 0)
            postfix.update({
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_acc:.3f}',
                'Val AUROC': f'{val_auroc:.3f}'
            })

            trial.report(val_loss, epoch)
            # trial.report(val_acc, epoch)
            # trial.report(val_rec, epoch)

            epoch_pbar.set_postfix(postfix)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    return objective
