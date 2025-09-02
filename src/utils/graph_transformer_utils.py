import torch
import os
import random
import numpy as np
import optuna
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score, Precision, Recall, Specificity, AUROC
from data.dataset.graph_codet import GraphCoDeTM4
from data.dataset.graph_aigcodeset import GraphAIGCodeSet
from models.GraphTransformer import GraphTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, optimizer, epoch, best_vloss, best_vacc, save_path='models/graph_transformer', scheduler=None):
    """Save model state dict, optimizer, scheduler, and architecture information"""
    model_name = getattr(model, 'name', 'GraphTransformer')
    model_save_path = os.path.join(save_path, model_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    checkpoint_filename = f"{model_name}_best.pth"
    checkpoint_filepath = os.path.join(model_save_path, checkpoint_filename)
    
    # Extract model architecture parameters
    model_config = {
        'num_node_features': model.num_node_features,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'num_heads': model.num_heads,
        'num_layers': model.num_layers,
        'pooling_method': model.pooling_method,
        'use_edge_attr': model.use_edge_attr
    }
    
    # Prepare checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'epoch': epoch,
        'best_vloss': best_vloss,
        'best_vacc': best_vacc
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint_data['scheduler_type'] = type(scheduler).__name__
    
    # Save everything in one checkpoint file
    torch.save(checkpoint_data, checkpoint_filepath)
    
    print(f"\nNew best model saved! Val Loss: {best_vloss:.4f}, Val Acc: {best_vacc:.4f}")
    print(f"Model architecture: {model_config}")
    if scheduler is not None:
        print(f"Scheduler: {type(scheduler).__name__}")
    return checkpoint_filepath

def load_model(model, optimizer, save_path='models/graph_transformer', model_name=None, scheduler=None):
    """Load model and optimizer state dictionaries with architecture verification"""
    if model_name is None:
        model_name = getattr(model, 'name', 'GraphTransformer')
    
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
            print(f"Warning: Model vocab size mismatch. Expected: {saved_config['num_node_features']}, Got: {model.num_node_features}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided and available in checkpoint
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Scheduler loaded: {checkpoint.get('scheduler_type', 'Unknown')}")
    elif scheduler is not None and 'scheduler_state_dict' not in checkpoint:
        print("Warning: Scheduler provided but no scheduler state found in checkpoint")
    
    print(f"Model loaded from {checkpoint_filepath}")
    print(f"Best validation loss: {checkpoint['best_vloss']:.4f}")
    print(f"Best validation accuracy: {checkpoint['best_vacc']:.4f}")
    print(f"Saved at epoch: {checkpoint['epoch']}")
    
    return checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']

def create_model_with_optuna_params(num_node_features, storage_url, study_name, model_name, 
                                   use_default_on_failure=True, source_study_name=None):
    """
    Create a GraphTransformer model, optimizer, and scheduler using best hyperparameters from Optuna study.
    
    Args:
        num_node_features: Number of input node features
        storage_url: Optuna storage URL
        study_name: Name of the Optuna study (used as fallback if source_study_name not provided)
        model_name: Name for the model (used for model.name attribute)
        use_default_on_failure: If True, create model with default params if Optuna loading fails
        source_study_name: Optional source study name to load parameters from (overrides study_name)
        
    Returns:
        tuple: (model, optimizer, scheduler, success_flag)
        success_flag: True if loaded from Optuna, False if using defaults
        scheduler: None if not used in optimization or if loading failed
    """
    # Use source study name if provided, otherwise use the default study name
    actual_study_name = source_study_name if source_study_name is not None else study_name
    
    try:
        study = optuna.load_study(storage=storage_url, study_name=actual_study_name)
        best_params = study.best_trial.params
        
        print(f"Loading model with best hyperparameters from Optuna study '{actual_study_name}':")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Create model with best hyperparameters
        model = GraphTransformer(
            num_node_features,
            embedding_dim=best_params["embedding_dim"],
            hidden_dim=best_params["hidden_dim"],
            num_heads=best_params["num_heads"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            pooling_method=best_params["pooling_method"]
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        model.name = model_name
        
        # Create scheduler if it was used in optimization
        scheduler = None
        if best_params.get('use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=5
            )
        
        return model, optimizer, scheduler, True
        
    except Exception as e:
        if use_default_on_failure:
            print(f"Failed to load Optuna parameters: {e}")
            print("Using default model configuration...")
            
            # Create model with default hyperparameters
            model = GraphTransformer(num_node_features).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.name = model_name
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=5
            )
            
            return model, optimizer, scheduler, False
        else:
            raise

def create_model_from_checkpoint(checkpoint_path, model_name=None):
    """
    Create a GraphTransformer model and optimizer from a saved checkpoint file.
    This function reconstructs the exact model architecture from saved configuration.
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        model_name: Optional model name override
        
    Returns:
        tuple: (model, optimizer, scheduler, epoch, best_vloss, best_vacc)
        Note: scheduler will be None if not saved in checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration. "
                        "This checkpoint was saved with an older version of the save function.")
    
    config = checkpoint['model_config']
    print(f"Creating model from saved configuration: {config}")
    
    # Create model with exact same architecture
    model = GraphTransformer(
        num_node_features=config['num_node_features'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        pooling_method=config['pooling_method'],
        use_edge_attr=config.get('use_edge_attr', False)
    ).to(DEVICE)
    
    # Create optimizer (we don't save lr in config, so use a default)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Load state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Create and load scheduler if available in checkpoint
    scheduler = None
    if 'scheduler_state_dict' in checkpoint:
        # For now, assume it's ReduceLROnPlateau - you might want to save scheduler config too
        scheduler_type = checkpoint.get('scheduler_type', 'ReduceLROnPlateau')
        print(f"Creating scheduler: {scheduler_type}")
        
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=5
            )
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print(f"Warning: Unknown scheduler type {scheduler_type}, skipping scheduler loading")
    
    # Set model name
    if model_name:
        model.name = model_name
    elif hasattr(model, 'name'):
        pass  # Keep existing name
    else:
        model.name = 'GraphTransformer'
    
    print(f"Model created and loaded from {checkpoint_path}")
    print(f"Best validation loss: {checkpoint['best_vloss']:.4f}")
    print(f"Best validation accuracy: {checkpoint['best_vacc']:.4f}")
    print(f"Saved at epoch: {checkpoint['epoch']}")
    if scheduler is not None:
        print(f"Scheduler loaded: {scheduler_type}")
    
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']

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
            
            if i % 10 == 0:
                eval_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{results.get("Acc", 0):.3f}'
                })

    avg_loss = eval_loss / len(dataloader)
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
            
            if i % 10 == 0:
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{results.get("Acc", 0):.3f}'
                })

    avg_loss = valid_loss / len(dataloader)
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
            batch_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{results.get("Acc", 0):.3f}'
            })

    final_metrics = {name: metric.compute().item() for name, metric in metrics.items()}
    return avg_loss, final_metrics

def train(model, optimizer, criterion, train_dataloader, val_dataloader=None, scheduler=None, 
          num_epochs=10, metrics={'Accuracy': Accuracy(task='binary')}, save_path='models/graph_transformer', 
          initial_best_vloss=None, initial_best_vacc=None, writer=None, start_epoch=0):
    model.train()
    best_vloss = initial_best_vloss if initial_best_vloss is not None else float('inf')
    best_vacc = initial_best_vacc if initial_best_vacc is not None else 0.0
    epoch_pbar = tqdm(range(num_epochs), desc="Training model...", unit=' epoch')
    
    for epoch in epoch_pbar:
        current_epoch = start_epoch + epoch
        
        avg_train_loss, train_results = train_epoch(
            model, optimizer, criterion, train_dataloader, metrics, epoch
        )

        postfix = {
            'Train Loss': f'{avg_train_loss:.4f}',
            'Train Acc': f'{train_results.get("Acc", 0):.3f}'
        }

        # Log training metrics to tensorboard
        if writer is not None:
            writer.add_scalar('train/loss', avg_train_loss, current_epoch)
            for metric_name, metric_value in train_results.items():
                writer.add_scalar(f'train/{metric_name.lower()}', metric_value, current_epoch)

        if val_dataloader:
            avg_val_loss, val_results = validate(model, val_dataloader, criterion, metrics)
            
            postfix.update({
                'Val Loss': f'{avg_val_loss:.4f}',
                'Val Acc': f'{val_results.get("Acc", 0):.3f}'
            })

            # Log validation metrics to tensorboard
            if writer is not None:
                writer.add_scalar('val/loss', avg_val_loss, current_epoch)
                for metric_name, metric_value in val_results.items():
                    writer.add_scalar(f'val/{metric_name.lower()}', metric_value, current_epoch)

            # Save best model
            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                best_vacc = val_results.get('Acc', 0)
                save_model(model, optimizer, current_epoch, best_vloss, best_vacc, save_path, scheduler)

            # Scheduler step
            if scheduler is not None:
                scheduler.step(avg_val_loss)

        epoch_pbar.set_postfix(postfix)

def load_single_data(data_dir='data/codet_graphs', split='train', shuffle=True, batch_size=128, suffix='', dataset='codet'):
    """Load a single data split and return a DataLoader."""
    if dataset == 'codet':
        data = GraphCoDeTM4(data_dir, split=split, suffix=suffix)
    elif dataset == 'aigcodeset':
        data = GraphAIGCodeSet(data_dir, split=split, suffix=suffix)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'codet' or 'aigcodeset'")
    
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    loader.dataset.num_node_features = data.num_node_features
    del data
    return loader

def load_multiple_data(data_dir='data/codet_graphs', splits=['train', 'val'], shuffles=None, batch_size=128, suffix='', dataset='codet'):
    """Load multiple data splits and return a list of DataLoaders."""
    if shuffles is None:
        shuffles = [True] * len(splits)
    
    # Ensure shuffles matches splits length
    if len(shuffles) != len(splits):
        if len(shuffles) < len(splits):
            shuffles.extend([True] * (len(splits) - len(shuffles)))
        else:
            shuffles = shuffles[:len(splits)]
    
    loaders = []
    for split, shuffle in zip(splits, shuffles):
        loader = load_single_data(data_dir, split, shuffle, batch_size, suffix, dataset)
        loaders.append(loader)
    
    return loaders

def load_data(data_dir='data/codet_graphs', splits=['train', 'val'], shuffles=None, batch_size=128, dataset='codet'):
    """
    Convenience function for backward compatibility.
    Consider using load_single_data() or load_multiple_data() directly.
    """
    if isinstance(splits, str):
        shuffle = shuffles if isinstance(shuffles, bool) else True
        return load_single_data(data_dir, splits, shuffle, batch_size, '', dataset)
    else:
        return load_multiple_data(data_dir, splits, shuffles, batch_size, '', dataset)

def get_metrics():
    metrics = {
        'Acc': Accuracy(task='binary').to(DEVICE),
        'Prec': Precision(task='binary').to(DEVICE),
        'Rec': Recall(task='binary').to(DEVICE),
        'Spec': Specificity(task='binary').to(DEVICE),
        'AUROC': AUROC(task='binary').to(DEVICE),
        'F1': F1Score(task='binary').to(DEVICE)
    }
    return metrics

def create_objective(train_dataloader, val_dataloader, num_epochs, writer=None):
    def objective(trial: optuna.Trial):
        # Suggest hyperparameters
        embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256, 512])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        pooling_method = trial.suggest_categorical('pooling_method', ['mean', 'max', 'add'])
        use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
        
        # Ensure hidden_dim is compatible with num_heads
        if hidden_dim % num_heads != 0:
            # Adjust hidden_dim to be divisible by num_heads
            hidden_dim = (hidden_dim // num_heads) * num_heads
            if hidden_dim == 0:
                hidden_dim = num_heads

        # Create model
        model = GraphTransformer(
            num_node_features=train_dataloader.dataset.num_node_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pooling_method=pooling_method
        ).to(DEVICE)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Create scheduler if suggested
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=3
            )

        # Training setup
        criterion = torch.nn.BCEWithLogitsLoss()
        metrics = get_metrics()
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(num_epochs):
            # Train
            avg_train_loss, train_results = train_epoch(
                model, optimizer, criterion, train_dataloader, metrics, epoch
            )
            
            # Validate
            avg_val_loss, val_results = validate(model, val_dataloader, criterion, metrics)
            
            # Update best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step(avg_val_loss)
            
            # Report intermediate value for pruning
            trial.report(avg_val_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_loss

    return objective
