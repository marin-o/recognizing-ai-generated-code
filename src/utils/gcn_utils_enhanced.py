import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import optuna
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from models.GCN import GCN, GCNWithPositionalEncoding
from data.dataset.graph_codet_enhanced import GraphCoDeTM4Enhanced
from data.dataset.graph_aigcodeset import GraphAIGCodeSet
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC, F1Score
from typing import Dict, Any, Optional, Tuple, Union
from tqdm import tqdm

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed: int = 872002):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

def get_metrics():
    """Initialize evaluation metrics."""
    return {
        'accuracy': Accuracy(task='binary').to(DEVICE),
        'precision': Precision(task='binary').to(DEVICE),
        'recall': Recall(task='binary').to(DEVICE),
        'specificity': Specificity(task='binary').to(DEVICE),
        'auroc': AUROC(task='binary').to(DEVICE),
        'f1': F1Score(task='binary').to(DEVICE),
    }

def load_enhanced_data(data_dir: str, split: str, batch_size: int = 128, 
                      suffix: str = '', dataset: str = 'codet', 
                      shuffle: bool = None) -> DataLoader:
    """Load enhanced dataset with positional encodings."""
    if shuffle is None:
        shuffle = split == 'train'
    
    print(f"Loading {split} dataset...")
    
    if dataset == 'codet':
        dataset_obj = GraphCoDeTM4Enhanced(root=data_dir, split=split, suffix=suffix)
    elif dataset == 'aigcodeset':
        dataset_obj = GraphAIGCodeSet(root=data_dir, split=split, suffix=suffix)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if hasattr(dataset_obj, 'print_dataset_info'):
        dataset_obj.print_dataset_info()
    
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True if str(DEVICE) == 'cuda' else False,
        num_workers=2
    )
    
    # Add dataset properties to dataloader for convenience
    dataloader.num_node_features = dataset_obj.num_node_features
    dataloader.dataset_obj = dataset_obj
    
    return dataloader

def load_multiple_enhanced_data(data_dir: str, batch_size: int = 128, 
                               suffix: str = '', dataset: str = 'codet') -> Tuple[DataLoader, DataLoader]:
    """Load training and validation datasets."""
    train_loader = load_enhanced_data(data_dir, 'train', batch_size, suffix, dataset, shuffle=True)
    val_loader = load_enhanced_data(data_dir, 'val', batch_size, suffix, dataset, shuffle=False)
    return train_loader, val_loader

def create_enhanced_model_from_config(num_node_features: int, config: Dict[str, Any], 
                                     model_name: str, use_positional_encoding: bool = True) -> nn.Module:
    """Create an enhanced model with optional positional encodings."""
    
    if use_positional_encoding:
        model = GCNWithPositionalEncoding(
            num_node_features=num_node_features,
            max_depth=config.get('max_depth', 50),
            max_child_index=config.get('max_child_index', 20),
            embedding_dim=config.get('embedding_dim', 256),
            hidden_dim_1=config.get('hidden_dim_1', 128),
            hidden_dim_2=config.get('hidden_dim_2', 128),
            sage=config.get('sage', False),
            use_two_layer_classifier=config.get('use_two_layer_classifier', False),
            dropout=config.get('dropout', 0.1),
            pooling_method=config.get('pooling_method', 'mean'),
            depth_embedding_dim=config.get('depth_embedding_dim', 32),
            child_embedding_dim=config.get('child_embedding_dim', 32)
        ).to(DEVICE)
    else:
        model = GCN(
            num_node_features=num_node_features,
            embedding_dim=config.get('embedding_dim', 256),
            hidden_dim_1=config.get('hidden_dim_1', 128),
            hidden_dim_2=config.get('hidden_dim_2', 128),
            sage=config.get('sage', False),
            use_two_layer_classifier=config.get('use_two_layer_classifier', False)
        ).to(DEVICE)
    
    model.name = model_name
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model

def create_enhanced_model_with_dataset_config(dataloader: DataLoader, model_name: str, 
                                             custom_config: Optional[Dict[str, Any]] = None,
                                             use_positional_encoding: bool = True) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Create model using dataset configuration for positional encodings."""
    
    config = {
        'embedding_dim': 256,
        'hidden_dim_1': 128,
        'hidden_dim_2': 128,
        'sage': False,
        'use_two_layer_classifier': False,
        'dropout': 0.1,
        'pooling_method': 'mean',
        'learning_rate': 0.001
    }
    
    # Update with custom config if provided
    if custom_config:
        config.update(custom_config)
    
    # Get positional encoding config from dataset if available
    if use_positional_encoding and hasattr(dataloader, 'dataset_obj'):
        dataset_obj = dataloader.dataset_obj
        
        if hasattr(dataset_obj, 'get_max_depth'):
            config['max_depth'] = dataset_obj.get_max_depth()
        if hasattr(dataset_obj, 'get_max_child_index'):
            config['max_child_index'] = dataset_obj.get_max_child_index()
        
        # Set default positional encoding dimensions if not specified
        config.setdefault('depth_embedding_dim', 32)
        config.setdefault('child_embedding_dim', 32)
        
        print(f"Positional encoding config: max_depth={config.get('max_depth')}, "
              f"max_child_index={config.get('max_child_index')}, "
              f"depth_embed_dim={config.get('depth_embedding_dim')}, "
              f"child_embed_dim={config.get('child_embedding_dim')}")
    
    # Create model
    model = create_enhanced_model_from_config(
        num_node_features=dataloader.num_node_features,
        config=config,
        model_name=model_name,
        use_positional_encoding=use_positional_encoding
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    
    return model, optimizer, scheduler

def enhanced_train(model: nn.Module, optimizer: torch.optim.Optimizer, 
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                  criterion: nn.Module, train_dataloader: DataLoader, 
                  val_dataloader: DataLoader, metrics: Dict[str, Any], 
                  num_epochs: int = 50, initial_best_vloss: float = float('inf'), 
                  initial_best_vacc: float = 0.0, writer=None, start_epoch: int = 0,
                  use_positional_encoding: bool = True):
    """Enhanced training function with positional encoding support."""
    
    best_vloss = initial_best_vloss
    best_vacc = initial_best_vacc
    
    # Check if dataset has positional encodings
    has_pos_encodings = False
    if hasattr(train_dataloader, 'dataset_obj') and hasattr(train_dataloader.dataset_obj, 'has_positional_encodings'):
        has_pos_encodings = train_dataloader.dataset_obj.has_positional_encodings()
    
    print(f"Training with positional encodings: {use_positional_encoding and has_pos_encodings}")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batch_count = 0
        
        # Reset metrics
        for metric in metrics.values():
            metric.reset()
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}")
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass with optional positional encodings
            if use_positional_encoding and has_pos_encodings and hasattr(batch, 'node_depth'):
                output = model(batch.x, batch.edge_index, batch.batch, 
                             batch.node_depth, batch.child_index)
            else:
                output = model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(output.squeeze(), batch.y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batch_count += 1
            
            # Update metrics
            predictions = torch.sigmoid(output.squeeze())
            for metric in metrics.values():
                metric.update(predictions, batch.y)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        avg_train_loss = train_loss / train_batch_count
        train_results = {name: metric.compute().item() for name, metric in metrics.items()}
        
        # Validation phase
        val_loss, val_results = enhanced_evaluate(
            model, val_dataloader, criterion, metrics, use_positional_encoding, has_pos_encodings
        )
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Logging
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_results['accuracy']:.4f}")
        
        if writer is not None:
            writer.add_scalar("train/loss", avg_train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            for name, value in train_results.items():
                writer.add_scalar(f"train/{name}", value, epoch)
            for name, value in val_results.items():
                writer.add_scalar(f"val/{name}", value, epoch)
        
        # Save best model
        if val_loss < best_vloss or (val_loss == best_vloss and val_results['accuracy'] > best_vacc):
            best_vloss = val_loss
            best_vacc = val_results['accuracy']
            save_enhanced_model(
                model, optimizer, scheduler, epoch, best_vloss, best_vacc,
                use_positional_encoding=use_positional_encoding
            )

def enhanced_evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                     metrics: Dict[str, Any], use_positional_encoding: bool = True,
                     has_pos_encodings: bool = True) -> Tuple[float, Dict[str, float]]:
    """Enhanced evaluation function with positional encoding support."""
    
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    # Reset metrics
    for metric in metrics.values():
        metric.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            
            # Forward pass with optional positional encodings
            if use_positional_encoding and has_pos_encodings and hasattr(batch, 'node_depth'):
                output = model(batch.x, batch.edge_index, batch.batch, 
                             batch.node_depth, batch.child_index)
            else:
                output = model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(output.squeeze(), batch.y.float())
            total_loss += loss.item()
            batch_count += 1
            
            # Update metrics
            predictions = torch.sigmoid(output.squeeze())
            for metric in metrics.values():
                metric.update(predictions, batch.y)
    
    avg_loss = total_loss / batch_count
    results = {name: metric.compute().item() for name, metric in metrics.items()}
    
    return avg_loss, results

def save_enhanced_model(model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int, best_vloss: float, best_vacc: float,
                       save_path: str = "models/gcn_enhanced",
                       use_positional_encoding: bool = True):
    """Save enhanced model with architecture information."""
    
    model_name = getattr(model, 'name', 'gcn_enhanced')
    model_save_path = os.path.join(save_path, model_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    checkpoint_filename = f"{model_name}_best.pth"
    checkpoint_filepath = os.path.join(model_save_path, checkpoint_filename)
    
    # Extract model architecture parameters
    if isinstance(model, GCNWithPositionalEncoding):
        model_config = {
            'model_type': 'GCNWithPositionalEncoding',
            'num_node_features': model.num_node_features,
            'max_depth': model.max_depth,
            'max_child_index': model.max_child_index,
            'embedding_dim': model.embedding_dim,
            'hidden_dim_1': model.hidden_dim_1,
            'hidden_dim_2': model.hidden_dim_2,
            'sage': model.sage,
            'use_two_layer_classifier': model.use_two_layer_classifier,
            'dropout': model.dropout_rate,
            'pooling_method': model.pooling_method,
            'depth_embedding_dim': model.depth_embedding_dim,
            'child_embedding_dim': model.child_embedding_dim
        }
    else:
        model_config = {
            'model_type': 'GCN',
            'num_node_features': model.num_node_features,
            'embedding_dim': model.embedding_dim,
            'hidden_dim_1': model.conv1.out_channels,
            'hidden_dim_2': model.conv2.out_channels,
            'sage': isinstance(model.conv1, SAGEConv),
            'use_two_layer_classifier': model.use_two_layer_classifier
        }
    
    # Prepare checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'epoch': epoch,
        'best_vloss': best_vloss,
        'best_vacc': best_vacc,
        'use_positional_encoding': use_positional_encoding
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save everything in one checkpoint file
    torch.save(checkpoint_data, checkpoint_filepath)
    
    print(f"\nNew best enhanced model saved! Val Loss: {best_vloss:.4f}, Val Acc: {best_vacc:.4f}")
    print(f"Model architecture: {model_config}")
    return checkpoint_filepath

def load_enhanced_model(model: nn.Module, optimizer: torch.optim.Optimizer,
                       save_path: str = "models/gcn_enhanced",
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[int, float, float]:
    """Load enhanced model and optimizer state dictionaries."""
    
    model_name = getattr(model, 'name', 'gcn_enhanced')
    model_save_path = os.path.join(save_path, model_name)
    checkpoint_filename = f"{model_name}_best.pth"
    checkpoint_filepath = os.path.join(model_save_path, checkpoint_filename)
    
    checkpoint = torch.load(checkpoint_filepath, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided and available in checkpoint
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Enhanced model loaded from {checkpoint_filepath}")
    print(f"Best validation loss: {checkpoint['best_vloss']:.4f}")
    print(f"Best validation accuracy: {checkpoint['best_vacc']:.4f}")
    print(f"Saved at epoch: {checkpoint['epoch']}")
    
    return checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']

def create_enhanced_model_from_checkpoint(checkpoint_path: str, model_name: str) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler], int, float, float]:
    """Create enhanced model from checkpoint file."""
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration information.")
    
    config = checkpoint['model_config']
    use_pos_encoding = checkpoint.get('use_positional_encoding', True)
    
    print(f"Creating enhanced model from saved configuration: {config}")
    
    # Create model with exact same architecture
    if config['model_type'] == 'GCNWithPositionalEncoding':
        model = GCNWithPositionalEncoding(
            num_node_features=config['num_node_features'],
            max_depth=config['max_depth'],
            max_child_index=config['max_child_index'],
            embedding_dim=config['embedding_dim'],
            hidden_dim_1=config['hidden_dim_1'],
            hidden_dim_2=config['hidden_dim_2'],
            sage=config['sage'],
            use_two_layer_classifier=config['use_two_layer_classifier'],
            dropout=config['dropout'],
            pooling_method=config['pooling_method'],
            depth_embedding_dim=config['depth_embedding_dim'],
            child_embedding_dim=config['child_embedding_dim']
        ).to(DEVICE)
    else:
        model = GCN(
            num_node_features=config['num_node_features'],
            embedding_dim=config['embedding_dim'],
            hidden_dim_1=config['hidden_dim_1'],
            hidden_dim_2=config['hidden_dim_2'],
            sage=config['sage'],
            use_two_layer_classifier=config['use_two_layer_classifier']
        ).to(DEVICE)
    
    model.name = model_name
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer and load state
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Create and load scheduler if available
    scheduler = None
    if 'scheduler_state_dict' in checkpoint:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']

# Re-export commonly used functions for backward compatibility
train = enhanced_train
evaluate = enhanced_evaluate
save_model = save_enhanced_model
load_model = load_enhanced_model
load_single_data = load_enhanced_data
load_multiple_data = load_multiple_enhanced_data
create_model_from_checkpoint = create_enhanced_model_from_checkpoint