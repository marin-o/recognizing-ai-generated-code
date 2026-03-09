import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.loader import DataLoader
from models.GraphTransformer import GraphTransformer, GraphTransformerWithPositionalEncoding
from data.dataset.graph_codet_enhanced import GraphCoDeTM4Enhanced
from data.dataset.graph_aigcodeset import GraphAIGCodeSet
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC, F1Score
import optuna
import random
import numpy as np
import gc
import json
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
        shuffle = (split == 'train')
    
    print(f"Loading {split} dataset...")
    
    if dataset == 'codet':
        dataset_obj = GraphCoDeTM4Enhanced(data_dir=data_dir, split=split, suffix=suffix)
    elif dataset == 'aigcodeset':
        # For AIGCodeSet, we'll use the regular loader since it might not have positional encodings yet
        dataset_obj = GraphAIGCodeSet(data_dir=data_dir, split=split, suffix=suffix)
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
        model = GraphTransformerWithPositionalEncoding(
            num_node_features=num_node_features,
            max_depth=config.get('max_depth', 50),
            max_child_index=config.get('max_child_index', 100),
            embedding_dim=config.get('embedding_dim', 256),
            hidden_dim=config.get('hidden_dim', 128),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            pooling_method=config.get('pooling_method', 'mean'),
            depth_embedding_dim=config.get('depth_embedding_dim', 32),
            child_embedding_dim=config.get('child_embedding_dim', 32)
        ).to(DEVICE)
        
        print(f"Created GraphTransformerWithPositionalEncoding")
        print(f"  Max depth: {config.get('max_depth', 50)}")
        print(f"  Max child index: {config.get('max_child_index', 100)}")
        print(f"  Depth embedding dim: {config.get('depth_embedding_dim', 32)}")
        print(f"  Child embedding dim: {config.get('child_embedding_dim', 32)}")
    else:
        model = GraphTransformer(
            num_node_features=num_node_features,
            embedding_dim=config.get('embedding_dim', 256),
            hidden_dim=config.get('hidden_dim', 128),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            pooling_method=config.get('pooling_method', 'mean')
        ).to(DEVICE)
        
        print(f"Created standard GraphTransformer")
    
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
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 2,
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
        if hasattr(dataset_obj, 'get_positional_encoding_config'):
            pos_config = dataset_obj.get_positional_encoding_config(use_percentile=True)
            if pos_config:
                config.update(pos_config)
                print("Using dataset-derived positional encoding configuration")
            else:
                print("Dataset positional encoding config not available, using defaults")
                config.update({
                    'max_depth': 50,
                    'max_child_index': 100,
                    'depth_embedding_dim': 32,
                    'child_embedding_dim': 32
                })
        else:
            print("Dataset does not support positional encoding configuration")
            use_positional_encoding = False
    
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
    
    print(f"Starting enhanced training for {num_epochs} epochs...")
    print(f"Using positional encodings: {use_positional_encoding}")
    
    # Check if dataset has positional encodings
    has_pos_encodings = False
    if hasattr(train_dataloader, 'dataset_obj'):
        dataset_obj = train_dataloader.dataset_obj
        if hasattr(dataset_obj, 'has_positional_encodings'):
            has_pos_encodings = dataset_obj.has_positional_encodings()
    
    if use_positional_encoding and not has_pos_encodings:
        print("Warning: Model expects positional encodings but dataset doesn't provide them")
        print("Model will use zero-padding for missing positional data")
    
    for epoch in range(num_epochs):
        epoch_num = start_epoch + epoch + 1
        print(f'\nEpoch {epoch_num}/{start_epoch + num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        
        train_bar = tqdm(train_dataloader, desc='Training', leave=False)
        for batch in train_bar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass with optional positional encodings
            if use_positional_encoding and isinstance(model, GraphTransformerWithPositionalEncoding):
                # Extract positional encoding data if available
                node_depth = getattr(batch, 'node_depth', None) if has_pos_encodings else None
                child_index = getattr(batch, 'child_index', None) if has_pos_encodings else None
                
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                               node_depth=node_depth, child_index=child_index)
            else:
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            
            loss = criterion(outputs.squeeze(), batch.y.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Collect predictions for metrics
            preds = torch.sigmoid(outputs.squeeze())
            train_preds.extend(preds.cpu().detach().numpy())
            train_targets.extend(batch.y.cpu().numpy())
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_loss = running_loss / len(train_dataloader)
        train_preds = torch.tensor(train_preds).to(DEVICE)
        train_targets = torch.tensor(train_targets).to(DEVICE)
        
        train_metrics = {}
        for name, metric in metrics.items():
            metric.reset()
            train_metrics[name] = metric(train_preds, train_targets).item()
        
        # Validation phase
        val_results = enhanced_evaluate(
            model, val_dataloader, criterion, metrics, use_positional_encoding, has_pos_encodings
        )
        if len(val_results) == 3:
            val_loss, val_metrics, _ = val_results
        else:
            val_loss, val_metrics = val_results
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
        print(f'Train Acc: {train_metrics["accuracy"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
        
        # Tensorboard logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch_num)
            writer.add_scalar('Loss/validation', val_loss, epoch_num)
            for name, value in train_metrics.items():
                writer.add_scalar(f'Train/{name}', value, epoch_num)
            for name, value in val_metrics.items():
                writer.add_scalar(f'Validation/{name}', value, epoch_num)
        
        # Save best model
        if val_loss < best_vloss or (val_loss == best_vloss and val_metrics['accuracy'] > best_vacc):
            best_vloss = val_loss
            best_vacc = val_metrics['accuracy']
            
            save_enhanced_model(
                model, optimizer, scheduler, epoch_num, best_vloss, best_vacc, 
                save_path="models/graph_transformer_enhanced", use_positional_encoding=use_positional_encoding
            )
            print(f'✓ New best model saved (Val Loss: {best_vloss:.5f}, Val Acc: {best_vacc:.4f})')
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def enhanced_evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                     metrics: Dict[str, Any], use_positional_encoding: bool = True,
                     has_pos_encodings: bool = True, perform_analysis: bool = False, 
                     analysis_dir: Optional[str] = None, model_name: Optional[str] = None) -> Union[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float], Dict[str, Any]]]:
    """Enhanced evaluation function with positional encoding support and optional misclassification analysis."""
    
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        eval_bar = tqdm(dataloader, desc='Evaluating', leave=False)
        for batch in eval_bar:
            batch = batch.to(DEVICE)
            
            # Forward pass with optional positional encodings
            if use_positional_encoding and isinstance(model, GraphTransformerWithPositionalEncoding):
                # Extract positional encoding data if available
                node_depth = getattr(batch, 'node_depth', None) if has_pos_encodings else None
                child_index = getattr(batch, 'child_index', None) if has_pos_encodings else None
                
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                               node_depth=node_depth, child_index=child_index)
            else:
                outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            
            loss = criterion(outputs.squeeze(), batch.y.float())
            total_loss += loss.item()
            
            # Collect predictions
            preds = torch.sigmoid(outputs.squeeze())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    all_preds = torch.tensor(all_preds).to(DEVICE)
    all_targets = torch.tensor(all_targets).to(DEVICE)
    
    eval_metrics = {}
    for name, metric in metrics.items():
        metric.reset()
        eval_metrics[name] = metric(all_preds, all_targets).item()
    
    # Perform misclassification analysis if requested
    if perform_analysis and analysis_dir and model_name:
        try:
            analysis_results = analyze_misclassified_samples(
                model, dataloader, criterion, metrics, analysis_dir, model_name
            )
            return avg_loss, eval_metrics, analysis_results
        except Exception as e:
            print(f"Analysis failed: {e}")
            return avg_loss, eval_metrics

    return avg_loss, eval_metrics

def analyze_misclassified_samples(model, dataloader, criterion, metrics, analysis_dir="analysis", model_name="GraphTransformerEnhanced"):
    """
    Analyze misclassified samples and create visualizations of graph size distributions.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing test samples
        criterion: Loss function
        metrics: Dictionary of evaluation metrics
        analysis_dir: Directory to save analysis results
        model_name: Name of the model for file naming
        
    Returns:
        dict: Analysis results including misclassification counts and statistics
    """
    print("Starting misclassification analysis...")
    
    # Determine device based on model location
    device = next(model.parameters()).device
    
    # Create analysis directory
    os.makedirs(analysis_dir, exist_ok=True)
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_graph_sizes = []
    all_probs = []
    
    # Collect predictions and graph information
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Collecting predictions", leave=False):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            y = data.y.to(device).float()
            
            # Get model predictions - handle enhanced model with positional encoding
            if hasattr(model, 'use_positional_encoding') and model.use_positional_encoding:
                # Extract positional encoding data if available
                node_depth = getattr(data, 'node_depth', None)
                child_index = getattr(data, 'child_index', None)
                out = model(x=x, edge_index=edge_index, batch=batch,
                           node_depth=node_depth, child_index=child_index)
            else:
                out = model(x, edge_index, batch)
                
            probs = torch.sigmoid(out.squeeze())
            predictions = (probs > 0.5).float()
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Calculate graph sizes for each sample in the batch
            batch_cpu = batch.cpu().numpy()
            unique_graphs = np.unique(batch_cpu)
            for graph_id in unique_graphs:
                graph_mask = batch_cpu == graph_id
                graph_size = np.sum(graph_mask)
                all_graph_sizes.append(graph_size)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    true_labels = np.array(all_true_labels)
    graph_sizes = np.array(all_graph_sizes)
    probs = np.array(all_probs)
    
    # Identify misclassified samples
    misclassified_mask = predictions != true_labels
    correctly_classified_mask = predictions == true_labels
    
    # Get misclassified sample information
    misclassified_sizes = graph_sizes[misclassified_mask]
    correctly_classified_sizes = graph_sizes[correctly_classified_mask]
    misclassified_true_labels = true_labels[misclassified_mask]
    misclassified_predictions = predictions[misclassified_mask]
    misclassified_probs = probs[misclassified_mask]
    
    # Calculate statistics
    total_samples = len(predictions)
    num_misclassified = np.sum(misclassified_mask)
    misclassification_rate = num_misclassified / total_samples
    
    # False positives (predicted AI-generated, actually human)
    false_positives = np.sum((predictions == 1) & (true_labels == 0))
    # False negatives (predicted human, actually AI-generated)
    false_negatives = np.sum((predictions == 0) & (true_labels == 1))
    
    print(f"Total samples: {total_samples}")
    print(f"Misclassified samples: {num_misclassified}")
    print(f"Misclassification rate: {misclassification_rate:.4f}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    
    # Create analysis dataframe
    analysis_data = {
        'graph_size': graph_sizes,
        'true_label': true_labels,
        'prediction': predictions,
        'probability': probs,
        'misclassified': misclassified_mask
    }
    df = pd.DataFrame(analysis_data)
    
    # Save detailed results
    results_file = os.path.join(analysis_dir, f"{model_name}_misclassification_analysis.csv")
    df.to_csv(results_file, index=False)
    print(f"Detailed analysis saved to: {results_file}")
    
    # Create visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Misclassification Analysis', fontsize=16, fontweight='bold')
    
    # 1. Graph size distribution comparison
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(graph_sizes), 50)
    ax1.hist(correctly_classified_sizes, bins=bins, alpha=0.7, label='Correctly Classified', 
             color='green', density=True)
    ax1.hist(misclassified_sizes, bins=bins, alpha=0.7, label='Misclassified', 
             color='red', density=True)
    ax1.set_xlabel('Graph Size (Number of Nodes)')
    ax1.set_ylabel('Density')
    ax1.set_title('Graph Size Distribution: Correct vs Misclassified')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Misclassification rate by graph size bins
    ax2 = axes[0, 1]
    size_bins = np.percentile(graph_sizes, [0, 25, 50, 75, 100])
    size_labels = [f'{int(size_bins[i])}-{int(size_bins[i+1])}' for i in range(len(size_bins)-1)]
    
    misclass_rates = []
    for i in range(len(size_bins)-1):
        mask = (graph_sizes >= size_bins[i]) & (graph_sizes < size_bins[i+1])
        if i == len(size_bins)-2:  # Last bin should include the maximum
            mask = (graph_sizes >= size_bins[i]) & (graph_sizes <= size_bins[i+1])
        
        if np.sum(mask) > 0:
            rate = np.sum(misclassified_mask[mask]) / np.sum(mask)
            misclass_rates.append(rate)
        else:
            misclass_rates.append(0)
    
    bars = ax2.bar(size_labels, misclass_rates, color='coral', alpha=0.7)
    ax2.set_xlabel('Graph Size Quartiles')
    ax2.set_ylabel('Misclassification Rate')
    ax2.set_title('Misclassification Rate by Graph Size')
    ax2.set_ylim(0, max(misclass_rates) * 1.1 if misclass_rates else 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, misclass_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3)
    
    # 3. False positive vs False negative analysis by size
    ax3 = axes[1, 0]
    fp_mask = (predictions == 1) & (true_labels == 0)
    fn_mask = (predictions == 0) & (true_labels == 1)
    
    fp_sizes = graph_sizes[fp_mask]
    fn_sizes = graph_sizes[fn_mask]
    
    ax3.hist(fp_sizes, bins=30, alpha=0.7, label=f'False Positives (n={len(fp_sizes)})', 
             color='orange', density=True)
    ax3.hist(fn_sizes, bins=30, alpha=0.7, label=f'False Negatives (n={len(fn_sizes)})', 
             color='purple', density=True)
    ax3.set_xlabel('Graph Size (Number of Nodes)')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Type Distribution by Graph Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confidence analysis for misclassified samples
    ax4 = axes[1, 1]
    
    # Separate misclassified by true label
    misclass_ai_mask = misclassified_mask & (true_labels == 1)  # AI samples misclassified as human
    misclass_human_mask = misclassified_mask & (true_labels == 0)  # Human samples misclassified as AI
    
    if np.sum(misclass_ai_mask) > 0:
        ax4.hist(probs[misclass_ai_mask], bins=20, alpha=0.7, 
                label=f'AI→Human (n={np.sum(misclass_ai_mask)})', color='blue')
    if np.sum(misclass_human_mask) > 0:
        ax4.hist(probs[misclass_human_mask], bins=20, alpha=0.7, 
                label=f'Human→AI (n={np.sum(misclass_human_mask)})', color='red')
    
    ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
    ax4.set_xlabel('Prediction Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Prediction Confidence for Misclassified Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(analysis_dir, f"{model_name}_misclassification_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to: {plot_file}")
    
    # Show the plot if in interactive mode
    try:
        plt.show()
    except:
        pass  # In case we're running in a non-interactive environment
    finally:
        plt.close()
    
    # Create summary statistics
    summary_stats = {
        'total_samples': total_samples,
        'misclassified_samples': num_misclassified,
        'misclassification_rate': misclassification_rate,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_graph_size_correct': np.mean(correctly_classified_sizes),
        'avg_graph_size_misclassified': np.mean(misclassified_sizes),
        'std_graph_size_correct': np.std(correctly_classified_sizes),
        'std_graph_size_misclassified': np.std(misclassified_sizes),
        'median_graph_size_correct': np.median(correctly_classified_sizes),
        'median_graph_size_misclassified': np.median(misclassified_sizes),
    }
    
    # Save summary statistics
    summary_file = os.path.join(analysis_dir, f"{model_name}_misclassification_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Misclassification Analysis Summary for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary_stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Summary statistics saved to: {summary_file}")
    print("\nMisclassification analysis completed!")
    
    return summary_stats

def save_enhanced_model(model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int, best_vloss: float, best_vacc: float,
                       save_path: str = "models/graph_transformer_enhanced",
                       use_positional_encoding: bool = True):
    """Save enhanced model with all necessary information for reconstruction."""
    
    os.makedirs(os.path.join(save_path, model.name), exist_ok=True)
    
    # Collect model architecture information
    model_config = {
        'model_type': 'GraphTransformerWithPositionalEncoding' if use_positional_encoding else 'GraphTransformer',
        'use_positional_encoding': use_positional_encoding,
        'num_node_features': getattr(model, 'num_node_features', None),
        'embedding_dim': getattr(model, 'embedding_dim', None),
        'hidden_dim': getattr(model, 'hidden_dim', None),
        'num_heads': getattr(model, 'num_heads', None),
        'num_layers': getattr(model, 'num_layers', None),
        'pooling_method': getattr(model, 'pooling_method', None),
    }
    
    # Add positional encoding specific config
    if use_positional_encoding:
        model_config.update({
            'max_depth': getattr(model, 'max_depth', None),
            'max_child_index': getattr(model, 'max_child_index', None),
            'depth_embedding_dim': getattr(model, 'depth_embedding_dim', None),
            'child_embedding_dim': getattr(model, 'child_embedding_dim', None),
        })
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_vloss': best_vloss,
        'best_vacc': best_vacc,
        'model_config': model_config,
        'model_name': model.name
    }
    
    checkpoint_path = os.path.join(save_path, model.name, f"{model.name}_best.pth")
    torch.save(checkpoint, checkpoint_path)

def load_enhanced_model(model: nn.Module, optimizer: torch.optim.Optimizer,
                       save_path: str = "models/graph_transformer_enhanced",
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[int, float, float]:
    """Load enhanced model checkpoint."""
    
    checkpoint_path = os.path.join(save_path, model.name, f"{model.name}_best.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No saved model found at: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_vloss = checkpoint['best_vloss']
    best_vacc = checkpoint['best_vacc']
    
    print(f"Loaded model from epoch {epoch}")
    print(f"Best validation loss: {best_vloss:.4f}")
    print(f"Best validation accuracy: {best_vacc:.4f}")
    
    return epoch, best_vloss, best_vacc

def create_enhanced_model_from_checkpoint(checkpoint_path: str, model_name: str) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler], int, float, float]:
    """Create and load an enhanced model from checkpoint with architecture reconstruction."""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration information")
    
    config = checkpoint['model_config']
    use_positional_encoding = config.get('use_positional_encoding', False)
    
    # Create model based on saved configuration
    model = create_enhanced_model_from_config(
        num_node_features=config['num_node_features'],
        config=config,
        model_name=model_name,
        use_positional_encoding=use_positional_encoding
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create scheduler if it was saved
    scheduler = None
    if checkpoint.get('scheduler_state_dict') is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_vloss = checkpoint['best_vloss']
    best_vacc = checkpoint['best_vacc']
    
    return model, optimizer, scheduler, epoch, best_vloss, best_vacc

# Re-export commonly used functions for backward compatibility
train = enhanced_train
evaluate = enhanced_evaluate
save_model = save_enhanced_model
load_model = load_enhanced_model
load_single_data = load_enhanced_data
load_multiple_data = load_multiple_enhanced_data
create_model_from_checkpoint = create_enhanced_model_from_checkpoint
