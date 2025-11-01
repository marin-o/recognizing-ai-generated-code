from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, Recall, Precision, Specificity, AUROC
from tqdm import tqdm
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import logging
from datasets import Dataset
from typing import Tuple, Optional, Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset.semeval2026_task13 import SemEval2026Task13
from models.cbm_starcoder_tree import CBMStarCoderTree, load_tokenizer

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg="auto"):
    """Get the appropriate device based on argument"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    return device


def get_metrics(device):
    """Get standard metrics for binary classification"""
    num_classes = 2
    metrics = {
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes).to(device),
        'f1': F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
        'recall': Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
        'precision': Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
        'auroc': AUROC(task="binary").to(device),
        'specificity': Specificity(task="multiclass", num_classes=num_classes, average="macro").to(device),
    }
    return metrics


def save_model_checkpoint(model, optimizer, scheduler, epoch, best_vloss, best_vacc, 
                         model_config, save_path, model_name):
    """Save model checkpoint with configuration"""
    model_save_path = os.path.join(save_path, model_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    checkpoint_filename = "best_model.pth"
    checkpoint_filepath = os.path.join(model_save_path, checkpoint_filename)
    
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
    
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_filepath)
    
    logger.info(f"Model saved with validation loss: {best_vloss:.4f}, validation accuracy: {best_vacc:.4f}")
    logger.info(f"Model saved to: {checkpoint_filepath}")
    return checkpoint_filepath


def load_model_checkpoint(checkpoint_path, model, optimizer, scheduler=None, device="auto"):
    """Load model checkpoint"""
    device = get_device(device) if isinstance(device, str) else device
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided and available in checkpoint
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    elif scheduler is not None and 'scheduler_state_dict' not in checkpoint:
        logger.warning("Scheduler state not found in checkpoint, using fresh scheduler")
    
    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Best validation loss: {checkpoint['best_vloss']:.4f}")
    logger.info(f"Best validation accuracy: {checkpoint['best_vacc']:.4f}")
    logger.info(f"Saved at epoch: {checkpoint['epoch']}")
    
    return checkpoint['epoch'], checkpoint['best_vloss'], checkpoint['best_vacc']


def save_hyperparameters(params, score, save_path, model_name):
    """Save the best hyperparameters to a JSON file"""
    hyperparams_dir = os.path.join(save_path, model_name, "hyperparameters")
    os.makedirs(hyperparams_dir, exist_ok=True)
    
    hyperparams_record = {
        "timestamp": datetime.now().isoformat(),
        "validation_f1_score": score,
        "parameters": params
    }
    
    best_params_file = os.path.join(hyperparams_dir, "best_hyperparameters.json")
    with open(best_params_file, "w") as f:
        json.dump(hyperparams_record, f, indent=4)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(hyperparams_dir, f"hyperparams_{timestamp_str}_f1_{score:.4f}.json")
    with open(history_file, "w") as f:
        json.dump(hyperparams_record, f, indent=4)
    
    logger.info(f"Saved best hyperparameters with F1 score: {score:.4f}")


def load_hyperparameters(save_path, model_name):
    """Load the best hyperparameters if they exist"""
    best_params_file = os.path.join(save_path, model_name, "hyperparameters", "best_hyperparameters.json")
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            data = json.load(f)
        return data["parameters"], data["validation_f1_score"]
    return None, None


def tokenize_fn(tokenizer, examples, max_length=512):
    """Tokenize function for dataset mapping"""
    return tokenizer(
        examples["code"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )


def load_dataset(
    train_subset=0.1,
    full_test_set=False,
    val_ratio=0.1,
    test_ratio=0.2,
    subtask="A"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the SemEval 2026 Task 13 dataset
    
    Args:
        train_subset: Fraction of training data to use
        full_test_set: Whether to use the full test set
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        subtask: Subtask for SemEval dataset ("A", "B", or "C")
    """
    logger.info(f"Loading SemEval 2026 Task 13 dataset - Subtask {subtask}")
    
    dataset_loader = SemEval2026Task13(subtask=subtask)
    
    # Load train, validation, and test splits
    train, val, test = dataset_loader.get_dataset(
        split=['train', 'val', 'test'],
        train_subset=train_subset,
        dynamic_split_sizing=True,
        val_ratio=val_ratio,
        test_ratio=test_ratio if not full_test_set else 1.0
    )
    
    logger.info(f"Dataset loaded - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    logger.info(f"Val ratio: {val_ratio:.1%}, Test ratio: {test_ratio if not full_test_set else 1.0:.1%}")
    
    return train, val, test  # type: ignore


def tokenize_datasets(train: Optional[Dataset], val: Optional[Dataset], test: Optional[Dataset], 
                     tokenizer, max_length):
    """Tokenize train, validation, and test datasets"""
    tokenize = lambda x: tokenize_fn(tokenizer, x, max_length=max_length)
    logger.info("Tokenizing data...")
    
    train_tokenized = train.map(tokenize, batched=True, num_proc=8) if train is not None else None
    val_tokenized = val.map(tokenize, batched=True, num_proc=8) if val is not None else None
    test_tokenized = test.map(tokenize, batched=True, num_proc=8) if test is not None else None
    
    return train_tokenized, val_tokenized, test_tokenized


def create_dataloaders(train, val, test, batch_size=16, num_workers=4):
    """Create DataLoaders from datasets"""
    
    def collate_fn(batch):
        """Custom collate function to handle variable-length sequences and extract raw code/language"""
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
        labels = torch.tensor([item['target_binary'] for item in batch])
        
        # Extract raw code and language for tree feature extraction
        codes = [item['code'] for item in batch]
        languages = [item.get('language', 'python') for item in batch]  # Default to python if not specified
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'codes': codes,
            'languages': languages
        }
    
    train_dataloader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=True, collate_fn=collate_fn
    ) if train is not None else None
    
    val_dataloader = DataLoader(
        val, batch_size=batch_size, num_workers=num_workers, 
        pin_memory=True, collate_fn=collate_fn
    ) if val is not None else None
    
    test_dataloader = DataLoader(
        test, batch_size=batch_size, num_workers=num_workers, 
        pin_memory=True, collate_fn=collate_fn
    ) if test is not None else None
    
    return train_dataloader, val_dataloader, test_dataloader


def train_one_epoch(model, train_dataloader, optimizer, criterion, device, 
                   gradient_clip=1.0, log_interval=50, overall_progress=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    total_batches = len(train_dataloader)

    for i, data in enumerate(train_dataloader):
        # Extract inputs and labels
        inputs = {
            'input_ids': data['input_ids'].to(device),
            'attention_mask': data['attention_mask'].to(device)
        }
        labels = data['labels'].to(device)
        codes = data['codes']
        languages = data['languages']
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with tree features
        outputs = model(inputs, codes=codes, languages=languages)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        if overall_progress is not None:
            overall_progress.update(1)
            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / (i + 1)
                overall_progress.set_postfix({"batch_loss": f"{avg_loss:.4f}"})

    avg_loss = running_loss / total_batches if total_batches > 0 else 0
    return avg_loss


def evaluate_model(model, dataloader, criterion, metrics, device, 
                  perform_analysis=False, analysis_dir=None, model_name="CBMTree"):
    """Evaluate model on a dataset"""
    model.eval()

    # Reset metrics
    for metric in metrics.values():
        metric.reset()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", leave=False):
            # Extract inputs and labels
            inputs = {
                'input_ids': data['input_ids'].to(device),
                'attention_mask': data['attention_mask'].to(device)
            }
            labels = data['labels'].to(device)
            codes = data['codes']
            languages = data['languages']
            
            # Forward pass with tree features
            outputs = model(inputs, codes=codes, languages=languages)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store for analysis
            all_predictions.append(preds)
            all_labels.append(labels)
            
            # Update metrics
            for metric in metrics.values():
                if isinstance(metric, AUROC):
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    metric.update(probs, labels)
                else:
                    metric.update(preds, labels)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    results = {}
    for name, metric in metrics.items():
        results[name] = metric.compute().item()
    
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    results['loss'] = avg_loss
    
    return results


def train_model(model, optimizer, scheduler, criterion, train_dataloader, val_dataloader,
               metrics, device, epochs, patience=5, gradient_clip=1.0, log_interval=50,
               save_path="models/cbm_tree", model_name="cbm_tree_baseline", writer=None,
               initial_best_vloss=None, initial_best_vacc=None, start_epoch=0):
    """Train the model"""
    best_vloss = initial_best_vloss if initial_best_vloss is not None else float("inf")
    best_vacc = initial_best_vacc if initial_best_vacc is not None else 0.0
    patience_counter = 0

    total_steps = epochs * len(train_dataloader)
    overall_progress = tqdm(total=total_steps, desc="Training Progress", position=0)

    for epoch in range(epochs):
        logger.info(f"Epoch {start_epoch + epoch + 1}/{start_epoch + epochs}")
        
        # Training phase
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, criterion, device,
            gradient_clip, log_interval, overall_progress
        )
        
        # Validation phase
        val_results = evaluate_model(model, val_dataloader, criterion, metrics, device)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, start_epoch + epoch)
            writer.add_scalar("loss/val", val_loss, start_epoch + epoch)
            for metric_name, metric_value in val_results.items():
                if metric_name != 'loss':
                    writer.add_scalar(f"val/{metric_name}", metric_value, start_epoch + epoch)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, StepLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
        
        # Log epoch results
        logger.info(f"Train Loss: {train_loss:.5f}")
        logger.info(f"Val Loss: {val_loss:.5f}")
        for metric_name, metric_value in val_results.items():
            if metric_name != 'loss':
                logger.info(f"Val {metric_name.capitalize()}: {metric_value:.6f}")
        
        # Early stopping and model saving
        if val_loss < best_vloss or val_acc > best_vacc:
            if val_loss < best_vloss:
                best_vloss = val_loss
                logger.info(f"New best validation loss: {best_vloss:.5f}")
            if val_acc > best_vacc:
                best_vacc = val_acc
                logger.info(f"New best validation accuracy: {best_vacc:.6f}")
            
            # Save model with configuration
            model_config = {
                'backbone_type': model.backbone_type,
                'embedding_dim': model.embedding_dim,
                'filter_sizes': model.filter_sizes,
                'lstm_hidden_dim': model.lstm_hidden_dim,
                'num_classes': model.num_classes,
                'dropout_rate': model.dropout_rate,
                'tree_feature_projection_dim': model.tree_feature_projection_dim,
                'model_name': model.model_name
            }
            
            save_model_checkpoint(
                model, optimizer, scheduler, start_epoch + epoch,
                best_vloss, best_vacc, model_config, save_path, model_name
            )
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {start_epoch + epoch + 1} epochs")
            break

    overall_progress.close()
    
    # Load best model
    checkpoint_path = os.path.join(save_path, model_name, "best_model.pth")
    load_model_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
    
    return model


def create_model_from_checkpoint(checkpoint_path, device="auto"):
    """Create model from checkpoint with all configuration"""
    device = get_device(device) if isinstance(device, str) else device
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this has model_config
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config. Cannot recreate model.")
    
    config = checkpoint['model_config']
    
    # Create model
    model = CBMStarCoderTree(
        backbone_type=config['backbone_type'],
        model_name=config.get('model_name'),
        embedding_dim=config['embedding_dim'],
        filter_sizes=config['filter_sizes'],
        lstm_hidden_dim=config['lstm_hidden_dim'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        freeze_backbone=True,  # Always freeze when loading for evaluation
        tree_feature_projection_dim=config['tree_feature_projection_dim']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer and scheduler (needed for consistency)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = None
    if 'scheduler_state_dict' in checkpoint:
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_vloss = checkpoint['best_vloss']
    best_vacc = checkpoint['best_vacc']
    
    logger.info(f"Model recreated from checkpoint")
    logger.info(f"Configuration: {config}")
    
    return model, optimizer, scheduler, epoch, best_vloss, best_vacc


def create_model_with_optuna_params(
    storage_url, 
    study_name, 
    model_name, 
    backbone_type="codebert",
    device="auto", 
    use_default_on_failure=True
):
    """Create model with best hyperparameters from Optuna study"""
    import optuna
    
    device = get_device(device) if isinstance(device, str) else device
    
    try:
        # Load study
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_params = study.best_params
        
        logger.info(f"Loaded best hyperparameters from Optuna study: {study_name}")
        logger.info(f"Best trial F1 score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Create model with best parameters
        model = CBMStarCoderTree(
            backbone_type=backbone_type,
            embedding_dim=768 if backbone_type == "codebert" else 2048,
            filter_sizes=best_params['filter_sizes'],
            lstm_hidden_dim=best_params['lstm_hidden_dim'],
            num_classes=2,
            dropout_rate=best_params['dropout_rate'],
            freeze_backbone=True,
            tree_feature_projection_dim=best_params.get('tree_feature_projection_dim', 128)
        ).to(device)
        
        # Create optimizer with best parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay'],
            eps=1e-8
        )
        
        # Create scheduler
        scheduler_type = best_params.get('scheduler', 'cosine')
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=40)
        else:
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        return model, optimizer, scheduler, best_params
        
    except Exception as e:
        logger.error(f"Failed to load Optuna study: {e}")
        if use_default_on_failure:
            logger.warning("Using default hyperparameters")
            model = CBMStarCoderTree(
                backbone_type=backbone_type,
                embedding_dim=768 if backbone_type == "codebert" else 2048,
                filter_sizes=768,
                lstm_hidden_dim=256,
                num_classes=2,
                dropout_rate=0.5,
                freeze_backbone=True,
                tree_feature_projection_dim=128
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=40)
            
            return model, optimizer, scheduler, {}
        else:
            raise
