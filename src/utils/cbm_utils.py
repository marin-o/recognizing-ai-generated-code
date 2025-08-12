import torch
import os
import json
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, Recall, Precision, Specificity, AUROC
from tqdm import tqdm
import logging
import optuna
from transformers import RobertaModel
from datasets import Dataset

from data.dataset.codet_m4 import CoDeTM4
from models.cbmclassifier import CBMClassifier
from utils.utils import tokenize_fn

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
            logger.warning("CUDA requested but not available, falling back to CPU")
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
        checkpoint_data['scheduler_type'] = type(scheduler).__name__
    
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
        logger.info(f"Scheduler loaded: {checkpoint.get('scheduler_type', 'Unknown')}")
    elif scheduler is not None and 'scheduler_state_dict' not in checkpoint:
        logger.warning("Scheduler provided but no scheduler state found in checkpoint")
    
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
        json.dump(hyperparams_record, f, indent=2)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(hyperparams_dir, f"hyperparams_{timestamp_str}_f1_{score:.4f}.json")
    with open(history_file, "w") as f:
        json.dump(hyperparams_record, f, indent=2)
    
    logger.info(f"Saved best hyperparameters with F1 score: {score:.4f}")

def load_hyperparameters(save_path, model_name):
    """Load the best hyperparameters if they exist"""
    best_params_file = os.path.join(save_path, model_name, "hyperparameters", "best_hyperparameters.json")
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            record = json.load(f)
            return record["parameters"], record["validation_f1_score"]
    return None, None

def train_one_epoch(model, train_dataloader, optimizer, criterion, device, 
                   gradient_clip=1.0, log_interval=50, overall_progress=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    total_batches = len(train_dataloader)

    for i, data in enumerate(train_dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["target_binary"].to(device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        running_loss += loss.item()

        if overall_progress is not None:
            overall_progress.update(1)
            overall_progress.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{running_loss / (i + 1):.4f}"
            })

        if i % log_interval == log_interval - 1:
            avg_batch_loss = running_loss / log_interval
            tqdm.write(f"[Batch {i + 1}] Loss: {avg_batch_loss:.4f}")
            running_loss = 0.0

    avg_loss = running_loss / total_batches if total_batches > 0 else 0
    return avg_loss

def evaluate_model(model, dataloader, criterion, metrics, device):
    """Evaluate model on a dataset"""
    model.eval()

    # Reset metrics
    for metric in metrics.values():
        metric.reset()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", position=0, leave=True)
        for data in progress_bar:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["target_binary"].to(device)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_predictions.append(predicted)
            all_labels.append(labels)
            
            # Update progress bar with current loss
            avg_loss = running_loss / (len(all_predictions))
            progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    results = {}
    for name, metric in metrics.items():
        metric.update(all_predictions, all_labels)
        results[name] = metric.compute().item()
    
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    results['loss'] = avg_loss
    
    return results

def train_model(model, optimizer, scheduler, criterion, train_dataloader, val_dataloader,
               metrics, device, epochs, patience=5, gradient_clip=1.0, log_interval=50,
               save_path="models/cbm", model_name="cbm_baseline", writer=None,
               initial_best_vloss=None, initial_best_vacc=None, start_epoch=0):
    """Train the model"""
    best_vloss = initial_best_vloss if initial_best_vloss is not None else float("inf")
    best_vacc = initial_best_vacc if initial_best_vacc is not None else 0.0
    patience_counter = 0

    total_steps = epochs * len(train_dataloader)
    overall_progress = tqdm(total=total_steps, desc="Training Progress", position=0)

    for epoch in range(epochs):
        current_epoch = start_epoch + epoch
        logger.info(f"Epoch {current_epoch + 1}")
        
        # Train for one epoch
        avg_loss = train_one_epoch(
            model, train_dataloader, optimizer, criterion, device, 
            gradient_clip, log_interval, overall_progress
        )
        logger.info(f"Average training loss: {avg_loss:.4f}")

        # Validate
        val_results = evaluate_model(model, val_dataloader, criterion, metrics, device)
        logger.info(f"Validation loss: {val_results['loss']:.4f}")
        logger.info(f"Validation accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"Validation F1: {val_results['f1']:.4f}")

        # Log to tensorboard
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, current_epoch)
            writer.add_scalar("val/loss", val_results['loss'], current_epoch)
            for metric_name, metric_value in val_results.items():
                if metric_name != 'loss':
                    writer.add_scalar(f"val/{metric_name}", metric_value, current_epoch)

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results['loss'])
            else:
                scheduler.step()

        # Save best model
        if val_results["loss"] < best_vloss:
            best_vloss = val_results["loss"]
            best_vacc = val_results["accuracy"]
            patience_counter = 0
            
            # Save model checkpoint
            model_config = {
                'lstm_hidden_dim': model.lstm_hidden_dim,
                'filter_sizes': model.filter_sizes,
                'dropout_rate': model.dropout_rate,
                'embedding_dim': model.embedding_dim,
            }
            save_model_checkpoint(
                model, optimizer, scheduler, current_epoch, best_vloss, best_vacc,
                model_config, save_path, model_name
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

    overall_progress.close()
    
    # Load best model
    checkpoint_path = os.path.join(save_path, model_name, "best_model.pth")
    load_model_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
    
    return model

def create_model_from_checkpoint(checkpoint_path, device="auto"):
    """Create model from checkpoint with all configuration"""
    from transformers import RobertaModel
    from models.cbmclassifier import CBMClassifier
    
    device = get_device(device) if isinstance(device, str) else device
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a very old format (just state_dict)
    if isinstance(checkpoint, dict) and 'base_model.embeddings.word_embeddings.weight' in checkpoint:
        logger.warning("Very old checkpoint format detected - loading direct state_dict")
        logger.info("Creating model with default configuration:")
        
        # Use default parameters for very old checkpoints
        config = {
            'lstm_hidden_dim': 256,
            'filter_sizes': 768,
            'dropout_rate': 0.5
        }
        
        logger.info(f"  lstm_hidden_dim: {config['lstm_hidden_dim']}")
        logger.info(f"  filter_sizes: {config['filter_sizes']}")
        logger.info(f"  dropout_rate: {config['dropout_rate']}")
        
        # Create base model and CBM classifier
        base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
        model = CBMClassifier(
            base_model=base_model,
            lstm_hidden_dim=config['lstm_hidden_dim'],
            filter_sizes=config['filter_sizes'],
            dropout_rate=config['dropout_rate']
        ).to(device)
        
        # Load the state dict directly
        model.load_state_dict(checkpoint)
        
        # Create new optimizer and scheduler since they're not saved
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, eps=1e-8)
        scheduler = None
        
        logger.info(f"Model loaded from very old checkpoint format: {checkpoint_path}")
        logger.info("Optimizer and scheduler initialized fresh (no saved state available)")
        
        # Return default values for missing metadata
        return model, optimizer, scheduler, 0, 0.0, 0.0
    
    # Check if this is a legacy checkpoint without model configuration
    elif isinstance(checkpoint, dict) and 'model_config' not in checkpoint:
        logger.warning("Legacy checkpoint detected - using default model parameters")
        logger.info("Creating model with default configuration:")
        
        # Use default parameters for legacy checkpoints
        config = {
            'lstm_hidden_dim': 256,
            'filter_sizes': 768,
            'dropout_rate': 0.5
        }
        
        logger.info(f"  lstm_hidden_dim: {config['lstm_hidden_dim']}")
        logger.info(f"  filter_sizes: {config['filter_sizes']}")
        logger.info(f"  dropout_rate: {config['dropout_rate']}")
    else:
        config = checkpoint['model_config']
        logger.info(f"Creating model from saved configuration: {config}")
    
    # Create base model and CBM classifier (for non-very-old checkpoints)
    if not (isinstance(checkpoint, dict) and 'base_model.embeddings.word_embeddings.weight' in checkpoint):
        base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
        model = CBMClassifier(
            base_model=base_model,
            lstm_hidden_dim=config['lstm_hidden_dim'],
            filter_sizes=config['filter_sizes'],
            dropout_rate=config['dropout_rate']
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, eps=1e-8)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # For legacy checkpoints, optimizer state might not be compatible
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load optimizer state (legacy checkpoint): {e}")
            logger.info("Optimizer will be reinitialized")
        
        # Create and load scheduler if available
        scheduler = None
        if 'scheduler_state_dict' in checkpoint:
            scheduler_type = checkpoint.get('scheduler_type', 'CosineAnnealingLR')
            logger.info(f"Creating scheduler: {scheduler_type}")
            
            if scheduler_type == 'CosineAnnealingLR':
                scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
            elif scheduler_type == 'StepLR':
                scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            
            if scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Scheduler state loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load scheduler state: {e}")
        else:
            logger.info("No scheduler information found in checkpoint")
        
        # Get checkpoint metrics, with fallbacks for legacy checkpoints
        best_vloss = checkpoint.get('best_vloss', 0.0)
        best_vacc = checkpoint.get('best_vacc', 0.0)
        epoch = checkpoint.get('epoch', 0)
        
        logger.info(f"Model created and loaded from {checkpoint_path}")
        logger.info(f"Best validation loss: {best_vloss:.4f}")
        logger.info(f"Best validation accuracy: {best_vacc:.4f}")
        logger.info(f"Saved at epoch: {epoch}")
        
        return model, optimizer, scheduler, epoch, best_vloss, best_vacc

def create_model_with_optuna_params(storage_url, study_name, model_name, 
                                   pretrained_model="microsoft/codebert-base", 
                                   device="auto", use_default_on_failure=True):
    """Create model with best hyperparameters from Optuna study"""
    from transformers import RobertaModel
    from models.cbmclassifier import CBMClassifier
    import optuna
    
    device = get_device(device) if isinstance(device, str) else device
    
    try:
        study = optuna.load_study(storage=storage_url, study_name=study_name)
        best_params = study.best_trial.params
        
        logger.info("Loading model with best hyperparameters from Optuna:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Create model with best hyperparameters
        base_model = RobertaModel.from_pretrained(pretrained_model)
        model = CBMClassifier(
            base_model=base_model,
            lstm_hidden_dim=best_params["lstm_hidden_dim"],
            filter_sizes=best_params["filter_sizes"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
            eps=1e-8
        )
        
        # Create scheduler if it was used in optimization
        scheduler = None
        scheduler_type = best_params.get('scheduler', 'cosine')
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
        elif scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        return model, optimizer, scheduler, True
        
    except Exception as e:
        logger.error(f"Failed to load Optuna parameters: {e}")
        if use_default_on_failure:
            logger.info("Creating model with default parameters...")
            base_model = RobertaModel.from_pretrained(pretrained_model)
            model = CBMClassifier(
                base_model=base_model,
                lstm_hidden_dim=256,
                filter_sizes=768,
                dropout_rate=0.5
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, eps=1e-8)
            scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
            
            return model, optimizer, scheduler, False
        else:
            raise e

def load_dataset(cache_dir, train_subset=0.1, full_test_set=False, val_ratio=0.1, test_ratio=0.2, 
                 use_cleaned=False, cleaned_data_path=None):
    """
    Load the CoDeTM4 dataset with configurable validation and test ratios
    
    Args:
        cache_dir: Directory to cache the original dataset
        train_subset: Fraction of training data to use
        full_test_set: Whether to use the full test set
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        use_cleaned: Whether to use the cleaned dataset (duplicates removed)
        cleaned_data_path: Path to cleaned dataset directory (auto-detected if None)
    """
    if use_cleaned:
        # Try to import and use the cleaned dataset
        try:
            from data.dataset.codet_m4_cleaned import CoDeTM4Cleaned
            
            # Auto-detect cleaned data path if not provided
            if cleaned_data_path is None:
                # Look for the most recent cleaned dataset
                import glob
                possible_paths = glob.glob("data/codet_cleaned_*")
                if possible_paths:
                    cleaned_data_path = max(possible_paths)  # Get the most recent one
                    logger.info(f"Auto-detected cleaned dataset: {cleaned_data_path}")
                else:
                    logger.warning("No cleaned dataset found, falling back to original dataset")
                    use_cleaned = False
            
            if use_cleaned:
                dataset = CoDeTM4Cleaned(cleaned_data_path=cleaned_data_path)
                logger.info(f"Using cleaned dataset from: {cleaned_data_path}")
                
                # Get dataset info if available
                try:
                    info = dataset.get_info()
                    if 'cleaning_metadata' in info:
                        metadata = info['cleaning_metadata']
                        logger.info(f"Cleaned dataset info:")
                        logger.info(f"  Original sizes - Train: {metadata.get('original_sizes', {}).get('train', 'N/A')}, "
                                  f"Val: {metadata.get('original_sizes', {}).get('val', 'N/A')}, "
                                  f"Test: {metadata.get('original_sizes', {}).get('test', 'N/A')}")
                        logger.info(f"  Cleaned sizes - Train: {metadata.get('cleaned_sizes', {}).get('train', 'N/A')}, "
                                  f"Val: {metadata.get('cleaned_sizes', {}).get('val', 'N/A')}, "
                                  f"Test: {metadata.get('cleaned_sizes', {}).get('test', 'N/A')}")
                        logger.info(f"  Total samples removed: {sum(metadata.get('removed_counts', {}).values())}")
                except Exception as e:
                    logger.debug(f"Could not load cleaning metadata: {e}")
                    
        except ImportError as e:
            logger.warning(f"Could not import cleaned dataset class: {e}")
            logger.info("Falling back to original dataset")
            use_cleaned = False
        except Exception as e:
            logger.warning(f"Error loading cleaned dataset: {e}")
            logger.info("Falling back to original dataset")
            use_cleaned = False
    
    # Use original dataset if not using cleaned or if cleaned failed
    if not use_cleaned:
        from data.dataset.codet_m4 import CoDeTM4
        dataset = CoDeTM4(cache_dir=cache_dir)
        logger.info(f"Using original dataset from cache: {cache_dir}")
    
    if full_test_set:
        # For evaluation, use full test set without downsampling
        train, val, test = dataset.get_dataset(
            split=['train', 'val', 'test'], 
            columns=['code', 'target_binary'],
            train_subset=train_subset,
            dynamic_split_sizing=False,
            max_split_ratio=0.5,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        logger.info(f"Using full test set with {len(test)} samples")
    else:
        # For training, use smaller subsets
        train, val, test = dataset.get_dataset(
            split=['train', 'val', 'test'], 
            columns=['code', 'target_binary'],
            train_subset=train_subset,
            dynamic_split_sizing=True,
            max_split_ratio=0.5,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    
    dataset_type = "cleaned" if use_cleaned else "original"
    logger.info(f"Dataset loaded ({dataset_type}) - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    logger.info(f"Val ratio: {val_ratio:.1%}, Test ratio: {test_ratio:.1%}")
    
    return train, val, test

def get_available_cleaned_datasets():
    """Get list of available cleaned datasets"""
    import glob
    import os
    
    possible_paths = glob.glob("data/codet_cleaned_*")
    datasets_info = []
    
    for path in possible_paths:
        if os.path.isdir(path):
            info = {"path": path, "name": os.path.basename(path)}
            
            # Try to get metadata
            metadata_path = os.path.join(path, 'cleaning_metadata.json')
            if os.path.exists(metadata_path):
                try:
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    info['metadata'] = metadata
                    info['timestamp'] = metadata.get('cleaning_timestamp', 'Unknown')
                    info['total_removed'] = sum(metadata.get('removed_counts', {}).values())
                except Exception as e:
                    logger.debug(f"Could not load metadata for {path}: {e}")
            
            datasets_info.append(info)
    
    # Sort by timestamp (most recent first)
    datasets_info.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return datasets_info

def use_cleaned_dataset_by_default():
    """
    Convenience function to check if cleaned datasets are available and should be used by default.
    Returns the path to the most recent cleaned dataset, or None if none available.
    """
    available = get_available_cleaned_datasets()
    if available:
        most_recent = available[0]
        logger.info(f"Most recent cleaned dataset available: {most_recent['name']}")
        if 'total_removed' in most_recent:
            logger.info(f"  Removed {most_recent['total_removed']} duplicate samples")
        return most_recent['path']
    return None

def tokenize_datasets(train: Dataset, val: Dataset, test: Dataset, tokenizer, max_length):
    """Tokenize train, validation, and test datasets"""
    tokenize = lambda x: tokenize_fn(tokenizer, x, max_length=max_length)
    logger.info("Tokenizing data...")
    train_tokenized = train.map(tokenize, batched=True, num_proc=8) if train is not None else None
    val_tokenized = val.map(tokenize, batched=True, num_proc=8) if val is not None else None
    test_tokenized = test.map(tokenize, batched=True, num_proc=8) if test is not None else None
    
    return train_tokenized, val_tokenized, test_tokenized

def create_dataloaders(train, val, test, batch_size=16, num_workers=4):
    """Create DataLoaders from datasets"""
    import torch
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function to handle HuggingFace dataset format"""
        # Convert list of dicts to dict of lists
        collated = {}
        for key in batch[0].keys():
            collated[key] = [item[key] for item in batch]
        
        # Convert to tensors for tokenizer outputs
        for key in ['input_ids', 'attention_mask']:
            if key in collated:
                collated[key] = torch.tensor(collated[key], dtype=torch.long)
        
        # Convert targets to tensors
        for key in ['target', 'target_binary']:
            if key in collated:
                collated[key] = torch.tensor(collated[key], dtype=torch.long)
        
        return collated
    
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
