import argparse
import logging
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, Recall, Precision
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CodeBERT-based classifier")
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/codebert-base",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/", help="Directory to cache dataset"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset for test set",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of training set for validation set",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Number of batches between logging metrics",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Only evaluate the model without training",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=500, help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--hyperparameter-search",
        action="store_true",
        help="Perform hyperparameter search using Optuna",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials for hyperparameter search",
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=15,
        help="Number of epochs per trial during hyperparameter search",
    )
    return parser.parse_args()


def train_one_epoch(
    epoch_index: int,
    train_dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    log_interval: int = 100,
    gradient_clip: float = 1.0,
    overall_progress: tqdm = None,
) -> float:
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
                "epoch": f"{epoch_index + 1}",
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{running_loss / (i + 1):.4f}"
            })

        if i % log_interval == log_interval - 1:
            avg_batch_loss = running_loss / log_interval
            tqdm.write(
                f"[Epoch {epoch_index + 1}, Batch {i + 1}] Loss: {avg_batch_loss:.4f}"
            )
            running_loss = 0.0

    avg_loss = running_loss / total_batches if total_batches > 0 else 0
    return avg_loss


def evaluate_model(
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    log_to_tensorboard: bool = False,
) -> dict:
    model.eval()

    num_classes = 2
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(
        device
    )
    precision = Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
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

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    acc = accuracy(all_predictions, all_labels)
    f1_score = f1(all_predictions, all_labels)
    recall_score = recall(all_predictions, all_labels)
    precision_score = precision(all_predictions, all_labels)
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0

    test_metrics = {
        "accuracy": acc.item(),
        "f1": f1_score.item(),
        "recall": recall_score.item(),
        "precision": precision_score.item(),
        "loss": avg_loss,
    }

    if log_to_tensorboard:
        # Set up TensorBoard logging
        tensorboard_run_name = "cbm_codet"
        log_dir = os.path.join("tensorboard_logs", "CoDeTM4", tensorboard_run_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        # Log metrics
        writer.add_scalar("Metrics/test_accuracy", test_metrics["accuracy"])
        writer.add_scalar("Metrics/test_f1_macro", test_metrics["f1"])
        writer.add_scalar("Metrics/recall", test_metrics["recall"])
        writer.add_scalar("Metrics/precision", test_metrics["precision"])
        writer.add_scalar("Metrics/test_loss", test_metrics["loss"])
        
        # Save model
        model_path = os.path.join(log_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        writer.close()

    return test_metrics


def train_model(
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    patience: int,
    log_interval: int = 100,
    gradient_clip: float = 1.0,
) -> torch.nn.Module:
    best_vloss = float("inf")
    patience_counter = 0
    models_dir = "models/cbm_codet"
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pth")

    total_steps = epochs * len(train_dataloader)
    overall_progress = tqdm(total=total_steps, desc="Training Progress", position=0)

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        avg_loss = train_one_epoch(
            epoch, train_dataloader, model, optimizer, criterion, device, log_interval, gradient_clip, overall_progress
        )
        logger.info(f"Average training loss: {avg_loss:.4f}")

        val_metrics = evaluate_model(val_dataloader, model, criterion, device, log_to_tensorboard=False)
        logger.info(f"Validation loss: {val_metrics['loss']:.4f}")
        logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
        logger.info(f"Validation recall: {val_metrics['recall']:.4f}")
        logger.info(f"Validation precision: {val_metrics['precision']:.4f}")

        scheduler.step()

        if val_metrics["loss"] < best_vloss:
            best_vloss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Model saved with validation loss: {best_vloss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement"
                )
                break

    overall_progress.close()
    model.load_state_dict(torch.load(best_model_path))
    return model


def save_best_hyperparameters(params: dict, score: float):
    """Save the best hyperparameters to a JSON file."""
    hyperparams_dir = "models/cbm_codet/hyperparameters"
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


def load_best_hyperparameters():
    """Load the best hyperparameters if they exist."""
    best_params_file = "models/cbm_codet/hyperparameters/best_hyperparameters.json"
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            record = json.load(f)
            return record["parameters"], record["validation_f1_score"]
    return None, None


def objective(trial, dataset, tokenizer, device, search_epochs):
    """Optuna objective function for hyperparameter optimization."""
    from transformers import RobertaModel
    from models.cbmclassifier import CBMClassifier
    from utils.utils import tokenize_fn
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7)
    lstm_hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [128, 256, 512])
    filter_sizes = trial.suggest_categorical("filter_sizes", [256, 512, 768, 1024])
    gradient_clip = trial.suggest_float("gradient_clip", 0.5, 2.0)
    scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "step"])
    
    train, val, test = dataset.get_dataset(
        split=['train', 'val', 'test'], 
        columns=['code', 'target_binary'],
        train_subset=1.0,
        dynamic_split_sizing=True,
        max_split_ratio=0.5
    )
    
    tokenize = lambda x: tokenize_fn(tokenizer, x, max_length=512)
    train = train.map(tokenize, batched=True)
    val = val.map(tokenize, batched=True)
    
    train.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])
    
    base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
    model = CBMClassifier(
        base_model=base_model,
        lstm_hidden_dim=lstm_hidden_dim,
        filter_sizes=filter_sizes,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=search_epochs, eta_min=1e-7)
    else:
        scheduler = StepLR(optimizer, step_size=search_epochs//3, gamma=0.1)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    train_dataloader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    epoch_progress = tqdm(range(search_epochs), desc=f"Trial {trial.number}", position=1, leave=False)
    
    for epoch in epoch_progress:

        model.train()
        total_loss = 0.0
        
        for data in train_dataloader:
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
            total_loss += loss.item()
        

        val_metrics = evaluate_model(val_dataloader, model, criterion, device, log_to_tensorboard=False)
        scheduler.step()
        
        current_f1 = val_metrics['f1']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            patience_counter = 0
        else:
            patience_counter += 1
        

        epoch_progress.set_postfix({
            "val_f1": f"{current_f1:.4f}",
            "best_f1": f"{best_val_f1:.4f}",
            "patience": f"{patience_counter}/{patience}"
        })
            

        if patience_counter >= patience:
            epoch_progress.set_description(f"Trial {trial.number} (Early Stop)")
            break
        

        trial.report(current_f1, epoch)
        

        if trial.should_prune():
            epoch_progress.set_description(f"Trial {trial.number} (Pruned)")
            epoch_progress.close()
            raise optuna.exceptions.TrialPruned()
    
    epoch_progress.close()
    return best_val_f1


def hyperparameter_search(args, dataset, tokenizer, device):
    """Perform hyperparameter search using Optuna."""
    logger.info("Starting hyperparameter search...")
    
    best_params, best_score = load_best_hyperparameters()
    if best_params:
        logger.info(f"Found existing best hyperparameters with F1 score: {best_score:.4f}")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    search_progress = tqdm(total=args.n_trials, desc="Hyperparameter Search", position=0)
    
    def callback(study, trial):
        if study.best_trial == trial:
            save_best_hyperparameters(trial.params, trial.value)
        

        search_progress.update(1)
        search_progress.set_postfix({
            "trial": f"{len(study.trials)}/{args.n_trials}",
            "best_f1": f"{study.best_value:.4f}" if study.best_value else "N/A",
            "current_f1": f"{trial.value:.4f}" if trial.value else "Pruned"
        })
    
    study.optimize(
        lambda trial: objective(trial, dataset, tokenizer, device, args.search_epochs),
        n_trials=args.n_trials,
        callbacks=[callback]
    )
    
    search_progress.close()
    
    logger.info("Hyperparameter search completed!")
    logger.info(f"Best F1 score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    return study.best_params, study.best_value


def main():
    from transformers import RobertaTokenizer, RobertaModel
    from data.dataset.codet_m4 import CoDeTM4
    from models.cbmclassifier import CBMClassifier
    from utils.utils import tokenize_fn
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = CoDeTM4(cache_dir=args.cache_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    if args.hyperparameter_search:
        best_params, best_score = hyperparameter_search(args, dataset, tokenizer, device)
        logger.info("Hyperparameter search completed. Use the found parameters for training.")
        return
    
    best_params, _ = load_best_hyperparameters()
    if best_params:
        logger.info("Using saved best hyperparameters for training")

        args.learning_rate = best_params.get("learning_rate", args.learning_rate)
        args.batch_size = best_params.get("batch_size", args.batch_size)
        args.weight_decay = best_params.get("weight_decay", args.weight_decay)
        args.gradient_clip = best_params.get("gradient_clip", args.gradient_clip)
        dropout_rate = best_params.get("dropout_rate", 0.5)
        lstm_hidden_dim = best_params.get("lstm_hidden_dim", 256)
        filter_sizes = best_params.get("filter_sizes", 768)
        scheduler_type = best_params.get("scheduler", "cosine")
    else:
        logger.info("No saved hyperparameters found, using default values")
        dropout_rate = 0.5
        lstm_hidden_dim = 256
        filter_sizes = 768
        scheduler_type = "cosine"

    print(f"Training with epochs: {args.epochs}")
    
    if args.eval:
        # For evaluation, use full test set without downsampling
        train, val, test = dataset.get_dataset(
            split=['train', 'val', 'test'], 
            columns=['code', 'target_binary'],
            train_subset=0.01,  # Keep small training set for consistency
            dynamic_split_sizing=False,  # Don't limit test set size
            max_split_ratio=0.5
        )
        logger.info(f"Evaluation mode: Using full test set with {len(test)} samples")
    else:
        # For training, use smaller subsets for faster training
        train, val, test = dataset.get_dataset(
            split=['train', 'val', 'test'], 
            columns=['code', 'target_binary'],
            train_subset=.1,
            dynamic_split_sizing=True,
            max_split_ratio=0.5
        )

    tokenize = lambda x: tokenize_fn(tokenizer, x, max_length=512)
    train = train.map(tokenize, batched=True)
    val = val.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])

    base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
    model = CBMClassifier(
        base_model=base_model,
        lstm_hidden_dim=lstm_hidden_dim,
        filter_sizes=filter_sizes,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use the scheduler type from hyperparameters
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    else:
        scheduler = StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)

    train_dataloader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )
    test_dataloader = DataLoader(
        test, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )

    models_dir = "models/cbm_codet"
    best_model_path = os.path.join(models_dir, "best_model.pth")

    if args.eval:
        if not os.path.exists(best_model_path):
            logger.error(
                f"Model file not found at {best_model_path}. Train the model first."
            )
            sys.exit(1)

        logger.info("Loading pre-trained model for evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Model loaded successfully.")
    else:
        # Load existing model if it exists before training
        if os.path.exists(best_model_path):
            logger.info("Loading existing model to continue training...")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logger.info("Existing model loaded successfully.")
        else:
            logger.info("No existing model found. Starting training from scratch.")
            
        model = train_model(
            args.epochs,
            train_dataloader,
            val_dataloader,
            model,
            optimizer,
            criterion,
            scheduler,
            device,
            device,
            args.patience,
            args.log_interval,
            args.gradient_clip,
        )

    test_metrics = evaluate_model(test_dataloader, model, criterion, device, log_to_tensorboard=True)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
