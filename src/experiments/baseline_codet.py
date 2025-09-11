import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy, F1Score, Recall, Precision, Specificity, AUROC
from transformers import RobertaTokenizer
from tqdm import tqdm
from data.dataset import CoDeTM4
from models.baseline_model import SimpleLinearHeadClassifier
from utils.utils import tokenize_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow_run_name = "baseline_codet_full_x" 
models_dir = "models/baseline_codet_full_x"


def parse_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a CodeBERT-based classifier")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer"
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
        "--patience", type=int, default=3, help="Patience for early stopping"
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
        default=100,
        help="Number of batches between logging metrics",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Only evaluate the model without training",
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
) -> float:
    """
    Train the model for one epoch with live progress output.

    Args:
        epoch_index (int): Current epoch number.
        train_dataloader (DataLoader): DataLoader for training data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): Device to run the model on.
        log_interval (int): Number of batches between logging metrics.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_dataloader)

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch_index + 1}", leave=False)
    for i, data in enumerate(progress_bar):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["target_binary"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

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
) -> dict:
    """
    Evaluate the model on a dataset.

    Args:
        dataloader (DataLoader): DataLoader for evaluation data.
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): Device to run the model on.

    Returns:
        dict: Dictionary containing accuracy, f1, recall, precision, and loss.
    """
    model.eval()

    # Initialize metrics for binary classification (matching enhanced transformer)
    accuracy = Accuracy(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    specificity = Specificity(task="binary").to(device)
    auroc = AUROC(task="binary").to(device)

    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []  # For AUROC calculation

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["target_binary"].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.append(predicted)
            all_labels.append(labels)
            all_probabilities.append(probabilities[:, 1])  # Probability of positive class

    # Concatenate all predictions, labels, and probabilities
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_probabilities = torch.cat(all_probabilities)

    # Calculate metrics
    acc = accuracy(all_predictions, all_labels)
    f1_score = f1(all_predictions, all_labels)
    recall_score = recall(all_predictions, all_labels)
    precision_score = precision(all_predictions, all_labels)
    specificity_score = specificity(all_predictions, all_labels)
    auroc_score = auroc(all_probabilities, all_labels)
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0

    test_metrics = {
        "accuracy": acc.item(),
        "f1": f1_score.item(),
        "recall": recall_score.item(),
        "precision": precision_score.item(),
        "specificity": specificity_score.item(),
        "auroc": auroc_score.item(),
        "loss": avg_loss,
    }

    # Set up TensorBoard logging
    log_dir = os.path.join("tensorboard_logs", "AIGCodeSet", mlflow_run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log metrics (matching enhanced transformer format)
    writer.add_scalar("Metrics/test_accuracy", test_metrics["accuracy"])
    writer.add_scalar("Metrics/test_f1", test_metrics["f1"])
    writer.add_scalar("Metrics/test_recall", test_metrics["recall"])
    writer.add_scalar("Metrics/test_precision", test_metrics["precision"])
    writer.add_scalar("Metrics/test_specificity", test_metrics["specificity"])
    writer.add_scalar("Metrics/test_auroc", test_metrics["auroc"])
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
) -> torch.nn.Module:
    """
    Train the model with early stopping, learning rate scheduling, and live metrics.

    Args:
        epochs (int): Number of epochs to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the model on.
        patience (int): Number of epochs to wait for improvement before stopping.
        log_interval (int): Number of batches between logging metrics.

    Returns:
        torch.nn.Module: Trained model with best weights.
    """
    best_vloss = float("inf")
    best_accuracy = 0.0
    patience_counter = 0
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pth")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        avg_loss = train_one_epoch(
            epoch, train_dataloader, model, optimizer, criterion, device, log_interval
        )
        logger.info(f"Average training loss: {avg_loss:.4f}")

        val_metrics = evaluate_model(val_dataloader, model, criterion, device)
        logger.info(f"Validation loss: {val_metrics['loss']:.4f}")
        logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
        logger.info(f"Validation recall: {val_metrics['recall']:.4f}")
        logger.info(f"Validation precision: {val_metrics['precision']:.4f}")

        scheduler.step()

        # Check if current model is better using both accuracy and loss
        current_accuracy = val_metrics["accuracy"]
        current_loss = val_metrics["loss"]
        
        # Model is better if accuracy improved OR (accuracy same but loss improved)
        is_better = (current_accuracy > best_accuracy) or \
               (current_accuracy == best_accuracy and current_loss < best_vloss)
        
        if is_better:
            best_vloss = current_loss
            best_accuracy = current_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            epoch_info_path = os.path.join(models_dir, "best_model_epoch.txt")
            with open(epoch_info_path, "w") as f:
                f.write(f"Best model saved from epoch: {epoch + 1}\n")
                f.write(f"Validation accuracy: {best_accuracy:.4f}\n")
                f.write(f"Validation loss: {best_vloss:.4f}\n")
            logger.info(f"Model saved with accuracy: {best_accuracy:.4f}, loss: {best_vloss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement"
                )
                break

    model.load_state_dict(torch.load(best_model_path))
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and preprocess dataset
    dataset = CoDeTM4(cache_dir=args.cache_dir)
    train, val, test = dataset.get_dataset(
        columns=['code','target_binary'],
        split=['train','val','test'],
        train_subset=1.0,
        dynamic_split_sizing=False,
    )
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    tokenize = lambda x: tokenize_fn(tokenizer, x)
    train = train.map(tokenize, batched=True, num_proc=12)
    val = val.map(tokenize, batched=True, num_proc=12)
    test = test.map(tokenize, batched=True, num_proc=12)

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask", "target_binary"])

    model = SimpleLinearHeadClassifier(freeze_codebert=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_dataloader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )
    test_dataloader = DataLoader(
        test, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )

    best_model_path = os.path.join(models_dir, "best_model.pth")

    if args.eval:
        # Evaluation only mode
        if not os.path.exists(best_model_path):
            logger.error(
                f"Model file not found at {best_model_path}. Train the model first."
            )
            sys.exit(1)

        logger.info("Loading pre-trained model for evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Model loaded successfully.")
    else:
        # Training mode
        model = train_model(
            args.epochs,
            train_dataloader,
            val_dataloader,
            model,
            optimizer,
            criterion,
            scheduler,
            device,
            args.patience,
            args.log_interval,
        )

    # Evaluate model (matching enhanced transformer output format)
    test_metrics = evaluate_model(test_dataloader, model, criterion, device)
    
    logger.info("\n" + "="*50)
    logger.info("BASELINE CODET EVALUATION RESULTS:")
    logger.info("="*50)
    logger.info(f"Test Loss: {test_metrics['loss']:.5f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.6f}")
    logger.info(f"Test precision: {test_metrics['precision']:.6f}")
    logger.info(f"Test recall: {test_metrics['recall']:.6f}")
    logger.info(f"Test specificity: {test_metrics['specificity']:.6f}")
    logger.info(f"Test auroc: {test_metrics['auroc']:.6f}")
    logger.info(f"Test f1: {test_metrics['f1']:.6f}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
