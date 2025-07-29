import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy, F1Score, Recall, Precision
from transformers import RobertaTokenizer
from tqdm import tqdm
from data.dataset import AIGCodeSet
from models.baseline_model import SimpleLinearHeadClassifier
from utils.utils import tokenize_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        labels = data["target"].to(device)

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

    # Initialize metrics - assuming binary classification (num_classes=2)
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
            labels = data["target"].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_predictions.append(predicted)
            all_labels.append(labels)

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Calculate metrics
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

    # Set up TensorBoard logging
    tensorboard_run_name = "baseline"
    log_dir = os.path.join("tensorboard_logs", "AIGCodeSet", tensorboard_run_name)
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
    patience_counter = 0
    models_dir = "models/baseline"
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

    model.load_state_dict(torch.load(best_model_path))
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and preprocess dataset
    dataset = AIGCodeSet(cache_dir=args.cache_dir)
    train, val, test = dataset.get_dataset(
        split=True, test_size=args.test_size, val_size=args.val_size
    )
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    tokenize = lambda x: tokenize_fn(tokenizer, x, max_length=512)
    train = train.map(tokenize, batched=True)
    val = val.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])

    # Initialize model, optimizer, and criterion
    model = SimpleLinearHeadClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Create dataloaders
    train_dataloader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )
    test_dataloader = DataLoader(
        test, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )

    models_dir = "models/baseline"
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

    # Evaluate model
    test_metrics = evaluate_model(test_dataloader, model, criterion, device)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
