import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, Recall, Precision
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from data.aigcodeset import AIGCodeSet
from models.cbmclassifier import CBMClassifier
from utils.utils import tokenize_fn
import mlflow

# Configure logging
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
        labels = data["target"].to(device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()

        running_loss += loss.item()
        
        # Update overall progress bar
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
            labels = data["target"].to(device)
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

    mlflow.set_experiment("AIGCodeSet")
    mlflow_run_name = "cbm"
    experiment_id = mlflow.get_experiment_by_name("AIGCodeSet").experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{mlflow_run_name}'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    )
    run_id = runs["run_id"].iloc[0] if not runs.empty else None
    with mlflow.start_run(run_id=run_id, run_name=mlflow_run_name) as run:
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test_f1_macro", test_metrics["f1"])
        mlflow.log_metric("recall", test_metrics["recall"])
        mlflow.log_metric("precision", test_metrics["precision"])
        mlflow.log_metric("test_loss", test_metrics["loss"])


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
    models_dir = "models/cbm"
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pth")

    # Calculate total training steps for overall progress bar
    total_steps = epochs * len(train_dataloader)
    overall_progress = tqdm(total=total_steps, desc="Training Progress", position=0)

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        avg_loss = train_one_epoch(
            epoch, train_dataloader, model, optimizer, criterion, device, log_interval, gradient_clip, overall_progress
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

    overall_progress.close()
    model.load_state_dict(torch.load(best_model_path))
    return model


def main():
    args = parse_args()
    print(args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
    model = CBMClassifier(base_model=base_model).to(device)  

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    train_dataloader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True
    )
    val_dataloader = DataLoader(
        val, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )
    test_dataloader = DataLoader(
        test, batch_size=args.batch_size, num_workers=6, pin_memory=True
    )

    models_dir = "models/cbm"
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
            args.gradient_clip,
        )

    test_metrics = evaluate_model(test_dataloader, model, criterion, device)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
