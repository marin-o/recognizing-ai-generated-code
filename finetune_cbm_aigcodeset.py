#!/usr/bin/env python3
"""
Minimalist script to finetune CBM CoDeTM4 model on AIGCodeSet dataset.
"""

import sys
import os
import torch
import logging
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset.aigcodeset import AIGCodeSet
from data.dataset.codet_m4 import CoDeTM4
from models.cbmclassifier import CBMClassifier
from utils.utils import tokenize_fn
from transformers import RobertaModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """Custom collate function to handle HuggingFace dataset format"""
    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    
    # Convert to tensors for tokenizer outputs
    for key in ['input_ids', 'attention_mask']:
        if key in collated:
            collated[key] = torch.tensor(collated[key], dtype=torch.long)
    
    # Convert targets to tensors (handle both target and target_binary)
    for key in ['target', 'target_binary']:
        if key in collated:
            collated[key] = torch.tensor(collated[key], dtype=torch.long)
    
    return collated

def evaluate_model(model, dataloader, criterion, device):
    """Quick evaluation function"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            
            # Handle both target and target_binary columns
            if "target" in data:
                labels = data["target"].to(device)
            elif "target_binary" in data:
                labels = data["target_binary"].to(device)
            else:
                raise KeyError("Neither 'target' nor 'target_binary' found in data")
                
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

def main():
    # Configuration
    PRETRAINED_MODEL_PATH = "models/cbm/cbm_codet_full/best_model.pth"  # Adjust path if needed
    LEARNING_RATE = 1e-5  # Low learning rate for finetuning
    EPOCHS = 3
    BATCH_SIZE = 8  # Small batch size to avoid memory issues
    MAX_LENGTH = 512
    CACHE_DIR = "data"  # Cache directory for datasets
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CATASTROPHIC_FORGETTING_CHECK_INTERVAL = 3  # Check every N epochs
    
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Finetuning CBM model on AIGCodeSet with LR={LEARNING_RATE}, epochs={EPOCHS}")
    logger.info(f"Will check for catastrophic forgetting on CoDeTM4 every {CATASTROPHIC_FORGETTING_CHECK_INTERVAL} epochs")
    
        # Load AIGCodeSet dataset
    print("Loading AIGCodeSet dataset...")
    aig_dataset = AIGCodeSet(cache_dir=CACHE_DIR)
    train_dataset, val_dataset, test_dataset = aig_dataset.get_dataset(
        split=True, 
        test_size=0.2, 
        val_size=0.1
    )
    
    logger.info(f"AIGCodeSet sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Load CoDeTM4 dataset for catastrophic forgetting evaluation
    print("Loading CoDeTM4 dataset for catastrophic forgetting evaluation...")
    codet_dataset = CoDeTM4(cache_dir=CACHE_DIR)
    codet_val_dataset = codet_dataset.get_dataset(
        split='val',
        val_ratio=0.1,
        test_ratio=0.2,
        columns=['code', 'target_binary']  
    )
    
    logger.info(f"CoDeTM4 validation size: {len(codet_val_dataset)}")
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    tokenize = lambda x: tokenize_fn(tokenizer, x, max_length=MAX_LENGTH)
    
    train = train_dataset.map(tokenize, batched=True)
    val = val_dataset.map(tokenize, batched=True)
    test = test_dataset.map(tokenize, batched=True)
    
    # Tokenize CoDeTM4 validation set for catastrophic forgetting check
    codet_val = codet_val_dataset.map(tokenize, batched=True)
    
    # Create dataloaders
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    codet_val_loader = DataLoader(codet_val, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Load pretrained CBM model
    logger.info(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
    
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        logger.error(f"Pretrained model not found at {PRETRAINED_MODEL_PATH}")
        logger.info("Available model paths:")
        for root, dirs, files in os.walk("models/"):
            for file in files:
                if file.endswith(".pth"):
                    logger.info(f"  {os.path.join(root, file)}")
        return
    
    # Check if it's a legacy model (direct state_dict) or new format
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'base_model.embeddings.word_embeddings.weight' in checkpoint:
        # Legacy format - direct state_dict
        logger.info("Loading legacy model format...")
        base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
        model = CBMClassifier(
            base_model=base_model,
            lstm_hidden_dim=256,
            filter_sizes=768,
            dropout_rate=0.5
        ).to(DEVICE)
        model.load_state_dict(checkpoint)
    else:
        # New format with metadata
        logger.info("Loading new model format...")
        config = checkpoint.get('model_config', {
            'lstm_hidden_dim': 256,
            'filter_sizes': 768,
            'dropout_rate': 0.5
        })
        base_model = RobertaModel.from_pretrained('microsoft/codebert-base')
        model = CBMClassifier(
            base_model=base_model,
            lstm_hidden_dim=config['lstm_hidden_dim'],
            filter_sizes=config['filter_sizes'],
            dropout_rate=config['dropout_rate']
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully!")
    
    # Setup optimizer with low learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=f"runs/cbm_finetune_aigcodeset")
    
    # Initial evaluation
    logger.info("Initial evaluation on AIGCodeSet...")
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)
    logger.info(f"Initial AIGCodeSet validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    # Initial evaluation on CoDeTM4 to establish baseline
    logger.info("Initial evaluation on CoDeTM4 (original dataset)...")
    codet_val_loss, codet_val_acc = evaluate_model(model, codet_val_loader, criterion, DEVICE)
    logger.info(f"Initial CoDeTM4 validation - Loss: {codet_val_loss:.4f}, Accuracy: {codet_val_acc:.2f}%")
    initial_codet_acc = codet_val_acc  # Store for catastrophic forgetting detection
    
    # Training loop
    logger.info("Starting finetuning...")
    best_val_acc = val_acc
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, data in enumerate(progress_bar):
            input_ids = data["input_ids"].to(DEVICE)
            attention_mask = data["attention_mask"].to(DEVICE)
            
            # Handle both target and target_binary columns
            if "target" in data:
                labels = data["target"].to(DEVICE)
            elif "target_binary" in data:
                labels = data["target_binary"].to(DEVICE)
            else:
                raise KeyError("Neither 'target' nor 'target_binary' found in data")
                
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("train/loss", loss.item(), global_step)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)
        avg_train_loss = total_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check for catastrophic forgetting every N epochs
        if (epoch + 1) % CATASTROPHIC_FORGETTING_CHECK_INTERVAL == 0:
            logger.info(f"Checking for catastrophic forgetting on CoDeTM4 (epoch {epoch+1})...")
            codet_val_loss, codet_val_acc = evaluate_model(model, codet_val_loader, criterion, DEVICE)
            accuracy_drop = initial_codet_acc - codet_val_acc
            logger.info(f"CoDeTM4 validation - Loss: {codet_val_loss:.4f}, Accuracy: {codet_val_acc:.2f}% (drop: {accuracy_drop:.2f}%)")
            
            # Log catastrophic forgetting metrics
            writer.add_scalar("catastrophic_forgetting/codet_loss", codet_val_loss, epoch)
            writer.add_scalar("catastrophic_forgetting/codet_accuracy", codet_val_acc, epoch)
            writer.add_scalar("catastrophic_forgetting/accuracy_drop", accuracy_drop, epoch)
            
            if accuracy_drop > 10.0:  # Alert if accuracy drops more than 10%
                logger.warning(f"⚠️  POTENTIAL CATASTROPHIC FORGETTING DETECTED! CoDeTM4 accuracy dropped by {accuracy_drop:.2f}%")
            elif accuracy_drop > 5.0:  # Warning for moderate drops
                logger.warning(f"⚠️  Moderate performance drop on CoDeTM4: {accuracy_drop:.2f}%")
            else:
                logger.info(f"✅ No significant catastrophic forgetting detected (drop: {accuracy_drop:.2f}%)")
        
        # Log to tensorboard
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "models/cbm_finetuned_aigcodeset"
            os.makedirs(save_path, exist_ok=True)
            
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': {
                    'lstm_hidden_dim': model.lstm_hidden_dim,
                    'filter_sizes': model.filter_sizes,
                    'dropout_rate': model.dropout_rate,
                    'embedding_dim': model.embedding_dim,
                },
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'finetuned_from': PRETRAINED_MODEL_PATH
            }
            
            torch.save(checkpoint_data, os.path.join(save_path, "best_model.pth"))
            logger.info(f"Saved best model with accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation on test set
    logger.info("Final evaluation on AIGCodeSet test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    logger.info(f"Final AIGCodeSet test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # Final evaluation on CoDeTM4 to check final state
    logger.info("Final evaluation on CoDeTM4 to check for catastrophic forgetting...")
    final_codet_loss, final_codet_acc = evaluate_model(model, codet_val_loader, criterion, DEVICE)
    final_accuracy_drop = initial_codet_acc - final_codet_acc
    logger.info(f"Final CoDeTM4 validation - Loss: {final_codet_loss:.4f}, Accuracy: {final_codet_acc:.2f}%")
    logger.info(f"Total CoDeTM4 accuracy drop: {final_accuracy_drop:.2f}% (from {initial_codet_acc:.2f}% to {final_codet_acc:.2f}%)")
    
    # Log final results
    writer.add_scalar("final/aigcodeset_test_loss", test_loss, EPOCHS)
    writer.add_scalar("final/aigcodeset_test_accuracy", test_acc, EPOCHS)
    writer.add_scalar("final/codet_loss", final_codet_loss, EPOCHS)
    writer.add_scalar("final/codet_accuracy", final_codet_acc, EPOCHS)
    writer.add_scalar("final/total_accuracy_drop", final_accuracy_drop, EPOCHS)
    writer.close()
    
    logger.info("Finetuning completed!")
    logger.info(f"Best AIGCodeSet validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Final AIGCodeSet test accuracy: {test_acc:.2f}%")
    logger.info(f"Final CoDeTM4 accuracy: {final_codet_acc:.2f}% (drop: {final_accuracy_drop:.2f}%)")
    logger.info(f"Model saved to: models/cbm_finetuned_aigcodeset/best_model.pth")
    
    # Catastrophic forgetting summary
    if final_accuracy_drop > 10.0:
        logger.warning(f"🚨 SIGNIFICANT CATASTROPHIC FORGETTING: {final_accuracy_drop:.2f}% accuracy drop on original dataset!")
    elif final_accuracy_drop > 5.0:
        logger.warning(f"⚠️  Moderate catastrophic forgetting: {final_accuracy_drop:.2f}% accuracy drop on original dataset")
    else:
        logger.info(f"✅ Minimal catastrophic forgetting: only {final_accuracy_drop:.2f}% accuracy drop on original dataset")

if __name__ == "__main__":
    main()
