"""
Embedding Generation Utility for CBM Tree Models
=================================================

This script generates and saves embeddings from transformer backbones (CodeBERT/StarCoder)
to a Hugging Face dataset format. This allows for decoupling embedding generation from
model training, significantly speeding up experiments.

Features:
- Generate embeddings for entire datasets
- Save to Hugging Face dataset format
- Support for CodeBERT and StarCoder backbones
- Preserves all original columns plus embeddings
- Batch processing for efficiency

Usage:
    python generate_embeddings.py --backbone-type codebert --subtask A --output-name semeval_codebert_embeddings
    python generate_embeddings.py --backbone-type starcoder --subtask A --output-name semeval_starcoder_embeddings
"""

import sys
import os
import argparse
import logging
import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset.semeval2026_task13 import SemEval2026Task13

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_backbone_and_tokenizer(backbone_type: str):
    """
    Load the appropriate backbone model and tokenizer.
    
    Args:
        backbone_type: "codebert" or "starcoder"
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading {backbone_type} backbone...")
    
    if backbone_type == "codebert":
        model = RobertaModel.from_pretrained("microsoft/codebert-base")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    elif backbone_type == "starcoder":
        from transformers import BitsAndBytesConfig
        model = AutoModel.from_pretrained(
            "bigcode/starcoder2-3b",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starcoder2-3b", trust_remote_code=True
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    return model, tokenizer


def generate_embeddings_for_split(
    dataset: Dataset,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 512
):
    """
    Generate embeddings for a dataset split.
    
    Args:
        dataset: Hugging Face Dataset
        model: Backbone model
        tokenizer: Tokenizer
        device: Device to use
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Dataset with added 'embedding' column
    """
    model.eval()
    model.to(device)
    
    embeddings_list = []
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating embeddings"):
        batch = dataset[i:i+batch_size]
        codes = batch['code']
        
        # Tokenize
        inputs = tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            if cls_embeddings.dtype == torch.bfloat16:
                cls_embeddings = cls_embeddings.float()
            embeddings_list.extend(cls_embeddings.numpy().tolist())
    
    # Add embeddings to dataset
    dataset = dataset.add_column("embedding", embeddings_list)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save embeddings for SemEval 2026 Task 13 dataset"
    )
    
    parser.add_argument(
        "--backbone-type",
        type=str,
        choices=["codebert", "starcoder"],
        required=True,
        help="Type of backbone model to use for generating embeddings"
    )
    parser.add_argument(
        "--subtask",
        type=str,
        choices=["A", "B", "C"],
        default="A",
        help="SemEval subtask"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name for the output dataset (e.g., semeval_codebert_embeddings)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Directory to save the embeddings dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load backbone and tokenizer
    model, tokenizer = load_backbone_and_tokenizer(args.backbone_type)
    
    # Load dataset
    logger.info(f"Loading SemEval 2026 Task 13 subtask {args.subtask}...")
    dataset_loader = SemEval2026Task13(subtask=args.subtask)
    
    # Load all splits
    train, val, test = dataset_loader.get_dataset(split=['train', 'val', 'test'])
    
    logger.info(f"Train size: {len(train)}")
    logger.info(f"Val size: {len(val)}")
    logger.info(f"Test size: {len(test)}")
    
    # Generate embeddings for each split
    logger.info("Generating embeddings for train split...")
    train_with_embeddings = generate_embeddings_for_split(
        train, model, tokenizer, device, args.batch_size, args.max_length
    )
    
    logger.info("Generating embeddings for val split...")
    val_with_embeddings = generate_embeddings_for_split(
        val, model, tokenizer, device, args.batch_size, args.max_length
    )
    
    logger.info("Generating embeddings for test split...")
    test_with_embeddings = generate_embeddings_for_split(
        test, model, tokenizer, device, args.batch_size, args.max_length
    )
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_with_embeddings,
        'validation': val_with_embeddings,
        'test': test_with_embeddings
    })
    
    # Save to disk
    output_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Saving embeddings dataset to {output_path}...")
    dataset_dict.save_to_disk(output_path)
    
    logger.info("Done!")
    logger.info(f"Dataset saved to: {output_path}")
    logger.info(f"Embedding dimension: {len(train_with_embeddings[0]['embedding'])}")
    logger.info(f"Available columns: {train_with_embeddings.column_names}")


if __name__ == "__main__":
    main()
