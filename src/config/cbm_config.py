import argparse
import sys


class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter that preserves raw formatting in epilog while showing defaults"""
    def _fill_text(self, text, width, indent):
        if text.startswith('\nExamples:'):
            return text
        return super()._fill_text(text, width, indent)


def show_command_help(command):
    """Show specific help for a command"""
    help_text = {
        'train': """
TRAIN MODE HELP
===============

Train a new CBM (CNN-BiLSTM) model with specified or default hyperparameters.

Usage:
  python cbm.py --train [OPTIONS]

Key Options:
  --model-name NAME         Name of the model (default: cbm_baseline)
  --use-best-params        Use best hyperparameters from Optuna study
  --epochs N               Number of training epochs (default: 40)
  --learning-rate RATE     Learning rate for optimizer (default: 1e-3)
  --batch-size SIZE        Batch size for training (default: 16)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --seed SEED              Random seed for reproducibility (default: 42)

Examples:
  # Basic training with default parameters
  python cbm.py --train --model-name my_cbm_model

  # Training with custom hyperparameters
  python cbm.py --train --model-name my_cbm_model --learning-rate 1e-4 --epochs 50

  # Training using optimized hyperparameters from previous optimization
  python cbm.py --train --model-name my_cbm_model --use-best-params

  # Training with specific device and larger batch size
  python cbm.py --train --model-name my_cbm_model --device cuda --batch-size 32

Note: If --use-best-params is specified, the script will load the best hyperparameters
from the Optuna study with the same name as the model.
        """,
        
        'resume': """
RESUME MODE HELP
================

Resume training from a saved CBM model checkpoint.

Usage:
  python cbm.py --resume [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to resume (default: cbm_baseline)
  --epochs N               Additional epochs to train (default: 40)
  --batch-size SIZE        Batch size for training (default: 16)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Resume training for 20 more epochs
  python cbm.py --resume --model-name my_cbm_model --epochs 20

  # Resume training with larger batch size
  python cbm.py --resume --model-name my_cbm_model --batch-size 32

Note: The model checkpoint must exist at models/cbm/{model_name}/best_model.pth
All hyperparameters (learning rate, architecture, scheduler) are loaded from the checkpoint.
        """,
        
        'optimize': """
OPTIMIZE MODE HELP
==================

Run hyperparameter optimization using Optuna to find the best CBM model configuration.

Usage:
  python cbm.py --optimize [OPTIONS]

Key Options:
  --model-name NAME         Name of the model/study (default: cbm_baseline)
  --n-trials N             Number of optimization trials (default: 50)
  --study-name NAME        Custom Optuna study name (default: same as model-name)
  --storage-url URL        Optuna database URL (default: sqlite:///optuna/cbm.db)
  --epochs N               Epochs per trial (default: 15)
  --batch-size SIZE        Batch size (default: 16)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic optimization with 100 trials
  python cbm.py --optimize --model-name my_cbm_model --n-trials 100

  # Optimization with custom study name
  python cbm.py --optimize --model-name my_cbm_model --study-name cbm_experiment_v2 --n-trials 50

  # Quick optimization with fewer epochs per trial
  python cbm.py --optimize --model-name my_cbm_model --n-trials 20 --epochs 10

  # Optimization with custom database location
  python cbm.py --optimize --model-name my_cbm_model --storage-url sqlite:///my_cbm_studies.db

Note: The optimization will resume from existing trials if a study with the same name exists.
        """,
        
        'eval': """
EVAL MODE HELP
==============

Evaluate a trained CBM model on the test dataset.

Usage:
  python cbm.py --eval [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to evaluate (default: cbm_baseline)
  --batch-size SIZE        Batch size for evaluation (default: 16)
  --cache-dir DIR          Directory for dataset cache (default: data/)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --full-test-set          Use full test set instead of downsampled version

Examples:
  # Basic evaluation
  python cbm.py --eval --model-name my_cbm_model

  # Evaluation on full test set
  python cbm.py --eval --model-name my_cbm_model --full-test-set

  # Evaluation with larger batch size for faster inference
  python cbm.py --eval --model-name my_cbm_model --batch-size 64

  # Force CPU evaluation
  python cbm.py --eval --model-name my_cbm_model --device cpu

Note: The model must have been trained and saved before evaluation. The script will
look for the model checkpoint at models/cbm/{model_name}/best_model.pth
        """
    }
    
    if command in help_text:
        print(help_text[command])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, resume, optimize, eval")
        print("Use 'python cbm.py --help' for general help")


def parse_args():
    """Parse command line arguments for CBM experiments"""
    # Check for command-specific help first
    if len(sys.argv) >= 3 and sys.argv[1] == '-h':
        show_command_help(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 3 and sys.argv[2] == '-h' and sys.argv[1] in ['train', 'resume', 'optimize', 'eval']:
        show_command_help(sys.argv[1])
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="CNN-BiLSTM (CBM) model training and evaluation script for code classification",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  Basic training with default parameters:
    python cbm.py --train --model-name my_cbm_model

  Training with custom learning rate and epochs:
    python cbm.py --train --model-name my_cbm_model --learning-rate 1e-4 --epochs 50

  Training using best hyperparameters from optimization:
    python cbm.py --train --model-name my_cbm_model --use-best-params

  Resume training from a checkpoint:
    python cbm.py --resume --model-name my_cbm_model --epochs 20

  Hyperparameter optimization with 100 trials:
    python cbm.py --optimize --model-name my_cbm_model --n-trials 100

  Evaluate a trained model on test set:
    python cbm.py --eval --model-name my_cbm_model

  Evaluate on full test set:
    python cbm.py --eval --model-name my_cbm_model --full-test-set

  Training with custom validation and test ratios:
    python cbm.py --train --model-name my_cbm_model --val-ratio 0.15 --test-ratio 0.25

  Complete workflow example:
    # 1. Optimize hyperparameters
    python cbm.py --optimize --model-name best_cbm_model --n-trials 100
    
    # 2. Train with optimized parameters
    python cbm.py --train --model-name best_cbm_model --use-best-params --epochs 50
    
    # 3. Evaluate the final model
    python cbm.py --eval --model-name best_cbm_model --full-test-set

Command-specific help:
    python cbm.py -h train      # Show detailed help for training mode
    python cbm.py -h resume     # Show detailed help for resume mode
    python cbm.py -h optimize   # Show detailed help for optimization mode
    python cbm.py -h eval       # Show detailed help for evaluation mode
        """,
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train a new model with default or specified hyperparameters"
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a saved checkpoint"
    )
    mode_group.add_argument(
        "--optimize",
        action="store_true", 
        help="Run hyperparameter optimization using Optuna"
    )
    mode_group.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a saved model on the test set"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-name",
        type=str,
        default="cbm_baseline",
        help="Name of the model (affects save location and Optuna study name)"
    )
    model_group.add_argument(
        "--use-best-params",
        action="store_true",
        help="Use best hyperparameters from Optuna study (only for --train mode)"
    )
    model_group.add_argument(
        "--pretrained-model",
        type=str,
        default="microsoft/codebert-base",
        help="Pretrained transformer model to use as base"
    )
    
    # Training parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer (only used when not using Optuna params)"
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    train_group.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value"
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping"
    )
    
    # Optimization parameters
    optim_group = parser.add_argument_group("Optimization Parameters")
    optim_group.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter optimization"
    )
    optim_group.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Custom Optuna study name (defaults to model name if not specified)"
    )
    optim_group.add_argument(
        "--storage-url", 
        type=str,
        default="sqlite:///optuna/cbm.db",
        help="Optuna storage database URL"
    )
    optim_group.add_argument(
        "--search-epochs",
        type=int,
        default=15,
        help="Number of epochs per trial during hyperparameter search"
    )
    
    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    data_group.add_argument(
        "--cache-dir",
        type=str,
        default="data/",
        help="Directory to cache dataset"
    )
    data_group.add_argument(
        "--train-subset",
        type=float,
        default=0.1,
        help="Fraction of training data to use (0.1 = 10%%)"
    )
    data_group.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio - fraction of available validation data to use (0.1 = 10%% of validation set)"
    )
    data_group.add_argument(
        "--test-ratio", 
        type=float,
        default=0.2,
        help="Test set ratio - fraction of available test data to use (0.2 = 20%% of test set)"
    )
    data_group.add_argument(
        "--full-test-set",
        action="store_true",
        help="Use full test set for evaluation (default: uses downsampled test set)"
    )
    data_group.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    data_group.add_argument(
        "--use-cleaned",
        action="store_true",
        help="Use cleaned (deduplicated) version of the dataset"
    )
    data_group.add_argument(
        "--cleaned-data-path",
        type=str,
        default=None,
        help="Path to cleaned dataset directory (if different from default)"
    )
    
    # System parameters
    system_group = parser.add_argument_group("System Parameters")
    system_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    system_group.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training/evaluation"
    )
    system_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    
    # Logging parameters
    logging_group = parser.add_argument_group("Logging Parameters")
    logging_group.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Base directory for tensorboard logs"
    )
    logging_group.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Number of batches between logging metrics during training"
    )
    logging_group.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Disable tensorboard logging"
    )
    
    args = parser.parse_args()
    
    # Set defaults and validate
    if args.study_name is None:
        args.study_name = args.model_name
    
    # Validate mode-specific arguments
    if args.use_best_params and not args.train:
        parser.error("--use-best-params can only be used with --train mode")
    
    if args.resume and args.use_best_params:
        parser.error("--use-best-params cannot be used with --resume mode (parameters are loaded from checkpoint)")
    
    if args.optimize and args.search_epochs < 5:
        print("Warning: Using less than 5 epochs for optimization may not give good results")
    
    return args
