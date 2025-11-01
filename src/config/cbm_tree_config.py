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

Train a new CBM StarCoder Tree model with specified or default hyperparameters.

Usage:
  python cbm_tree.py --train [OPTIONS]

Key Options:
  --model-name NAME         Name of the model (default: cbm_tree_baseline)
  --backbone-type TYPE      Backbone type: 'codebert' or 'starcoder' (default: codebert)
  --use-best-params        Use best hyperparameters from Optuna study
  --epochs N               Number of training epochs (default: 40)
  --learning-rate RATE     Learning rate for optimizer (default: 1e-3)
  --batch-size SIZE        Batch size for training (default: 16)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --seed SEED              Random seed for reproducibility (default: 42)
  --freeze-backbone        Freeze backbone model parameters (default: True)

Examples:
  # Basic training with CodeBERT backbone
  python cbm_tree.py --train --model-name my_cbm_tree_model

  # Training with StarCoder 3B backbone (requires more VRAM)
  python cbm_tree.py --train --model-name cbm_starcoder --backbone-type starcoder

  # Training with custom hyperparameters
  python cbm_tree.py --train --model-name my_cbm_tree_model --learning-rate 1e-4 --epochs 50

  # Training using optimized hyperparameters from previous optimization
  python cbm_tree.py --train --model-name my_cbm_tree_model --use-best-params

  # Training with unfrozen backbone for fine-tuning
  python cbm_tree.py --train --model-name my_cbm_tree_model --no-freeze-backbone

Note: If --use-best-params is specified, the script will load the best hyperparameters
from the Optuna study with the same name as the model.
        """,
        
        'resume': """
RESUME MODE HELP
================

Resume training from a saved CBM Tree model checkpoint.

Usage:
  python cbm_tree.py --resume [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to resume (default: cbm_tree_baseline)
  --epochs N               Additional epochs to train (default: 40)
  --batch-size SIZE        Batch size for training (default: 16)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Resume training for 20 more epochs
  python cbm_tree.py --resume --model-name my_cbm_tree_model --epochs 20

  # Resume training with larger batch size
  python cbm_tree.py --resume --model-name my_cbm_tree_model --batch-size 32

Note: The model checkpoint must exist at models/cbm_tree/{model_name}/best_model.pth
All hyperparameters (learning rate, architecture, scheduler) are loaded from the checkpoint.
        """,
        
        'optimize': """
OPTIMIZE MODE HELP
==================

Run hyperparameter optimization using Optuna to find the best CBM Tree model configuration.

Usage:
  python cbm_tree.py --optimize [OPTIONS]

Key Options:
  --model-name NAME         Name of the model/study (default: cbm_tree_baseline)
  --backbone-type TYPE      Backbone type: 'codebert' or 'starcoder' (default: codebert)
  --n-trials N             Number of optimization trials (default: 50)
  --study-name NAME        Custom Optuna study name (default: same as model-name)
  --storage-url URL        Optuna database URL (default: sqlite:///optuna/cbm_tree.db)
  --search-epochs N        Epochs per trial (default: 15)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic optimization with 100 trials
  python cbm_tree.py --optimize --model-name my_cbm_tree_model --n-trials 100

  # Optimization with StarCoder backbone
  python cbm_tree.py --optimize --model-name cbm_starcoder --backbone-type starcoder --n-trials 50

  # Quick optimization with fewer epochs per trial
  python cbm_tree.py --optimize --model-name my_cbm_tree_model --n-trials 20 --search-epochs 10

Note: The optimization will resume from existing trials if a study with the same name exists.
        """,
        
        'eval': """
EVAL MODE HELP
==============

Evaluate a trained CBM Tree model on the test dataset.

Usage:
  python cbm_tree.py --eval [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to evaluate (default: cbm_tree_baseline)
  --batch-size SIZE        Batch size for evaluation (default: 16)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --full-test-set          Use full test set instead of downsampled version

Examples:
  # Basic evaluation
  python cbm_tree.py --eval --model-name my_cbm_tree_model

  # Evaluation on full test set
  python cbm_tree.py --eval --model-name my_cbm_tree_model --full-test-set

  # Evaluation with larger batch size for faster inference
  python cbm_tree.py --eval --model-name my_cbm_tree_model --batch-size 64

Note: The model must have been trained and saved before evaluation. The script will
look for the model checkpoint at models/cbm_tree/{model_name}/best_model.pth
        """
    }
    
    if command in help_text:
        print(help_text[command])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, resume, optimize, eval")
        print("Use 'python cbm_tree.py --help' for general help")


def parse_args():
    """Parse command line arguments for CBM Tree experiments"""
    # Check for command-specific help first
    if len(sys.argv) >= 3 and sys.argv[1] == '-h':
        show_command_help(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 3 and sys.argv[2] == '-h' and sys.argv[1] in ['train', 'resume', 'optimize', 'eval']:
        show_command_help(sys.argv[1])
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="CBM StarCoder Tree model training and evaluation script for code classification with tree-sitter features",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  Basic training with CodeBERT backbone:
    python cbm_tree.py --train --model-name my_cbm_tree_model

  Training with StarCoder 3B backbone:
    python cbm_tree.py --train --model-name cbm_starcoder --backbone-type starcoder

  Training with custom learning rate and epochs:
    python cbm_tree.py --train --model-name my_cbm_tree_model --learning-rate 1e-4 --epochs 50

  Training using best hyperparameters from optimization:
    python cbm_tree.py --train --model-name my_cbm_tree_model --use-best-params

  Resume training from a checkpoint:
    python cbm_tree.py --resume --model-name my_cbm_tree_model --epochs 20

  Hyperparameter optimization with 100 trials:
    python cbm_tree.py --optimize --model-name my_cbm_tree_model --n-trials 100

  Evaluate a trained model on test set:
    python cbm_tree.py --eval --model-name my_cbm_tree_model

  Evaluate on full test set:
    python cbm_tree.py --eval --model-name my_cbm_tree_model --full-test-set

  Training on SemEval dataset with specific subtask:
    python cbm_tree.py --train --model-name semeval_cbm_tree --subtask A

  Complete workflow example:
    # 1. Optimize hyperparameters
    python cbm_tree.py --optimize --model-name best_cbm_tree --n-trials 100
    
    # 2. Train with optimized parameters
    python cbm_tree.py --train --model-name best_cbm_tree --use-best-params --epochs 50
    
    # 3. Evaluate the final model
    python cbm_tree.py --eval --model-name best_cbm_tree --full-test-set

Command-specific help:
    python cbm_tree.py -h train      # Show detailed help for training mode
    python cbm_tree.py -h resume     # Show detailed help for resume mode
    python cbm_tree.py -h optimize   # Show detailed help for optimization mode
    python cbm_tree.py -h eval       # Show detailed help for evaluation mode
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
        default="cbm_tree_baseline",
        help="Name of the model (affects save location and Optuna study name)"
    )
    model_group.add_argument(
        "--use-best-params",
        action="store_true",
        help="Use best hyperparameters from Optuna study (only for --train mode)"
    )
    model_group.add_argument(
        "--backbone-type",
        type=str,
        choices=["codebert", "starcoder"],
        default="codebert",
        help="Type of backbone model to use"
    )
    model_group.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze backbone model parameters during training"
    )
    model_group.add_argument(
        "--no-freeze-backbone",
        action="store_false",
        dest="freeze_backbone",
        help="Do not freeze backbone model parameters (allow fine-tuning)"
    )
    model_group.add_argument(
        "--tree-feature-projection-dim",
        type=int,
        default=128,
        help="Dimension to project tree features to before concatenation"
    )
    
    # Architecture parameters
    arch_group = parser.add_argument_group("Architecture Parameters")
    arch_group.add_argument(
        "--filter-sizes",
        type=int,
        default=768,
        help="Number of filters for CNN layers"
    )
    arch_group.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension of BiLSTM"
    )
    arch_group.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Dropout rate for regularization"
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
        default="sqlite:///optuna/cbm_tree.db",
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
        "--subtask",
        type=str,
        choices=["A", "B", "C"],
        default="A",
        help="Subtask for SemEval dataset"
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
    logging_group.add_argument(
        "--enable-misclassification-analysis",
        action="store_true",
        help="Enable detailed misclassification analysis during evaluation (creates plots and statistics)"
    )
    logging_group.add_argument(
        "--analysis-dir",
        type=str,
        default="analysis",
        help="Directory to save misclassification analysis results"
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
