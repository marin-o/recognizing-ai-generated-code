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

Train a new GAT model with specified or default hyperparameters.

Usage:
  python gat.py --train [OPTIONS]

Key Options:
  --model-name NAME         Name of the model (default: gat_baseline)
  --use-best-params        Use best hyperparameters from Optuna study
  --epochs N               Number of training epochs (default: 50)
  --learning-rate RATE     Learning rate for optimizer (default: 0.001)
  --batch-size SIZE        Batch size for training (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --seed SEED              Random seed for reproducibility (default: 872002)

Examples:
  # Basic training with default parameters
  python gat.py --train --model-name my_gat_model

  # Training with custom hyperparameters
  python gat.py --train --model-name my_gat_model --learning-rate 0.01 --epochs 100

  # Training using optimized hyperparameters from previous optimization
  python gat.py --train --model-name my_gat_model --use-best-params

  # Training with specific device and larger batch size
  python gat.py --train --model-name my_gat_model --device cuda --batch-size 256

Note: If --use-best-params is specified, the script will load the best hyperparameters
from the Optuna study with the same name as the model.
        """,
        
        'resume': """
RESUME MODE HELP
================

Resume training from a saved model checkpoint.

Usage:
  python gat.py --resume [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to resume (default: gat_baseline)
  --epochs N               Additional epochs to train (default: 50)
  --batch-size SIZE        Batch size for training (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Resume training for 100 more epochs
  python gat.py --resume --model-name my_gat_model --epochs 100

  # Resume training with larger batch size
  python gat.py --resume --model-name my_gat_model --batch-size 256

Note: The model checkpoint must exist at models/gat/{model_name}/{model_name}_best.pth
All hyperparameters (learning rate, architecture, scheduler) are loaded from the checkpoint.
        """,
        
        'optimize': """
OPTIMIZE MODE HELP
==================

Run hyperparameter optimization using Optuna to find the best model configuration.

Usage:
  python gat.py --optimize [OPTIONS]

Key Options:
  --model-name NAME         Name of the model/study (default: gat_baseline)
  --n-trials N             Number of optimization trials (default: 50)
  --study-name NAME        Custom Optuna study name (default: same as model-name)
  --storage-url URL        Optuna database URL (default: sqlite:///optuna/gat.db)
  --epochs N               Epochs per trial (default: 50)
  --batch-size SIZE        Batch size (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic optimization with 100 trials
  python gat.py --optimize --model-name my_gat_model --n-trials 100

  # Optimization with custom study name
  python gat.py --optimize --model-name my_gat_model --study-name gat_experiment_v2 --n-trials 50

  # Quick optimization with fewer epochs per trial
  python gat.py --optimize --model-name my_gat_model --n-trials 20 --epochs 20

  # Optimization with custom database location
  python gat.py --optimize --model-name my_gat_model --storage-url sqlite:///my_gat_studies.db

Note: The optimization will resume from existing trials if a study with the same name exists.
        """,
        
        'eval': """
EVAL MODE HELP
==============

Evaluate a trained model on the test dataset.

Usage:
  python gat.py --eval [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to evaluate (default: gat_baseline)
  --batch-size SIZE        Batch size for evaluation (default: 128)
  --data-dir DIR           Directory containing graph data (default: data/codet_graphs)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic evaluation
  python gat.py --eval --model-name my_gat_model

  # Evaluation with larger batch size for faster inference
  python gat.py --eval --model-name my_gat_model --batch-size 512

  # Evaluation with custom data directory
  python gat.py --eval --model-name my_gat_model --data-dir custom_data/

  # Force CPU evaluation
  python gat.py --eval --model-name my_gat_model --device cpu

Note: The model must have been trained and saved before evaluation. The script will
look for the model checkpoint at models/gat/{model_name}/{model_name}_best.pth
        """
    }
    
    if command in help_text:
        print(help_text[command])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, resume, optimize, eval")
        print("Use 'python gat.py --help' for general help")


def parse_args():
    """Parse command line arguments for GAT experiments"""
    # Check for command-specific help first
    if len(sys.argv) >= 3 and sys.argv[1] == '-h':
        show_command_help(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 3 and sys.argv[2] == '-h' and sys.argv[1] in ['train', 'resume', 'optimize', 'eval']:
        show_command_help(sys.argv[1])
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Graph Attention Network (GAT) training and evaluation script",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  Basic training with default parameters (CoDeTM4):
    python gat.py --train --model-name my_gat_model

  Training with AIGCodeSet dataset:
    python gat.py --train --model-name my_gat_model --dataset aigcodeset

  Training with custom learning rate and epochs:
    python gat.py --train --model-name my_gat_model --learning-rate 0.01 --epochs 100

  Training using best hyperparameters from optimization:
    python gat.py --train --model-name my_gat_model --use-best-params

  Resume training from a checkpoint:
    python gat.py --resume --model-name my_gat_model --epochs 100

  Hyperparameter optimization with 100 trials:
    python gat.py --optimize --model-name my_gat_model --n-trials 100

  Evaluate a trained model on test set:
    python gat.py --eval --model-name my_gat_model

  Training on AIGCodeSet with custom data directory:
    python gat.py --train --model-name aig_gat_model --dataset aigcodeset --data-dir data/aigcodeset_graphs

  Complete workflow example:
    # 1. Optimize hyperparameters
    python gat.py --optimize --model-name best_gat_model --n-trials 100
    
    # 2. Train with optimized parameters
    python gat.py --train --model-name best_gat_model --use-best-params --epochs 200
    
    # 3. Evaluate the final model
    python gat.py --eval --model-name best_gat_model

Command-specific help:
    python gat.py -h train      # Show detailed help for training mode
    python gat.py -h resume     # Show detailed help for resume mode
    python gat.py -h optimize   # Show detailed help for optimization mode
    python gat.py -h eval       # Show detailed help for evaluation mode
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
        default="gat_baseline",
        help="Name of the model (affects save location and Optuna study name)"
    )
    model_group.add_argument(
        "--use-best-params",
        action="store_true",
        help="Use best hyperparameters from Optuna study (only for --train mode)"
    )
    
    # Training parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer (only used when not using Optuna params)"
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
        default="sqlite:///optuna/gat.db",
        help="Optuna storage database URL"
    )
    
    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and evaluation"
    )
    data_group.add_argument(
        "--data-dir",
        type=str,
        default="data/codet_graphs",
        help="Directory containing the graph data"
    )
    data_group.add_argument(
        "--data-suffix",
        type=str,
        help="The suffix in the data file's filename (eg with 'comments' passed -> train_graphs_comments.pt)"
    )
    data_group.add_argument(
        "--dataset",
        type=str,
        choices=["codet", "aigcodeset"],
        default="codet",
        help="Dataset to use: 'codet' for CoDeTM4 or 'aigcodeset' for AIGCodeSet"
    )
    
    # System parameters
    system_group = parser.add_argument_group("System Parameters")
    system_group.add_argument(
        "--seed",
        type=int,
        default=872002,
        help="Random seed for reproducibility"
    )
    system_group.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training/evaluation"
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
        "--disable-tensorboard",
        action="store_true",
        help="Disable tensorboard logging"
    )
    
    args = parser.parse_args()
    
    # Set defaults and validate
    if args.study_name is None:
        args.study_name = args.model_name
    
    # Set default data directory based on dataset choice if not explicitly provided
    if args.data_dir == "data/codet_graphs" and args.dataset == "aigcodeset":
        args.data_dir = "data/aigcodeset_graphs"
    
    # Validate mode-specific arguments
    if args.use_best_params and not args.train:
        parser.error("--use-best-params can only be used with --train mode")
    
    if args.resume and args.use_best_params:
        parser.error("--use-best-params cannot be used with --resume mode (parameters are loaded from checkpoint)")
    
    if args.optimize and args.epochs < 10:
        print("Warning: Using less than 10 epochs for optimization may not give good results")
    
    return args
