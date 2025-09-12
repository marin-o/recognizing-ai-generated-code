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

Train a new GCN model with specified or default hyperparameters.

Usage:
  python gcn.py --train [OPTIONS]

Key Options:
  --model-name NAME         Name of the model (default: gcn_baseline)
  --use-best-params        Use best hyperparameters from Optuna study
  --source-model-name NAME  Source model name for loading best params (use with --use-best-params)
  --epochs N               Number of training epochs (default: 50)
  --learning-rate RATE     Learning rate for optimizer (default: 0.001)
  --batch-size SIZE        Batch size for training (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --seed SEED              Random seed for reproducibility (default: 872002)

Examples:
  # Basic training with default parameters
  python gcn.py --train --model-name my_model

  # Training with custom hyperparameters
  python gcn.py --train --model-name my_model --learning-rate 0.01 --epochs 100

  # Training using optimized hyperparameters from previous optimization
  python gcn.py --train --model-name my_model --use-best-params

  # Training using parameters from a different model's optimization
  python gcn.py --train --model-name new_model --use-best-params --source-model-name baseline_v2

  # Training with specific device and larger batch size
  python gcn.py --train --model-name my_model --device cuda --batch-size 256

Note: If --use-best-params is specified, the script will load the best hyperparameters
from the Optuna study. If --source-model-name is also specified, it will load parameters
from that study instead of the current model's study.
        """,
        
        'resume': """
RESUME MODE HELP
================

Resume training from a saved model checkpoint.

Usage:
  python gcn.py --resume [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to resume (default: gcn_baseline)
  --epochs N               Additional epochs to train (default: 50)
  --batch-size SIZE        Batch size for training (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Resume training for 100 more epochs
  python gcn.py --resume --model-name my_model --epochs 100

  # Resume training with larger batch size
  python gcn.py --resume --model-name my_model --batch-size 256

Note: The model checkpoint must exist at models/gnn/{model_name}/{model_name}_best.pth
All hyperparameters (learning rate, architecture, scheduler) are loaded from the checkpoint.
        """,
        
        'optimize': """
OPTIMIZE MODE HELP
==================

Run hyperparameter optimization using Optuna to find the best model configuration.

Usage:
  python gcn.py --optimize [OPTIONS]

Key Options:
  --model-name NAME         Name of the model/study (default: gcn_baseline)
  --n-trials N             Number of optimization trials (default: 50)
  --study-name NAME        Custom Optuna study name (default: same as model-name)
  --storage-url URL        Optuna database URL (default: sqlite:///optuna/gcn.db)
  --epochs N               Epochs per trial (default: 50)
  --batch-size SIZE        Batch size (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic optimization with 100 trials
  python gcn.py --optimize --model-name my_model --n-trials 100

  # Optimization with custom study name
  python gcn.py --optimize --model-name my_model --study-name experiment_v2 --n-trials 50

  # Quick optimization with fewer epochs per trial
  python gcn.py --optimize --model-name my_model --n-trials 20 --epochs 20

  # Optimization with custom database location
  python gcn.py --optimize --model-name my_model --storage-url sqlite:///my_studies.db

Note: The optimization will resume from existing trials if a study with the same name exists.
        """,
        
        'eval': """
EVAL MODE HELP
==============

Evaluate a trained model on the test dataset.

Usage:
  python gcn.py --eval [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to evaluate (default: gcn_baseline)
  --batch-size SIZE        Batch size for evaluation (default: 128)
  --data-dir DIR           Directory containing graph data (default: data/codet_graphs)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic evaluation
  python gcn.py --eval --model-name my_model

  # Evaluation with larger batch size for faster inference
  python gcn.py --eval --model-name my_model --batch-size 512

  # Evaluation with custom data directory
  python gcn.py --eval --model-name my_model --data-dir custom_data/

  # Force CPU evaluation
  python gcn.py --eval --model-name my_model --device cpu

Note: The model must have been trained and saved before evaluation. The script will
look for the model checkpoint at models/gnn/{model_name}/{model_name}_best.pth
        """
    }
    
    if command in help_text:
        print(help_text[command])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, resume, optimize, eval")
        print("Use 'python gcn.py --help' for general help")


def parse_args():
    """Parse command line arguments for GCN experiments"""
    # Check for command-specific help first
    if len(sys.argv) >= 3 and sys.argv[1] == '-h':
        show_command_help(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 3 and sys.argv[2] == '-h' and sys.argv[1] in ['train', 'resume', 'optimize', 'eval']:
        show_command_help(sys.argv[1])
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Graph Convolutional Network (GCN) training and evaluation script",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  Basic training with default parameters (CoDeTM4):
    python gcn.py --train --model-name my_model

  Training with AIGCodeSet dataset:
    python gcn.py --train --model-name my_model --dataset aigcodeset

  Training with custom learning rate and epochs:
    python gcn.py --train --model-name my_model --learning-rate 0.01 --epochs 100

  Training using best hyperparameters from optimization:
    python gcn.py --train --model-name my_model --use-best-params

  Training using parameters from another model's optimization:
    python gcn.py --train --model-name new_model --use-best-params --source-model-name baseline_v2

  Resume training from a checkpoint:
    python gcn.py --resume --model-name my_model --epochs 100

  Hyperparameter optimization with 100 trials:
    python gcn.py --optimize --model-name my_model --n-trials 100

  Evaluate a trained model on test set:
    python gcn.py --eval --model-name my_model

  Training on AIGCodeSet with custom data directory:
    python gcn.py --train --model-name aig_model --dataset aigcodeset --data-dir data/aigcodeset_graphs

  Evaluate a trained model on test set:
    python gcn.py --eval --model-name my_model

  Complete workflow example:
    # 1. Optimize hyperparameters
    python gcn.py --optimize --model-name best_model --n-trials 100
    
    # 2. Train with optimized parameters
    python gcn.py --train --model-name best_model --use-best-params --epochs 200
    
    # 3. Evaluate the final model
    python gcn.py --eval --model-name best_model

Command-specific help:
    python gcn.py -h train      # Show detailed help for training mode
    python gcn.py -h resume     # Show detailed help for resume mode
    python gcn.py -h optimize   # Show detailed help for optimization mode
    python gcn.py -h eval       # Show detailed help for evaluation mode
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
        default="gcn_baseline",
        help="Name of the model (affects save location and Optuna study name)"
    )
    model_group.add_argument(
        "--use-best-params",
        action="store_true",
        help="Use best hyperparameters from Optuna study (only for --train mode)"
    )
    model_group.add_argument(
        "--source-model-name",
        type=str,
        default=None,
        help="Source model name for loading best parameters (use with --use-best-params). If not specified, uses --model-name"
    )
    model_group.add_argument(
        "--force-two-layer-classifier",
        action="store_true",
        help="Force use of two-layer classifier even when loading params from single-layer study"
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
        default="sqlite:///optuna/gcn.db",
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
    
    if args.source_model_name and not args.use_best_params:
        parser.error("--source-model-name can only be used with --use-best-params")
    
    if args.resume and args.use_best_params:
        parser.error("--use-best-params cannot be used with --resume mode (parameters are loaded from checkpoint)")
    
    if args.resume and args.source_model_name:
        parser.error("--source-model-name cannot be used with --resume mode")
    
    if args.optimize and args.epochs < 10:
        print("Warning: Using less than 10 epochs for optimization may not give good results")
    
    return args
