import argparse
import sys
import os


class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter that preserves raw formatting in epilog while showing defaults"""
    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))


def show_command_help(command):
    """Show specific help for a command"""
    help_text = {
        'train': """
ENHANCED TRAIN MODE HELP
========================

Train a new Enhanced GCN model with positional encodings and specified or default hyperparameters.

Usage:
  python gcn_enhanced.py --train [OPTIONS]

Key Options:
  --model-name NAME         Name of the model (default: gcn_enhanced_baseline)
  --use-best-params        Use best hyperparameters from Optuna study
  --source-model-name NAME  Source model name for loading best params (use with --use-best-params)
  --epochs N               Number of training epochs (default: 50)
  --learning-rate RATE     Learning rate for optimizer (default: 0.001)
  --batch-size SIZE        Batch size for training (default: 128)
  
  Architecture Options:
  --hidden-dim-1 N         First hidden dimension size (default: 128)
  --hidden-dim-2 N         Second hidden dimension size (default: 128)
  --embedding-dim N        Embedding dimension size (default: 256)
  --dropout RATE           Dropout rate (default: 0.1)
  --pooling-method METHOD  Pooling method: mean/max/add (default: mean)
  --sage                   Use SAGE convolution instead of GCN
  --two-layer-classifier   Use two-layer classifier
  
  Positional Encoding Options:
  --disable-pos-encoding   Disable positional encodings (use standard GCN)
  --max-depth N            Maximum depth for depth embeddings (auto-detected from data)
  --max-child-index N      Maximum child index for child embeddings (auto-detected from data)
  --depth-embed-dim N      Dimension of depth embeddings (default: 32)
  --child-embed-dim N      Dimension of child embeddings (default: 32)
  
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --seed SEED              Random seed for reproducibility (default: 872002)

Examples:
  # Basic training with positional encodings (uses dataset statistics)
  python gcn_enhanced.py --train --model-name my_enhanced_gcn

  # Training with custom architecture and positional encodings
  python gcn_enhanced.py --train --model-name my_gcn --hidden-dim-1 256 --hidden-dim-2 256 --sage

  # Training without positional encodings (standard GCN)
  python gcn_enhanced.py --train --model-name standard_gcn --disable-pos-encoding

  # Training with custom positional encoding dimensions
  python gcn_enhanced.py --train --model-name my_gcn --depth-embed-dim 64 --child-embed-dim 64

  # Training with optimized hyperparameters from previous optimization
  python gcn_enhanced.py --train --model-name my_gcn --use-best-params

Note: If --use-best-params is specified, the script will load the best hyperparameters
from the Optuna study. Positional encoding parameters are automatically configured
based on dataset statistics unless overridden.
        """,
        
        'resume': """
ENHANCED RESUME MODE HELP
=========================

Resume training from a saved Enhanced GCN model checkpoint.

Usage:
  python gcn_enhanced.py --resume [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to resume (default: gcn_enhanced_baseline)
  --epochs N               Additional epochs to train (default: 50)
  --batch-size SIZE        Batch size for training (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Resume training for 100 more epochs
  python gcn_enhanced.py --resume --model-name my_enhanced_gcn --epochs 100

Note: The model checkpoint must exist at models/gcn_enhanced/{model_name}/{model_name}_best.pth
All hyperparameters and architectural choices (including positional encodings) are loaded from the checkpoint.
        """,
        
        'optimize': """
ENHANCED OPTIMIZE MODE HELP
===========================

Run hyperparameter optimization using Optuna to find the best Enhanced GCN configuration.

Usage:
  python gcn_enhanced.py --optimize [OPTIONS]

Key Options:
  --model-name NAME         Name of the model/study (default: gcn_enhanced_baseline)
  --n-trials N             Number of optimization trials (default: 50)
  --study-name NAME        Custom Optuna study name (default: same as model-name)
  --storage-url URL        Optuna database URL (default: sqlite:///optuna/gcn_enhanced.db)
  --epochs N               Epochs per trial (default: 50)
  --batch-size SIZE        Batch size (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  
  Optimization Space Options:
  --optimize-pos-encoding  Include positional encoding parameters in optimization
  --disable-pos-encoding   Disable positional encodings during optimization

Examples:
  # Basic optimization with positional encodings
  python gcn_enhanced.py --optimize --model-name my_enhanced_gcn --n-trials 100

  # Optimization including positional encoding hyperparameters
  python gcn_enhanced.py --optimize --model-name my_gcn --optimize-pos-encoding --n-trials 100

  # Optimization without positional encodings (standard GCN space)
  python gcn_enhanced.py --optimize --model-name standard_gcn --disable-pos-encoding --n-trials 50

Note: The optimization will resume from existing trials if a study with the same name exists.
When --optimize-pos-encoding is used, the optimization will also tune positional encoding dimensions.
        """,
        
        'eval': """
ENHANCED EVAL MODE HELP
=======================

Evaluate a trained Enhanced GCN model on the test dataset.

Usage:
  python gcn_enhanced.py --eval [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to evaluate (default: gcn_enhanced_baseline)
  --batch-size SIZE        Batch size for evaluation (default: 128)
  --data-dir DIR           Directory containing graph data (default: data/codet_graphs)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic evaluation
  python gcn_enhanced.py --eval --model-name my_enhanced_gcn

  # Evaluation with custom data directory and batch size
  python gcn_enhanced.py --eval --model-name my_gcn --batch-size 512 --data-dir data/custom_graphs

Note: The model must have been trained and saved before evaluation. The model architecture
and configuration (including positional encodings) are automatically loaded from the checkpoint.
        """
    }
    
    if command in help_text:
        print(help_text[command])
    else:
        print(f"No help available for command: {command}")


def parse_args():
    """Parse command line arguments for Enhanced GCN experiments"""
    # Check for command-specific help first
    if len(sys.argv) >= 3 and sys.argv[1] == '-h':
        show_command_help(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 3 and sys.argv[2] == '-h' and sys.argv[1] in ['train', 'resume', 'optimize', 'eval']:
        show_command_help(sys.argv[1])
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description='Enhanced GCN with Positional Encodings - Train, optimize, or evaluate GCN models with optional depth and child embeddings',
        formatter_class=CustomHelpFormatter,
        epilog="""
USAGE EXAMPLES:
===============

Training Examples:
  python gcn_enhanced.py --train --model-name my_enhanced_gcn
  python gcn_enhanced.py --train --model-name my_gcn --disable-pos-encoding
  python gcn_enhanced.py --train --model-name my_gcn --sage --two-layer-classifier
  python gcn_enhanced.py --train --model-name my_gcn --use-best-params

Optimization Examples:
  python gcn_enhanced.py --optimize --model-name my_gcn --n-trials 100
  python gcn_enhanced.py --optimize --model-name my_gcn --optimize-pos-encoding

Evaluation Examples:
  python gcn_enhanced.py --eval --model-name my_enhanced_gcn
  python gcn_enhanced.py --eval --model-name my_gcn --batch-size 512

Resume Examples:
  python gcn_enhanced.py --resume --model-name my_enhanced_gcn --epochs 50

COMMAND-SPECIFIC HELP:
======================
For detailed help on specific commands, use:
  python gcn_enhanced.py -h train
  python gcn_enhanced.py -h optimize
  python gcn_enhanced.py -h eval
  python gcn_enhanced.py -h resume

        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train a new model')
    mode_group.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    mode_group.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    mode_group.add_argument('--eval', action='store_true', help='Evaluate a trained model')
    
    # Model configuration
    parser.add_argument('--model-name', type=str, default='gcn_enhanced_baseline',
                       help='Name of the model (used for saving/loading)')
    parser.add_argument('--data-dir', type=str, default='data/codet_graphs',
                       help='Directory containing graph data')
    parser.add_argument('--data-suffix', type=str, default='',
                       help='Suffix for data files')
    parser.add_argument('--dataset', type=str, default='codet', choices=['codet', 'aigcodeset'],
                       help='Dataset to use')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training/evaluation')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    
    # Model architecture
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Dimension of node embeddings')
    parser.add_argument('--hidden-dim-1', type=int, default=128,
                       help='First hidden layer dimension')
    parser.add_argument('--hidden-dim-2', type=int, default=128,
                       help='Second hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--pooling-method', type=str, default='mean', 
                       choices=['mean', 'max', 'add'],
                       help='Graph pooling method')
    parser.add_argument('--sage', action='store_true',
                       help='Use SAGE convolution instead of GCN')
    parser.add_argument('--two-layer-classifier', action='store_true',
                       help='Use two-layer classifier')
    
    # Positional encoding options
    parser.add_argument('--disable-pos-encoding', action='store_true',
                       help='Disable positional encodings (use standard GCN)')
    parser.add_argument('--max-depth', type=int, default=None,
                       help='Maximum depth for positional encodings (auto-detected if not specified)')
    parser.add_argument('--max-child-index', type=int, default=None,
                       help='Maximum child index for positional encodings (auto-detected if not specified)')
    parser.add_argument('--depth-embed-dim', type=int, default=32,
                       help='Dimension of depth embeddings')
    parser.add_argument('--child-embed-dim', type=int, default=32,
                       help='Dimension of child embeddings')
    
    # Optimization-specific options
    parser.add_argument('--use-best-params', action='store_true',
                       help='Use best hyperparameters from Optuna study (for training)')
    parser.add_argument('--source-model-name', type=str, default=None,
                       help='Source model name for loading best params (use with --use-best-params)')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Optuna study name (defaults to model-name)')
    parser.add_argument('--storage-url', type=str, default='sqlite:///optuna/gcn_enhanced.db',
                       help='Optuna database URL')
    parser.add_argument('--optimize-pos-encoding', action='store_true',
                       help='Include positional encoding parameters in optimization')
    
    # System options
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training/evaluation')
    parser.add_argument('--seed', type=int, default=872002,
                       help='Random seed for reproducibility')
    
    # Logging options
    parser.add_argument('--disable-tensorboard', action='store_true',
                       help='Disable tensorboard logging')
    parser.add_argument('--log-dir', type=str, default='tensorboard_logs',
                       help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Set default study name if not provided
    if args.study_name is None:
        args.study_name = args.model_name
    
    # Validate arguments
    if args.use_best_params and not args.train:
        parser.error("--use-best-params can only be used with --train")
    
    if args.source_model_name and not args.use_best_params:
        parser.error("--source-model-name requires --use-best-params")
    
    if args.optimize_pos_encoding and not args.optimize:
        parser.error("--optimize-pos-encoding can only be used with --optimize")
    
    if args.disable_pos_encoding and (args.max_depth or args.max_child_index or 
                                      args.depth_embed_dim != 32 or args.child_embed_dim != 32):
        parser.error("Positional encoding parameters cannot be used with --disable-pos-encoding")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print("Enhanced GCN Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")