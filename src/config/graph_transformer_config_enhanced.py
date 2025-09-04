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
ENHANCED TRAIN MODE HELP
========================

Train a new Enhanced Graph Transformer model with positional encodings and specified or default hyperparameters.

Usage:
  python graph_transformer_enhanced.py --train [OPTIONS]

Key Options:
  --model-name NAME         Name of the model (default: graph_transformer_enhanced_baseline)
  --use-best-params        Use best hyperparameters from Optuna study
  --source-model-name NAME  Source model name for loading best params (use with --use-best-params)
  --epochs N               Number of training epochs (default: 50)
  --learning-rate RATE     Learning rate for optimizer (default: 0.001)
  --batch-size SIZE        Batch size for training (default: 128)
  
  Architecture Options:
  --num-heads N            Number of attention heads (default: 8)
  --num-layers N           Number of transformer layers (default: 2)
  --hidden-dim N           Hidden dimension size (default: 128)
  --embedding-dim N        Embedding dimension size (default: 256)
  --dropout RATE           Dropout rate (default: 0.1)
  --pooling-method METHOD  Pooling method: mean/max/add (default: mean)
  
  Positional Encoding Options:
  --disable-pos-encoding   Disable positional encodings (use standard GraphTransformer)
  --max-depth N            Maximum depth for depth embeddings (auto-detected from data)
  --max-child-index N      Maximum child index for child embeddings (auto-detected from data)
  --depth-embed-dim N      Dimension of depth embeddings (default: 32)
  --child-embed-dim N      Dimension of child embeddings (default: 32)
  --use-percentile         Use 99th percentile instead of absolute max for pos encoding limits
  
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  --seed SEED              Random seed for reproducibility (default: 872002)

Examples:
  # Basic training with positional encodings (uses dataset statistics)
  python graph_transformer_enhanced.py --train --model-name my_enhanced_transformer

  # Training with custom architecture and positional encodings
  python graph_transformer_enhanced.py --train --model-name my_transformer --num-heads 16 --num-layers 4 --hidden-dim 256

  # Training without positional encodings (standard GraphTransformer)
  python graph_transformer_enhanced.py --train --model-name standard_transformer --disable-pos-encoding

  # Training with custom positional encoding dimensions
  python graph_transformer_enhanced.py --train --model-name my_transformer --depth-embed-dim 64 --child-embed-dim 64

  # Training with optimized hyperparameters from previous optimization
  python graph_transformer_enhanced.py --train --model-name my_transformer --use-best-params

Note: If --use-best-params is specified, the script will load the best hyperparameters
from the Optuna study. Positional encoding parameters are automatically configured
based on dataset statistics unless overridden.
        """,
        
        'resume': """
ENHANCED RESUME MODE HELP
=========================

Resume training from a saved Enhanced Graph Transformer model checkpoint.

Usage:
  python graph_transformer_enhanced.py --resume [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to resume (default: graph_transformer_enhanced_baseline)
  --epochs N               Additional epochs to train (default: 50)
  --batch-size SIZE        Batch size for training (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Resume training for 100 more epochs
  python graph_transformer_enhanced.py --resume --model-name my_enhanced_transformer --epochs 100

Note: The model checkpoint must exist at models/graph_transformer_enhanced/{model_name}/{model_name}_best.pth
All hyperparameters and architectural choices (including positional encodings) are loaded from the checkpoint.
        """,
        
        'optimize': """
ENHANCED OPTIMIZE MODE HELP
===========================

Run hyperparameter optimization using Optuna to find the best Enhanced Graph Transformer configuration.

Usage:
  python graph_transformer_enhanced.py --optimize [OPTIONS]

Key Options:
  --model-name NAME         Name of the model/study (default: graph_transformer_enhanced_baseline)
  --n-trials N             Number of optimization trials (default: 50)
  --study-name NAME        Custom Optuna study name (default: same as model-name)
  --storage-url URL        Optuna database URL (default: sqlite:///optuna/graph_transformer_enhanced.db)
  --epochs N               Epochs per trial (default: 50)
  --batch-size SIZE        Batch size (default: 128)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)
  
  Optimization Space Options:
  --optimize-pos-encoding  Include positional encoding parameters in optimization
  --disable-pos-encoding   Disable positional encodings during optimization

Examples:
  # Basic optimization with positional encodings
  python graph_transformer_enhanced.py --optimize --model-name my_enhanced_transformer --n-trials 100

  # Optimization including positional encoding hyperparameters
  python graph_transformer_enhanced.py --optimize --model-name my_transformer --optimize-pos-encoding --n-trials 100

  # Optimization without positional encodings (standard GraphTransformer space)
  python graph_transformer_enhanced.py --optimize --model-name standard_transformer --disable-pos-encoding --n-trials 50

Note: The optimization will resume from existing trials if a study with the same name exists.
When --optimize-pos-encoding is used, the optimization will also tune positional encoding dimensions.
        """,
        
        'eval': """
ENHANCED EVAL MODE HELP
=======================

Evaluate a trained Enhanced Graph Transformer model on the test dataset.

Usage:
  python graph_transformer_enhanced.py --eval [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to evaluate (default: graph_transformer_enhanced_baseline)
  --batch-size SIZE        Batch size for evaluation (default: 128)
  --data-dir DIR           Directory containing graph data (default: data/codet_graphs)
  --device DEVICE          Device to use: auto/cpu/cuda (default: auto)

Examples:
  # Basic evaluation
  python graph_transformer_enhanced.py --eval --model-name my_enhanced_transformer

  # Evaluation with custom data directory and batch size
  python graph_transformer_enhanced.py --eval --model-name my_transformer --batch-size 512 --data-dir data/custom_graphs

Note: The model must have been trained and saved before evaluation. The model architecture
and configuration (including positional encodings) are automatically loaded from the checkpoint.
        """
    }
    
    if command in help_text:
        print(help_text[command])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, resume, optimize, eval")
        print("Use 'python graph_transformer_enhanced.py --help' for general help")


def parse_args():
    """Parse command line arguments for Enhanced Graph Transformer experiments"""
    # Check for command-specific help first
    if len(sys.argv) >= 3 and sys.argv[1] == '-h':
        show_command_help(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 3 and sys.argv[2] == '-h' and sys.argv[1] in ['train', 'resume', 'optimize', 'eval']:
        show_command_help(sys.argv[1])
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Enhanced Graph Transformer training and evaluation script for AST classification with positional encodings",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  Basic training with positional encodings (CoDeTM4):
    python graph_transformer_enhanced.py --train --model-name my_enhanced_transformer

  Training with AIGCodeSet dataset:
    python graph_transformer_enhanced.py --train --model-name my_transformer --dataset aigcodeset

  Training without positional encodings (standard GraphTransformer):
    python graph_transformer_enhanced.py --train --model-name standard_transformer --disable-pos-encoding

  Training with custom positional encoding dimensions:
    python graph_transformer_enhanced.py --train --model-name my_transformer --depth-embed-dim 64 --child-embed-dim 64

  Training using best hyperparameters from optimization:
    python graph_transformer_enhanced.py --train --model-name my_transformer --use-best-params

  Resume training from a checkpoint:
    python graph_transformer_enhanced.py --resume --model-name my_transformer --epochs 100

  Hyperparameter optimization including positional encodings:
    python graph_transformer_enhanced.py --optimize --model-name my_transformer --optimize-pos-encoding --n-trials 100

  Evaluate a trained model on test set:
    python graph_transformer_enhanced.py --eval --model-name my_transformer

Command-specific help:
    python graph_transformer_enhanced.py -h train      # Show detailed help for training mode
    python graph_transformer_enhanced.py -h resume     # Show detailed help for resume mode
    python graph_transformer_enhanced.py -h optimize   # Show detailed help for optimization mode
    python graph_transformer_enhanced.py -h eval       # Show detailed help for evaluation mode
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
        default="graph_transformer_enhanced_baseline",
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
    
    # Enhanced Graph Transformer specific parameters
    transformer_group = parser.add_argument_group("Graph Transformer Architecture")
    transformer_group.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension size"
    )
    transformer_group.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size"
    )
    transformer_group.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    transformer_group.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of transformer layers"
    )
    transformer_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    transformer_group.add_argument(
        "--pooling-method",
        type=str,
        choices=["mean", "max", "add"],
        default="mean",
        help="Global pooling method for graph-level representation"
    )
    
    # Positional encoding parameters
    pos_encoding_group = parser.add_argument_group("Positional Encoding Configuration")
    pos_encoding_group.add_argument(
        "--disable-pos-encoding",
        action="store_true",
        help="Disable positional encodings (use standard GraphTransformer)"
    )
    pos_encoding_group.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for depth embeddings (auto-detected from dataset if not specified)"
    )
    pos_encoding_group.add_argument(
        "--max-child-index",
        type=int,
        default=None,
        help="Maximum child index for child embeddings (auto-detected from dataset if not specified)"
    )
    pos_encoding_group.add_argument(
        "--depth-embed-dim",
        type=int,
        default=32,
        help="Dimension of depth embeddings"
    )
    pos_encoding_group.add_argument(
        "--child-embed-dim",
        type=int,
        default=32,
        help="Dimension of child embeddings"
    )
    pos_encoding_group.add_argument(
        "--use-percentile",
        action="store_true",
        help="Use 99th percentile instead of absolute maximum for positional encoding limits"
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
        default="sqlite:///optuna/graph_transformer_enhanced.db",
        help="Optuna storage database URL"
    )
    optim_group.add_argument(
        "--optimize-pos-encoding",
        action="store_true",
        help="Include positional encoding parameters in optimization search space"
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
        default="cleaned_comments_depth",
        help="The suffix in the data file's filename (default includes depth data for positional encodings)"
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
    
    # Adjust data suffix for positional encodings
    if not args.disable_pos_encoding and args.data_suffix == "cleaned_comments_depth":
        print("Using depth-enhanced data for positional encodings")
    elif args.disable_pos_encoding and args.data_suffix == "cleaned_comments_depth":
        print("Warning: Using depth data but positional encodings are disabled")
        print("Consider using --data-suffix cleaned_comments for consistency")
    
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
    
    # Validate transformer-specific parameters
    if args.num_heads < 1:
        parser.error("--num-heads must be at least 1")
    
    if args.num_layers < 1:
        parser.error("--num-layers must be at least 1")
    
    if args.hidden_dim % args.num_heads != 0:
        print(f"Warning: hidden_dim ({args.hidden_dim}) should be divisible by num_heads ({args.num_heads}) for optimal performance")
    
    # Validate positional encoding parameters
    if not args.disable_pos_encoding:
        if args.max_depth is not None and args.max_depth < 1:
            parser.error("--max-depth must be at least 1")
        
        if args.max_child_index is not None and args.max_child_index < 1:
            parser.error("--max-child-index must be at least 1")
        
        if args.depth_embed_dim < 1:
            parser.error("--depth-embed-dim must be at least 1")
        
        if args.child_embed_dim < 1:
            parser.error("--child-embed-dim must be at least 1")
    
    if args.optimize_pos_encoding and args.disable_pos_encoding:
        parser.error("Cannot use --optimize-pos-encoding with --disable-pos-encoding")
    
    return args
