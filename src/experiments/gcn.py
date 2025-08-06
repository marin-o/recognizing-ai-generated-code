import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter that preserves raw formatting in epilog while showing defaults"""
    def _fill_text(self, text, width, indent):
        if text.startswith('\nExamples:'):
            # Preserve raw formatting for examples section
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
  --model-name NAME         Name of the model (default: baseline_gcn)
  --use-best-params        Use best hyperparameters from Optuna study
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

  # Training with specific device and larger batch size
  python gcn.py --train --model-name my_model --device cuda --batch-size 256

Note: If --use-best-params is specified, the script will load the best hyperparameters
from the Optuna study with the same name as the model.
        """,
        
        'resume': """
RESUME MODE HELP
================

Resume training from a saved model checkpoint.

Usage:
  python gcn.py --resume [OPTIONS]

Key Options:
  --model-name NAME         Name of the model to resume (default: baseline_gcn)
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
  --model-name NAME         Name of the model/study (default: baseline_gcn)
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
  --model-name NAME         Name of the model to evaluate (default: baseline_gcn)
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
  Basic training with default parameters:
    python gcn.py --train --model-name my_model

  Training with custom learning rate and epochs:
    python gcn.py --train --model-name my_model --learning-rate 0.01 --epochs 100

  Training using best hyperparameters from optimization:
    python gcn.py --train --model-name my_model --use-best-params

  Resume training from a checkpoint:
    python gcn.py --resume --model-name my_model --epochs 100

  Hyperparameter optimization with 100 trials:
    python gcn.py --optimize --model-name my_model --n-trials 100

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
        default="baseline_gcn",
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
    
    args = parser.parse_args()
    
    # Set defaults and validate
    if args.study_name is None:
        args.study_name = args.model_name
    
    # Validate mode-specific arguments
    if args.use_best_params and not args.train:
        parser.error("--use-best-params can only be used with --train mode")
    
    if args.resume and args.use_best_params:
        parser.error("--use-best-params cannot be used with --resume mode (parameters are loaded from checkpoint)")
    
    if args.optimize and args.epochs < 10:
        print("Warning: Using less than 10 epochs for optimization may not give good results")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    import optuna
    from optuna.trial import TrialState
    import torch
    from models.GCN import GCN
    from data.dataset import GraphCoDeTM4
    from torch_geometric.loader import DataLoader
    from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC
    from utils.gnn_utils import (
        save_model,
        load_model,
        get_metrics,
        set_seed,
        evaluate,
        train,
        load_single_data,
        load_multiple_data,
        create_objective,
        create_model_with_optuna_params,
        create_model_from_checkpoint,
        DEVICE,
    )
    from tqdm import tqdm
    import gc

    set_seed(args.seed)
    
    if args.eval:
        # Evaluation mode only - load model and evaluate on test set
        print("Running in evaluation mode...")
        
        # Load test data
        test_dataloader = load_single_data(
            data_dir=args.data_dir,
            split="test", 
            shuffle=False,
            batch_size=args.batch_size
        )
        
        # Load model directly from checkpoint (includes architecture info)
        checkpoint_path = f"models/gnn/{MODEL_NAME}/{MODEL_NAME}_best.pth"
        
        try:
            model, optimizer, scheduler, epoch, best_vloss, best_vacc = create_model_from_checkpoint(
                checkpoint_path, model_name=MODEL_NAME
            )
            print(f"Loaded model from epoch {epoch}")
            print(f"Best validation loss: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
        except FileNotFoundError:
            print(f"Error: No saved model found for {MODEL_NAME}")
            print(f"Expected path: {checkpoint_path}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            print("This checkpoint was saved with an older version that doesn't include architecture information.")
            print("Please re-train the model or use the migration script.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Evaluate on test set
        criterion = torch.nn.BCEWithLogitsLoss()
        metrics = get_metrics()
        test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS:")
        print("=" * 50)
        print(f"Test Loss: {test_loss:.4f}")
        for metric_name, metric_value in test_metrics.items():
            print(f"Test {metric_name}: {metric_value:.4f}")
        print("=" * 50)
        
    else:
        # Training or optimization mode
        train_loader, val_loader = load_multiple_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )

        if args.optimize:
            print("Running hyperparameter optimization...")
            os.makedirs("optuna", exist_ok=True)
            study = optuna.create_study(
                storage=args.storage_url, 
                study_name=args.study_name,
                direction="minimize", 
                load_if_exists=True
            )
            objective = create_objective(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                num_epochs=args.epochs,
            )
            study.optimize(objective, n_trials=args.n_trials)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            print(f"  Value: {study.best_trial.value}")
            print("  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")

        elif args.train:
            print("Training model...")
            if args.use_best_params:
                # Create model with best hyperparameters from Optuna
                model, optimizer, scheduler, optuna_success = create_model_with_optuna_params(
                    num_node_features=train_loader.dataset.num_node_features,
                    storage_url=args.storage_url,
                    study_name=args.study_name,
                    model_name=MODEL_NAME,
                    use_default_on_failure=False  # Don't fallback to defaults in training mode
                )
                
                if not optuna_success:
                    print("Failed to load Optuna parameters for training mode")
                    print(f"Make sure the study '{args.study_name}' exists in {args.storage_url}")
                    sys.exit(1)
                    
                # If Optuna didn't use a scheduler, create default one
                if scheduler is None:
                    print("Optuna study didn't use a scheduler, creating default scheduler...")
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=optimizer, patience=5
                    )
            else:
                # Use default model architecture
                model = GCN(
                    train_loader.dataset.num_node_features,
                ).to(DEVICE)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                model.name = MODEL_NAME
                
                # Create default scheduler
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, patience=5
                )
                
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                metrics=metrics,
                num_epochs=args.epochs,
            )

            # Clean up RAM to make room for the evaluation data
            del train_loader, val_loader
            gc.collect()

            # Final evaluation on test set
            test_dataloader = load_single_data(
                data_dir=args.data_dir,
                split="test", 
                shuffle=False,
                batch_size=args.batch_size
            )

            epoch, best_vloss, best_vacc = load_model(
                model, optimizer, save_path="models/gnn", scheduler=scheduler
            )
            test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

            print("\n" + "=" * 50)
            print("FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.4f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.4f}")
            print("=" * 50)

        elif args.resume:
            print("Resuming model training...")
            
            # Load model directly from checkpoint (includes architecture info)
            checkpoint_path = f"models/gnn/{MODEL_NAME}/{MODEL_NAME}_best.pth"
            
            try:
                model, optimizer, scheduler, start_epoch, best_vloss, best_vacc = create_model_from_checkpoint(
                    checkpoint_path, model_name=MODEL_NAME
                )
                print(f"Resuming from epoch {start_epoch}")
                print(f"Best validation loss so far: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
            except FileNotFoundError:
                print(f"Error: No saved model found for {MODEL_NAME}")
                print(f"Expected path: {checkpoint_path}")
                print("Use --train mode to train a new model")
                sys.exit(1)
            except ValueError as e:
                print(f"Error: {e}")
                print("This checkpoint was saved with an older version that doesn't include architecture information.")
                print("Please re-train the model or use the migration script.")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)
            
            # If no scheduler was saved, create a default one
            if scheduler is None:
                print("No scheduler found in checkpoint, creating default scheduler...")
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, patience=5
                )
                
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            
            # Continue training for additional epochs
            print(f"Continuing training for {args.epochs} more epochs...")
            train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                metrics=metrics,
                num_epochs=args.epochs,
            )

            # Clean up RAM to make room for the evaluation data
            del train_loader, val_loader
            gc.collect()

            # Final evaluation on test set
            test_dataloader = load_single_data(
                data_dir=args.data_dir,
                split="test", 
                shuffle=False,
                batch_size=args.batch_size
            )

            epoch, best_vloss, best_vacc = load_model(
                model, optimizer, save_path="models/gnn", scheduler=scheduler
            )
            test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

            print("\n" + "=" * 50)
            print("RESUMED TRAINING - FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.4f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.4f}")
            print("=" * 50)
