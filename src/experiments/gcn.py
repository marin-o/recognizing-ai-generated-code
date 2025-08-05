import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN training and optimization script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization instead of single training run",
    )
    parser.add_argument(
        "-b",
        "--use-best",
        action="store_true",
        help="Use best hyperparameters from Optuna study instead of hardcoded values",
    )
    parser.add_argument(
        "-s",
        "--storage-url",
        help="Path to Optuna storage url",
        default="sqlite:///optuna/gcn.db",
    )
    parser.add_argument(
        "-n",
        "--model-name",
        help="Specifies the name of the model. This has an effect on the location where the model is saved.",
        default="baseline_gcn",
    )
    parser.add_argument(
        "--study-name",
        help="Name of the Optuna study. If not provided, defaults to model name.",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--num-epochs",
        help="The number of epochs to train the model. Applicable both for optimization and normal training modes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run in evaluation mode only - load saved model and evaluate on test set without training",
    )
    args = parser.parse_args()
    
    # Set study name to model name if not provided
    if args.study_name is None:
        args.study_name = args.model_name
    
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
        DEVICE,
    )
    from tqdm import tqdm
    import gc

    set_seed(872002)
    
    if args.eval:
        # Evaluation mode only - load model and evaluate on test set
        print("Running in evaluation mode...")
        
        # Load test data
        test_dataloader = load_single_data(split="test", shuffle=False)
        
        # Create model with best hyperparameters from Optuna
        model, optimizer, optuna_success = create_model_with_optuna_params(
            num_node_features=test_dataloader.dataset.num_node_features,
            storage_url=args.storage_url,
            study_name=args.study_name,
            model_name=MODEL_NAME,
            use_default_on_failure=True
        )
        
        # Load the saved model
        try:
            epoch, best_vloss, best_vacc = load_model(
                model, optimizer, save_path="models/gnn"
            )
            print(f"Loaded model from epoch {epoch}")
        except FileNotFoundError:
            print(f"Error: No saved model found for {MODEL_NAME}")
            print(f"Expected path: models/gnn/{MODEL_NAME}/{MODEL_NAME}_best.pth")
            sys.exit(1)
        except RuntimeError as e:
            print(f"Error loading model state: {e}")
            print("This usually means the saved model has different hyperparameters than expected.")
            print("Make sure the Optuna study contains the correct hyperparameters for this model.")
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
        # Training mode
        train_loader, val_loader = load_multiple_data()

        if args.optimize:
            os.makedirs("optuna", exist_ok=True)
            study = optuna.create_study(
                storage=args.storage_url, direction="minimize", load_if_exists=True
            )
            objective = create_objective(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                num_epochs=args.num_epochs,
            )
            study.optimize(objective, n_trials=20)

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

        else:
            if args.use_best:
                # Create model with best hyperparameters from Optuna
                model, optimizer, optuna_success = create_model_with_optuna_params(
                    num_node_features=train_loader.dataset.num_node_features,
                    storage_url=args.storage_url,
                    study_name=args.study_name,
                    model_name=MODEL_NAME,
                    use_default_on_failure=False  # Don't fallback to defaults in training mode
                )
                
                if not optuna_success:
                    print("Failed to load Optuna parameters for training mode")
                    sys.exit(1)
            else:

                model = GCN(
                    train_loader.dataset.num_node_features,
                ).to(DEVICE)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001317217987013023)
                model.name = MODEL_NAME
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=5
            )
            train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                metrics=metrics,
                num_epochs=args.num_epochs,
            )

            # Clean up RAM to make room for the evaluation data
            # Not necessary if you have about 14GB to dedicate just to the training environment
            del train_loader, val_loader
            gc.collect()

            test_dataloader = load_single_data(split="test", shuffle=False)

            epoch, best_vloss, best_vacc = load_model(
                model, optimizer, save_path=f"models/gnn"
            )
            test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

            print("\n" + "=" * 50)
            print("FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.4f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.4f}")
            print("=" * 50)
