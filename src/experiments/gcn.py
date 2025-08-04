import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_NAME = "baseline_gcn"


def parse_args():
    parser = argparse.ArgumentParser(description="GNN training and optimization script")
    parser.add_argument(
        "-o",
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization instead of single training run",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

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
        DEVICE,
    )
    from tqdm import tqdm
    import gc

    set_seed(872002)
    train_loader, val_loader = load_multiple_data()

    if args.optimize:
        os.makedirs('optuna', exist_ok=True)
        study = optuna.create_study(storage='sqlite:///optuna/gcn_base.db', direction="minimize", load_if_exists=True)  
        objective = create_objective(
            train_dataloader=train_loader, val_dataloader=val_loader, num_epochs=10
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

        model = GCN(
            train_loader.num_node_features,
            hidden_dim_1=256,
            hidden_dim_2=128,
            embedding_dim=128,
            sage=True,
        ).to(DEVICE)
        model.name = MODEL_NAME

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        metrics = get_metrics()

        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            metrics=metrics,
            num_epochs=30,
        )

        # Clean up RAM to make room for the evaluation data
        # Not necessary if you have about 14GB to dedicate just to the training environment
        del train_loader, val_loader
        gc.collect()

        test_dataloader = load_single_data(split="test", shuffle=False)

        epoch, best_vloss, best_vacc = load_model(
            model, optimizer, save_path="models/gnn"
        )
        test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS:")
        print("=" * 50)
        print(f"Test Loss: {test_loss:.4f}")
        for metric_name, metric_value in test_metrics.items():
            print(f"Test {metric_name}: {metric_value:.4f}")
        print("=" * 50)
