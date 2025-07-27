import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from data.dataset.aigcodeset_cst import AIGCodeSet_WithCSTFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier on CST features"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/", help="Directory to cache dataset"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset for test set",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of training set for validation",
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        default=True,
        help="Whether to perform hyperparameter tuning using GridSearchCV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and preprocess dataset
    dataset = AIGCodeSet_WithCSTFeatures(
        cache_dir=args.cache_dir, features_as_tensor=False
    )
    train, val, test = dataset.get_dataset(
        split=True, test_size=args.test_size, val_size=args.val_size
    )
    logger.info(f"Train columns: {train.column_names}")
    logger.info(f"Train dataset size: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Define feature columns (exclude 'code' and 'target')
    feature_columns = [
        "function_defs",
        "if_statements",
        "loops",
        "imports",
        "comments",
        "class_defs",
        "max_nesting_depth",
        "binary_ops",
        "errors",
    ]

    # Verify all feature columns exist
    missing_cols = [col for col in feature_columns if col not in train.column_names]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Extract features
    train_cst_features = np.array(
        [train[col] for col in feature_columns]
    ).T  # Shape: (n_samples, 9)
    val_cst_features = np.array([val[col] for col in feature_columns]).T
    test_cst_features = np.array([test[col] for col in feature_columns]).T

    # Scale features
    scaler = StandardScaler()
    train_cst_features = scaler.fit_transform(train_cst_features)
    val_cst_features = scaler.transform(val_cst_features)
    test_cst_features = scaler.transform(test_cst_features)

    # Get labels
    train_labels = np.array(train["target"])
    val_labels = np.array(val["target"])
    test_labels = np.array(test["target"])

    # Check class distribution
    class_counts = np.bincount(train_labels)
    logger.info(
        f"Class distribution (train): Human={class_counts[0]}, AI={class_counts[1]}"
    )

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    # Optional hyperparameter tuning
    if args.tune_hyperparameters:
        logger.info("Starting hyperparameter tuning...")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
        )
        grid_search.fit(train_cst_features, train_labels)
        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score (F1 macro): {grid_search.best_score_:.4f}")

    # Train model
    logger.info("Starting training...")
    model.fit(train_cst_features, train_labels)
    logger.info("Training completed.")

    # Evaluate model
    val_predictions = model.predict(val_cst_features)
    test_predictions = model.predict(test_cst_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info("Validation Classification Report:")
    logger.info(
        classification_report(
            val_labels,
            val_predictions,
            target_names=["human", "ai"],
        )
    )
    logger.info("Test Classification Report:")
    logger.info(
        classification_report(
            test_labels,
            test_predictions,
            target_names=["human", "ai"],
        )
    )

    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    logger.info("Feature Importance:")
    for feat, imp in sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(f"{feat}: {imp:.4f}")

    mlflow.set_experiment("AIGCodeSet")
    experiment_id = mlflow.get_experiment_by_name("AIGCodeSet").experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName = 'random_forest'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    )
    run_id = runs["run_id"].iloc[0] if not runs.empty else None
    with mlflow.start_run(run_id=run_id, run_name="random_forest") as run:
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric(
            "test_f1_macro",
            classification_report(test_labels, test_predictions, output_dict=True)[
                "macro avg"
            ]["f1-score"],
        )
        mlflow.log_metric(
            "val_f1_macro",
            classification_report(val_labels, val_predictions, output_dict=True)[
                "macro avg"
            ]["f1-score"],
        )
        mlflow.log_metric(
            "f1_macro",
            classification_report(test_labels, test_predictions, output_dict=True)[
                "macro avg"
            ]["f1-score"],
        )
        mlflow.log_metric(
            "recall",
            classification_report(test_labels, test_predictions, output_dict=True)[
                "macro avg"
            ]["recall"],
        )
        mlflow.log_metric(
            "precision",
            classification_report(test_labels, test_predictions, output_dict=True)[
                "macro avg"
            ]["precision"],
        )
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("min_samples_split", model.min_samples_split)
        mlflow.log_param("min_samples_leaf", model.min_samples_leaf)

        # Log feature importances as CSV artifact
        feature_df = pd.DataFrame(
            list(feature_importance.items()), columns=["Feature", "Importance"]
        )
        feature_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
