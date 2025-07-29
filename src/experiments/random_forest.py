import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from data.dataset import AIGCodeSet_WithCSTFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

from torch.utils.tensorboard import SummaryWriter
import os


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

    # Set up TensorBoard logging
    log_dir = os.path.join("tensorboard_logs", "AIGCodeSet", "random_forest")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log metrics
    writer.add_scalar("Metrics/test_accuracy", test_accuracy)
    writer.add_scalar("Metrics/val_accuracy", val_accuracy)
    writer.add_scalar("Metrics/test_f1_macro", 
        classification_report(test_labels, test_predictions, output_dict=True)["macro avg"]["f1-score"])
    writer.add_scalar("Metrics/val_f1_macro",
        classification_report(val_labels, val_predictions, output_dict=True)["macro avg"]["f1-score"])
    writer.add_scalar("Metrics/f1_macro",
        classification_report(test_labels, test_predictions, output_dict=True)["macro avg"]["f1-score"])
    writer.add_scalar("Metrics/recall",
        classification_report(test_labels, test_predictions, output_dict=True)["macro avg"]["recall"])
    writer.add_scalar("Metrics/precision",
        classification_report(test_labels, test_predictions, output_dict=True)["macro avg"]["precision"])
    
    # Log hyperparameters
    writer.add_text("Hyperparameters/n_estimators", str(model.n_estimators))
    writer.add_text("Hyperparameters/max_depth", str(model.max_depth))
    writer.add_text("Hyperparameters/min_samples_split", str(model.min_samples_split))
    writer.add_text("Hyperparameters/min_samples_leaf", str(model.min_samples_leaf))

    # Log feature importances as CSV artifact
    feature_df = pd.DataFrame(
        list(feature_importance.items()), columns=["Feature", "Importance"]
    )
    feature_csv_path = os.path.join(log_dir, "feature_importances.csv")
    feature_df.to_csv(feature_csv_path, index=False)
    
    # Save model
    model_path = os.path.join(log_dir, "random_forest_model.pkl")
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    writer.close()


if __name__ == "__main__":
    main()
