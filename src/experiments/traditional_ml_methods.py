import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from data.dataset.codet_m4 import CoDeTM4
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from torch.utils.tensorboard import SummaryWriter
import os


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate traditional ML classifiers (Random Forest, SVM, Logistic Regression) on code features"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/", help="Directory to cache dataset"
    )
    parser.add_argument(
        "--train-subset",
        type=float,
        default=0.1,
        help="Fraction of training data to use (for faster experimentation)",
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        default=False,
        help="Whether to perform hyperparameter tuning using GridSearchCV",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "svm", "lr", "nb"],
        choices=["rf", "svm", "lr", "nb"],
        help="Models to evaluate: rf (Random Forest), svm (SVM), lr (Logistic Regression), nb (Naive Bayes)",
    )
    return parser.parse_args()


def extract_features(dataset):
    """Extract TF-IDF features from code text."""
    # Get code text from HuggingFace dataset
    code_texts = dataset['code']
    
    # Use TF-IDF to extract features from code
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        lowercase=True,
        token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',  # Match identifiers
        ngram_range=(1, 2),
    )
    
    features = vectorizer.fit_transform(code_texts)
    return features, vectorizer


def get_model_and_params(model_name):
    """Get model instance and hyperparameter grid for tuning."""
    if model_name == "rf":
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
    elif model_name == "svm":
        model = SVC(random_state=42, probability=True)
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }
    elif model_name == "lr":
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }
    elif model_name == "nb":
        model = MultinomialNB()
        param_grid = {
            "alpha": [0.1, 0.5, 1.0, 2.0],
            "fit_prior": [True, False],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, param_grid


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name, tune_hyperparameters, log_dir):
    """Train and evaluate a single model."""
    logger.info(f"\n{'='*20} Evaluating {model_name.upper()} {'='*20}")
    
    # Get model and hyperparameters
    model_instance, param_grid = get_model_and_params(model_name)
    
    # Hyperparameter tuning
    if tune_hyperparameters:
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(
            model_instance, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        model_instance = grid_search.best_estimator_
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score (F1 macro): {grid_search.best_score_:.4f}")
    
    # Train model
    logger.info(f"Training {model_name}...")
    model_instance.fit(X_train, y_train)
    
    # Make predictions
    val_predictions = model_instance.predict(X_val)
    test_predictions = model_instance.predict(X_test)
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    val_f1 = f1_score(y_val, val_predictions, average='macro')
    test_f1 = f1_score(y_test, test_predictions, average='macro')
    test_precision = precision_score(y_test, test_predictions, average='macro')
    test_recall = recall_score(y_test, test_predictions, average='macro')
    
    # Log results
    logger.info(f"{model_name.upper()} Results:")
    logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Test F1 (macro): {test_f1:.4f}")
    logger.info(f"  Test Precision (macro): {test_precision:.4f}")
    logger.info(f"  Test Recall (macro): {test_recall:.4f}")
    
    logger.info(f"Test Classification Report for {model_name.upper()}:")
    logger.info(
        classification_report(
            y_test,
            test_predictions,
            target_names=["human", "ai"],
        )
    )
    
    # TensorBoard logging
    model_log_dir = os.path.join(log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=model_log_dir)
    
    # Log metrics
    writer.add_scalar("Metrics/test_accuracy", test_accuracy)
    writer.add_scalar("Metrics/val_accuracy", val_accuracy)
    writer.add_scalar("Metrics/test_f1_macro", test_f1)
    writer.add_scalar("Metrics/val_f1_macro", val_f1)
    writer.add_scalar("Metrics/test_precision_macro", test_precision)
    writer.add_scalar("Metrics/test_recall_macro", test_recall)
    
    # Save model
    model_path = os.path.join(model_log_dir, f"{model_name}_model.pkl")
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model_instance, f)
    
    writer.close()
    
    return {
        'model': model_instance,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
    }


def main():
    args = parse_args()

    # Load dataset
    dataset = CoDeTM4(cache_dir=args.cache_dir)
    train, val, test = dataset.get_dataset(
        split=['train', 'val', 'test'], 
        columns=['code', 'target', 'target_binary'],
        train_subset=args.train_subset
    )
    
    logger.info(f"Dataset loaded - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Extract features using TF-IDF
    logger.info("Extracting TF-IDF features from code...")
    X_train, vectorizer = extract_features(train)
    # Use the same vectorizer for val and test (transform only, don't fit)
    X_val = vectorizer.transform(val['code'])
    X_test = vectorizer.transform(test['code'])
    
    # Get labels (using binary targets: 0=human, 1=ai)
    y_train = np.array(train['target_binary'])
    y_val = np.array(val['target_binary'])
    y_test = np.array(test['target_binary'])
    
    logger.info(f"Feature extraction complete. Feature dimensionality: {X_train.shape[1]}")
    
    # Set up logging directory
    log_dir = os.path.join("tensorboard_logs", "CoDeTM4", "traditional_ml")
    os.makedirs(log_dir, exist_ok=True)
    
    # Evaluate each model
    results = {}
    for model_name in args.models:
        try:
            result = evaluate_model(
                model_name, X_train, X_val, X_test, y_train, y_val, y_test,
                model_name, args.tune_hyperparameters, log_dir
            )
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    # Summary of all models
    logger.info("\n" + "="*50)
    logger.info("SUMMARY OF ALL MODELS")
    logger.info("="*50)
    
    for model_name, result in results.items():
        logger.info(f"{model_name.upper()}: Accuracy={result['test_accuracy']:.4f}, "
                   f"F1={result['test_f1']:.4f}, "
                   f"Precision={result['test_precision']:.4f}, "
                   f"Recall={result['test_recall']:.4f}")
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        logger.info(f"\nBest model by F1-score: {best_model_name.upper()} "
                   f"(F1: {results[best_model_name]['test_f1']:.4f})")


if __name__ == "__main__":
    main()
