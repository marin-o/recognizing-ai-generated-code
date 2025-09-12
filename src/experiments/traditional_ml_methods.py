import argparse
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.dataset.codet_m4_cleaned_cst import CoDeTM4CleanedWithCSTFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from torch.utils.tensorboard import SummaryWriter
import os


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate traditional ML classifiers on multi-language CST features from cleaned CoDet-M4 dataset"
    )
    parser.add_argument(
        "--cleaned-data-path", 
        type=str, 
        default="data/codet_cleaned_20250812_201438/", 
        help="Path to cleaned dataset directory"
    )
    parser.add_argument(
        "--train-subset",
        type=float,
        default=1.0,
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
        default=["rf", "lr", "nb", "svm"],
        choices=["rf", "svm", "lr", "nb"],
        help="Models to evaluate: rf (Random Forest), svm (SVM), lr (Logistic Regression), nb (Naive Bayes)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="combined",
        choices=["tfidf", "cst", "combined"],
        help="Type of features to use: tfidf (text features only), cst (CST features only), combined (both)"
    )
    return parser.parse_args()


def extract_features(train_dataset, val_dataset, test_dataset, feature_type="combined"):
    """Extract features from datasets based on feature_type."""
    
    if feature_type == "cst":
        # Use only CST features
        logger.info("Using CST features only...")
        
        # Extract CST features from datasets
        if 'cst_features' in train_dataset.column_names:
            # Features are in tensor format
            X_train = np.array(train_dataset['cst_features'])
            X_val = np.array(val_dataset['cst_features'])
            X_test = np.array(test_dataset['cst_features'])
        else:
            # Features are in individual columns
            cst_feature_names = ['function_defs', 'class_defs', 'if_statements', 'loops', 
                               'imports', 'comments', 'binary_ops', 'errors', 
                               'max_nesting_depth', 'language_encoded']
            
            X_train = np.array([[sample[feat] for feat in cst_feature_names] for sample in train_dataset])
            X_val = np.array([[sample[feat] for feat in cst_feature_names] for sample in val_dataset])
            X_test = np.array([[sample[feat] for feat in cst_feature_names] for sample in test_dataset])
        
        # Normalize CST features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        return X_train, X_val, X_test, None, scaler
    
    elif feature_type == "tfidf":
        # Use only TF-IDF features
        logger.info("Using TF-IDF features only...")
        
        # Extract TF-IDF features from code
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',  # Match identifiers
            ngram_range=(1, 2),
        )
        
        X_train = vectorizer.fit_transform(train_dataset['code'])
        X_val = vectorizer.transform(val_dataset['code'])
        X_test = vectorizer.transform(test_dataset['code'])
        
        return X_train, X_val, X_test, vectorizer, None
    
    elif feature_type == "combined":
        # Use both TF-IDF and CST features
        logger.info("Using combined TF-IDF and CST features...")
        
        # Get TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduce to make room for CST features
            stop_words='english',
            lowercase=True,
            token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
            ngram_range=(1, 2),
        )
        
        tfidf_train = vectorizer.fit_transform(train_dataset['code'])
        tfidf_val = vectorizer.transform(val_dataset['code'])
        tfidf_test = vectorizer.transform(test_dataset['code'])
        
        # Get CST features
        if 'cst_features' in train_dataset.column_names:
            cst_train = np.array(train_dataset['cst_features'])
            cst_val = np.array(val_dataset['cst_features'])
            cst_test = np.array(test_dataset['cst_features'])
        else:
            cst_feature_names = ['function_defs', 'class_defs', 'if_statements', 'loops', 
                               'imports', 'comments', 'binary_ops', 'errors', 
                               'max_nesting_depth', 'language_encoded']
            
            cst_train = np.array([[sample[feat] for feat in cst_feature_names] for sample in train_dataset])
            cst_val = np.array([[sample[feat] for feat in cst_feature_names] for sample in val_dataset])
            cst_test = np.array([[sample[feat] for feat in cst_feature_names] for sample in test_dataset])
        
        # Normalize CST features
        scaler = StandardScaler()
        cst_train_scaled = scaler.fit_transform(cst_train)
        cst_val_scaled = scaler.transform(cst_val)
        cst_test_scaled = scaler.transform(cst_test)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        X_train = hstack([tfidf_train, csr_matrix(cst_train_scaled)])
        X_val = hstack([tfidf_val, csr_matrix(cst_val_scaled)])
        X_test = hstack([tfidf_test, csr_matrix(cst_test_scaled)])
        
        return X_train, X_val, X_test, vectorizer, scaler
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def get_model_and_params(model_name, feature_type="combined"):
    """Get model instance and hyperparameter grid for tuning."""
    if model_name == "rf":
        model = RandomForestClassifier(random_state=872002, n_jobs=-1)
        param_grid = {
            "n_estimators": [10, 20, 30, 50],
            "max_depth": [10, 20, 30, 40],
            "min_samples_split": [2, 3, 5],
        }
    elif model_name == "svm":
        model = SVC(random_state=872002, probability=True)
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }
    elif model_name == "lr":
        model = LogisticRegression(random_state=872002, max_iter=1000)
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }
    elif model_name == "nb":
        # Choose appropriate Naive Bayes variant based on feature type
        if feature_type == "cst" or feature_type == "combined":
            # Use GaussianNB for scaled/continuous features (CST or combined with CST)
            model = GaussianNB()
            param_grid = {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            }
        else:
            # Use MultinomialNB for discrete/count features (TF-IDF only)
            model = MultinomialNB()
            param_grid = {
                "alpha": [0.1, 0.5, 1.0, 2.0],
                "fit_prior": [True, False],
            }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, param_grid


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name, tune_hyperparameters, log_dir, feature_type="combined"):
    """Train and evaluate a single model."""
    logger.info(f"\n{'='*20} Evaluating {model_name.upper()} {'='*20}")
    
    # Get model and hyperparameters
    model_instance, param_grid = get_model_and_params(model_name, feature_type)
    
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
    
    # Get prediction probabilities for AUROC (if available)
    try:
        if hasattr(model_instance, "predict_proba"):
            test_proba = model_instance.predict_proba(X_test)[:, 1]  # Probability of positive class
            val_proba = model_instance.predict_proba(X_val)[:, 1]
        elif hasattr(model_instance, "decision_function"):
            test_proba = model_instance.decision_function(X_test)
            val_proba = model_instance.decision_function(X_val)
        else:
            test_proba = None
            val_proba = None
    except:
        test_proba = None
        val_proba = None
    
    # Calculate metrics (binary classification, same as enhanced transformer)
    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    # Use binary averaging to match enhanced transformer metrics
    val_f1 = f1_score(y_val, val_predictions, average='binary')
    test_f1 = f1_score(y_test, test_predictions, average='binary')
    test_precision = precision_score(y_test, test_predictions, average='binary')
    test_recall = recall_score(y_test, test_predictions, average='binary')
    
    # Calculate specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate AUROC
    test_auroc = None
    if test_proba is not None:
        try:
            test_auroc = roc_auc_score(y_test, test_proba)
        except:
            test_auroc = None
    
    # Log results (matching enhanced transformer format)
    logger.info(f"\n{'='*50}")
    logger.info(f"{model_name.upper()} EVALUATION RESULTS:")
    logger.info(f"{'='*50}")
    logger.info(f"Validation Accuracy: {val_accuracy:.6f}")
    logger.info(f"Test Accuracy: {test_accuracy:.6f}")
    logger.info(f"Test Precision: {test_precision:.6f}")
    logger.info(f"Test Recall: {test_recall:.6f}")
    logger.info(f"Test Specificity: {test_specificity:.6f}")
    if test_auroc is not None:
        logger.info(f"Test AUROC: {test_auroc:.6f}")
    else:
        logger.info(f"Test AUROC: N/A (model doesn't support probability prediction)")
    logger.info(f"Test F1: {test_f1:.6f}")
    logger.info(f"{'='*50}")
    
    logger.info(f"\nDetailed Classification Report for {model_name.upper()}:")
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
    
    # Log model parameters/hyperparameters
    model_params = model_instance.get_params()
    logger.info(f"Final parameters for {model_name}: {model_params}")
    
    # Log hyperparameters to TensorBoard
    for param_name, param_value in model_params.items():
        if isinstance(param_value, (int, float)):
            writer.add_scalar(f"Hyperparameters/{param_name}", param_value)
        elif isinstance(param_value, str):
            # For string parameters, we can log them as text or encode them
            writer.add_text(f"Hyperparameters/{param_name}", str(param_value))
        elif param_value is None:
            writer.add_text(f"Hyperparameters/{param_name}", "None")
        elif isinstance(param_value, bool):
            writer.add_scalar(f"Hyperparameters/{param_name}", int(param_value))
        else:
            # For other types, convert to string
            writer.add_text(f"Hyperparameters/{param_name}", str(param_value))
    
    # Log metrics (matching enhanced transformer format)
    writer.add_scalar("Metrics/test_accuracy", test_accuracy)
    writer.add_scalar("Metrics/val_accuracy", val_accuracy)
    writer.add_scalar("Metrics/test_f1", test_f1)  # Changed from f1_macro to f1
    writer.add_scalar("Metrics/val_f1", val_f1)    # Changed from val_f1_macro to val_f1
    writer.add_scalar("Metrics/test_precision", test_precision)  # Changed from precision_macro
    writer.add_scalar("Metrics/test_recall", test_recall)        # Changed from recall_macro
    writer.add_scalar("Metrics/test_specificity", test_specificity)  # New metric
    if test_auroc is not None:
        writer.add_scalar("Metrics/test_auroc", test_auroc)      # New metric
    
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
        'test_specificity': test_specificity,
        'test_auroc': test_auroc,
    }


def main():
    args = parse_args()

    # Check if cleaned data path exists
    if not os.path.exists(args.cleaned_data_path):
        logger.error(f"Cleaned data path not found: {args.cleaned_data_path}")
        logger.info("Available directories in data/:")
        if os.path.exists("data/"):
            for item in os.listdir("data/"):
                if os.path.isdir(os.path.join("data/", item)) and "cleaned" in item:
                    logger.info(f"  - {item}")
        return

    # Load dataset with CST features
    logger.info(f"Loading cleaned dataset with CST features from: {args.cleaned_data_path}")
    dataset = CoDeTM4CleanedWithCSTFeatures(
        cleaned_data_path=args.cleaned_data_path, 
        features_as_tensor=True  # Use tensor format for easier processing
    )
    
    train, val, test = dataset.get_dataset(
        split=['train', 'val', 'test'], 
        columns='all',  # Get all columns including CST features
        train_subset=args.train_subset
    )
    
    logger.info(f"Dataset loaded - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Check language distribution
    if 'language' in train.column_names:
        train_languages = {}
        for lang in train['language']:
            train_languages[lang] = train_languages.get(lang, 0) + 1
        logger.info(f"Training language distribution: {train_languages}")

    # Extract features based on feature type
    logger.info(f"Extracting {args.feature_type} features...")
    X_train, X_val, X_test, vectorizer, scaler = extract_features(
        train, val, test, feature_type=args.feature_type
    )
    
    # Get labels (using binary targets: 0=human, 1=ai)
    y_train = np.array(train['target_binary'])
    y_val = np.array(val['target_binary'])
    y_test = np.array(test['target_binary'])
    
    # Debug: Check class distribution
    train_class_dist = np.bincount(y_train)
    val_class_dist = np.bincount(y_val)
    test_class_dist = np.bincount(y_test)
    
    logger.info(f"Feature extraction complete. Feature dimensionality: {X_train.shape[1] if X_train is not None else 'None'}")
    logger.info(f"Training class distribution: {dict(enumerate(train_class_dist))} (0=human, 1=ai)")
    logger.info(f"Validation class distribution: {dict(enumerate(val_class_dist))} (0=human, 1=ai)")
    logger.info(f"Test class distribution: {dict(enumerate(test_class_dist))} (0=human, 1=ai)")
    
    # Set up logging directory
    log_dir = os.path.join("runs", "CoDeTM4_Cleaned_CST", f"traditional_ml_{args.feature_type}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Evaluate each model
    results = {}
    total_models = len(args.models)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"STARTING EVALUATION OF {total_models} MODELS")
    logger.info(f"Models to evaluate: {', '.join([m.upper() for m in args.models])}")
    logger.info(f"{'='*50}")
    
    for i, model_name in enumerate(args.models, 1):
        logger.info(f"\n🔄 [{i}/{total_models}] Starting evaluation of {model_name.upper()}...")
        try:
            result = evaluate_model(
                model_name, X_train, X_val, X_test, y_train, y_val, y_test,
                model_name, args.tune_hyperparameters, log_dir, args.feature_type
            )
            results[model_name] = result
            logger.info(f"✅ [{i}/{total_models}] {model_name.upper()} evaluation completed successfully!")
        except Exception as e:
            logger.error(f"❌ [{i}/{total_models}] Error evaluating {model_name}: {e}")
            continue
    
    # Summary of all models
    logger.info("\n" + "="*50)
    logger.info("SUMMARY OF ALL MODELS")
    logger.info("="*50)
    
    for model_name, result in results.items():
        auroc_str = f"AUROC={result['test_auroc']:.4f}" if result['test_auroc'] is not None else "AUROC=N/A"
        logger.info(f"{model_name.upper()}: Accuracy={result['test_accuracy']:.4f}, "
                   f"Precision={result['test_precision']:.4f}, "
                   f"Recall={result['test_recall']:.4f}, "
                   f"Specificity={result['test_specificity']:.4f}, "
                   f"{auroc_str}, "
                   f"F1={result['test_f1']:.4f}")
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        best_auroc = results[best_model_name]['test_auroc']
        auroc_info = f", AUROC: {best_auroc:.4f}" if best_auroc is not None else ""
        logger.info(f"\nBest model by F1-score: {best_model_name.upper()} "
                   f"(F1: {results[best_model_name]['test_f1']:.4f}{auroc_info})")


if __name__ == "__main__":
    main()
