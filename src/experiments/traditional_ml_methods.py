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
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: wordcloud package not available. Word cloud visualizations will be skipped.")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
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
        choices=["rf", "svm", "lr", "nb", "catboost"],
        help="Models to evaluate: rf (Random Forest), svm (SVM), lr (Logistic Regression), nb (Naive Bayes), catboost (CatBoost)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="combined",
        choices=["tfidf", "cst", "combined"],
        help="Type of features to use: tfidf (text features only), cst (CST features only), combined (both)"
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        default=False,
        help="Save feature visualizations to disk"
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
    """Get model instance and hyperpa rameter grid for tuning."""
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
    elif model_name == "catboost":
        model = CatBoostClassifier(
            random_state=872002,
            verbose=False,  # Reduce output verbosity
            allow_writing_files=False  # Prevent CatBoost from writing files
        )
        param_grid = {
            "iterations": [100, 200, 300],
            "learning_rate": [0.03, 0.1, 0.2],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, param_grid


def visualize_tfidf_features(X_train, y_train, vectorizer, save_dir, dataset):
    """Create TF-IDF feature visualizations."""
    logger.info("Creating TF-IDF feature visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(save_dir, "tfidf_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert sparse matrix to dense for easier manipulation
    X_train_dense = X_train.toarray()
    
    # Split by class
    human_indices = np.where(y_train == 0)[0]
    ai_indices = np.where(y_train == 1)[0]
    
    # 1. Top discriminative terms
    plt.figure(figsize=(15, 8))
    
    # Calculate mean TF-IDF scores for each class
    human_mean = np.mean(X_train_dense[human_indices], axis=0)
    ai_mean = np.mean(X_train_dense[ai_indices], axis=0)
    
    # Calculate difference (AI - Human)
    diff_scores = ai_mean - human_mean
    
    # Get top terms for each class
    top_ai_indices = np.argsort(diff_scores)[-20:]
    top_human_indices = np.argsort(diff_scores)[:20]
    
    # Plot top AI terms
    plt.subplot(1, 2, 1)
    top_ai_terms = [feature_names[i] for i in top_ai_indices]
    top_ai_scores = diff_scores[top_ai_indices]
    plt.barh(range(len(top_ai_terms)), top_ai_scores, color='red', alpha=0.7)
    plt.yticks(range(len(top_ai_terms)), top_ai_terms)
    plt.xlabel('AI-preference score')
    plt.title('Top 20 AI-favored terms')
    plt.tight_layout()
    
    # Plot top Human terms
    plt.subplot(1, 2, 2)
    top_human_terms = [feature_names[i] for i in top_human_indices]
    top_human_scores = diff_scores[top_human_indices]
    plt.barh(range(len(top_human_terms)), top_human_scores, color='blue', alpha=0.7)
    plt.yticks(range(len(top_human_terms)), top_human_terms)
    plt.xlabel('Human-preference score')
    plt.title('Top 20 Human-favored terms')
    plt.tight_layout()
    
    plt.savefig(os.path.join(viz_dir, "top_discriminative_terms.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Word clouds (if available)
    if WORDCLOUD_AVAILABLE:
        # Combine code for each class - fix indexing issue
        train_codes = dataset['code']
        
        # Convert numpy indices to regular Python integers
        human_indices_list = [int(i) for i in human_indices]
        ai_indices_list = [int(i) for i in ai_indices]
        
        # Take a smaller sample for performance
        human_sample_size = min(500, len(human_indices_list))
        ai_sample_size = min(500, len(ai_indices_list))
        
        human_sample_indices = human_indices_list[:human_sample_size]
        ai_sample_indices = ai_indices_list[:ai_sample_size]
        
        human_codes = [train_codes[i] for i in human_sample_indices]
        ai_codes = [train_codes[i] for i in ai_sample_indices]
        
        human_text = ' '.join(human_codes)
        ai_text = ' '.join(ai_codes)
        
        # Create word clouds
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        human_wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                   colormap='Blues').generate(human_text)
        plt.imshow(human_wordcloud, interpolation='bilinear')
        plt.title('Human Code Word Cloud', fontsize=16)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        ai_wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                colormap='Reds').generate(ai_text)
        plt.imshow(ai_wordcloud, interpolation='bilinear')
        plt.title('AI Code Word Cloud', fontsize=16)
        plt.axis('off')
        
        plt.savefig(os.path.join(viz_dir, "word_clouds.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Feature distribution
    plt.figure(figsize=(12, 6))
    
    # Overall TF-IDF distribution
    all_scores = X_train_dense.flatten()
    all_scores = all_scores[all_scores > 0]  # Remove zeros
    
    plt.subplot(1, 2, 1)
    plt.hist(all_scores, bins=50, alpha=0.7, color='green')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of TF-IDF Scores')
    plt.yscale('log')
    
    # Feature sparsity
    plt.subplot(1, 2, 2)
    sparsity_per_sample = np.sum(X_train_dense > 0, axis=1) / X_train_dense.shape[1]
    plt.hist(sparsity_per_sample, bins=50, alpha=0.7, color='orange')
    plt.xlabel('Feature Density (non-zero ratio)')
    plt.ylabel('Number of Samples')
    plt.title('Feature Density Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "tfidf_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"TF-IDF visualizations saved to {viz_dir}")


def visualize_cst_features(X_train, y_train, save_dir, feature_type="cst"):
    """Create CST feature visualizations."""
    logger.info("Creating CST feature visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(save_dir, "cst_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # CST feature names (original list)
    cst_feature_names_full = ['function_defs', 'class_defs', 'if_statements', 'loops', 
                             'imports', 'comments', 'binary_ops', 'errors', 
                             'max_nesting_depth', 'language_encoded']
    
    # If combined features, extract only CST part
    if feature_type == "combined":
        # CST features are the last columns
        X_cst = X_train[:, -len(cst_feature_names_full):]
    else:
        X_cst = X_train
    
    # Convert to dense if sparse
    if hasattr(X_cst, 'toarray'):
        X_cst = X_cst.toarray()
    
    # Adjust feature names to match actual features
    actual_num_features = X_cst.shape[1]
    cst_feature_names = cst_feature_names_full[:actual_num_features]
    
    logger.info(f"Actual CST features: {actual_num_features}, Names: {cst_feature_names}")
    
    # Split by class
    human_indices = np.where(y_train == 0)[0]
    ai_indices = np.where(y_train == 1)[0]
    
    human_features = X_cst[human_indices]
    ai_features = X_cst[ai_indices]
    
    # 1. Feature comparison bar chart
    plt.figure(figsize=(15, 10))
    
    human_means = np.mean(human_features, axis=0)
    ai_means = np.mean(ai_features, axis=0)
    human_stds = np.std(human_features, axis=0)
    ai_stds = np.std(ai_features, axis=0)
    
    x = np.arange(len(cst_feature_names))
    width = 0.35
    
    plt.bar(x - width/2, human_means, width, label='Human', alpha=0.8, 
            yerr=human_stds, capsize=5, color='blue')
    plt.bar(x + width/2, ai_means, width, label='AI', alpha=0.8, 
            yerr=ai_stds, capsize=5, color='red')
    
    plt.xlabel('CST Features')
    plt.ylabel('Average Count')
    plt.title('CST Feature Comparison: Human vs AI Code')
    plt.xticks(x, cst_feature_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "cst_feature_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature distributions
    n_features = len(cst_feature_names)
    n_cols = min(5, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_features == 1 else axes
    else:
        axes = axes.ravel()
    
    for i, feature_name in enumerate(cst_feature_names):
        ax = axes[i] if n_features > 1 else axes
        
        # Plot histograms
        ax.hist(human_features[:, i], bins=30, alpha=0.7, label='Human', color='blue', density=True)
        ax.hist(ai_features[:, i], bins=30, alpha=0.7, label='AI', color='red', density=True)
        
        ax.set_title(f'{feature_name}')
        ax.set_xlabel('Count')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Add statistics
        human_median = np.median(human_features[:, i])
        ai_median = np.median(ai_features[:, i])
        ax.axvline(human_median, color='blue', linestyle='--', alpha=0.8)
        ax.axvline(ai_median, color='red', linestyle='--', alpha=0.8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "cst_feature_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation matrix
    plt.figure(figsize=(12, 10))
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_cst.T)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                xticklabels=cst_feature_names, yticklabels=cst_feature_names,
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('CST Features Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "cst_correlation_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"CST visualizations saved to {viz_dir}")


def visualize_combined_features(X_train, y_train, save_dir, n_samples=2000):
    """Create combined feature visualizations."""
    logger.info("Creating combined feature visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(save_dir, "combined_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert to dense if sparse
    if hasattr(X_train, 'toarray'):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train
    
    # Sample data for performance (dimensionality reduction can be slow)
    if len(X_train_dense) > n_samples:
        indices = np.random.choice(len(X_train_dense), n_samples, replace=False)
        X_sample = X_train_dense[indices]
        y_sample = y_train[indices]
    else:
        X_sample = X_train_dense
        y_sample = y_train
    
    # 1. PCA visualization
    logger.info("Performing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    colors = ['blue' if label == 0 else 'red' for label in y_sample]
    labels = ['Human' if label == 0 else 'AI' for label in y_sample]
    
    for label_val, color, label_name in [(0, 'blue', 'Human'), (1, 'red', 'AI')]:
        mask = y_sample == label_val
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.6, label=label_name, s=20)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA: Combined Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PCA explained variance
    plt.subplot(1, 2, 2)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "pca_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE visualization (if feasible)
    if X_sample.shape[0] <= 1000 and X_sample.shape[1] <= 1000:  # t-SNE can be slow
        logger.info("Performing t-SNE...")
        
        # First reduce dimensionality with PCA if needed
        if X_sample.shape[1] > 50:
            pca_pre = PCA(n_components=50)
            X_for_tsne = pca_pre.fit_transform(X_sample)
        else:
            X_for_tsne = X_sample
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample)//4))
        X_tsne = tsne.fit_transform(X_for_tsne)
        
        plt.figure(figsize=(10, 8))
        
        for label_val, color, label_name in [(0, 'blue', 'Human'), (1, 'red', 'AI')]:
            mask = y_sample == label_val
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, alpha=0.6, label=label_name, s=20)
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE: Combined Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "tsne_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        logger.info("Skipping t-SNE due to large dataset size (performance consideration)")
    
    # 3. Feature importance analysis (simple variance-based)
    feature_variances = np.var(X_train_dense, axis=0)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(feature_variances, bins=50, alpha=0.7, color='green')
    plt.xlabel('Feature Variance')
    plt.ylabel('Count')
    plt.title('Distribution of Feature Variances')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    # Show top variant features
    top_indices = np.argsort(feature_variances)[-20:]
    plt.barh(range(len(top_indices)), feature_variances[top_indices])
    plt.ylabel('Feature Index')
    plt.xlabel('Variance')
    plt.title('Top 20 Most Variant Features')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "feature_variance_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Combined feature visualizations saved to {viz_dir}")


def visualize_feature_pairplots(X_train, y_train, save_dir, feature_names=None, max_features=10, sample_size=1000):
    """Create pairwise scatterplots for features."""
    logger.info("Creating pairwise scatterplot visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(save_dir, "pairplot_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert to dense if sparse
    if hasattr(X_train, 'toarray'):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train
    
    # Sample data for performance
    if len(X_train_dense) > sample_size:
        indices = np.random.choice(len(X_train_dense), sample_size, replace=False)
        X_sample = X_train_dense[indices]
        y_sample = y_train[indices]
    else:
        X_sample = X_train_dense
        y_sample = y_train
    
    # Limit number of features for visualization
    n_features = min(max_features, X_sample.shape[1])
    
    # If we have too many features, select the most variant ones
    if X_sample.shape[1] > max_features:
        feature_variances = np.var(X_sample, axis=0)
        top_indices = np.argsort(feature_variances)[-max_features:]
        X_sample = X_sample[:, top_indices]
        if feature_names is not None and len(feature_names) > max_features:
            feature_names = [feature_names[i] for i in top_indices]
    
    if feature_names is None or len(feature_names) == 0:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    else:
        feature_names = feature_names[:n_features]
    
    # Create pairplot
    fig, axes = plt.subplots(n_features, n_features, figsize=(3*n_features, 3*n_features))
    
    colors = ['blue' if label == 0 else 'red' for label in y_sample]
    color_map = {0: 'blue', 1: 'red'}
    label_map = {0: 'Human', 1: 'AI'}
    
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j] if n_features > 1 else axes
            
            if i == j:
                # Diagonal: histograms
                for label_val in [0, 1]:
                    mask = y_sample == label_val
                    ax.hist(X_sample[mask, i], bins=20, alpha=0.6, 
                           color=color_map[label_val], label=label_map[label_val], density=True)
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel('Density')
                if i == 0:
                    ax.legend()
            else:
                # Off-diagonal: scatterplots
                for label_val in [0, 1]:
                    mask = y_sample == label_val
                    ax.scatter(X_sample[mask, j], X_sample[mask, i], 
                             alpha=0.6, s=20, color=color_map[label_val], 
                             label=label_map[label_val] if i == 0 and j == 1 else None)
                ax.set_xlabel(feature_names[j])
                ax.set_ylabel(feature_names[i])
                if i == 0 and j == 1:
                    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "feature_pairplots.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Pairwise scatterplots saved to {viz_dir}")


def visualize_nonlinear_dependencies(X_train, y_train, save_dir, feature_names=None, max_features=15):
    """Detect and visualize nonlinear dependencies between features."""
    logger.info("Analyzing nonlinear dependencies between features...")
    
    # Create visualization directory
    viz_dir = os.path.join(save_dir, "nonlinear_dependencies")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Convert to dense if sparse
    if hasattr(X_train, 'toarray'):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train
    
    # Limit number of features for computational efficiency
    n_features = min(max_features, X_train_dense.shape[1])
    
    if X_train_dense.shape[1] > max_features:
        # Select most variant features
        feature_variances = np.var(X_train_dense, axis=0)
        top_indices = np.argsort(feature_variances)[-max_features:]
        X_subset = X_train_dense[:, top_indices]
        if feature_names is not None and len(feature_names) > max_features:
            feature_names_subset = [feature_names[i] for i in top_indices]
        else:
            feature_names_subset = [f'Feature_{i}' for i in top_indices]
    else:
        X_subset = X_train_dense
        feature_names_subset = feature_names if feature_names is not None and len(feature_names) > 0 else [f'Feature_{i}' for i in range(n_features)]
    
    # 1. Mutual Information Matrix (nonlinear dependencies)
    logger.info("Computing mutual information matrix...")
    
    # Compute mutual information between features and target
    mi_scores = mutual_info_classif(X_subset, y_train, random_state=42)
    
    # Compute pairwise mutual information between features (approximation)
    # For computational efficiency, we'll use correlation as a proxy and compare with Spearman
    pearson_corr = np.corrcoef(X_subset.T)
    spearman_corr = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                spearman_corr[i, j] = 1.0
            else:
                try:
                    corr_result = spearmanr(X_subset[:, i], X_subset[:, j])
                    corr = corr_result.correlation if hasattr(corr_result, 'correlation') else corr_result[0]
                    # Handle NaN correlations
                    if np.isnan(float(corr)):
                        corr = 0.0
                    spearman_corr[i, j] = float(corr)
                    spearman_corr[j, i] = float(corr)
                except:
                    spearman_corr[i, j] = 0.0
                    spearman_corr[j, i] = 0.0
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pearson correlation
    im1 = axes[0, 0].imshow(pearson_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title('Pearson Correlation (Linear Dependencies)')
    axes[0, 0].set_xticks(range(n_features))
    axes[0, 0].set_yticks(range(n_features))
    axes[0, 0].set_xticklabels(feature_names_subset, rotation=45, ha='right')
    axes[0, 0].set_yticklabels(feature_names_subset)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Spearman correlation
    im2 = axes[0, 1].imshow(spearman_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 1].set_title('Spearman Correlation (Monotonic Dependencies)')
    axes[0, 1].set_xticks(range(n_features))
    axes[0, 1].set_yticks(range(n_features))
    axes[0, 1].set_xticklabels(feature_names_subset, rotation=45, ha='right')
    axes[0, 1].set_yticklabels(feature_names_subset)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference (Spearman - Pearson): indicates nonlinear monotonic relationships
    nonlinear_indicator = np.abs(spearman_corr) - np.abs(pearson_corr)
    im3 = axes[1, 0].imshow(nonlinear_indicator, cmap='viridis', vmin=0)
    axes[1, 0].set_title('Nonlinearity Indicator (|Spearman| - |Pearson|)')
    axes[1, 0].set_xticks(range(n_features))
    axes[1, 0].set_yticks(range(n_features))
    axes[1, 0].set_xticklabels(feature_names_subset, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(feature_names_subset)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Mutual information with target
    axes[1, 1].bar(range(n_features), mi_scores)
    axes[1, 1].set_title('Mutual Information with Target (Feature Relevance)')
    axes[1, 1].set_xlabel('Features')
    axes[1, 1].set_ylabel('Mutual Information')
    axes[1, 1].set_xticks(range(n_features))
    axes[1, 1].set_xticklabels(feature_names_subset, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "nonlinear_dependency_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Identify and plot top nonlinear relationships
    # Find pairs with highest nonlinearity indicator
    nonlinear_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            nonlinear_score = nonlinear_indicator[i, j]
            if nonlinear_score > 0.1:  # Threshold for significant nonlinearity
                nonlinear_pairs.append((i, j, nonlinear_score))
    
    # Sort by nonlinearity score
    nonlinear_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if nonlinear_pairs:
        # Plot top nonlinear relationships
        n_plots = min(6, len(nonlinear_pairs))
        if n_plots > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for plot_idx in range(n_plots):
                i, j, score = nonlinear_pairs[plot_idx]
                ax = axes[plot_idx]
                
                # Scatterplot with color coding by class
                for label_val in [0, 1]:
                    mask = y_train == label_val
                    color = 'blue' if label_val == 0 else 'red'
                    label = 'Human' if label_val == 0 else 'AI'
                    ax.scatter(X_subset[mask, i], X_subset[mask, j], 
                             alpha=0.6, s=20, color=color, label=label)
                
                ax.set_xlabel(feature_names_subset[i])
                ax.set_ylabel(feature_names_subset[j])
                ax.set_title(f'Nonlinear Score: {score:.3f}')
                ax.legend()
            
            # Hide unused subplots
            for plot_idx in range(n_plots, 6):
                axes[plot_idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "top_nonlinear_relationships.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Found {len(nonlinear_pairs)} significant nonlinear relationships")
    else:
        logger.info("No significant nonlinear relationships detected")
    
    # Save summary statistics
    summary_file = os.path.join(viz_dir, "dependency_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("FEATURE DEPENDENCY ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("CORRELATION MATRIX ANALYSIS:\n")
        f.write(f"- Pearson correlation captures LINEAR relationships\n")
        f.write(f"- Spearman correlation captures MONOTONIC relationships\n")
        f.write(f"- Nonlinearity indicator (|Spearman| - |Pearson|) shows NONLINEAR monotonic patterns\n\n")
        
        # Find strongest linear correlations
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
        strong_linear = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(pearson_corr[i, j]) > 0.7:
                    strong_linear.append((i, j, pearson_corr[i, j]))
        
        f.write(f"STRONG LINEAR CORRELATIONS (|r| > 0.7):\n")
        if strong_linear:
            for i, j, corr in sorted(strong_linear, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"- {feature_names_subset[i]} <-> {feature_names_subset[j]}: {corr:.3f}\n")
        else:
            f.write("- None found\n")
        f.write("\n")
        
        f.write(f"SIGNIFICANT NONLINEAR RELATIONSHIPS:\n")
        if nonlinear_pairs:
            for i, j, score in nonlinear_pairs[:10]:  # Top 10
                f.write(f"- {feature_names_subset[i]} <-> {feature_names_subset[j]}: {score:.3f}\n")
        else:
            f.write("- None found\n")
        f.write("\n")
        
        f.write(f"FEATURE RELEVANCE (Mutual Information with Target):\n")
        mi_sorted = sorted(enumerate(mi_scores), key=lambda x: x[1], reverse=True)
        for idx, score in mi_sorted:
            f.write(f"- {feature_names_subset[idx]}: {score:.3f}\n")
    
    logger.info(f"Nonlinear dependency analysis saved to {viz_dir}")


def save_feature_visualizations(X_train, X_val, X_test, y_train, y_val, y_test, 
                               vectorizer, scaler, feature_type, log_dir, train_dataset):
    """Save comprehensive feature visualizations including pairplots and nonlinear analysis."""
    logger.info(f"\n{'='*50}")
    logger.info("CREATING FEATURE VISUALIZATIONS")
    logger.info(f"{'='*50}")
    
    viz_base_dir = os.path.join(log_dir, "feature_visualizations")
    os.makedirs(viz_base_dir, exist_ok=True)
    
    # Get feature names for better labeling
    feature_names = None
    
    if feature_type == "tfidf":
        visualize_tfidf_features(X_train, y_train, vectorizer, viz_base_dir, train_dataset)
        feature_names = vectorizer.get_feature_names_out()
        
        # For TF-IDF, limit to top features for pairplots
        visualize_feature_pairplots(X_train, y_train, viz_base_dir, feature_names, max_features=8)
        visualize_nonlinear_dependencies(X_train, y_train, viz_base_dir, feature_names, max_features=10)
        
    elif feature_type == "cst":
        visualize_cst_features(X_train, y_train, viz_base_dir, feature_type)
        
        # CST feature names
        cst_feature_names_full = ['function_defs', 'class_defs', 'if_statements', 'loops', 
                                 'imports', 'comments', 'binary_ops', 'errors', 
                                 'max_nesting_depth', 'language_encoded']
        actual_num_features = X_train.shape[1]
        feature_names = cst_feature_names_full[:actual_num_features]
        
        visualize_feature_pairplots(X_train, y_train, viz_base_dir, feature_names)
        visualize_nonlinear_dependencies(X_train, y_train, viz_base_dir, feature_names)
        
    elif feature_type == "combined":
        # Create TF-IDF visualizations (first part of combined features)
        # CST features are the last N columns, where N is determined by actual CST features
        cst_feature_names_full = ['function_defs', 'class_defs', 'if_statements', 'loops', 
                                 'imports', 'comments', 'binary_ops', 'errors', 
                                 'max_nesting_depth', 'language_encoded']
        
        # Determine actual number of CST features based on X_train dimensions
        # We know the CST features are at the end
        if hasattr(X_train, 'toarray'):
            X_dense_sample = X_train[:10].toarray()  # Sample to check
        else:
            X_dense_sample = X_train[:10]
        
        # For combined features, we need to figure out how many CST features there are
        # The dataset likely has fewer than 10 CST features
        total_features = X_train.shape[1]
        
        # Estimate CST features count (should be <= 10)
        estimated_cst_features = min(10, total_features)  # Safety check
        
        # Try different numbers to find the right split
        if total_features > 10:
            n_cst_features = 9  # Based on the error we saw earlier
        else:
            n_cst_features = total_features
        
        tfidf_part = X_train[:, :-n_cst_features] if n_cst_features > 0 else X_train
        
        # Create a temporary vectorizer-like object for TF-IDF part
        class TempVectorizer:
            def get_feature_names_out(self):
                return [f"tfidf_feature_{i}" for i in range(tfidf_part.shape[1])]
        
        temp_vectorizer = TempVectorizer()
        
        # TF-IDF visualizations (but skip word clouds for combined to avoid confusion)
        viz_dir = os.path.join(viz_base_dir, "tfidf_part")
        os.makedirs(viz_dir, exist_ok=True)
        
        logger.info("Creating TF-IDF part visualizations...")
        feature_names = temp_vectorizer.get_feature_names_out()
        
        # Convert sparse matrix to dense for easier manipulation
        if hasattr(tfidf_part, 'toarray'):
            tfidf_dense = tfidf_part.toarray()
        else:
            tfidf_dense = tfidf_part
        
        # Feature distribution
        plt.figure(figsize=(12, 6))
        
        all_scores = tfidf_dense.flatten()
        all_scores = all_scores[all_scores > 0]
        
        plt.subplot(1, 2, 1)
        plt.hist(all_scores, bins=50, alpha=0.7, color='green')
        plt.xlabel('TF-IDF Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of TF-IDF Scores (Combined Features)')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        sparsity_per_sample = np.sum(tfidf_dense > 0, axis=1) / tfidf_dense.shape[1]
        plt.hist(sparsity_per_sample, bins=50, alpha=0.7, color='orange')
        plt.xlabel('Feature Density (non-zero ratio)')
        plt.ylabel('Number of Samples')
        plt.title('TF-IDF Feature Density Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "tfidf_part_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # CST visualizations
        visualize_cst_features(X_train, y_train, viz_base_dir, feature_type)
        
        # Combined features visualization
        visualize_combined_features(X_train, y_train, viz_base_dir)
        
        # For combined features, create pairplots with a subset of most important features
        visualize_feature_pairplots(X_train, y_train, viz_base_dir, max_features=8)
        visualize_nonlinear_dependencies(X_train, y_train, viz_base_dir, max_features=12)
    
    logger.info(f"All feature visualizations saved to {viz_base_dir}")


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
    
    logger.info(f"Feature extraction complete. Feature dimensionality: {X_train.shape[1] if hasattr(X_train, 'shape') else 'Unknown'}")
    logger.info(f"Training class distribution: {dict(enumerate(train_class_dist))} (0=human, 1=ai)")
    logger.info(f"Validation class distribution: {dict(enumerate(val_class_dist))} (0=human, 1=ai)")
    logger.info(f"Test class distribution: {dict(enumerate(test_class_dist))} (0=human, 1=ai)")
    
    # Set up logging directory
    log_dir = os.path.join("runs", "CoDeTM4_Cleaned_CST", f"traditional_ml_{args.feature_type}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Save feature visualizations if requested
    if args.save_visualizations:
        save_feature_visualizations(
            X_train, X_val, X_test, y_train, y_val, y_test,
            vectorizer, scaler, args.feature_type, log_dir, train
        )
    
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
