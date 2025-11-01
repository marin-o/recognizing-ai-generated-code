"""
CBM Model with Precomputed Embeddings
======================================

This is a variant of the CBM StarCoder Tree model that uses precomputed embeddings
instead of computing them on-the-fly. This significantly speeds up training by
eliminating the need to run the transformer backbone during training.

The model receives precomputed CLS token embeddings and processes them through
the CNN-BiLSTM architecture with tree-sitter features, exactly like the original
model but without the embedding computation step.
"""

import torch
import torch.nn as nn
import sys
import os
import logging
from typing import Optional, List

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.tree_features import extract_tree_features, get_feature_dimension

logger = logging.getLogger(__name__)


class CBMPrecomputed(nn.Module):
    """
    Multi-Channel CNN-BiLSTM classifier with precomputed embeddings and tree-sitter features.

    This model is architecturally identical to CBMStarCoderTree, but receives precomputed
    CLS token embeddings instead of computing them from raw text.

    Architecture:
    1. Receive precomputed embeddings (CLS tokens)
    2. Tree-sitter extracts AST/statistical features
    3. Tree features are concatenated with embeddings
    4. CNN layers extract local patterns
    5. BiLSTM with attention captures sequential dependencies
    6. Features are merged and classified
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        filter_sizes: int = 768,
        lstm_hidden_dim: int = 256,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        tree_feature_projection_dim: int = 128,
    ):
        """
        Initialize the CBM model with precomputed embeddings.

        Args:
            embedding_dim: Dimension of precomputed embeddings
            filter_sizes: Number of filters for CNN layers
            lstm_hidden_dim: Hidden dimension of BiLSTM
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            tree_feature_projection_dim: Dimension to project tree features to
        """
        super(CBMPrecomputed, self).__init__()

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.tree_feature_dim = get_feature_dimension()  # 11 features
        self.tree_feature_projection_dim = tree_feature_projection_dim

        logger.info(f"Initializing CBM with precomputed embeddings (dim={embedding_dim})")

        # Tree feature projection layer
        self.tree_projection = nn.Sequential(
            nn.Linear(self.tree_feature_dim, tree_feature_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # After concatenating tree features with embeddings
        self.enhanced_embedding_dim = self.embedding_dim + tree_feature_projection_dim

        # CNN1: multiple kernel sizes
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(self.enhanced_embedding_dim, filter_sizes, kernel_size=k)
                for k in [2, 3, 4, 5]
            ]
        )

        # CNN2: second layer of convs
        self.conv21 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)
        self.conv22 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)
        self.conv23 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.enhanced_embedding_dim,
            hidden_size=self.lstm_hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        # Attention layer for LSTM outputs
        self.attention_fc = nn.Linear(self.lstm_hidden_dim, 1)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()

        # Final classifier
        self.classifier = nn.Linear(filter_sizes * 4 + lstm_hidden_dim, num_classes)

        logger.info(
            f"Model initialized with {self.count_parameters()} trainable parameters"
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_tree_features_batch(self, codes: List[str], languages: List[str]) -> torch.Tensor:
        """
        Extract tree-sitter features for a batch of code samples.

        Args:
            codes: List of code strings
            languages: List of language strings

        Returns:
            Tensor of shape (batch_size, tree_feature_dim)
        """
        batch_features = []

        for code, lang in zip(codes, languages):
            try:
                features = extract_tree_features(code, lang)
                feature_vector = [
                    features["function_count"],
                    features["class_count"],
                    features["if_count"],
                    features["loop_count"],
                    features["import_count"],
                    features["comment_count"],
                    features["binary_op_count"],
                    features["error_count"],
                    features["max_nesting_depth"],
                    features["total_nodes"],
                    features["avg_node_depth"],
                ]
                batch_features.append(feature_vector)
            except Exception as e:
                logger.warning(f"Failed to extract tree features: {e}")
                # Use default features (all zeros except error_count=1)
                batch_features.append([0.0] * 10 + [1.0])

        return torch.tensor(batch_features, dtype=torch.float32)

    def forward(
        self,
        embeddings: torch.Tensor,
        codes: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ):
        """
        Forward pass of the model.

        Args:
            embeddings: Precomputed embeddings, shape (batch_size, embedding_dim)
                       These are CLS token embeddings from the transformer backbone
            codes: Optional list of raw code strings for tree feature extraction
            languages: Optional list of language strings for tree feature extraction

        Returns:
            Logits tensor of shape (batch_size, num_classes)

        Note:
            If codes and languages are not provided, zero tree features will be used.
            The embeddings are unsqueezed to add a sequence dimension since the
            original model expects (batch_size, seq_len, embedding_dim) but we only
            have the CLS token, so seq_len=1.
        """
        batch_size = embeddings.size(0)
        
        # Embeddings come as (batch_size, embedding_dim)
        # For CNN compatibility, we need a sequence. We'll replicate the CLS token
        # to create a pseudo-sequence. We need at least length 6 for the two-layer CNNs:
        # - First layer: kernel sizes 2,3,4,5 produce outputs of length 5,4,3,2
        # - Second layer: kernel size 2 needs input length >= 2
        seq_len = 10  # Use 10 for safety
        emb = embeddings.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)

        # Extract and project tree features if codes and languages provided
        if codes is not None and languages is not None:
            tree_features = self.extract_tree_features_batch(codes, languages)
            tree_features = tree_features.to(emb.device)
            tree_features_projected = self.tree_projection(tree_features)
        else:
            # Use zero tree features if not provided
            tree_features_projected = torch.zeros(
                batch_size,
                self.tree_feature_projection_dim,
                device=emb.device,
                dtype=emb.dtype,
            )

        # Expand tree features to match sequence length and concatenate
        tree_features_expanded = tree_features_projected.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (batch_size, seq_len, tree_projection_dim)

        # Concatenate embeddings with tree features
        emb = torch.cat([emb, tree_features_expanded], dim=2)  # (batch_size, seq_len, embedding_dim + tree_projection_dim)

        # CNN branch
        emb_cnn = emb.transpose(1, 2)  # (batch_size, enhanced_embedding_dim, 1)
        conv1_outputs = [self.relu(conv(emb_cnn)) for conv in self.convs1]
        pooled = [
            torch.max_pool1d(c, kernel_size=c.size(2)).squeeze(2) for c in conv1_outputs
        ]

        x_1 = pooled[0]

        c2_2 = self.relu(self.conv21(conv1_outputs[1]))
        x_2 = torch.max_pool1d(c2_2, kernel_size=c2_2.size(2)).squeeze(2)

        c2_3 = self.relu(self.conv22(conv1_outputs[2]))
        x_3 = torch.max_pool1d(c2_3, kernel_size=c2_3.size(2)).squeeze(2)

        c2_4 = self.relu(self.conv23(conv1_outputs[3]))
        x_4 = torch.max_pool1d(c2_4, kernel_size=c2_4.size(2)).squeeze(2)

        cnn_features = torch.cat([x_1, x_2, x_3, x_4], dim=1)

        # LSTM branch with attention
        lstm_out, _ = self.lstm(emb)

        attn_weights = torch.softmax(self.attention_fc(lstm_out).squeeze(-1), dim=1)
        attn_applied = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        # Merge features
        features = torch.cat([cnn_features, attn_applied], dim=1)
        features = self.dropout(features)

        # Classification
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    """
    Test the CBM Precomputed model.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 80)
    print("Testing CBM Precomputed Model")
    print("=" * 80)

    # Test with dummy precomputed embeddings (simulating CodeBERT)
    print("\n1. Testing with dummy precomputed embeddings...")
    print("-" * 80)

    try:
        # Initialize model
        model = CBMPrecomputed(
            embedding_dim=768,
            filter_sizes=256,
            lstm_hidden_dim=128,
            num_classes=2,
            dropout_rate=0.3,
            tree_feature_projection_dim=64,
        )

        # Create dummy data
        batch_size = 4
        embedding_dim = 768
        
        # Simulate precomputed embeddings (CLS tokens)
        dummy_embeddings = torch.randn(batch_size, embedding_dim)

        # Test codes and languages
        test_codes = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class Calculator:\n    def add(self, a, b):\n        return a + b",
            "for i in range(10):\n    print(i)",
            "import os\nimport sys\n\ndef main():\n    pass"
        ]
        test_languages = ["python", "python", "python", "python"]

        print(f"Input embeddings shape: {dummy_embeddings.shape}")
        print(f"Model parameters: {model.count_parameters():,}")

        # Forward pass without tree features
        print("\nForward pass WITHOUT tree features...")
        model.eval()
        with torch.no_grad():
            outputs_no_tree = model(dummy_embeddings)
        print(f"Output shape: {outputs_no_tree.shape}")
        print(f"Sample output: {outputs_no_tree[0]}")

        # Forward pass with tree features
        print("\nForward pass WITH tree features...")
        with torch.no_grad():
            outputs_with_tree = model(
                dummy_embeddings, codes=test_codes, languages=test_languages
            )
        print(f"Output shape: {outputs_with_tree.shape}")
        print(f"Sample output: {outputs_with_tree[0]}")

        # Check that outputs are different
        diff = torch.abs(outputs_no_tree - outputs_with_tree).sum().item()
        print(f"\nDifference between with/without tree features: {diff:.4f}")
        if diff > 0:
            print("✓ Tree features are being properly integrated!")
        else:
            print("✗ Warning: Tree features may not be properly integrated")

        print("\n✓ Basic test PASSED")

    except Exception as e:
        print(f"\n✗ Basic test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
