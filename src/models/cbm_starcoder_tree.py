"""
Enhanced CBM Model with StarCoder/CodeBERT backbone and Tree-sitter features.

This is an improved version of the CBM classifier from:
"A text classification method based on a convolutional and bidirectional long short-term memory models"
DOI: https://doi.org/10.1080/09540091.2022.2098926

Enhancements:
- Support for StarCoder 3B or CodeBERT backbones
- Integration of AST and statistical features from tree-sitter
- Dynamic language support via tree-sitter parsers
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer, BitsAndBytesConfig
import sys
import os
import logging
from typing import Dict, Optional, Literal


# Add src to path for tree features
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.tree_features import extract_tree_features, get_feature_dimension

logger = logging.getLogger(__name__)

BackboneType = Literal["starcoder", "codebert"]


class CBMStarCoderTree(nn.Module):
    """
    Multi-Channel CNN-BiLSTM classifier with transformer backbone and tree-sitter features.

    Architecture:
    1. Transformer backbone (StarCoder/CodeBERT) extracts embeddings
    2. Tree-sitter extracts AST/statistical features
    3. Tree features are concatenated with CLS token embedding
    4. CNN layers extract local patterns
    5. BiLSTM with attention captures sequential dependencies
    6. Features are merged and classified
    """

    def __init__(
        self,
        backbone_type: BackboneType = "codebert",
        model_name: Optional[str] = None,
        embedding_dim: int = 768,
        filter_sizes: int = 768,
        lstm_hidden_dim: int = 256,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = True,
        tree_feature_projection_dim: int = 128,
    ):
        """
        Initialize the CBM model with tree-sitter features.

        Args:
            backbone_type: Type of backbone model ("starcoder" or "codebert")
            model_name: Specific model name (overrides backbone_type if provided)
            embedding_dim: Dimension of word embeddings from backbone
            filter_sizes: Number of filters for CNN layers
            lstm_hidden_dim: Hidden dimension of BiLSTM
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone parameters
            tree_feature_projection_dim: Dimension to project tree features to
        """
        super(CBMStarCoderTree, self).__init__()

        self.backbone_type = backbone_type
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.tree_feature_dim = get_feature_dimension()  # 11 features
        self.tree_feature_projection_dim = tree_feature_projection_dim

        # Select model based on backbone_type or explicit model_name
        if model_name is not None:
            self.model_name = model_name
        elif backbone_type == "starcoder":
            self.model_name = "bigcode/starcoder2-3b"
        elif backbone_type == "codebert":
            self.model_name = "microsoft/codebert-base"
        else:
            raise ValueError(
                f"Invalid backbone_type: {backbone_type}. Must be 'starcoder' or 'codebert'"
            )

        logger.info(f"Initializing CBM with backbone: {self.model_name}")

        # Load backbone model
        if backbone_type == "codebert" and model_name is None:
            # Use RobertaModel for CodeBERT (more efficient)
            self.backbone = RobertaModel.from_pretrained(self.model_name)
        else:
            # Use AutoModel for other models including StarCoder
            self.backbone = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                dtype=torch.bfloat16,
            )

        # Get actual embedding dimension from model
        if hasattr(self.backbone.config, "hidden_size"):
            self.embedding_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "d_model"):
            self.embedding_dim = self.backbone.config.d_model

        logger.info(f"Backbone embedding dimension: {self.embedding_dim}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone parameters frozen")

        # Tree feature projection layer
        # Projects tree features to a smaller dimension and prepares them for concatenation
        self.tree_projection = nn.Sequential(
            nn.Linear(self.tree_feature_dim, tree_feature_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # After concatenating tree features with embeddings, we have:
        # embedding_dim + tree_feature_projection_dim
        self.enhanced_embedding_dim = self.embedding_dim + tree_feature_projection_dim

        # CNN1: multiple kernel sizes (operates on enhanced embeddings)
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(self.enhanced_embedding_dim, filter_sizes, kernel_size=k)
                for k in [2, 3, 4, 5]
            ]
        )

        # CNN2: second layer of convs applied to outputs of CNN1
        self.conv21 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)
        self.conv22 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)
        self.conv23 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)

        # BiLSTM operates on the enhanced embeddings
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
        # Input: concatenated CNN features (filter_sizes * 4) + LSTM features (lstm_hidden_dim)
        self.classifier = nn.Linear(filter_sizes * 4 + lstm_hidden_dim, num_classes)

        logger.info(
            f"Model initialized with {self.count_parameters()} trainable parameters"
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_tree_features_batch(self, codes: list, languages: list) -> torch.Tensor:
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
                # Extract values in consistent order
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
        inputs: Dict[str, torch.Tensor],
        codes: Optional[list] = None,
        languages: Optional[list] = None,
    ):
        """
        Forward pass of the model.

        Args:
            inputs: Dictionary containing 'input_ids' and 'attention_mask'
            codes: Optional list of raw code strings for tree feature extraction
            languages: Optional list of language strings for tree feature extraction

        Returns:
            Logits tensor of shape (batch_size, num_classes)

        Note:
            If codes and languages are not provided, zero tree features will be used.
        """
        # Get embeddings from backbone
        outputs = self.backbone(**inputs)
        emb = outputs.last_hidden_state  # (batch_size, seq_len, embedding_dim)

        batch_size = emb.size(0)

        # Extract and project tree features if codes and languages provided
        if codes is not None and languages is not None:
            tree_features = self.extract_tree_features_batch(codes, languages)
            tree_features = tree_features.to(emb.device)
            tree_features_projected = self.tree_projection(
                tree_features
            )  # (batch_size, tree_projection_dim)
        else:
            # Use zero tree features if not provided
            tree_features_projected = torch.zeros(
                batch_size,
                self.tree_feature_projection_dim,
                device=emb.device,
                dtype=emb.dtype,
            )

        # Expand tree features to match sequence length and concatenate with embeddings
        # We concatenate tree features across all tokens
        tree_features_expanded = tree_features_projected.unsqueeze(1).expand(
            batch_size, emb.size(1), -1
        )  # (batch_size, seq_len, tree_projection_dim)

        # Concatenate embeddings with tree features
        emb = torch.cat(
            [emb, tree_features_expanded], dim=2
        )  # (batch_size, seq_len, embedding_dim + tree_projection_dim)

        # CNN branch
        emb_cnn = emb.transpose(1, 2)  # (batch_size, enhanced_embedding_dim, seq_len)
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


def load_tokenizer(
    backbone_type: BackboneType = "codebert", model_name: Optional[str] = None
):
    """
    Load appropriate tokenizer for the backbone model.

    Args:
        backbone_type: Type of backbone ("starcoder" or "codebert")
        model_name: Specific model name (overrides backbone_type if provided)

    Returns:
        Tokenizer instance
    """
    if model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif backbone_type == "codebert":
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    elif backbone_type == "starcoder":
        tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starcoder2-3b", trust_remote_code=True
        )
    else:
        raise ValueError(f"Invalid backbone_type: {backbone_type}")

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


if __name__ == "__main__":
    """
    Test the CBM StarCoder Tree model with both backbones.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 80)
    print("Testing CBM StarCoder Tree Model")
    print("=" * 80)

    # Test with CodeBERT (small and fast for testing)
    print("\n1. Testing with CodeBERT backbone...")
    print("-" * 80)

    try:
        # Initialize model
        model_codebert = CBMStarCoderTree(
            backbone_type="codebert",
            embedding_dim=768,
            filter_sizes=256,
            lstm_hidden_dim=128,
            num_classes=2,
            dropout_rate=0.3,
            freeze_backbone=True,
            tree_feature_projection_dim=64,
        )

        # Load tokenizer
        tokenizer_codebert = load_tokenizer("codebert")

        # Test data
        test_codes = [
            """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
            """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
""",
        ]

        test_languages = ["python", "python"]

        # Tokenize
        inputs = tokenizer_codebert(
            test_codes,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Model parameters: {model_codebert.count_parameters():,}")

        # Test tree feature extraction
        print("\nTesting tree feature extraction...")
        tree_features = model_codebert.extract_tree_features_batch(
            test_codes, test_languages
        )
        print(f"Tree features shape: {tree_features.shape}")
        print(f"Sample features for code 1: {tree_features[0]}")

        # Forward pass without tree features
        print("\nForward pass WITHOUT tree features...")
        model_codebert.eval()
        with torch.no_grad():
            outputs_no_tree = model_codebert(inputs)
        print(f"Output shape: {outputs_no_tree.shape}")
        print(f"Sample output: {outputs_no_tree[0]}")

        # Forward pass with tree features
        print("\nForward pass WITH tree features...")
        with torch.no_grad():
            outputs_with_tree = model_codebert(
                inputs, codes=test_codes, languages=test_languages
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

        print("\n✓ CodeBERT backbone test PASSED")

    except Exception as e:
        print(f"\n✗ CodeBERT backbone test FAILED: {e}")
        import traceback

        traceback.print_exc()

    # Test with SemEval dataset
    print("\n\n2. Testing with SemEval Dataset...")
    print("-" * 80)

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
        from data.dataset.semeval2026_task13 import SemEval2026Task13

        # Load dataset
        print("Loading SemEval dataset (subtask A, small sample)...")
        dataset_loader = SemEval2026Task13(subtask="A")

        # Get small train sample
        train_dataset = dataset_loader.get_dataset(
            split="train", train_subset=0.001
        )  # 0.1% for quick test
        print(f"Loaded {len(train_dataset)} samples")

        # Get first few samples
        test_samples = [train_dataset[i] for i in range(min(4, len(train_dataset)))]

        # Extract codes and languages
        codes = [sample["code"] for sample in test_samples]
        languages = [
            sample.get("language", "python").lower() for sample in test_samples
        ]
        labels = [sample["target_binary"] for sample in test_samples]

        print(f"\nSample languages: {languages}")
        print(f"Sample labels: {labels}")

        # Tokenize
        inputs = tokenizer_codebert(
            codes, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Forward pass
        print("\nRunning forward pass with SemEval data...")
        model_codebert.eval()
        with torch.no_grad():
            outputs = model_codebert(inputs, codes=codes, languages=languages)

        print(f"Output shape: {outputs.shape}")
        predictions = torch.argmax(outputs, dim=1)
        print(f"Predictions: {predictions.tolist()}")
        print(f"Actual labels: {labels}")

        # Calculate accuracy
        correct = sum(p == l for p, l in zip(predictions.tolist(), labels))
        accuracy = correct / len(labels) * 100
        print(f"Random accuracy on {len(labels)} samples: {accuracy:.1f}%")

        print("\n✓ SemEval dataset test PASSED")

    except Exception as e:
        print(f"\n✗ SemEval dataset test FAILED: {e}")
        print(
            "Note: You may need to login with 'huggingface-cli login' to access the dataset"
        )
        import traceback

        traceback.print_exc()

    # Test with multi-language samples
    print("\n\n3. Testing with Multi-language Support...")
    print("-" * 80)

    try:
        multi_lang_codes = [
            "def test(): pass",  # Python
            "public class Test { }",  # Java
            "function test() { }",  # JavaScript
        ]

        multi_lang_languages = ["python", "java", "javascript"]

        print("Testing languages:", multi_lang_languages)

        # Test tree feature extraction for each language
        for code, lang in zip(multi_lang_codes, multi_lang_languages):
            features = extract_tree_features(code, lang)
            print(f"\n{lang.upper()} features:")
            print(f"  Functions: {features['function_count']}")
            print(f"  Classes: {features['class_count']}")
            print(f"  Errors: {features['error_count']}")

        # Test model with multi-language batch
        inputs = tokenizer_codebert(
            multi_lang_codes,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        model_codebert.eval()
        with torch.no_grad():
            outputs = model_codebert(
                inputs, codes=multi_lang_codes, languages=multi_lang_languages
            )

        print(f"\nMulti-language batch output shape: {outputs.shape}")
        print(f"Predictions: {torch.argmax(outputs, dim=1).tolist()}")

        print("\n✓ Multi-language test PASSED")

    except Exception as e:
        print(f"\n✗ Multi-language test FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n\n" + "=" * 80)
    print("All basic tests completed!")
    print("=" * 80)

    # Optional: Test StarCoder if user wants (commented out by default as it's large)
    print(
        "\n\nNOTE: StarCoder 3B testing is skipped by default as it requires ~6GB download."
    )
    print("To test StarCoder, uncomment the code section below and run again.")
    print("The model architecture is identical, just with a different backbone.")

    """
    # Uncomment to test StarCoder
    print("\n\n4. Testing with StarCoder 3B backbone...")
    print("-" * 80)
    
    try:
        model_starcoder = CBMStarCoderTree(
            backbone_type="starcoder",
            embedding_dim=2048,  # StarCoder has larger embedding
            filter_sizes=512,
            lstm_hidden_dim=256,
            num_classes=2,
            dropout_rate=0.3,
            freeze_backbone=True,
            tree_feature_projection_dim=128
        )
        
        tokenizer_starcoder = load_tokenizer("starcoder")
        
        inputs = tokenizer_starcoder(
            test_codes,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        model_starcoder.eval()
        with torch.no_grad():
            outputs = model_starcoder(inputs, codes=test_codes, languages=test_languages)
        
        print(f"StarCoder output shape: {outputs.shape}")
        print("✓ StarCoder backbone test PASSED")
        
    except Exception as e:
        print(f"✗ StarCoder backbone test FAILED: {e}")
        import traceback
        traceback.print_exc()
    """
