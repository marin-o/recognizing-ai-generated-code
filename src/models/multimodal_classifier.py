import torch
import torch.nn as nn
import warnings
from transformers import RobertaModel


class SimpleMultimodalClassifier(nn.Module):
    """
    A simple classifier built on top of CodeBERT with a linear head.

    Args:
        dropout_rate (float): Dropout probability for regularization.
        reduced_dim (int): Dimensionality of the reduced representation.
        num_numerical_features (int): Number of numerical features to concatenate with CodeBERT embeddings.
        num_classes (int): Number of output classes.
        freeze_codebert (bool): Whether to freeze CodeBERT weights.
    """

    def __init__(
        self,
        reduce: bool,
        dim: int = None,
        dropout_rate: float = 0.2,
        num_numerical_features: int = 9,
        num_classes: int = 2,
        freeze_codebert: bool = True,
    ):
        super(SimpleMultimodalClassifier, self).__init__()
        
        if reduce and dim is None:
            raise ValueError("dim must be specified when reduce=True")
        
        if not reduce and dim is not None:
            warnings.warn(
                f"dim ({dim}) will be ignored because reduce=False",
                UserWarning
            )
        self.reduce = reduce
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        
        if dim is not None and dim >= self.codebert.config.hidden_size // 4:
            raise ValueError(
                f"dim ({dim}) must be smaller than "
                f"{self.codebert.config.hidden_size // 4} "
                f"(hidden_size // 4 = {self.codebert.config.hidden_size} // 4). "
                f"Consider using a smaller value like 64 or 128."
            )


        self.dim = dim
        self.num_classes = num_classes
        self.num_features = num_numerical_features

        if freeze_codebert:
            for param in self.codebert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)

        if reduce:
            self.reduce_dims_mlp = nn.Sequential(
                nn.Linear(
                    self.codebert.config.hidden_size, self.codebert.config.hidden_size // 2
                ),
                nn.ReLU(),
                nn.Linear(
                    self.codebert.config.hidden_size // 2,
                    self.codebert.config.hidden_size // 4,
                ),
                nn.ReLU(),
                nn.Linear(self.codebert.config.hidden_size // 4, self.dim),
            )
        else:
            self.dim = self.codebert.config.hidden_size

        self.classifier = nn.Linear(
            self.dim + self.num_features, self.num_classes
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Logits for each class.

        Raises:
            ValueError: If input_ids or attention_mask is None or shapes are incompatible.
        """
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")
        if input_ids.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                "input_ids and attention_mask must have the same batch size"
            )
        if features is None:
            raise ValueError("features must be provided")
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        emb = self.dropout(emb)
        if self.reduce:
            emb = self.reduce_dims_mlp(emb)
        # Concatenate numerical features
        if features is None:
            raise ValueError("features must be provided")
        concatenated = torch.cat((emb, features), dim=1)
        if concatenated.shape[1] != self.dim + self.num_features:
            raise ValueError(
                f"Concatenated features shape {concatenated.shape[1]} does not match "
                f"expected shape {self.dim + self.num_features}"
            )
        # Pass through classifier layer
        logits = self.classifier(concatenated)
        return logits
