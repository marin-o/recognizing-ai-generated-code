import torch
import torch.nn as nn
from transformers import RobertaModel
from typing import Optional


class SimpleLinearHeadClassifier(nn.Module):
    """
    A simple classifier built on top of CodeBERT with a linear head.

    Args:
        dropout_rate (float): Dropout probability for regularization.
        num_classes (int): Number of output classes.
        freeze_codebert (bool): Whether to freeze CodeBERT weights.
    """

    def __init__(
        self,
        dropout_rate: float = 0.2,
        num_classes: int = 2,
        freeze_codebert: bool = True,
    ):
        super(SimpleLinearHeadClassifier, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        if freeze_codebert:
            for param in self.codebert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        
        # 2-layer classifier head
        hidden_size = self.codebert.config.hidden_size
        classifier_hidden_dim = hidden_size // 2  # Half the input size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
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
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state[:, 0, :]  
        dropped = self.dropout(emb)
        logits = self.classifier(dropped)
        return logits
