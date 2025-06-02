import torch
import torch.nn as nn

from transformers import RobertaModel, RobertaTokenizer

class SimpleLinearHeadClassifier(nn.Module):
    def __init__(self):
        super(SimpleLinearHeadClassifier, self).__init__()
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        for param in self.codebert.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.codebert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state[:, 0, :]
        dropped = self.dropout(emb)
        logits = self.classifier(dropped)
        return logits
