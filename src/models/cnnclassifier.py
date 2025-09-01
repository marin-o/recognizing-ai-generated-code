'''
Model implementation based on CNN architecture only (without LSTM/attention components).
This is a simplified version of the CBM classifier that uses only the CNN components
for text classification.
'''

import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, base_model, embedding_dim=768, filter_sizes=768, num_classes=2, dropout_rate=0.5):
        """
        Initialize the CNN-only classifier.
        
        Args:
            base_model: Model used to extract word embeddings
            embedding_dim: Dimension of word embeddings
            filter_sizes: Number of filters for CNN
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(CNNClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.base_model = base_model
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # CNN1: multiple kernel sizes
        self.convs1 = nn.ModuleList([
            nn.Conv1d(embedding_dim, filter_sizes, kernel_size=k)
            for k in [2, 3, 4, 5]
        ])

        # CNN2: second layer of convs applied to outputs of CNN1
        self.conv21 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)
        self.conv22 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)
        self.conv23 = nn.Conv1d(filter_sizes, filter_sizes, kernel_size=2)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.pool = nn.functional.max_pool1d
        self.relu = nn.ReLU()

        # Classifier uses only CNN features (no LSTM features)
        self.classifier = nn.Linear(filter_sizes*4, num_classes)

    def forward(self, inputs):
        emb = self.base_model(**inputs).last_hidden_state  # (batch, seq_len, embed_dim)

        emb_cnn = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv1_outputs = [self.relu(conv(emb_cnn)) for conv in self.convs1]
        pooled = [self.pool(c, kernel_size=c.size(2)).squeeze(2) for c in conv1_outputs]
        x_1 = pooled[0]
        c2_2 = self.relu(self.conv21(conv1_outputs[1]))
        x_2 = self.pool(c2_2, kernel_size=c2_2.size(2)).squeeze(2)

        c2_3 = self.relu(self.conv22(conv1_outputs[2]))
        x_3 = self.pool(c2_3, kernel_size=c2_3.size(2)).squeeze(2)

        c2_4 = self.relu(self.conv23(conv1_outputs[3]))
        x_4 = self.pool(c2_4, kernel_size=c2_4.size(2)).squeeze(2)

        # Concatenate only CNN features (no LSTM/attention features)
        cnn_features = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        features = self.dropout(cnn_features)

        logits = self.classifier(features)
        return logits
        

class DummyBaseModel(nn.Module):
    def __init__(self, seq_len=10, embedding_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def forward(self, **inputs):
        batch_size = inputs['input_ids'].shape[0]
        class Output:
            def __init__(self, tensor):
                self.last_hidden_state = tensor
        dummy_tensor = torch.randn(batch_size, self.seq_len, self.embedding_dim)
        return Output(dummy_tensor)


if __name__ == "__main__":
    dummy_input = {
        'input_ids': torch.randint(0, 1000, (4, 10))
    }

    base_model = DummyBaseModel(seq_len=10, embedding_dim=768)
    model = CNNClassifier(base_model)

    with torch.no_grad():
        output = model(dummy_input)

    print("Output shape:", output.shape)  # Expected: (4, num_classes)
