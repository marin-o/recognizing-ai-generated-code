import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.aigcodeset import AIGCodeSet
from models.baseline_model import SimpleLinearHeadClassifier
from utils.utils import tokenize_fn
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

train, val, test = AIGCodeSet().get_dataset(split=True)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

tokenize = lambda x: tokenize_fn(tokenizer, x)
train, val, test = (
    train.map(tokenize, batched=True),
    val.map(tokenize, batched=True),
    test.map(tokenize, batched=True),
)

train.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])
val.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])
test.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])

model = SimpleLinearHeadClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
criterion = torch.nn.CrossEntropyLoss()

train_dataloader, val_dataloader, test_dataloader = (
    DataLoader(train, 32, shuffle=True),
    DataLoader(val, 32),
    DataLoader(test, 32),
)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        input_ids, attention_masks, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['target'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_masks, labels)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()


        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")

    model.train()
    avg_loss = train_one_epoch(epoch)
    # print(f"Average loss for epoch {epoch+1}: {avg_loss:.4f}")

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            input_ids, attention_masks, labels = vdata['input_ids'].to(device), vdata['attention_mask'].to(device), vdata['target'].to(device)
            outputs = model(input_ids, attention_masks, labels)
            vloss = criterion(outputs, labels)
            running_vloss += vloss.item()
        avg_vloss = running_vloss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        print(f"Validation loss: {avg_vloss:.4f}")
    


# test the model and calculate accuracy 
def evaluate_model(dataloader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            input_ids, attention_masks, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['target'].to(device)
            outputs = model(input_ids, attention_masks)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Evaluate the model on the test set
test_accuracy = evaluate_model(test_dataloader)
print(f"Test Accuracy: {test_accuracy:.4f}")


