import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.GCN import GCN
from data.dataset import GraphCoDeTM4
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC
from utils.gnn_utils import (
    save_model, load_model, set_seed, compute_batch, 
    evaluate, validate, train_epoch, train, load_data, load_single_data, load_multiple_data, DEVICE
)

import gc

MODEL_NAME = 'baseline_gcn'

if __name__ == '__main__':
    set_seed(872002)

    train_loader, val_loader = load_multiple_data()
    
    model = GCN(train_loader.num_node_features, hidden_dim_1=256, hidden_dim_2=128, embedding_dim=128, sage=True).to(DEVICE)
    model.name = MODEL_NAME

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    metrics = {
        'Acc': Accuracy(task='binary').to(DEVICE),
        'Prec': Precision(task='binary').to(DEVICE),
        'Rec': Recall(task='binary').to(DEVICE),
        'Spec': Specificity(task='binary').to(DEVICE),
        'AUROC': AUROC(task='binary').to(DEVICE)
    }

    train(model=model, optimizer=optimizer, criterion=criterion, train_dataloader=train_loader, val_dataloader=val_loader, metrics=metrics, num_epochs=30)
    
    # Clean up RAM to make room for the evaluation data
    # Not necessary if you have about 14GB to dedicate just to the training environment
    del train_loader, val_loader
    gc.collect()

    test_dataloader = load_single_data(split='test', shuffle=False)
    
    epoch, best_vloss, best_vacc = load_model(model, optimizer, save_path='models/gnn')
    test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS:")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    for metric_name, metric_value in test_metrics.items():
        print(f"Test {metric_name}: {metric_value:.4f}")
    print("="*50)
                                  
