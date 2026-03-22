"""Improved WISDM experiment with better model."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
import sys

class DeeperCNN(nn.Module):
    def __init__(self, in_ch, n_cls):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_cls)
        )
    
    def forward(self, x):
        x = self.features(x).squeeze(-1)
        return self.classifier(x)

def main():
    print('Loading WISDM...', flush=True)
    d = np.load('data/wisdm/processed/wisdm_processed.npz')
    data = d['data']
    labels = d['labels']
    splits = d['split_info']
    
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    train_data = torch.from_numpy(data[train_mask]).float()
    train_labels = torch.from_numpy(labels[train_mask]).long()
    test_data = torch.from_numpy(data[test_mask]).float()
    test_labels = torch.from_numpy(labels[test_mask]).long()
    
    print(f'Train: {len(train_data)}, Test: {len(test_data)}', flush=True)
    
    # Check class balance
    train_classes, train_counts = np.unique(train_labels.numpy(), return_counts=True)
    print(f'Train classes: {len(train_classes)}', flush=True)
    
    num_classes = len(np.unique(labels))
    print(f'Total classes: {num_classes}', flush=True)
    
    # Compute class weights for balanced sampling
    class_weights = 1.0 / train_counts
    sample_weights = class_weights[train_labels.numpy()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    model = DeeperCNN(3, num_classes)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=64)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()
    
    print('Training for 50 epochs...', flush=True)
    best_acc = 0
    best_f1 = 0
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            model.eval()
            all_preds = []
            with torch.no_grad():
                for bx, by in test_loader:
                    pred = model(bx).argmax(1)
                    all_preds.extend(pred.numpy())
            
            acc = accuracy_score(test_labels.numpy(), all_preds) * 100
            f1 = f1_score(test_labels.numpy(), all_preds, average='macro') * 100
            
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
            
            print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.2f}%, F1={f1:.2f}%', flush=True)
    
    print(f'\nBest Accuracy: {best_acc:.2f}%', flush=True)
    print(f'Best F1 Macro: {best_f1:.2f}%', flush=True)

if __name__ == '__main__':
    main()
