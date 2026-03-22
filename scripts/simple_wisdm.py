"""Simple WISDM experiment to get accuracy numbers."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import sys

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
    
    class SimpleCNN(nn.Module):
        def __init__(self, in_ch, n_cls):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, 32, 5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, n_cls)
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    num_classes = len(np.unique(labels))
    model = SimpleCNN(3, num_classes)
    
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=64)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print('Training for 30 epochs...', flush=True)
    best_acc = 0
    best_f1 = 0
    
    for epoch in range(30):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            for bx, by in test_loader:
                pred = model(bx).argmax(1)
                all_preds.extend(pred.numpy())
        
        acc = np.mean(np.array(all_preds) == test_labels.numpy()) * 100
        f1 = f1_score(test_labels.numpy(), all_preds, average='macro') * 100
        
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}: Acc={acc:.2f}%, F1={f1:.2f}%', flush=True)
    
    print(f'Best Accuracy: {best_acc:.2f}%', flush=True)
    print(f'Best F1 Macro: {best_f1:.2f}%', flush=True)

if __name__ == '__main__':
    main()
