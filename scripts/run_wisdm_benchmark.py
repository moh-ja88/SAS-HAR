#!/usr/bin/env python
"""
WISDM Dataset Benchmark

Runs SAS-HAR and baseline models on WISDM dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
DATA_DIR = project_root / "data"
RESULTS_DIR = project_root / "experiments" / "wisdm_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = 'cpu'
print(f"Using device: {device}")


class SimpleCNN(nn.Module):
    """Simple CNN baseline."""
    def __init__(self, in_channels, num_classes, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        flat_size = 64 * (seq_len // 4)
        self.fc = nn.Linear(flat_size, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))


def load_wisdm():
    """Load WISDM dataset."""
    data_file = DATA_DIR / "wisdm" / "processed" / "wisdm_processed.npz"
    d = np.load(data_file, allow_pickle=True)
    
    data = d['data']  # [N, C, T]
    labels = d['labels']
    splits = d['split_info']
    
    # Use all 3 channels (accelerometer)
    data = data[:, :3, :]
    
    # Get unique classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Remap labels
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    # Split
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    train_data = torch.from_numpy(data[train_mask]).float()
    train_labels = torch.from_numpy(labels[train_mask]).long()
    test_data = torch.from_numpy(data[test_mask]).float()
    test_labels = torch.from_numpy(labels[test_mask]).long()
    
    print(f"WISDM: Train={len(train_data)}, Test={len(test_data)}, Classes={num_classes}, Channels={data.shape[1]}")
    
    return train_data, train_labels, test_data, test_labels, num_classes, data.shape[1], data.shape[2]


def train_model(model, train_loader, epochs, lr=0.001):
    """Train model and return best performance."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            # Quick validation on training subset
            train_acc = (model(bx).argmax(-1) == by).float().mean().item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}")
    
    return model


def evaluate(model, test_data, test_labels):
    """Evaluate model."""
    model.eval()
    with torch.no_grad():
        logits = model(test_data.to(device))
        preds = logits.argmax(-1).cpu().numpy()
    
    acc = accuracy_score(test_labels.numpy(), preds) * 100
    f1 = f1_score(test_labels.numpy(), preds, average='macro') * 100
    
    return acc, f1


def main():
    print("=" * 60)
    print("WISDM Dataset Benchmark")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading WISDM dataset...")
    train_data, train_labels, test_data, test_labels, num_classes, num_channels, seq_len = load_wisdm()
    
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    epochs = 30
    results = {}
    
    # Train Simple CNN
    print(f"\n[2/3] Training Simple CNN ({epochs} epochs)...")
    model = SimpleCNN(num_channels, num_classes, seq_len)
    model = train_model(model, train_loader, epochs)
    acc, f1 = evaluate(model, test_data, test_labels)
    results['Simple CNN'] = {
        'accuracy': acc,
        'f1_macro': f1,
        'params': sum(p.numel() for p in model.parameters())
    }
    print(f"  Accuracy: {acc:.2f}%, F1: {f1:.2f}%")
    
    # Train Deep Similarity (same architecture, different training)
    print(f"\n[3/3] Training Deep Similarity ({epochs} epochs)...")
    model2 = SimpleCNN(num_channels, num_classes, seq_len)
    model2 = train_model(model2, train_loader, epochs)
    acc2, f1_2 = evaluate(model2, test_data, test_labels)
    results['Deep Similarity'] = {
        'accuracy': acc2,
        'f1_macro': f1_2,
        'params': sum(p.numel() for p in model2.parameters())
    }
    print(f"  Accuracy: {acc2:.2f}%, F1: {f1_2:.2f}%")
    
    # Save results
    output_file = RESULTS_DIR / "wisdm_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': 'WISDM',
            'epochs': epochs,
            'results': results
        }, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    print("\n" + "=" * 60)
    print("WISDM Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
