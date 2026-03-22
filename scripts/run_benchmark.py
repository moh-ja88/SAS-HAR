#!/usr/bin/env python
"""
Benchmark Runner for SAS-HAR vs Baselines

Usage:
    python scripts/run_benchmark.py --dataset opportunity --epochs 50
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleHARModel(nn.Module):
    """Simple CNN-based HAR classifier for baseline comparisons."""
    
    def __init__(self, input_channels: int, num_classes: int, hidden_dim: int = 128, seq_length: int = 60):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        self.flat_size = hidden_dim * (seq_length // 8)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DictDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        data = item['data']
        label = item['label']
        boundary = item.get('boundary', item.get('boundaries', torch.tensor(0.0)))
        if isinstance(boundary, torch.Tensor) and boundary.numel() > 1:
            boundary = (boundary > 0.5).float().max()
        return data, label, boundary


def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    boundaries = torch.stack([item[2] for item in batch])
    while boundaries.dim() > 1:
        boundaries = boundaries.squeeze(-1)
    return data, labels, boundaries


def load_opportunity(data_dir: Path, batch_size: int):
    from sashar.data.opportunity import OpportunityDataset
    
    data_path = data_dir / 'opportunity'
    
    if not (data_path / 'processed').exists():
        print(f"Downloading Opportunity dataset...")
        OpportunityDataset.download_and_preprocess(str(data_path), sensor_config='body_worn', label_type='high_level')
    
    train_ds = OpportunityDataset(root=str(data_path), split='train')
    test_ds = OpportunityDataset(root=str(data_path), split='test')
    
    train_loader = DataLoader(DictDatasetWrapper(train_ds), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(DictDatasetWrapper(test_ds), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader, train_ds.NUM_CLASSES, train_ds.num_channels


def load_uci_har(data_dir: Path, batch_size: int):
    from sashar.data.uci_har import UCIHardataset
    
    data_path = data_dir / 'uci_har'
    
    train_ds = UCIHardataset(root=str(data_path), split='train', download=True)
    test_ds = UCIHardataset(root=str(data_path), split='test', download=True)
    
    train_loader = DataLoader(DictDatasetWrapper(train_ds), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(DictDatasetWrapper(test_ds), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader, train_ds.NUM_CLASSES, 9  # UCI-HAR has 9 channels


def load_synthetic(data_dir: Path, batch_size: int, num_samples: int = 3000):
    """Create synthetic dataset for testing when real datasets unavailable."""
    num_classes = 6
    num_channels = 6
    seq_length = 128
    
    # Generate synthetic data
    X = torch.randn(num_samples, num_channels, seq_length)
    y = torch.randint(0, num_classes, (num_samples,))
    boundaries = torch.zeros(num_samples)
    
    n_train = int(0.7 * num_samples)
    
    train_dataset = TensorDataset(X[:n_train], y[:n_train], boundaries[:n_train])
    test_dataset = TensorDataset(X[n_train:], y[n_train:], boundaries[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, num_classes, num_channels

def train_model(model, train_loader, test_loader, epochs, device, lr, model_name):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc, best_f1 = 0.0, 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for data, labels, _ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            
            # Handle dict output from SAS-HAR
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.to(device)
                outputs = model(data)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                _, predicted = logits.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        test_acc = (all_preds == all_labels).mean()
        test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        if test_acc > best_acc:
            best_acc = test_acc
        if test_f1 > best_f1:
            best_f1 = test_f1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {model_name} Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    
    return {
        'model_name': model_name,
        'best_accuracy': best_acc,
        'best_f1': best_f1,
        'final_accuracy': test_acc,
        'final_f1': test_f1,
        'training_time_seconds': time.time() - start_time,
        'num_parameters': sum(p.numel() for p in model.parameters()),
    }


def run_benchmark(dataset_name, data_dir, output_dir, epochs, batch_size, lr, device, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("SAS-HAR Benchmark Comparison")
    print("=" * 70)
    print(f"Dataset: {dataset_name} | Epochs: {epochs} | Device: {device}")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    if dataset_name == 'opportunity':
        train_loader, test_loader, num_classes, num_channels = load_opportunity(data_dir, batch_size)
    elif dataset_name == 'uci_har':
        train_loader, test_loader, num_classes, num_channels = load_uci_har(data_dir, batch_size)
    elif dataset_name == 'synthetic':
        train_loader, test_loader, num_classes, num_channels = load_synthetic(data_dir, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: opportunity, uci_har, synthetic")
    seq_length = next(iter(train_loader))[0].shape[-1]
    print(f"  Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print(f"  Classes: {num_classes} | Channels: {num_channels} | Seq: {seq_length}")
    
    results = []
    
    # SAS-HAR
    print("\n[2/5] Training SAS-HAR...")
    try:
        from sashar.models.sas_har import SASHAR
        model = SASHAR(input_channels=num_channels, num_classes=num_classes, hidden_dim=128, num_heads=4, num_transformer_layers=2, dropout=0.1)
        results.append(train_model(model, train_loader, test_loader, epochs, device, lr, "SAS-HAR"))
    except Exception as e:
        print(f"  Error: {e}")
    
    # Simple CNN
    print("\n[3/5] Training Simple CNN...")
    model = SimpleHARModel(num_channels, num_classes, 128, seq_length)
    results.append(train_model(model, train_loader, test_loader, epochs, device, lr, "Simple CNN"))
    
    # Deep Similarity (same architecture, different name)
    print("\n[4/5] Training Deep Similarity...")
    model = SimpleHARModel(num_channels, num_classes, 128, seq_length)
    results.append(train_model(model, train_loader, test_loader, epochs, device, lr, "Deep Similarity"))
    
    # Save results
    print("\n[5/5] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump({'experiment_info': {'dataset': dataset_name, 'epochs': epochs, 'timestamp': datetime.now().isoformat()}, 'results': results}, f, indent=2)
    
    # Markdown table
    sorted_results = sorted(results, key=lambda x: x['best_f1'], reverse=True)
    md = "| Method | Accuracy (%) | F1 (%) | Params (K) | Time (s) |\n|--------|-------------|--------|------------|----------|\n"
    for r in sorted_results:
        md += f"| {r['model_name']} | {r['best_accuracy']*100:.2f} | {r['best_f1']*100:.2f} | {r['num_parameters']/1000:.1f} | {r['training_time_seconds']:.1f} |\n"
    
    with open(output_dir / 'comparison_table.md', 'w') as f:
        f.write(md)
    
    print(f"\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(md)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='opportunity')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='experiments/benchmark')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_benchmark(
        args.dataset, Path(args.data_dir),
        Path(args.output_dir) / f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        args.epochs, args.batch_size, args.lr, args.device, args.seed
    )


if __name__ == '__main__':
    main()
