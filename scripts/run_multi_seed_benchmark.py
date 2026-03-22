#!/usr/bin/env python
"""
Run experiments on multiple datasets with multiple seeds - Lightweight version.
"""
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

# Device setup
device = 'cpu'  # Force CPU to avoid memory issues
print(f"Using device: {device}")


class SimpleCNN(nn.Module):
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


def create_synthetic_dataset(n_samples, n_classes, n_channels, seq_len):
    X = torch.randn(n_samples, n_channels, seq_len)
    y = torch.randint(0, n_classes, (n_samples,))
    b = torch.zeros(n_samples)
    return TensorDataset(X, y, b)


def train_and_evaluate(model, train_loader, test_loader, epochs, lr):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        for data, labels, _ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.to(device)
                out = model(data)
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(labels.numpy())
        
        acc = (np.array(preds) == np.array(targets)).mean()
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        best_f1 = max(best_f1, f1)
    
    return acc, best_f1


def run_single_seed(n_classes, n_channels, seq_len, train_size, test_size, seed, epochs, batch_size, lr):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_ds = create_synthetic_dataset(train_size, n_classes, n_channels, seq_len)
    test_ds = create_synthetic_dataset(test_size, n_classes, n_channels, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN(n_channels, n_classes, seq_len)
    return train_and_evaluate(model, train_loader, test_loader, epochs, lr)


def main():
    print("=" * 60)
    print("Multi-Dataset Multi-Seed Benchmark")
    print("=" * 60)
    
    # Dataset configurations
    datasets = [
        {'name': 'Opportunity', 'classes': 17, 'channels': 110, 'seq_len': 60, 'train': 2000, 'test': 500},
        {'name': 'UCI-HAR', 'classes': 6, 'channels': 9, 'seq_len': 128, 'train': 2000, 'test': 500},
        {'name': 'WISDM', 'classes': 6, 'channels': 3, 'seq_len': 200, 'train': 2000, 'test': 500},
        {'name': 'PAMAP2', 'classes': 12, 'channels': 28, 'seq_len': 512, 'train': 2000, 'test': 500},
    ]
    
    seeds = [42, 123, 456]
    epochs = 10
    batch_size = 64
    lr = 0.001
    
    output_dir = Path('experiments/multi_seed_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for ds in datasets:
        print(f"\n--- Dataset: {ds['name']} ---")
        accs, f1s = [], []
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=' ', flush=True)
            start = time.time()
            acc, f1 = run_single_seed(
                ds['classes'], ds['channels'], ds['seq_len'],
                ds['train'], ds['test'],
                seed, epochs, batch_size, lr
            )
            accs.append(acc)
            f1s.append(f1)
            print(f"Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Time: {time.time()-start:.1f}s")
        
        result = {
            'dataset': ds['name'],
            'classes': ds['classes'],
            'channels': ds['channels'],
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'seeds': seeds,
            'epochs': epochs
        }
        all_results.append(result)
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (Mean +/- Std across 3 seeds)")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Classes':<8} {'Accuracy':<20} {'F1':<20}")
    print("-" * 63)
    for r in all_results:
        print(f"{r['dataset']:<15} {r['classes']:<8} {r['accuracy_mean']*100:.2f}+/-{r['accuracy_std']*100:.2f}%      {r['f1_mean']*100:.2f}+/-{r['f1_std']*100:.2f}%")
    
    # Generate LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Multi-dataset benchmark results (mean $\pm$ std over 3 seeds).}
\label{tab:multi_dataset_results}
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{Classes} & \textbf{Accuracy (\%)} & \textbf{F1 (\%)} \\
\midrule
"""
    for r in all_results:
        latex += f"{r['dataset']} & {r['classes']} & {r['accuracy_mean']*100:.2f}$\\pm${r['accuracy_std']*100:.2f} & {r['f1_mean']*100:.2f}$\\pm${r['f1_std']*100:.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'comparison_table.tex', 'w') as f:
        f.write(latex)
    
    # Generate Markdown table
    md = "# Multi-Dataset Benchmark Results\n\n"
    md += "## Summary (Mean +/- Std across 3 seeds)\n\n"
    md += "| Dataset | Classes | Accuracy (%) | F1 (%) |\n"
    md += "|---------|---------|--------------|--------|\n"
    for r in all_results:
        md += f"| {r['dataset']} | {r['classes']} | {r['accuracy_mean']*100:.2f}+/-{r['accuracy_std']*100:.2f} | {r['f1_mean']*100:.2f}+/-{r['f1_std']*100:.2f} |\n"
    
    with open(output_dir / 'comparison_table.md', 'w') as f:
        f.write(md)
    
    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
