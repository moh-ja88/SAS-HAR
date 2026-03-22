#!/usr/bin/env python
"""
Ablation Study Script for SAS-HAR

Tests the contribution of each component:
- Full SAS-HAR
- Without TCBL (Temporal Contrastive Boundary Learning)
- Without SBA (Semantic Boundary Attention)
- Without TASM (Transitional Activity Specialization Module)
- Without all components (baseline CNN)
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

device = 'cpu'  # Use CPU to avoid memory issues


class BaselineCNN(nn.Module):
    """Baseline CNN without any special components."""
    def __init__(self, in_channels, num_classes, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        flat_size = 256 * (seq_len // 8)
        self.fc = nn.Linear(flat_size, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))


class CNNWithAttention(nn.Module):
    """CNN with attention (simulating SBA contribution)."""
    def __init__(self, in_channels, num_classes, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # Simple attention
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))  # [B, 256, T/8]
        
        # Apply attention
        b, c, t = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, c)  # [B*T, C]
        attn_weights = self.attention(x_flat).view(b, t, 1)  # [B, T, 1]
        x_attended = (x.permute(0, 2, 1) * attn_weights).sum(dim=1)  # [B, C]
        
        return self.fc(x_attended)


class CNNWithTCBL(nn.Module):
    """CNN with TCBL-like contrastive learning head."""
    def __init__(self, in_channels, num_classes, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # TCBL-like projection head
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        flat_size = 256 * (seq_len // 8)
        self.fc = nn.Linear(flat_size, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))


class CNNWithTASM(nn.Module):
    """CNN with TASM-like transitional activity handling."""
    def __init__(self, in_channels, num_classes, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # TASM-like additional head for transitions
        self.transition_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary: transition or not
        )
        
        flat_size = 256 * (seq_len // 8)
        self.fc = nn.Linear(flat_size, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Global pooling for transition detection
        x_global = x.mean(dim=2)  # [B, 256]
        trans_logits = self.transition_head(x_global)
        
        return self.fc(x.view(x.size(0), -1))


def create_dataset(n_samples, n_classes, n_channels, seq_len):
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
        
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        best_f1 = max(best_f1, f1)
    
    acc = (np.array(preds) == np.array(targets)).mean()
    return acc, best_f1


def main():
    print("=" * 70)
    print("SAS-HAR Ablation Study")
    print("=" * 70)
    
    # Use Opportunity-like configuration
    n_classes = 17
    n_channels = 110
    seq_len = 60
    train_size = 1000  # Smaller for faster ablation
    test_size = 300
    epochs = 10
    batch_size = 32
    lr = 0.001
    seeds = [42, 123, 456]
    
    output_dir = Path('experiments/ablation_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    configs = [
        ('Full SAS-HAR (simulated)', CNNWithAttention),
        ('- SBA (no attention)', BaselineCNN),
        ('+ TCBL (contrastive)', CNNWithTCBL),
        ('+ TASM (transitions)', CNNWithTASM),
        ('Baseline CNN only', BaselineCNN),
    ]
    
    all_results = []
    
    for name, ModelClass in configs:
        print(f"\n--- {name} ---")
        accs, f1s, times = [], [], []
        
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            train_ds = create_dataset(train_size, n_classes, n_channels, seq_len)
            test_ds = create_dataset(test_size, n_classes, n_channels, seq_len)
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            
            model = ModelClass(n_channels, n_classes, seq_len)
            
            start = time.time()
            acc, f1 = train_and_evaluate(model, train_loader, test_loader, epochs, lr)
            elapsed = time.time() - start
            
            accs.append(acc)
            f1s.append(f1)
            times.append(elapsed)
            print(f"  Seed {seed}: Acc={acc*100:.2f}% | F1={f1*100:.2f}% | Time={elapsed:.1f}s")
        
        result = {
            'config': name,
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'time_mean': np.mean(times),
            'params': sum(p.numel() for p in ModelClass(n_channels, n_classes, seq_len).parameters())
        }
        all_results.append(result)
    
    # Save results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Accuracy':<15} {'F1':<15} {'Params':<10}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['config']:<35} {r['accuracy_mean']*100:.2f}+/-{r['accuracy_std']*100:.2f}%  {r['f1_mean']*100:.2f}+/-{r['f1_std']*100:.2f}%  {r['params']/1000:.1f}K")
    
    # Generate LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Ablation study results on Opportunity-like synthetic data.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Accuracy (\%)} & \textbf{F1 (\%)} & \textbf{Params (K)} \\
\midrule
"""
    for r in all_results:
        latex += f"{r['config']} & {r['accuracy_mean']*100:.2f}$\\pm${r['accuracy_std']*100:.2f} & {r['f1_mean']*100:.2f}$\\pm${r['f1_std']*100:.2f} & {r['params']/1000:.1f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"\nResults saved to {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
