#!/usr/bin/env python
"""
Quick Start Script for SAS-HAR Research Framework

This script:
1. Downloads real HAR datasets (or generates synthetic data if unavailable)
2. Runs a quick training experiment
3. Generates results and figures
4. Creates a summary report

Usage:
    python scripts/quick_start.py --dataset synthetic --epochs 10
    python scripts/quick_start.py --dataset opportunity --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sashar.models.sas_har import SASHAR, SASHARLite
    from sashar.evaluation.metrics import (
        compute_classification_metrics,
        compute_boundary_metrics,
        compute_all_metrics
    )
    from sashar.evaluation.visualization import (
        plot_confusion_matrix,
        plot_training_curves,
        plot_embeddings,
        plot_class_distribution,
        generate_all_paper_figures,
    )
    HAS_SASHAR = True
except ImportError as e:
    print(f"Warning: Could not import SAS-HAR modules: {e}")
    HAS_SASHAR = False


# ============================================================================
# Fallback Model (when SASHAR import fails)
# ============================================================================

class SimpleHARModel(nn.Module):
    """Simple HAR model as fallback when SASHAR is not available."""
    
    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 6,
        hidden_dim: int = 128,
        seq_length: int = 128,
    ):
        super().__init__()
        
        # CNN encoder
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size
        self.flat_size = hidden_dim * (seq_length // 8)
        
        # Classification head
        self.fc_cls = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Boundary head
        self.fc_bnd = nn.Sequential(
            nn.Linear(self.flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # CNN encoding
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Heads
        logits = self.fc_cls(x)
        boundaries = self.fc_bnd(x)
        
        return {'logits': logits, 'boundaries': boundaries}


# ============================================================================
# Fallback Visualization Functions (when sashar.visualization not available)
# ============================================================================

def _fallback_plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = 'Confusion Matrix',
    save_path: Path | None = None,
):
    """Fallback confusion matrix plot using matplotlib directly."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names[:num_classes],
           yticklabels=class_names[:num_classes],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _fallback_plot_training_curves(
    history: dict[str, list[float]],
    metrics: list[str] | None = None,
    title: str = 'Training Progress',
    save_path: Path | None = None,
):
    """Fallback training curves plot using matplotlib directly."""
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    available = [m for m in metrics if m in history or f'train_{m}' in history]
    n_plots = len(available)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available):
        ax = axes[idx]
        epochs = range(1, len(history.get('train_loss', history.get('loss', []))) + 1)
        
        train_key = f'train_{metric}' if f'train_{metric}' in history else metric
        val_key = f'val_{metric}' if f'val_{metric}' in history else None
        
        if train_key in history:
            ax.plot(epochs, history[train_key], 'b-', label=f'Train {metric}', linewidth=2)
        if val_key and val_key in history:
            ax.plot(epochs, history[val_key], 'r-', label=f'Val {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _fallback_plot_class_distribution(
    labels: np.ndarray,
    class_names: list[str] | None = None,
    title: str = 'Class Distribution',
    save_path: Path | None = None,
):
    """Fallback class distribution plot using matplotlib directly."""
    unique, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [str(i) for i in unique]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([class_names[i] if i < len(class_names) else str(i) for i in unique], counts)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# Use fallback functions if imports failed
if not HAS_SASHAR:
    plot_confusion_matrix = _fallback_plot_confusion_matrix
    plot_training_curves = _fallback_plot_training_curves
    plot_class_distribution = _fallback_plot_class_distribution


# ============================================================================
# Synthetic Data Generator
# ============================================================================

class SyntheticHARDataset:
    """Generate synthetic HAR data for testing."""
    
    ACTIVITY_PATTERNS = {
        0: {'freq': 1.0, 'amp': 0.5, 'name': 'walking'},
        1: {'freq': 2.0, 'amp': 1.0, 'name': 'running'},
        2: {'freq': 0.1, 'amp': 0.1, 'name': 'sitting'},
        3: {'freq': 0.05, 'amp': 0.05, 'name': 'standing'},
        4: {'freq': 0.02, 'amp': 0.02, 'name': 'lying'},
        5: {'freq': 0.5, 'amp': 0.8, 'name': 'stairs_up'},
        6: {'freq': 0.6, 'amp': 0.7, 'name': 'stairs_down'},
        7: {'freq': 3.0, 'amp': 1.5, 'name': 'jumping'},
    }
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 6,
        seq_length: int = 128,
        num_channels: int = 6,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_channels = num_channels
        
        self.data, self.labels, self.boundaries = self._generate_data()
    
    def _generate_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic sensor data with activity patterns."""
        data = np.zeros((self.num_samples, self.num_channels, self.seq_length), dtype=np.float32)
        labels = np.zeros(self.num_samples, dtype=np.int64)
        boundaries = np.zeros(self.num_samples, dtype=np.float32)
        
        t = np.linspace(0, 4, self.seq_length)  # 4 seconds of data
        
        for i in range(self.num_samples):
            # Random activity
            activity = np.random.randint(0, self.num_classes)
            labels[i] = activity
            
            pattern = self.ACTIVITY_PATTERNS.get(activity, {'freq': 1.0, 'amp': 0.5})
            freq = pattern['freq']
            amp = pattern['amp']
            
            # Generate sensor data for each channel
            for c in range(self.num_channels):
                # Base signal with activity-specific frequency
                phase = np.random.uniform(0, 2 * np.pi)
                signal = amp * np.sin(2 * np.pi * freq * t + phase)
                
                # Add harmonics
                signal += 0.3 * amp * np.sin(2 * np.pi * 2 * freq * t + phase * 1.5)
                
                # Add noise
                signal += np.random.normal(0, 0.1 * amp, self.seq_length)
                
                # Add drift
                signal += 0.1 * amp * np.linspace(-1, 1, self.seq_length)
                
                data[i, c] = signal
            
            # Randomly set some boundaries
            if np.random.random() < 0.2:
                boundaries[i] = 1.0
        
        return data, labels, boundaries
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders."""
        n = self.num_samples
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Convert to tensors
        data_tensor = torch.from_numpy(self.data)
        labels_tensor = torch.from_numpy(self.labels)
        boundaries_tensor = torch.from_numpy(self.boundaries)
        
        # Split
        train_data = TensorDataset(
            data_tensor[:train_end],
            labels_tensor[:train_end],
            boundaries_tensor[:train_end]
        )
        val_data = TensorDataset(
            data_tensor[train_end:val_end],
            labels_tensor[train_end:val_end],
            boundaries_tensor[train_end:val_end]
        )
        test_data = TensorDataset(
            data_tensor[val_end:],
            labels_tensor[val_end:],
            boundaries_tensor[val_end:]
        )
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


# ============================================================================
# Training Functions
# ============================================================================
# Dataset Wrapper for Dict-based datasets
# ============================================================================

class DictDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to convert dict-based dataset to tuple-based for training loop."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Convert dict to tuple: (data, label, boundary)
        data = item['data']
        label = item['label']
        boundary = item.get('boundary', torch.tensor(0.0))
        # Ensure boundary is a scalar tensor
        if isinstance(boundary, torch.Tensor) and boundary.numel() > 1:
            boundary = (boundary > 0.5).float().max().unsqueeze(0)
        return data, label, boundary


def collate_fn(batch):
    """Collate function that handles both dict and tuple-based datasets."""
    # Handle dict-based dataset (returns dict per sample)
    if isinstance(batch[0], dict):
        data = torch.stack([item['data'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        boundaries_list = []
        for item in batch:
            bnd = item.get('boundary', item.get('boundaries', torch.tensor(0.0)))
            if isinstance(bnd, torch.Tensor):
                # Handle different boundary shapes: [T], [1], or scalar
                if bnd.numel() > 1:
                    bnd = (bnd > 0.5).float().max().unsqueeze(0)
                elif bnd.dim() > 0 and bnd.shape[-1] == 1:
                    # Squeeze [1] -> scalar
                    bnd = bnd.squeeze(-1)
                boundaries_list.append(bnd)
            else:
                boundaries_list.append(torch.tensor(0.0))
        boundaries = torch.stack(boundaries_list)
    else:
        # Handle tuple-based dataset (returns tuple per sample)
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        boundaries = torch.stack([item[2] for item in batch])
    
    # Ensure boundaries is 1D [B] - squeeze any extra dimensions
    while boundaries.dim() > 1:
        boundaries = boundaries.squeeze(-1)
    
    return data, labels, boundaries

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_cls: nn.Module,
    criterion_bnd: nn.Module,
    device: str,
    boundary_weight: float = 0.5,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bnd_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels, boundaries in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        boundaries = boundaries.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(data)
        logits = outputs['logits']
        pred_boundaries = outputs['boundaries']
        
        # Classification loss
        cls_loss = criterion_cls(logits, labels)
        
        # Boundary loss (BCE) - aggregate predictions to match target shape
        # pred_boundaries: [B, T] or [B, T, 1], boundaries: [B]
        if pred_boundaries.dim() == 3:
            pred_boundaries = pred_boundaries.squeeze(-1)  # [B, T]
        # Take max probability across time as the boundary score for this sample
        pred_boundary_score = pred_boundaries.max(dim=1)[0]  # [B]
        bnd_loss = criterion_bnd(pred_boundary_score, boundaries.float())
        
        # Total loss
        loss = cls_loss + boundary_weight * bnd_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bnd_loss += bnd_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'cls_loss': total_cls_loss / n_batches,
        'bnd_loss': total_bnd_loss / n_batches,
        'accuracy': correct / total,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion_cls: nn.Module,
    criterion_bnd: nn.Module,
    device: str,
    boundary_weight: float = 0.5,
) -> dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_boundaries_pred = []
    all_boundaries_true = []
    
    with torch.no_grad():
        for data, labels, boundaries in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            boundaries = boundaries.to(device)
            
            outputs = model(data)
            logits = outputs['logits']
            pred_boundaries = outputs['boundaries'].squeeze(-1)
            
            # Aggregate predictions to match target shape
            if pred_boundaries.dim() == 2:
                pred_boundary_score = pred_boundaries.max(dim=1)[0]  # [B]
            else:
                pred_boundary_score = pred_boundaries
            
            cls_loss = criterion_cls(logits, labels)
            bnd_loss = criterion_bnd(pred_boundary_score, boundaries.float())
            loss = cls_loss + boundary_weight * bnd_loss
            
            total_loss += loss.item()
            
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_boundaries_pred.append((pred_boundary_score > 0.5).float().cpu().numpy())
            all_boundaries_true.append(boundaries.cpu().numpy())
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_boundaries_pred = np.concatenate(all_boundaries_pred)
    all_boundaries_true = np.concatenate(all_boundaries_true)
    
    # Compute metrics
    accuracy = (all_preds == all_labels).mean()
    
    # F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Boundary metrics
    boundary_precision = (all_boundaries_pred * all_boundaries_true).sum() / (all_boundaries_pred.sum() + 1e-8)
    boundary_recall = (all_boundaries_pred * all_boundaries_true).sum() / (all_boundaries_true.sum() + 1e-8)
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-8)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'f1': f1,
        'boundary_precision': boundary_precision,
        'boundary_recall': boundary_recall,
        'boundary_f1': boundary_f1,
        'predictions': all_preds,
        'labels': all_labels,
    }


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(
    dataset_name: str = 'synthetic',
    data_dir: str = 'data',
    output_dir: str = 'experiments/quick_start',
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    num_classes: int = 6,
    num_channels: int = 6,
    seq_length: int = 128,
    device: str = 'auto',
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run a complete experiment.
    
    Args:
        dataset_name: 'synthetic', 'opportunity', 'pamap2', etc.
        data_dir: Directory for data storage
        output_dir: Directory for results
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Model hidden dimension
        num_classes: Number of activity classes
        num_channels: Number of sensor channels
        seq_length: Sequence length
        device: 'auto', 'cpu', or 'cuda'
        seed: Random seed
    
    Returns:
        Dictionary with experiment results
    """
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path = output_path / 'figures'
    figures_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SAS-HAR Quick Start Experiment")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {epochs}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # Step 1: Load/Create Dataset
    # =========================================================================
    print("Step 1: Loading dataset...")
    
    if dataset_name == 'synthetic':
        print("  Using synthetic data")
        dataset = SyntheticHARDataset(
            num_samples=2000,
            num_classes=num_classes,
            seq_length=seq_length,
            num_channels=num_channels,
            seed=seed
        )
        train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=batch_size)
        class_names = [SyntheticHARDataset.ACTIVITY_PATTERNS.get(i, {'name': f'class_{i}'})['name'] 
                       for i in range(num_classes)]
    else:
        # Try to load real dataset
        print(f"  Attempting to load {dataset_name} dataset...")
        try:
            if dataset_name == 'opportunity':
                from sashar.data.opportunity import OpportunityDataset
                data_path = Path(data_dir) / 'opportunity'
                
                # Try to download
                if not (data_path / 'processed').exists():
                    print(f"  Downloading {dataset_name} dataset (this may take a few minutes)...")
                    OpportunityDataset.download_and_preprocess(
                        str(data_path),
                        sensor_config='body_worn',
                        label_type='high_level'
                    )
                
                train_ds = OpportunityDataset(root=str(data_path), split='train')
                test_ds = OpportunityDataset(root=str(data_path), split='test')
                
                # Wrap datasets to convert dict to tuple
                train_ds_wrapped = DictDatasetWrapper(train_ds)
                test_ds_wrapped = DictDatasetWrapper(test_ds)
                
                train_loader = DataLoader(train_ds_wrapped, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(test_ds_wrapped, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_ds_wrapped, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                
                num_classes = train_ds.NUM_CLASSES
                num_channels = train_ds.num_channels
                class_names = list(train_ds.ACTIVITY_LABELS.values())[:num_classes]
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")
            print("  Falling back to synthetic data...")
            dataset = SyntheticHARDataset(
                num_samples=2000,
                num_classes=num_classes,
                seq_length=seq_length,
                num_channels=num_channels,
                seed=seed
            )
            train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=batch_size)
            class_names = [SyntheticHARDataset.ACTIVITY_PATTERNS.get(i, {'name': f'class_{i}'})['name'] 
                           for i in range(num_classes)]
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Num classes: {num_classes}")
    print(f"  Num channels: {num_channels}")
    
    # =========================================================================
    # Step 2: Create Model
    # =========================================================================
    print("\nStep 2: Creating model...")
    
    if HAS_SASHAR:
        model = SASHAR(
            input_channels=num_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_transformer_layers=2,
            dropout=0.1,
        ).to(device)
        model_name = "SAS-HAR"
    else:
        model = SimpleHARModel(
            input_channels=num_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            seq_length=seq_length,
        ).to(device)
        model_name = "SimpleHARModel"
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Hidden dim: {hidden_dim}")
    
    # =========================================================================
    # Step 3: Training Setup
    # =========================================================================
    print("\nStep 3: Setting up training...")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bnd = nn.BCELoss()
    
    # =========================================================================
    # Step 4: Training Loop
    # =========================================================================
    print(f"\nStep 4: Training for {epochs} epochs...")
    print("-" * 60)
    
    history: dict[str, list[float]] = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_boundary_f1': [],
    }
    
    best_val_f1 = 0.0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            criterion_cls, criterion_bnd, device
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader,
            criterion_cls, criterion_bnd, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_boundary_f1'].append(val_metrics['boundary_f1'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Bnd F1: {val_metrics['boundary_f1']:.4f}")
    
    training_time = time.time() - start_time
    print("-" * 60)
    print(f"  Training completed in {training_time:.1f}s")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # =========================================================================
    # Step 5: Final Evaluation
    # =========================================================================
    print("\nStep 5: Final evaluation on test set...")
    
    test_metrics = validate(
        model, test_loader,
        criterion_cls, criterion_bnd, device
    )
    
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (Macro): {test_metrics['f1']:.4f}")
    print(f"  Test Boundary F1: {test_metrics['boundary_f1']:.4f}")
    
    # =========================================================================
    # Step 6: Generate Figures
    # =========================================================================
    print("\nStep 6: Generating figures...")
    
    # Confusion matrix
    try:
        fig = plot_confusion_matrix(
            test_metrics['labels'],
            test_metrics['predictions'],
            class_names=class_names[:num_classes],
            title='SAS-HAR Test Confusion Matrix',
            save_path=figures_path / 'confusion_matrix.pdf'
        )
        plt.close(fig)
        print("  Saved: confusion_matrix.pdf")
    except Exception as e:
        print(f"  Warning: Could not generate confusion matrix: {e}")
    
    # Training curves
    try:
        fig = plot_training_curves(
            history,
            metrics=['loss', 'accuracy', 'f1'],
            title='Training Progress',
            save_path=figures_path / 'training_curves.pdf'
        )
        plt.close(fig)
        print("  Saved: training_curves.pdf")
    except Exception as e:
        print(f"  Warning: Could not generate training curves: {e}")
    
    # Class distribution
    try:
        fig = plot_class_distribution(
            test_metrics['labels'],
            class_names=class_names[:num_classes],
            title='Test Set Class Distribution',
            save_path=figures_path / 'class_distribution.pdf'
        )
        plt.close(fig)
        print("  Saved: class_distribution.pdf")
    except Exception as e:
        print(f"  Warning: Could not generate class distribution: {e}")
    
    # =========================================================================
    # Step 7: Save Results
    # =========================================================================
    print("\nStep 7: Saving results...")
    
    results = {
        'experiment_info': {
            'dataset': dataset_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'num_channels': num_channels,
            'device': device,
            'seed': seed,
            'training_time_seconds': training_time,
            'timestamp': datetime.now().isoformat(),
        },
        'model_info': {
            'name': model_name,
            'num_parameters': num_params,
        },
        'training_history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'test_results': {
            'accuracy': float(test_metrics['accuracy']),
            'f1_macro': float(test_metrics['f1']),
            'boundary_precision': float(test_metrics['boundary_precision']),
            'boundary_recall': float(test_metrics['boundary_recall']),
            'boundary_f1': float(test_metrics['boundary_f1']),
        },
        'best_val_f1': float(best_val_f1),
    }
    
    # Save results JSON
    results_path = output_path / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")
    
    # Save model
    model_path = output_path / 'model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_channels': num_channels,
            'num_classes': num_classes,
            'hidden_dim': hidden_dim,
        },
        'results': results,
    }, model_path)
    print(f"  Saved: {model_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset:           {dataset_name}")
    print(f"Model Parameters:  {num_params:,}")
    print(f"Training Time:     {training_time:.1f}s")
    print(f"Best Val F1:       {best_val_f1:.4f}")
    print(f"Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (Macro):   {test_metrics['f1']:.4f}")
    print(f"Test Boundary F1:  {test_metrics['boundary_f1']:.4f}")
    print(f"Output Directory:  {output_path}")
    print(f"{'='*60}\n")
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SAS-HAR Quick Start Experiment')
    
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'opportunity'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, default='experiments/quick_start',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
        seed=args.seed,
    )
    
    return results


if __name__ == '__main__':
    main()
