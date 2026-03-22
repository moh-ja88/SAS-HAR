"""
Cross-dataset transfer learning experiments.

Simple version: Train on source, evaluate on target using KNN.
Uses channel and time-step normalization for compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
DATA_DIR = project_root / "data"
RESULTS_DIR = project_root / "experiments" / "analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Common configuration
NUM_CHANNELS = 3  # Use accelerometer only
TARGET_LEN = 64   # Target time series length


def normalize_time_series(data, target_len=TARGET_LEN):
    """Resample time series to target length."""
    if data.shape[-1] == target_len:
        return data
    
    # Simple interpolation using torch
    data_tensor = torch.from_numpy(data).float()
    data_tensor = data_tensor.permute(0, 2, 1)  # [N, T, C]
    
    # Interpolate
    indices = torch.linspace(0, data_tensor.shape[1] - 1, target_len)
    indices = indices.long().clamp(0, data_tensor.shape[1] - 1)
    result = data_tensor[:, indices, :]
    
    return result.permute(0, 2, 1).numpy()  # [N, C, T]


def load_dataset(name, max_samples=10000):
    """Load and normalize dataset."""
    data_dir = DATA_DIR / name / "processed"
    
    if name == 'opportunity':
        data_file = data_dir / "opportunity_body_worn_high_level.npz"
    else:
        data_file = data_dir / f"{name}_processed.npz"
    
    if not data_file.exists():
        return None
    
    d = np.load(data_file, allow_pickle=True)
    data = d['data']  # [N, C, T]
    labels = d['labels']
    splits = d['split_info']
    
    # Subsample if too large (for memory efficiency)
    if len(data) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
        labels = labels[indices]
        splits = splits[indices]
    
    # Get number of classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Remap labels to continuous range
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])
    
    # Use only first 3 channels (accelerometer)
    if data.shape[1] >= 3:
        data = data[:, :3, :]
    
    # Normalize time series length
    data = normalize_time_series(data)
    
    print(f"  {name}: {data.shape[0]} samples, {data.shape[1]} channels, {data.shape[2]} time steps, {num_classes} classes")
    
    return data, remapped_labels, splits, num_classes
    """Load and normalize dataset."""
    data_dir = DATA_DIR / name / "processed"
    
    if name == 'opportunity':
        data_file = data_dir / "opportunity_body_worn_high_level.npz"
    else:
        data_file = data_dir / f"{name}_processed.npz"
    
    if not data_file.exists():
        return None
    
    d = np.load(data_file, allow_pickle=True)
    data = d['data']  # [N, C, T]
    labels = d['labels']
    splits = d['split_info']
    
    # Get number of classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Remap labels to continuous range
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])
    
    # Use only first 3 channels (accelerometer)
    if data.shape[1] >= 3:
        data = data[:, :3, :]
    
    # Normalize time series length
    data = normalize_time_series(data)
    
    print(f"  {name}: {data.shape[0]} samples, {data.shape[1]} channels, {data.shape[2]} time steps, {num_classes} classes")
    
    return data, remapped_labels, splits, num_classes


class FeatureExtractor(nn.Module):
    """CNN feature extractor."""
    
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        return self.features(x).squeeze(-1)


def train_model(data, labels, splits, epochs=15):
    """Train feature extractor."""
    model = FeatureExtractor(in_channels=3)
    
    train_mask = splits == 'train'
    train_data = data[train_mask]
    train_labels = labels[train_mask]
    
    train_dataset = TensorDataset(
        torch.from_numpy(train_data).float(),
        torch.from_numpy(train_labels).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    
    return model


def extract_features(model, data):
    """Extract features."""
    model.eval()
    with torch.no_grad():
        features = model(torch.from_numpy(data).float()).numpy()
    return features


def run_transfer():
    """Run cross-dataset transfer."""
    print("\n[Transfer] Cross-Dataset Experiments")
    print("Using accelerometer (3 channels), normalized to 64 time steps\n")
    
    # Load all datasets
    datasets = {}
    for name in ['uci_har', 'pamap2', 'opportunity', 'wisdm']:
        result = load_dataset(name)
        if result is not None:
            datasets[name] = {
                'data': result[0],
                'labels': result[1],
                'splits': result[2],
                'num_classes': result[3]
            }
    
    print()
    results = {}
    
    # Transfer experiments
    for source in datasets:
        for target in datasets:
            if source == target:
                continue
            
            print(f"  {source.upper()} -> {target.upper()}")
            
            src = datasets[source]
            tgt = datasets[target]
            
            # Train on source
            model = train_model(src['data'], src['labels'], src['splits'])
            
            # Extract features
            src_train_mask = src['splits'] == 'train'
            src_features = extract_features(model, src['data'][src_train_mask])
            src_labels = src['labels'][src_train_mask]
            
            tgt_test_mask = tgt['splits'] == 'test'
            tgt_features = extract_features(model, tgt['data'][tgt_test_mask])
            tgt_labels = tgt['labels'][tgt_test_mask]
            
            # KNN transfer
            min_classes = min(src['num_classes'], tgt['num_classes'])
            src_labels_mapped = src_labels % min_classes
            tgt_labels_mapped = tgt_labels % min_classes
            
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(src_features, src_labels_mapped)
            predictions = knn.predict(tgt_features)
            
            acc = accuracy_score(tgt_labels_mapped, predictions) * 100
            
            results[f"{source}_to_{target}"] = {
                'accuracy': float(round(acc, 2)),
                'source_classes': int(src['num_classes']),
                'target_classes': int(tgt['num_classes']),
                'mapped_classes': int(min_classes)
            }
            
            print(f"    Accuracy: {acc:.2f}%")
    
    # Save
    output_file = RESULTS_DIR / "cross_dataset_transfer.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Dataset Transfer Learning")
    print("=" * 60)
    run_transfer()
