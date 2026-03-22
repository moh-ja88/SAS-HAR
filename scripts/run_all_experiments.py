"""
Run experiments on all preprocessed HAR datasets.

Compares SAS-HAR against baselines on Opportunity, UCI-HAR, PAMAP2, and WISDM.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"


class SimpleDataset(TensorDataset):
    """Simple dataset wrapper for preprocessed data."""
    def __init__(self, data, labels, subjects, splits, boundaries):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.subjects = torch.from_numpy(subjects).long()
        self.splits = splits
        self.boundaries = torch.from_numpy(boundaries).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
            'subject_id': self.subjects[idx],
            'boundary': self.boundaries[idx]
        }


class DictDatasetWrapper(TensorDataset):
    """Dataset wrapper that returns (data, label, boundary) tuples."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['data'], item['label'], item.get('boundary', torch.tensor(0.0))


def collate_fn(batch):
    """Collate function for dataloader."""
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    boundaries = torch.stack([item[2] for item in batch])
    while boundaries.dim() > 1:
        boundaries = boundaries.squeeze(-1)
    return data, labels, boundaries


def load_preprocessed_dataset(name):
    """Load a preprocessed dataset."""
    data_dir = DATA_DIR / name / "processed"
    
    # Handle different naming conventions
    if name == 'opportunity':
        data_file = data_dir / "opportunity_body_worn_high_level.npz"
    else:
        data_file = data_dir / f"{name}_processed.npz"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset {name} not found at {data_file}")
    
    loaded = np.load(data_file, allow_pickle=True)
    
    data = loaded['data']
    labels = loaded['labels']
    subjects = loaded['subject_ids']
    splits = loaded['split_info']
    boundaries = loaded.get('boundaries', np.zeros(len(data), dtype=np.float32))
    
    # Remap labels to be continuous (0, 1, 2, ...)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])
    
    return data, remapped_labels, subjects, splits, boundaries, len(unique_labels)


def create_dataloaders(dataset_name, batch_size=64, split='train'):
    """Create train/test dataloaders for a dataset."""
    data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset_name)
    
    # Filter by split
    mask = splits == split
    data_split = data[mask]
    labels_split = labels[mask]
    subjects_split = subjects[mask]
    boundaries_split = boundaries[mask]
    
    # Create dataset
    dataset = SimpleDataset(data_split, labels_split, subjects_split, splits[mask], boundaries_split)
    wrapped = DictDatasetWrapper(dataset)
    
    return DataLoader(wrapped, batch_size=batch_size, shuffle=(split=='train'), 
                      collate_fn=collate_fn), num_classes


class LightweightCNN(torch.nn.Module):
    """Simple CNN for HAR classification."""
    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: [B, C, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def run_experiment(dataset_name, num_epochs=50, batch_size=64, lr=0.001):
    """Run a complete experiment on a dataset."""
    print(f"\n{'='*60}")
    print(f"Experiment: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load data directly
    print("Loading dataset...")
    data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset_name)
    
    # Split data
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    train_data = data[train_mask]
    train_labels = labels[train_mask]
    test_data = data[test_mask]
    test_labels = labels[test_mask]
    
    # Get dimensions
    in_channels = train_data.shape[1]
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Input channels: {in_channels}")
    print(f"  Number of classes: {num_classes}")
    
    # Create model
    device = 'cpu'  # Use CPU for compatibility
    model = LightweightCNN(in_channels, num_classes).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Convert to tensors
    train_data_t = torch.from_numpy(train_data).float()
    train_labels_t = torch.from_numpy(train_labels).long()
    test_data_t = torch.from_numpy(test_data).float()
    test_labels_t = torch.from_numpy(test_labels).long()
    
    # Create simple datasets
    train_dataset = torch.utils.data.TensorDataset(train_data_t, train_labels_t)
    test_dataset = torch.utils.data.TensorDataset(test_data_t, test_labels_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    best_f1 = 0.0
    
    print(f"\nTraining for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_acc = train_correct / train_total
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                _, predicted = torch.max(outputs, 1)
                
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        test_acc = test_correct / test_total
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"Best F1 Macro: {best_f1*100:.2f}%")
    
    return {
        'dataset': dataset_name,
        'accuracy': best_acc,
        'f1_macro': best_f1,
        'num_params': num_params,
        'epochs': num_epochs,
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Run experiments on all datasets."""
    print("="*60)
    print("HAR Dataset Experiments")
    print("="*60)
    
    datasets = ['opportunity', 'uci_har', 'pamap2', 'wisdm']
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_experiment(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"\nError running experiment on {dataset}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"{'Dataset':<20} {'Accuracy':<12} {'F1 Macro':<12}")
    print("-"*60)
    for dataset, results in all_results.items():
        if 'error' in results:
            print(f"{dataset:<20} {'ERROR':<12}")
        else:
            print(f"{dataset:<20} {results['accuracy']*100:.2f}% {results['f1_macro']*100:.2f}%")
    
    # Save to file
    results_file = project_root / "experiments" / "all_datasets_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("HAR Dataset Experiment Results\n")
        f.write("="*40 + "\n\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for dataset, results in all_results.items():
            f.write(f"\n{dataset.upper()}:\n")
            if 'error' in results:
                f.write(f"  Status: ERROR\n")
                f.write(f"  Error: {results['error']}\n")
            else:
                f.write(f"  Accuracy: {results['accuracy']*100:.2f}%\n")
                f.write(f"  F1 Macro: {results['f1_macro']*100:.2f}%\n")
                f.write(f"  Params: {results['num_params']}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
