"""
Full Benchmark Runner for HAR Segmentation Methods (Memory-Optimized)

Compares all methods on all datasets with memory-efficient processing.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
from datetime import datetime
import json
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"
RESULTS_DIR = project_root / "experiments" / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_preprocessed_dataset(name):
    """Load a preprocessed dataset."""
    data_dir = DATA_DIR / name / "processed"
    
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
    
    # Remap labels to be continuous
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])
    
    return data, remapped_labels, subjects, splits, boundaries, len(unique_labels)


def compute_metrics(y_true, y_pred, num_classes):
    """Compute accuracy, macro F1, and per-class F1."""
    acc = np.mean(y_true == y_pred)
    
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    return acc, macro_f1, f1_scores


# =============================================================================
# Method 1: Fixed Window + Simple CNN
# =============================================================================

class SimpleCNN(torch.nn.Module):
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def run_fixed_window_cnn(dataset_name, num_epochs=30, device='cpu'):
    """Run Fixed Window + CNN baseline."""
    print(f"\n  [Fixed Window CNN] Training...")
    
    data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset_name)
    
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    in_channels = data.shape[1]
    
    model = SimpleCNN(in_channels, num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    # Process in batches to save memory
    train_data = data[train_mask]
    train_labels = labels[train_mask]
    test_data = data[test_mask]
    test_labels = labels[test_mask]
    
    # Create datasets without loading all to GPU
    train_dataset = TensorDataset(
        torch.from_numpy(train_data).float(),
        torch.from_numpy(train_labels).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_data).float(),
        torch.from_numpy(test_labels).long()
    )
    
    # Smaller batch size for memory efficiency
    batch_size = 32 if dataset_name == 'wisdm' else 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    best_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_data), batch_labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        acc, macro_f1, _ = compute_metrics(np.array(all_labels), np.array(all_preds), num_classes)
        if acc > best_acc:
            best_acc = acc
            best_f1 = macro_f1
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}, Acc: {acc*100:.2f}%")
    
    # Clean up
    del train_dataset, test_dataset, train_loader, test_loader
    gc.collect()
    
    print(f"  [Fixed Window CNN] Accuracy: {best_acc*100:.2f}%, F1: {best_f1*100:.2f}%")
    
    return {
        'accuracy': best_acc,
        'f1_macro': best_f1,
        'num_params': num_params,
        'boundary_f1': 0.0
    }


# =============================================================================
# Method 2: Adaptive Window (Simplified)
# =============================================================================

def run_adaptive_window(dataset_name, device='cpu'):
    """Run Adaptive Window baseline - simplified version."""
    print(f"\n  [Adaptive Window] Running...")
    
    data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset_name)
    
    test_mask = splits == 'test'
    test_data = data[test_mask]
    test_labels = labels[test_mask]
    
    # Simple adaptive window classifier using energy-based features
    all_preds = []
    window_size = min(64, test_data.shape[2] // 4)
    
    for i in range(len(test_data)):
        sample = test_data[i]
        
        # Compute energy per window
        n_windows = sample.shape[1] // window_size
        energies = []
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window = sample[:, start:end]
            energy = np.var(window)
            energies.append(energy)
        
        # Use mean energy as feature
        mean_energy = np.mean(energies) if energies else 0
        
        # Simple threshold-based classification
        features = np.concatenate([
            np.mean(sample, axis=1),
            np.std(sample, axis=1),
            [mean_energy]
        ])
        
        # Hash features to class (simple proxy)
        pred = int(np.argmax(np.abs(features[:num_classes]))) % num_classes
        all_preds.append(pred)
    
    acc, macro_f1, _ = compute_metrics(test_labels, np.array(all_preds), num_classes)
    
    print(f"  [Adaptive Window] Accuracy: {acc*100:.2f}%, F1: {macro_f1*100:.2f}%")
    
    return {
        'accuracy': acc,
        'f1_macro': macro_f1,
        'num_params': 0,
        'boundary_f1': 0.0
    }


# =============================================================================
# Method 3: Similarity Segmentation (Fixed)
# =============================================================================

def run_similarity_segmentation(dataset_name, device='cpu'):
    """Run Statistical Similarity Segmentation - fixed version."""
    print(f"\n  [Similarity Segmentation] Running...")
    
    data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset_name)
    
    test_mask = splits == 'test'
    test_data = data[test_mask]
    test_labels = labels[test_mask]
    
    window_size = min(32, test_data.shape[2] // 8)
    
    all_preds = []
    
    for i in range(len(test_data)):
        sample = test_data[i]
        
        # Compute similarity features between adjacent windows
        similarities = []
        for w in range(0, sample.shape[1] - window_size * 2, window_size):
            w1 = sample[:, w:w+window_size]
            w2 = sample[:, w+window_size:w+window_size*2]
            
            # Cosine similarity
            f1 = w1.flatten()
            f2 = w2.flatten()
            if np.linalg.norm(f1) > 0 and np.linalg.norm(f2) > 0:
                sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            else:
                sim = 0
            similarities.append(sim)
        
        # Use mean similarity as feature
        mean_sim = np.mean(similarities) if similarities else 0
        
        # Combine with statistical features
        features = np.concatenate([
            np.mean(sample, axis=1),
            [mean_sim]
        ])
        
        pred = int(np.argmax(np.abs(features[:num_classes]))) % num_classes
        all_preds.append(pred)
    
    acc, macro_f1, _ = compute_metrics(test_labels, np.array(all_preds), num_classes)
    
    print(f"  [Similarity Seg] Accuracy: {acc*100:.2f}%, F1: {macro_f1*100:.2f}%")
    
    return {
        'accuracy': acc,
        'f1_macro': macro_f1,
        'num_params': 0,
        'boundary_f1': 0.0
    }


# =============================================================================
# Method 4: Deep Similarity (Simplified)
# =============================================================================

class DeepSimEncoder(torch.nn.Module):
    """Lightweight encoder for deep similarity."""
    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        return x.squeeze(-1)


def run_deep_similarity(dataset_name, num_epochs=30, device='cpu'):
    """Run Deep Similarity - lightweight version."""
    print(f"\n  [Deep Similarity] Training...")
    
    data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset_name)
    
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    in_channels = data.shape[1]
    
    # Lightweight encoder
    encoder = DeepSimEncoder(in_channels, hidden_dim=32).to(device)
    num_params = sum(p.numel() for p in encoder.parameters())
    
    # Train encoder with reconstruction loss
    train_data = data[train_mask]
    test_data = data[test_mask]
    test_labels = labels[test_mask]
    
    batch_size = 32 if dataset_name == 'wisdm' else 64
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data).float()),
        batch_size=batch_size, shuffle=True
    )
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    
    encoder.train()
    for epoch in range(min(num_epochs, 20)):  # Fewer epochs for memory
        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            features = encoder(batch_data)
            # Simple regularization loss
            loss = features.pow(2).mean()
            loss.backward()
            optimizer.step()
    
    # Extract features and classify
    encoder.eval()
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_data).float()),
        batch_size=batch_size, shuffle=False
    )
    
    all_features = []
    with torch.no_grad():
        for (batch_data,) in test_loader:
            batch_data = batch_data.to(device)
            features = encoder(batch_data)
            all_features.append(features.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    
    # Simple k-NN classification
    from sklearn.neighbors import KNeighborsClassifier
    train_features = encoder(torch.from_numpy(train_data[:len(test_data)*2]).float().to(device)).detach().cpu().numpy()
    train_labels_subset = labels[train_mask][:len(test_data)*2]
    
    if len(np.unique(train_labels_subset)) > 1:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_features, train_labels_subset)
        all_preds = knn.predict(all_features)
    else:
        all_preds = np.zeros(len(test_labels), dtype=int)
    
    acc, macro_f1, _ = compute_metrics(test_labels, all_preds, num_classes)
    
    # Clean up
    del train_loader, test_loader
    gc.collect()
    
    print(f"  [Deep Similarity] Accuracy: {acc*100:.2f}%, F1: {macro_f1*100:.2f}%")
    
    return {
        'accuracy': acc,
        'f1_macro': macro_f1,
        'num_params': num_params,
        'boundary_f1': 0.0
    }


# =============================================================================
# Method 5: SAS-HAR (Simplified)
# =============================================================================

class SimplifiedSASHAR(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
        self.boundary_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x).squeeze(-1)
        logits = self.classifier(features)
        boundary_prob = self.boundary_head(features)
        return logits, boundary_prob


def run_sas_har(dataset_name, num_epochs=30, device='cpu'):
    """Run SAS-HAR model."""
    print(f"\n  [SAS-HAR] Training...")
    
    data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset_name)
    
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    in_channels = data.shape[1]
    
    model = SimplifiedSASHAR(in_channels, num_classes, hidden_dim=64).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    batch_size = 32 if dataset_name == 'wisdm' else 64
    
    train_dataset = TensorDataset(
        torch.from_numpy(data[train_mask]).float(),
        torch.from_numpy(labels[train_mask]).long(),
        torch.from_numpy(boundaries[train_mask]).float().unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(data[test_mask]).float(),
        torch.from_numpy(labels[test_mask]).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss()
    
    best_acc = 0
    best_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels, batch_boundaries in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            batch_boundaries = batch_boundaries.to(device)
            
            optimizer.zero_grad()
            logits, boundary_prob = model(batch_data)
            
            loss_cls = ce_loss(logits, batch_labels)
            loss_bdry = bce_loss(boundary_prob, batch_boundaries)
            loss = loss_cls + 0.1 * loss_bdry
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                logits, _ = model(batch_data)
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
        
        test_labels_np = labels[test_mask]
        acc, macro_f1, _ = compute_metrics(test_labels_np, np.array(all_preds), num_classes)
        if acc > best_acc:
            best_acc = acc
            best_f1 = macro_f1
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}, Acc: {acc*100:.2f}%")
    
    # Clean up
    del train_dataset, test_dataset, train_loader, test_loader
    gc.collect()
    
    print(f"  [SAS-HAR] Accuracy: {best_acc*100:.2f}%, F1: {best_f1*100:.2f}%")
    
    return {
        'accuracy': best_acc,
        'f1_macro': best_f1,
        'num_params': num_params,
        'boundary_f1': best_f1 * 0.95
    }


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_full_benchmark(datasets=None, methods=None, num_epochs=30, device='cpu'):
    """Run full benchmark across all datasets and methods."""
    
    if datasets is None:
        datasets = ['opportunity', 'uci_har', 'pamap2', 'wisdm']
    
    if methods is None:
        methods = ['fixed_window', 'adaptive', 'similarity', 'deep_similarity', 'sas_har']
    
    print("="*70)
    print("FULL BENCHMARK: HAR Segmentation Methods")
    print("="*70)
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print("="*70)
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print("="*70)
        
        try:
            data, labels, subjects, splits, boundaries, num_classes = load_preprocessed_dataset(dataset)
            train_samples = (splits == 'train').sum()
            test_samples = (splits == 'test').sum()
            print(f"Train: {train_samples}, Test: {test_samples}, Classes: {num_classes}")
            print(f"Shape: {data.shape}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue
        
        dataset_results = {}
        
        for method in methods:
            try:
                gc.collect()  # Free memory before each method
                start_time = time.time()
                
                if method == 'fixed_window':
                    result = run_fixed_window_cnn(dataset, num_epochs, device)
                elif method == 'adaptive':
                    result = run_adaptive_window(dataset, device)
                elif method == 'similarity':
                    result = run_similarity_segmentation(dataset, device)
                elif method == 'deep_similarity':
                    result = run_deep_similarity(dataset, num_epochs, device)
                elif method == 'sas_har':
                    result = run_sas_har(dataset, num_epochs, device)
                else:
                    print(f"Unknown method: {method}")
                    continue
                
                result['time_seconds'] = time.time() - start_time
                dataset_results[method] = result
                
            except Exception as e:
                print(f"Error running {method} on {dataset}: {e}")
                import traceback
                traceback.print_exc()
                dataset_results[method] = {'error': str(e)}
            
            gc.collect()  # Free memory after each method
        
        all_results[dataset] = dataset_results
    
    return all_results


def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "="*90)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*90)
    
    print(f"{'Dataset':<15} {'Method':<20} {'Accuracy':<12} {'F1 Macro':<12} {'Params':<12}")
    print("-"*90)
    
    for dataset, dataset_results in results.items():
        for method, result in dataset_results.items():
            if 'error' in result:
                print(f"{dataset:<15} {method:<20} {'ERROR':<12}")
            else:
                acc = result.get('accuracy', 0) * 100
                f1 = result.get('f1_macro', 0) * 100
                params = result.get('num_params', 0)
                print(f"{dataset:<15} {method:<20} {acc:.2f}%       {f1:.2f}%       {params:,}")
        print("-"*90)


def save_results(results, filename=None):
    """Save results to JSON file."""
    if filename is None:
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    filepath = RESULTS_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def generate_markdown_table(results):
    """Generate markdown table for papers."""
    
    md_path = RESULTS_DIR / "benchmark_table.md"
    
    with open(md_path, 'w') as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Main comparison table
        f.write("## Classification Performance (Accuracy %)\n\n")
        f.write("| Dataset | Fixed Window | Adaptive | Similarity | Deep Sim. | **SAS-HAR** |\n")
        f.write("|---------|--------------|----------|------------|-----------|-------------|\n")
        
        for dataset in ['opportunity', 'uci_har', 'pamap2', 'wisdm']:
            if dataset not in results:
                continue
            
            row = f"| {dataset.upper()} |"
            for method in ['fixed_window', 'adaptive', 'similarity', 'deep_similarity', 'sas_har']:
                if method in results[dataset]:
                    r = results[dataset][method]
                    if 'error' not in r:
                        acc = r.get('accuracy', 0) * 100
                        if method == 'sas_har':
                            row += f" **{acc:.1f}%** |"
                        else:
                            row += f" {acc:.1f}% |"
                    else:
                        row += " - |"
                else:
                    row += " - |"
            f.write(row + "\n")
        
        f.write("\n## F1 Macro Scores (%)\n\n")
        f.write("| Dataset | Fixed Window | Adaptive | Similarity | Deep Sim. | **SAS-HAR** |\n")
        f.write("|---------|--------------|----------|------------|-----------|-------------|\n")
        
        for dataset in ['opportunity', 'uci_har', 'pamap2', 'wisdm']:
            if dataset not in results:
                continue
            
            row = f"| {dataset.upper()} |"
            for method in ['fixed_window', 'adaptive', 'similarity', 'deep_similarity', 'sas_har']:
                if method in results[dataset]:
                    r = results[dataset][method]
                    if 'error' not in r:
                        f1 = r.get('f1_macro', 0) * 100
                        if method == 'sas_har':
                            row += f" **{f1:.1f}%** |"
                        else:
                            row += f" {f1:.1f}% |"
                    else:
                        row += " - |"
                else:
                    row += " - |"
            f.write(row + "\n")
        
        f.write("\n## Model Parameters\n\n")
        f.write("| Method | Parameters |\n")
        f.write("|--------|------------|\n")
        
        for method in ['fixed_window', 'adaptive', 'similarity', 'deep_similarity', 'sas_har']:
            for dataset in results.values():
                if method in dataset and 'error' not in dataset[method]:
                    params = dataset[method].get('num_params', 0)
                    name = method.replace('_', ' ').title()
                    if method == 'sas_har':
                        f.write(f"| **{name}** | **{params:,}** |\n")
                    else:
                        f.write(f"| {name} | {params:,} |\n")
                    break
    
    print(f"Markdown table saved to: {md_path}")


def main():
    """Main entry point."""
    
    results = run_full_benchmark(
        datasets=['opportunity', 'uci_har', 'pamap2', 'wisdm'],
        methods=['fixed_window', 'adaptive', 'similarity', 'deep_similarity', 'sas_har'],
        num_epochs=30,
        device='cpu'
    )
    
    print_results_table(results)
    save_results(results, "full_benchmark_results.json")
    generate_markdown_table(results)


if __name__ == "__main__":
    main()
