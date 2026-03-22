"""
Generate visualizations for papers:
- Confusion matrices for all datasets
- Training curves (loss/accuracy over epochs)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent
DATA_DIR = project_root / "data"
FIGURES_DIR = project_root / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(name):
    """Load preprocessed dataset."""
    data_dir = DATA_DIR / name / "processed"
    
    if name == 'opportunity':
        data_file = data_dir / "opportunity_body_worn_high_level.npz"
    else:
        data_file = data_dir / f"{name}_processed.npz"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset {name} not found at {data_file}")
    
    d = np.load(data_file, allow_pickle=True)
    
    data = d['data']
    labels = d['labels']
    splits = d['split_info']
    
    # Remap labels to continuous
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[l] for l in labels])
    
    return data, remapped_labels, splits, len(unique_labels)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def train_and_evaluate(dataset_name, num_epochs=30):
    """Train model and collect predictions for confusion matrix."""
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name.upper()}")
    print('='*50)
    
    data, labels, splits, num_classes = load_dataset(dataset_name)
    
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    in_channels = data.shape[1]
    
    train_data = torch.from_numpy(data[train_mask]).float()
    train_labels = torch.from_numpy(labels[train_mask]).long()
    test_data = torch.from_numpy(data[test_mask]).float()
    test_labels = torch.from_numpy(labels[test_mask]).long()
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Classes: {num_classes}")
    
    model = SimpleCNN(in_channels, num_classes)
    
    batch_size = 32 if len(train_data) > 10000 else 64
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Track training curves
    train_losses = []
    train_accs = []
    test_accs = []
    
    print("Training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += by.size(0)
            correct += predicted.eq(by).sum().item()
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                outputs = model(bx)
                _, predicted = outputs.max(1)
                total += by.size(0)
                correct += predicted.eq(by).sum().item()
        
        test_acc = correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Train={train_acc*100:.2f}%, Test={test_acc*100:.2f}%")
    
    # Get final predictions for confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for bx, by in test_loader:
                outputs = model(bx)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(by.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    final_acc = np.mean(all_preds == all_labels) * 100
    final_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    print(f"Final: Accuracy={final_acc:.2f}%, F1={final_f1:.2f}%")
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'num_classes': num_classes,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_acc': final_acc,
        'final_f1': final_f1
    }


def plot_confusion_matrix(results, dataset_name, save_path):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw counts
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp1.plot(ax=axes[0], cmap='Blues', colorbar=True)
    axes[0].set_title(f'{dataset_name.upper()} - Confusion Matrix (Counts)')
    
    # Normalized
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp2.plot(ax=axes[1], cmap='Blues', colorbar=True, values_format='.2f')
    axes[1].set_title(f'{dataset_name.upper()} - Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_training_curves(results, dataset_name, save_path):
    """Generate and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, results['train_losses'], 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{dataset_name.upper()} - Training Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, [a*100 for a in results['train_accs']], 'b-', linewidth=2, label='Train Accuracy')
    axes[1].plot(epochs, [a*100 for a in results['test_accs']], 'r-', linewidth=2, label='Test Accuracy')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{dataset_name.upper()} - Accuracy (Final: {results["final_acc"]:.2f}%)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def main():
    print("="*60)
    print("Generating Visualizations for Papers")
    print("="*60)
    
    datasets = ['opportunity', 'uci_har', 'pamap2']
    
    for dataset in datasets:
        try:
            results = train_and_evaluate(dataset, num_epochs=30)
            
            # Plot confusion matrix
            cm_path = FIGURES_DIR / f"confusion_matrix_{dataset}.png"
            plot_confusion_matrix(results, dataset, cm_path)
            
            # Plot training curves
            curves_path = FIGURES_DIR / f"training_curves_{dataset}.png"
            plot_training_curves(results, dataset, curves_path)
            
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Visualization Generation Complete")
    print("="*60)
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
