"""
Training script for SAS-HAR model.

Usage:
    python train.py --dataset uci_har --config configs/standard.yaml
    python train.py --dataset wisdm --pretrain --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sashar.models.sas_har import SASHAR
from sashar.models.encoder import CNNEncoder, TransformerEncoder
from sashar.models.heads import (
    MultiTaskHead, BoundaryLoss, ConsistencyLoss
)
from sashar.models.tcbl import (
    TCBLPretrainer, ActivityAugmentation, PseudoLabelGenerator
)


class Config:
    """Configuration class for training."""
    
    def __init__(self, config_dict: Dict):
        # Model config
        self.input_channels = config_dict.get('input_channels', 6)
        self.d_model = config_dict.get('d_model', 512)
        self.n_heads = config_dict.get('n_heads', 8)
        self.n_layers = config_dict.get('n_layers', 4)
        self.d_ff = config_dict.get('d_ff', 2048)
        self.dropout = config_dict.get('dropout', 0.1)
        self.num_classes = config_dict.get('num_classes', 6)
        
        # Training config
        self.batch_size = config_dict.get('batch_size', 64)
        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.epochs = config_dict.get('epochs', 100)
        self.warmup_epochs = config_dict.get('warmup_epochs', 5)
        self.early_stopping = config_dict.get('early_stopping', 10)
        self.gradient_clip = config_dict.get('gradient_clip', 1.0)
        
        # Loss config
        self.boundary_weight = config_dict.get('boundary_weight', 1.0)
        self.classification_weight = config_dict.get('classification_weight', 1.0)
        self.consistency_weight = config_dict.get('consistency_weight', 0.2)
        
        # Data config
        self.window_size = config_dict.get('window_size', 256)
        self.stride = config_dict.get('stride', 128)
        self.augmentation = config_dict.get('augmentation', True)
        
        # Pre-training config
        self.pretrain_epochs = config_dict.get('pretrain_epochs', 50)
        self.pretrain_lr = config_dict.get('pretrain_lr', 1e-4)
        
        # System config
        self.device = config_dict.get('device', 'cuda')
        self.seed = config_dict.get('seed', 42)
        self.num_workers = config_dict.get('num_workers', 4)
        self.save_dir = config_dict.get('save_dir', 'checkpoints')
        self.log_dir = config_dict.get('log_dir', 'logs')
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class HARTrainer:
    """Trainer class for SAS-HAR model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = config.device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Loss functions
        self.boundary_loss = BoundaryLoss(pos_weight=5.0)
        self.classification_loss = nn.CrossEntropyLoss()
        self.consistency_loss = ConsistencyLoss()
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        boundary_loss_sum = 0.0
        class_loss_sum = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            if len(batch) == 3:
                data, labels, boundaries = batch
            else:
                data, labels = batch
                boundaries = None
            
            data = data.to(self.device)
            labels = labels.to(self.device)
            if boundaries is not None:
                boundaries = boundaries.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(data, return_attention=False)
            boundary_scores = output['boundary_scores']
            class_logits = output['class_logits']
            
            # Compute losses
            loss_dict = {}
            
            # Classification loss
            class_loss = self.classification_loss(class_logits, labels)
            loss_dict['class_loss'] = class_loss.item()
            class_loss_sum += class_loss.item()
            
            # Boundary loss
            if boundaries is not None:
                boundary_loss = self.boundary_loss(boundary_scores, boundaries)
                loss_dict['boundary_loss'] = boundary_loss.item()
                boundary_loss_sum += boundary_loss.item()
            else:
                boundary_loss = 0.0
            
            # Total loss
            total = (
                self.config.classification_weight * class_loss +
                self.config.boundary_weight * boundary_loss
            )
            
            # Backward pass
            total.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += total.item()
            preds = class_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total.item(),
                'class_loss': class_loss.item()
            })
        
        # Compute epoch metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_class_loss': class_loss_sum / len(self.train_loader),
            'train_boundary_loss': boundary_loss_sum / len(self.train_loader),
            'train_accuracy': accuracy,
            'train_f1': f1
        }
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_boundary_preds = []
        all_boundary_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if len(batch) == 3:
                    data, labels, boundaries = batch
                else:
                    data, labels = batch
                    boundaries = None
                
                data = data.to(self.device)
                labels = labels.to(self.device)
                if boundaries is not None:
                    boundaries = boundaries.to(self.device)
                
                # Forward pass
                output = self.model(data)
                boundary_scores = output['boundary_scores']
                class_logits = output['class_logits']
                
                # Compute loss
                class_loss = self.classification_loss(class_logits, labels)
                
                if boundaries is not None:
                    boundary_loss = self.boundary_loss(boundary_scores, boundaries)
                    total = (
                        self.config.classification_weight * class_loss +
                        self.config.boundary_weight * boundary_loss
                    )
                    
                    # Track boundary predictions
                    boundary_preds = (boundary_scores > 0.5).float()
                    all_boundary_preds.extend(boundary_preds.cpu().numpy().flatten())
                    all_boundary_labels.extend(boundaries.cpu().numpy().flatten())
                else:
                    total = class_loss
                
                total_loss += total.item()
                
                # Track predictions
                preds = class_logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_accuracy': accuracy,
            'val_f1': f1
        }
        
        # Boundary metrics if available
        if len(all_boundary_preds) > 0:
            boundary_acc = accuracy_score(all_boundary_labels, all_boundary_preds)
            boundary_f1 = f1_score(all_boundary_labels, all_boundary_preds, average='binary')
            metrics['val_boundary_acc'] = boundary_acc
            metrics['val_boundary_f1'] = boundary_f1
        
        return metrics
    
    def train(self) -> Dict[str, float]:
        """Full training loop."""
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            for k, v in train_metrics.items():
                self.writer.add_scalar(k, v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(k, v, epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"  Train - Loss: {train_metrics['train_loss']:.4f}, "
                  f"Acc: {train_metrics['train_accuracy']:.4f}, "
                  f"F1: {train_metrics['train_f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, "
                  f"Acc: {val_metrics['val_accuracy']:.4f}, "
                  f"F1: {val_metrics['val_f1']:.4f}")
            
            # Save best model
            if val_metrics['val_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['val_f1']
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_metrics['val_f1'],
                    'config': self.config.to_dict()
                }
                torch.save(
                    checkpoint,
                    os.path.join(self.config.save_dir, 'best_model.pt')
                )
                print(f"  Saved best model (F1: {val_metrics['val_f1']:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Load best model for final evaluation
        checkpoint = torch.load(os.path.join(self.config.save_dir, 'best_model.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation if test loader provided
        final_metrics = {'best_val_f1': self.best_val_f1}
        
        if self.test_loader is not None:
            test_metrics = self.validate(0)  # epoch doesn't matter for test
            final_metrics.update({f'test_{k[4:]}': v for k, v in test_metrics.items()})
            print(f"\nFinal Test Metrics:")
            print(f"  Acc: {test_metrics['val_accuracy']:.4f}, "
                  f"F1: {test_metrics['val_f1']:.4f}")
        
        return final_metrics


class Pretrainer:
    """Self-supervised pre-training using TCBL."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.device = config.device
        
        # TCBL components
        self.pretrainer = TCBLPretrainer(
            d_model=config.d_model
        ).to(config.device)
        
        self.augmenter = ActivityAugmentation()
        self.pseudo_generator = PseudoLabelGenerator()
        
        # Optimizer
        params = list(model.parameters()) + list(self.pretrainer.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=config.pretrain_lr,
            weight_decay=config.weight_decay
        )
    
    def pretrain(self, epochs: int) -> Dict[str, float]:
        """Run self-supervised pre-training."""
        print(f"Starting TCBL pre-training for {epochs} epochs")
        
        self.model.train()
        
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Pretrain Epoch {epoch}")
            
            for batch in pbar:
                if len(batch) == 3:
                    data, _, _ = batch  # Don't need labels for pre-training
                else:
                    data, _ = batch
                
                data = data.to(self.device)
                
                # Generate augmented view
                data_aug = self.augmenter(data)
                
                # Get features from model
                with torch.no_grad():
                    # Extract features using CNN encoder
                    features = self.model.cnn_encoder(data)
                    features = features.permute(0, 2, 1)  # (B, T, D)
                
                # Generate pseudo labels
                boundary_pseudo, activity_pseudo = self.pseudo_generator(features)
                
                # Forward pass through pretrainer
                features_aug = self.model.cnn_encoder(data_aug)
                features_aug = features_aug.permute(0, 2, 1)
                
                self.optimizer.zero_grad()
                loss, info = self.pretrainer(
                    features,
                    features_aug,
                    boundary_pseudo_labels=boundary_pseudo,
                    activity_pseudo_labels=activity_pseudo
                )
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch}: Pretrain Loss = {avg_loss:.4f}")
        
        return {'pretrain_loss': avg_loss}


def create_dummy_dataloader(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Create dummy dataloaders for testing."""
    class DummyDataset(Dataset):
        def __init__(self, num_samples: int = 1000):
            self.num_samples = num_samples
            self.data = torch.randn(num_samples, config.input_channels, config.window_size)
            self.labels = torch.randint(0, config.num_classes, (num_samples,))
            self.boundaries = torch.randint(0, 2, (num_samples, config.window_size)).float()
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx], self.boundaries[idx]
    
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train SAS-HAR model')
    parser.add_argument('--dataset', type=str, default='uci_har',
                        help='Dataset name')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--pretrain', action='store_true',
                        help='Run TCBL pre-training first')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config({})
    
    # Override config with command line args
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    config.device = args.device
    
    print("Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Create model
    model = SASHAR(
        input_channels=config.input_channels,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders (replace with actual data loading)
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dummy_dataloader(config)
    
    # Pre-training
    if args.pretrain:
        print("\n=== Starting Pre-training ===")
        pretrainer = Pretrainer(model, config, train_loader)
        pretrainer.pretrain(config.pretrain_epochs)
    
    # Training
    print("\n=== Starting Training ===")
    trainer = HARTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    final_metrics = trainer.train()
    
    print("\n=== Training Complete ===")
    print(f"Best Validation F1: {final_metrics['best_val_f1']:.4f}")
    
    # Save final config
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
