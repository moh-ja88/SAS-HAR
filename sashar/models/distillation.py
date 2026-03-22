"""
Knowledge Distillation for SAS-HAR

This module implements knowledge distillation techniques for compressing
the SAS-HAR model into a lightweight student model suitable for edge deployment.

Key Components:
- Response-based distillation (soft targets)
- Feature-based distillation (intermediate representations)
- Attention transfer (attention map matching)
- Cross-modal distillation (for multi-sensor to single-sensor)

Reference:
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Romero et al. (2015): "FitNets: Hints for Thin Deep Nets"
- Zagoruyko & Komodakis (2017): "Paying More Attention to Attention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    
    # Temperature for soft targets
    temperature: float = 3.0
    
    # Loss weights
    alpha_ce: float = 0.3  # Weight for hard targets (cross-entropy)
    alpha_kd: float = 0.7  # Weight for soft targets (KL divergence)
    alpha_feat: float = 0.5  # Weight for feature distillation
    alpha_attn: float = 0.3  # Weight for attention transfer
    
    # Feature matching
    feature_layers: List[int] = None  # Layers to match features
    proj_hidden: int = 64  # Hidden dim for projection heads
    
    # Attention transfer
    attention_layers: List[int] = None
    
    # Training
    epochs: int = 50
    lr: float = 1e-4
    warmup_epochs: int = 5
    
    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = [1, 2, 3]
        if self.attention_layers is None:
            self.attention_layers = [1, 2, 3]


class DistillationLoss(nn.Module):
    """
    Comprehensive knowledge distillation loss combining:
    1. Response-based distillation (soft targets)
    2. Feature-based distillation (hints)
    3. Attention transfer
    
    Total Loss = α_ce * L_CE + α_kd * L_KD + α_feat * L_feat + α_attn * L_attn
    """
    
    def __init__(self, config: Optional[DistillationConfig] = None):
        super().__init__()
        self.config = config or DistillationConfig()
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: Optional[List[torch.Tensor]] = None,
        teacher_features: Optional[List[torch.Tensor]] = None,
        student_attention: Optional[List[torch.Tensor]] = None,
        teacher_attention: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total distillation loss.
        
        Args:
            student_logits: Student model outputs [B, K]
            teacher_logits: Teacher model outputs [B, K]
            labels: Ground truth labels [B]
            student_features: List of student feature maps
            teacher_features: List of teacher feature maps
            student_attention: List of student attention maps
            teacher_attention: List of teacher attention maps
            
        Returns:
            total_loss: Combined distillation loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Hard target loss (Cross-Entropy)
        loss_ce = F.cross_entropy(student_logits, labels)
        loss_dict['loss_ce'] = loss_ce.item()
        total_loss = total_loss + self.config.alpha_ce * loss_ce
        
        # 2. Soft target loss (KL Divergence)
        loss_kd = self._kd_loss(student_logits, teacher_logits)
        loss_dict['loss_kd'] = loss_kd.item()
        total_loss = total_loss + self.config.alpha_kd * loss_kd
        
        # 3. Feature distillation loss
        if student_features is not None and teacher_features is not None:
            loss_feat = self._feature_loss(student_features, teacher_features)
            loss_dict['loss_feat'] = loss_feat.item()
            total_loss = total_loss + self.config.alpha_feat * loss_feat
        
        # 4. Attention transfer loss
        if student_attention is not None and teacher_attention is not None:
            loss_attn = self._attention_loss(student_attention, teacher_attention)
            loss_dict['loss_attn'] = loss_attn.item()
            total_loss = total_loss + self.config.alpha_attn * loss_attn
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Knowledge distillation loss (KL divergence with temperature).
        
        L_KD = T^2 * KL(σ(z_T/T) || σ(z_S/T))
        """
        T = self.config.temperature
        
        # Soft targets
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        soft_student = F.log_softmax(student_logits / T, dim=1)
        
        # KL divergence (scaled by T^2 to match gradient magnitude)
        loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        
        return loss
    
    def _feature_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Feature-based distillation loss (FitNets-style).
        
        L_feat = Σ_l || f_S^l - f_T^l ||^2
        """
        total_loss = 0.0
        num_layers = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Handle dimension mismatch with projection
            if s_feat.shape != t_feat.shape:
                # Project student to match teacher
                s_feat = self._project_features(s_feat, t_feat.shape[-1])
            
            # L2 loss
            loss = F.mse_loss(s_feat, t_feat)
            total_loss = total_loss + loss
            num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def _project_features(
        self,
        features: torch.Tensor,
        target_dim: int
    ) -> torch.Tensor:
        """Project features to target dimension."""
        if features.dim() == 3:
            # [B, T, D] -> project D
            B, T, D = features.shape
            features = features.reshape(B * T, D)
            
            # Simple linear projection
            if not hasattr(self, 'feat_proj'):
                self.feat_proj = nn.Linear(D, target_dim).to(features.device)
            
            features = self.feat_proj(features)
            features = features.reshape(B, T, target_dim)
        
        return features
    
    def _attention_loss(
        self,
        student_attention: List[torch.Tensor],
        teacher_attention: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Attention transfer loss (AT).
        
        L_AT = Σ_l || A_S^l - A_T^l ||^2
        
        where A is the normalized attention map.
        """
        total_loss = 0.0
        num_layers = 0
        
        for s_attn, t_attn in zip(student_attention, teacher_attention):
            # Normalize attention maps
            s_attn_norm = F.normalize(s_attn.view(s_attn.size(0), -1), p=2, dim=1)
            t_attn_norm = F.normalize(t_attn.view(t_attn.size(0), -1), p=2, dim=1)
            
            # MSE loss on normalized attention
            loss = F.mse_loss(s_attn_norm, t_attn_norm)
            total_loss = total_loss + loss
            num_layers += 1
        
        return total_loss / max(num_layers, 1)


class FeatureProjectionHead(nn.Module):
    """
    Projection head for feature distillation.
    
    Projects student features to match teacher feature dimensions.
    """
    
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or max(student_dim, teacher_dim) // 2
        
        self.proj = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, teacher_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class KnowledgeDistillator:
    """
    Main class for knowledge distillation training.
    
    Handles the full distillation pipeline from teacher to student.
    
    Example:
        >>> teacher = SASHAR(input_channels=6, num_classes=6)
        >>> student = SASHARLite(input_channels=6, num_classes=6)
        >>> distillator = KnowledgeDistillator(teacher, student)
        >>> 
        >>> for epoch in range(epochs):
        ...     loss = distillator.train_step(batch)
        ...     if epoch % 10 == 0:
        ...         metrics = distillator.evaluate(val_loader)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
        device: str = 'cuda'
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.config = config or DistillationConfig()
        self.device = device
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Distillation loss
        self.distill_loss = DistillationLoss(self.config)
        
        # Optimizer for student
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.lr
        )
        
        # Feature projection heads (created dynamically)
        self.projection_heads = nn.ModuleDict()
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_features: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with distillation.
        
        Args:
            batch: Dictionary with 'data' and 'label'
            return_features: Whether to return intermediate features
            
        Returns:
            loss: Total distillation loss
            metrics: Dictionary of loss components
        """
        self.student.train()
        self.teacher.eval()
        
        x = batch['data'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Forward pass
        student_outputs = self.student(x, return_attention=return_features)
        with torch.no_grad():
            teacher_outputs = self.teacher(x, return_attention=return_features)
        
        # Extract features and attention if available
        student_features = student_outputs.get('temporal_features', None)
        teacher_features = teacher_outputs.get('temporal_features', None)
        student_attention = student_outputs.get('attention', None)
        teacher_attention = teacher_outputs.get('attention', None)
        
        # Convert to lists if single tensors
        if student_features is not None and not isinstance(student_features, list):
            student_features = [student_features]
            teacher_features = [teacher_features] if teacher_features is not None else None
        
        if student_attention is not None and not isinstance(student_attention, list):
            student_attention = [student_attention]
            teacher_attention = [teacher_attention] if teacher_attention is not None else None
        
        # Compute distillation loss
        loss, metrics = self.distill_loss(
            student_logits=student_outputs['logits'],
            teacher_logits=teacher_outputs['logits'],
            labels=labels,
            student_features=student_features,
            teacher_features=teacher_features,
            student_attention=student_attention,
            teacher_attention=teacher_attention
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, metrics
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate student model.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.student.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.student(x)
                loss = F.cross_entropy(outputs['logits'], labels)
                
                preds = outputs['logits'].argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(dataloader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def distill(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: Optional[int] = None,
        early_stopping: int = 10,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full distillation training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs (default from config)
            early_stopping: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs
        
        history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
        
        best_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                loss, metrics = self.train_step(batch)
                epoch_loss += metrics['total_loss']
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            
            # Logging
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
            
            # Early stopping
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    self.save_student(checkpoint_dir + "/best_student.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def save_student(self, path: str):
        """Save student model."""
        torch.save(self.student.state_dict(), path)
    
    def load_student(self, path: str):
        """Load student model."""
        self.student.load_state_dict(torch.load(path))
    
    def get_compression_ratio(self) -> float:
        """Get parameter compression ratio."""
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        return teacher_params / student_params


class CrossModalDistillator(KnowledgeDistillator):
    """
    Cross-modal knowledge distillation.
    
    Distills knowledge from multi-sensor teacher to single-sensor student.
    Useful for deploying on devices with limited sensors.
    
    Example:
        >>> # Teacher: Acc + Gyro + Mag (9 channels)
        >>> # Student: Acc only (3 channels)
        >>> distillator = CrossModalDistillator(teacher, student)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_modalities: List[str] = ['acc', 'gyro', 'mag'],
        student_modalities: List[str] = ['acc'],
        **kwargs
    ):
        super().__init__(teacher, student, **kwargs)
        
        self.teacher_modalities = teacher_modalities
        self.student_modalities = student_modalities
        
        # Channel indices
        self.channel_mapping = self._create_channel_mapping()
    
    def _create_channel_mapping(self) -> Dict[str, Tuple[int, int]]:
        """Create mapping from modality to channel indices."""
        channels_per_modality = {
            'acc': (0, 3),
            'gyro': (3, 6),
            'mag': (6, 9)
        }
        return channels_per_modality
    
    def extract_modality(
        self,
        x: torch.Tensor,
        modalities: List[str]
    ) -> torch.Tensor:
        """Extract specific modalities from multi-modal input."""
        channels = []
        for mod in modalities:
            start, end = self.channel_mapping[mod]
            channels.append(x[:, start:end, :])
        
        return torch.cat(channels, dim=1)


def create_distilled_model(
    teacher: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    student_params: int = 25000,
    epochs: int = 50,
    device: str = 'cuda'
) -> nn.Module:
    """
    Convenience function to create a distilled student model.
    
    Args:
        teacher: Pre-trained teacher model
        train_loader: Training data
        val_loader: Validation data
        student_params: Target number of parameters for student
        epochs: Distillation epochs
        device: Device to use
        
    Returns:
        Distilled student model
    """
    from sashar.models import SASHARLite
    
    # Create student
    student = SASHARLite(
        input_channels=getattr(teacher, 'input_channels', 6),
        num_classes=getattr(teacher, 'num_classes', 6)
    ).to(device)
    
    # Configure distillation
    config = DistillationConfig(
        temperature=3.0,
        alpha_ce=0.3,
        alpha_kd=0.7,
        epochs=epochs
    )
    
    # Distill
    distillator = KnowledgeDistillator(
        teacher, student, config, device
    )
    
    history = distillator.distill(
        train_loader, val_loader,
        epochs=epochs
    )
    
    return student


# Example usage
if __name__ == "__main__":
    # Create teacher and student
    from sashar.models import SASHAR, SASHARLite
    
    teacher = SASHAR(input_channels=6, num_classes=6)
    student = SASHARLite(input_channels=6, num_classes=6)
    
    # Create distillator
    config = DistillationConfig(
        temperature=3.0,
        alpha_ce=0.3,
        alpha_kd=0.7,
        alpha_feat=0.5,
        epochs=50
    )
    
    distillator = KnowledgeDistillator(teacher, student, config)
    
    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Compression ratio: {distillator.get_compression_ratio():.1f}x")
