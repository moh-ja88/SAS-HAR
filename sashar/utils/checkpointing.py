"""
Checkpointing utilities for SAS-HAR experiments.

Provides model checkpoint saving, loading, and management.
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    save_dir: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    is_best: bool = False,
    save_optimizer: bool = True,
    extra_state: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save a training checkpoint.
    
    Args:
        save_dir: Directory to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics
        config: Configuration dictionary
        filename: Custom filename (default: checkpoint_epoch_{epoch}.pt)
        is_best: If True, also save as 'best_model.pt'
        save_optimizer: Whether to save optimizer state
        extra_state: Additional state to save
    
    Returns:
        Path to saved checkpoint
    
    Example:
        >>> save_checkpoint(
        ...     'checkpoints/', model, optimizer,
        ...     epoch=10, metrics={'val_f1': 0.95}, is_best=True
        ... )
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_name': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
    }
    
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_name'] = optimizer.__class__.__name__
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        checkpoint['config'] = config
    
    if extra_state is not None:
        checkpoint['extra_state'] = extra_state
    
    # Determine filename
    if filename is None:
        filename = f'checkpoint_epoch_{epoch:04d}.pt'
    
    filepath = save_dir / filename
    torch.save(checkpoint, filepath)
    
    # Save as best if needed
    if is_best:
        best_path = save_dir / 'best_model.pt'
        shutil.copy(filepath, best_path)
    
    # Save latest
    latest_path = save_dir / 'latest_model.pt'
    shutil.copy(filepath, latest_path)
    
    return filepath


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[Union[str, torch.device]] = None,
    load_optimizer: bool = True,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
        load_optimizer: Whether to load optimizer state
        strict: Whether to strictly enforce state dict matching
    
    Returns:
        Checkpoint dictionary
    
    Example:
        >>> checkpoint = load_checkpoint('checkpoints/best_model.pt', model)
        >>> print(f"Loaded epoch {checkpoint['epoch']}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    map_location = device if device is not None else None
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load model state
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def get_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    metric_name: str = 'val_f1',
    mode: str = 'max'
) -> Optional[Path]:
    """
    Find the best checkpoint based on a metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Name of metric to compare
        mode: 'max' or 'min'
    
    Returns:
        Path to best checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # First check for explicit best_model.pt
    best_path = checkpoint_dir / 'best_model.pt'
    if best_path.exists():
        return best_path
    
    # Search through all checkpoints
    best_file = None
    best_value = float('-inf') if mode == 'max' else float('inf')
    
    for ckpt_file in checkpoint_dir.glob('checkpoint_epoch_*.pt'):
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            if 'metrics' in checkpoint and metric_name in checkpoint['metrics']:
                value = checkpoint['metrics'][metric_name]
                if (mode == 'max' and value > best_value) or \
                   (mode == 'min' and value < best_value):
                    best_value = value
                    best_file = ckpt_file
        except (RuntimeError, KeyError):
            continue
    
    return best_file


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 5,
    keep_best: bool = True,
    metric_name: str = 'val_f1',
    mode: str = 'max'
) -> List[Path]:
    """
    Remove old checkpoints, keeping only recent and best ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
        metric_name: Metric name for finding best checkpoint
        mode: 'max' or 'min' for best checkpoint selection
    
    Returns:
        List of removed checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    # Find all checkpoints
    checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    # Identify checkpoints to keep
    to_keep = set()
    
    # Keep latest N
    for ckpt in checkpoints[:keep_last_n]:
        to_keep.add(ckpt)
    
    # Keep best
    if keep_best:
        best = get_best_checkpoint(checkpoint_dir, metric_name, mode)
        if best is not None:
            to_keep.add(best)
    
    # Keep special files
    for special in ['best_model.pt', 'latest_model.pt']:
        special_path = checkpoint_dir / special
        if special_path.exists():
            to_keep.add(special_path)
    
    # Remove others
    removed = []
    for ckpt in checkpoints:
        if ckpt not in to_keep:
            ckpt.unlink()
            removed.append(ckpt)
    
    return removed


def list_checkpoints(
    checkpoint_dir: Union[str, Path],
    include_metrics: bool = True
) -> List[Dict[str, Any]]:
    """
    List all checkpoints with their metadata.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        include_metrics: Whether to include metrics in output
    
    Returns:
        List of checkpoint info dictionaries
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    
    for ckpt_file in sorted(checkpoint_dir.glob('*.pt')):
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            info = {
                'path': str(ckpt_file),
                'filename': ckpt_file.name,
                'epoch': checkpoint.get('epoch', -1),
                'timestamp': checkpoint.get('timestamp', None),
            }
            
            if include_metrics and 'metrics' in checkpoint:
                info['metrics'] = checkpoint['metrics']
            
            checkpoints.append(info)
        except (RuntimeError, KeyError):
            continue
    
    return checkpoints


class CheckpointManager:
    """
    Manages checkpoint saving with automatic cleanup.
    
    Example:
        >>> manager = CheckpointManager('checkpoints/', keep_last_n=5)
        >>> for epoch in range(100):
        ...     # ... training ...
        ...     manager.save(model, optimizer, epoch=epoch, metrics=metrics)
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        keep_last_n: int = 5,
        keep_best: bool = True,
        best_metric: str = 'val_f1',
        mode: str = 'max',
        save_optimizer: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.best_metric = best_metric
        self.mode = mode
        self.save_optimizer = save_optimizer
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._best_value = float('-inf') if mode == 'max' else float('inf')
        self._best_epoch = -1
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Save a checkpoint and manage cleanup.
        
        Returns:
            Dictionary with paths to saved checkpoints
        """
        # Check if this is the best
        is_best = False
        if metrics is not None and self.best_metric in metrics:
            value = metrics[self.best_metric]
            if (self.mode == 'max' and value > self._best_value) or \
               (self.mode == 'min' and value < self._best_value):
                self._best_value = value
                self._best_epoch = epoch
                is_best = True
        
        # Save checkpoint
        filepath = save_checkpoint(
            self.save_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            config=config,
            is_best=is_best,
            save_optimizer=self.save_optimizer,
            extra_state=extra_state
        )
        
        # Cleanup old checkpoints
        removed = cleanup_old_checkpoints(
            self.save_dir,
            keep_last_n=self.keep_last_n,
            keep_best=self.keep_best,
            metric_name=self.best_metric,
            mode=self.mode
        )
        
        result = {'checkpoint': filepath}
        if is_best:
            result['best'] = self.save_dir / 'best_model.pt'
        
        return result
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.save_dir / 'best_model.pt'
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found in {self.save_dir}")
        return load_checkpoint(best_path, model, optimizer, device=device)
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """Load the latest checkpoint."""
        latest_path = self.save_dir / 'latest_model.pt'
        if not latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint found in {self.save_dir}")
        return load_checkpoint(latest_path, model, optimizer, device=device)
    
    @property
    def best_value(self) -> float:
        """Get the best metric value seen so far."""
        return self._best_value
    
    @property
    def best_epoch(self) -> int:
        """Get the epoch with the best metric."""
        return self._best_epoch


if __name__ == "__main__":
    # Test checkpointing utilities
    print("Testing checkpointing utilities...")
    
    import tempfile
    
    # Create test model and optimizer
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test save_checkpoint
        metrics = {'val_f1': 0.85, 'val_acc': 0.90}
        path = save_checkpoint(tmpdir, model, optimizer, epoch=1, metrics=metrics, is_best=True)
        print(f"✓ Saved checkpoint to: {path}")
        
        # Test load_checkpoint
        checkpoint = load_checkpoint(path, model, optimizer)
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint['metrics']}")
        
        # Test CheckpointManager
        manager = CheckpointManager(tmpdir, keep_last_n=3)
        for epoch in range(5):
            metrics = {'val_f1': 0.80 + epoch * 0.02}
            manager.save(model, optimizer, epoch=epoch, metrics=metrics)
        
        print(f"✓ Best F1: {manager.best_value:.4f} at epoch {manager.best_epoch}")
        
        # Test list_checkpoints
        ckpts = list_checkpoints(tmpdir)
        print(f"✓ Found {len(ckpts)} checkpoints")
    
    print("\n✓ All checkpointing tests passed!")
