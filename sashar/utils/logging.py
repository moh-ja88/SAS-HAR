"""
Logging utilities for SAS-HAR experiments.

Provides structured logging with support for console, file, TensorBoard,
and Weights & Biases.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

import torch
import torch.nn as nn


# Global logger registry
_LOGGERS: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = 'sas_har',
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (None = no file logging)
        level: Overall logger level
        console: Whether to log to console
        console_level: Console handler level
        file_level: File handler level
        format_string: Custom format string
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = setup_logger('experiment', log_dir='logs/')
        >>> logger.info("Training started")
    """
    global _LOGGERS
    
    # Return existing logger if already configured
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = 'sas_har') -> logging.Logger:
    """
    Get an existing logger or create a default one.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name)


def log_config(
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Log configuration dictionary.
    
    Args:
        config: Configuration dictionary
        logger: Logger to use (default: get default logger)
        save_path: Optional path to save config as JSON
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 50)
    logger.info("Configuration")
    logger.info("=" * 50)
    
    def _log_dict(d: Dict, indent: int = 0):
        for k, v in d.items():
            if isinstance(v, dict):
                logger.info("  " * indent + f"{k}:")
                _log_dict(v, indent + 1)
            else:
                logger.info("  " * indent + f"{k}: {v}")
    
    _log_dict(config)
    logger.info("=" * 50)
    
    # Save to file
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = '',
    logger: Optional[logging.Logger] = None,
    tensorboard_writer: Optional[Any] = None,
    wandb_run: Optional[Any] = None
) -> None:
    """
    Log metrics to console, TensorBoard, and/or W&B.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current step/epoch (for TensorBoard/W&B)
        prefix: Prefix for metric names (e.g., 'train_', 'val_')
        logger: Logger instance
        tensorboard_writer: TensorBoard SummaryWriter
        wandb_run: Weights & Biases run object
    """
    if logger is None:
        logger = get_logger()
    
    # Format metrics string
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    if prefix:
        logger.info(f"[{prefix.rstrip('_')}] {metrics_str}")
    else:
        logger.info(metrics_str)
    
    # Log to TensorBoard
    if tensorboard_writer is not None and step is not None:
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            tensorboard_writer.add_scalar(full_name, value, step)
    
    # Log to W&B
    if wandb_run is not None:
        import wandb
        log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
        if step is not None:
            log_dict['step'] = step
        wandb_run.log(log_dict)


def log_model_summary(
    model: nn.Module,
    input_size: Optional[tuple] = None,
    logger: Optional[logging.Logger] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Log model summary including parameter counts.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for detailed summary
        logger: Logger instance
        save_path: Optional path to save summary
    
    Returns:
        Dictionary with model statistics
    """
    if logger is None:
        logger = get_logger()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    summary = {
        'model_name': model.__class__.__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': model_size_mb,
    }
    
    # Log summary
    logger.info("=" * 50)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info("=" * 50)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    # Detailed layer breakdown
    logger.info("\nLayer breakdown:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                logger.info(f"  {name}: {params:,} params")
    
    logger.info("=" * 50)
    
    # Save to file
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return summary


class MetricTracker:
    """
    Track and aggregate metrics over epochs.
    
    Example:
        >>> tracker = MetricTracker()
        >>> tracker.update('train_loss', 0.5, n=32)
        >>> tracker.update('train_loss', 0.4, n=32)
        >>> print(tracker.avg('train_loss'))  # 0.45
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[tuple]] = {}
    
    def update(self, name: str, value: float, n: int = 1) -> None:
        """Update a metric with a new value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((value, n))
    
    def avg(self, name: str) -> float:
        """Get weighted average of a metric."""
        if name not in self.metrics:
            return 0.0
        total = sum(v * n for v, n in self.metrics[name])
        count = sum(n for _, n in self.metrics[name])
        return total / count if count > 0 else 0.0
    
    def reset(self, name: Optional[str] = None) -> None:
        """Reset metric(s)."""
        if name is None:
            self.metrics = {}
        elif name in self.metrics:
            del self.metrics[name]
    
    def get_all_avg(self) -> Dict[str, float]:
        """Get averages of all tracked metrics."""
        return {name: self.avg(name) for name in self.metrics}


class ProgressLogger:
    """
    Context manager for logging operation progress.
    
    Example:
        >>> with ProgressLogger("Training epoch", total=100) as p:
        ...     for i in range(100):
        ...         # ... training code ...
        ...         p.update(1, loss=0.5)
    """
    
    def __init__(
        self,
        desc: str,
        total: int = 0,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10
    ):
        self.desc = desc
        self.total = total
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.current = 0
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.desc}")
        return self
    
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        self.logger.info(f"Completed: {self.desc} ({elapsed:.2f}s)")
    
    def update(self, n: int = 1, **metrics) -> None:
        """Update progress."""
        self.current += n
        if self.current % self.log_interval == 0 or self.current == self.total:
            msg = f"{self.desc}: {self.current}/{self.total}"
            if metrics:
                metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                msg += f" | {metrics_str}"
            self.logger.info(msg)


if __name__ == "__main__":
    # Test logging utilities
    print("Testing logging utilities...")
    
    # Create logger
    logger = setup_logger('test', log_dir='logs/')
    
    # Test config logging
    config = {
        'model': {'hidden_dim': 256, 'num_layers': 3},
        'training': {'lr': 1e-4, 'epochs': 100}
    }
    log_config(config, logger)
    
    # Test model summary
    model = nn.Linear(10, 5)
    summary = log_model_summary(model, logger)
    print(f"\nModel summary: {summary}")
    
    # Test metric tracker
    tracker = MetricTracker()
    tracker.update('loss', 0.5, n=32)
    tracker.update('loss', 0.4, n=32)
    print(f"\nAverage loss: {tracker.avg('loss'):.4f}")
    
    # Test progress logger
    with ProgressLogger("Test operation", total=10, logger=logger) as p:
        for i in range(10):
            p.update(1, metric=i * 0.1)
    
    print("\n✓ All logging tests passed!")
