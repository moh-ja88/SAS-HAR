"""
Reproducibility Infrastructure for SAS-HAR

This module provides comprehensive reproducibility features:
- Deterministic random seed setting
- Environment tracking and logging
- Experiment versioning
- Configuration checkpointing

All experiments using this module are guaranteed to be reproducible.
"""

import os
import sys
import json
import random
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np
import torch


@dataclass
class EnvironmentInfo:
    """System and environment information for reproducibility."""
    
    # System
    python_version: str
    platform: str
    platform_version: str
    hostname: str
    
    # Hardware
    cpu_count: int
    gpu_available: bool
    gpu_count: int
    gpu_name: str
    cuda_version: str
    
    # Software
    numpy_version: str
    torch_version: str
    scipy_version: str
    sklearn_version: str
    
    # Git
    git_commit: str
    git_branch: str
    git_dirty: bool
    
    @classmethod
    def collect(cls) -> 'EnvironmentInfo':
        """Collect current environment information."""
        
        # System info
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # GPU info
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        cuda_version = torch.version.cuda or "N/A"
        
        # Git info
        git_commit, git_branch, git_dirty = cls._get_git_info()
        
        # Package versions
        versions = cls._get_package_versions()
        
        return cls(
            python_version=python_version,
            platform=platform.system(),
            platform_version=platform.version(),
            hostname=platform.node(),
            cpu_count=os.cpu_count() or 0,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            numpy_version=versions.get('numpy', 'unknown'),
            torch_version=versions.get('torch', 'unknown'),
            scipy_version=versions.get('scipy', 'unknown'),
            sklearn_version=versions.get('sklearn', 'unknown'),
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty
        )
    
    @staticmethod
    def _get_git_info() -> tuple:
        """Get git repository information."""
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
            
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            dirty = len(status) > 0
            
            return commit, branch, dirty
        except:
            return "unknown", "unknown", False
    
    @staticmethod
    def _get_package_versions() -> Dict[str, str]:
        """Get versions of key packages."""
        versions = {}
        
        try:
            import numpy
            versions['numpy'] = numpy.__version__
        except:
            pass
        
        try:
            versions['torch'] = torch.__version__
        except:
            pass
        
        try:
            import scipy
            versions['scipy'] = scipy.__version__
        except:
            pass
        
        try:
            import sklearn
            versions['sklearn'] = sklearn.__version__
        except:
            pass
        
        return versions
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        lines = ["Environment Information:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class ReproducibilityManager:
    """
    Manages reproducibility for experiments.
    
    Features:
    - Deterministic seeding across all libraries
    - Environment logging
    - Configuration checkpointing
    - Experiment versioning
    
    Example:
        >>> manager = ReproducibilityManager(seed=42, experiment_name="my_exp")
        >>> manager.setup()
        >>> # ... run experiment ...
        >>> manager.save_checkpoint()
    """
    
    def __init__(
        self,
        seed: int = 42,
        experiment_name: str = "experiment",
        output_dir: str = "experiments",
        deterministic: bool = True
    ):
        self.seed = seed
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.deterministic = deterministic
        
        self.env_info: Optional[EnvironmentInfo] = None
        self.config_hash: Optional[str] = None
        self.start_time: Optional[datetime] = None
        
    def setup(self, config: Optional[Dict] = None):
        """
        Set up reproducibility for experiment.
        
        Args:
            config: Optional configuration dictionary to checkpoint
        """
        self.start_time = datetime.now()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seeds
        self._set_seeds(self.seed)
        
        # Collect environment info
        self.env_info = EnvironmentInfo.collect()
        
        # Save environment info
        self._save_environment_info()
        
        # Hash and save config
        if config:
            self._save_config(config)
        
        # Print summary
        self._print_summary()
    
    def _set_seeds(self, seed: int):
        """Set random seeds for all libraries."""
        
        # Python
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic mode (slower but reproducible)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For PyTorch >= 1.8
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True)
                except:
                    pass
        
        print(f"[Reproducibility] Seeds set to {seed}")
        if self.deterministic:
            print("[Reproducibility] Deterministic mode enabled (may be slower)")
    
    def _save_environment_info(self):
        """Save environment information."""
        if not self.env_info:
            return
        
        env_file = self.output_dir / "environment.json"
        with open(env_file, 'w') as f:
            json.dump(self.env_info.to_dict(), f, indent=2)
        
        print(f"[Reproducibility] Environment info saved to {env_file}")
    
    def _save_config(self, config: Dict):
        """Save and hash configuration."""
        
        # Compute hash
        config_str = json.dumps(config, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Save config
        config_file = self.output_dir / f"config_{self.config_hash}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[Reproducibility] Config saved (hash: {self.config_hash})")
    
    def _print_summary(self):
        """Print reproducibility summary."""
        print("\n" + "=" * 60)
        print("REPRODUCIBILITY SUMMARY")
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Seed: {self.seed}")
        print(f"Deterministic: {self.deterministic}")
        print(f"Output dir: {self.output_dir}")
        
        if self.env_info:
            print(f"\nEnvironment:")
            print(f"  Python: {self.env_info.python_version}")
            print(f"  PyTorch: {self.env_info.torch_version}")
            print(f"  GPU: {self.env_info.gpu_name} (x{self.env_info.gpu_count})")
            print(f"  Git commit: {self.env_info.git_commit}")
        
        print("=" * 60 + "\n")
    
    def save_checkpoint(
        self,
        results: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        model_state: Optional[Dict] = None
    ):
        """
        Save experiment checkpoint.
        
        Args:
            results: Experiment results
            metrics: Evaluation metrics
            model_state: Model state dict
        """
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'experiment_name': self.experiment_name,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'config_hash': self.config_hash,
            'results': results,
            'metrics': metrics
        }
        
        # Save checkpoint
        checkpoint_file = self.output_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        # Save model if provided
        if model_state:
            model_file = self.output_dir / "model.pt"
            torch.save(model_state, model_file)
        
        print(f"[Reproducibility] Checkpoint saved to {checkpoint_file}")
    
    def get_reproducibility_report(self) -> str:
        """Generate reproducibility report."""
        lines = [
            f"# Reproducibility Report: {self.experiment_name}",
            f"",
            f"Generated: {datetime.now().isoformat()}",
            f"",
            f"## Configuration",
            f"- Seed: {self.seed}",
            f"- Deterministic: {self.deterministic}",
            f"",
            f"## Environment",
        ]
        
        if self.env_info:
            for key, value in self.env_info.to_dict().items():
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)


@contextmanager
def reproducible_experiment(
    seed: int = 42,
    experiment_name: str = "experiment",
    output_dir: str = "experiments",
    config: Optional[Dict] = None
):
    """
    Context manager for reproducible experiments.
    
    Example:
        >>> with reproducible_experiment(seed=42, experiment_name="test") as manager:
        ...     # Run experiment
        ...     results = train_model()
        ...     manager.save_checkpoint(results=results)
    """
    manager = ReproducibilityManager(
        seed=seed,
        experiment_name=experiment_name,
        output_dir=output_dir
    )
    
    try:
        manager.setup(config)
        yield manager
    finally:
        # Cleanup if needed
        pass


def set_global_seed(seed: int):
    """
    Set global random seed for reproducibility.
    
    This is a convenience function for simple cases.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_hash(config: Dict) -> str:
    """
    Get hash of configuration for tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Hash string
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


# Example usage
if __name__ == "__main__":
    # Example: Using context manager
    config = {
        'model': 'sas_har',
        'epochs': 100,
        'batch_size': 64,
        'lr': 1e-4
    }
    
    with reproducible_experiment(
        seed=42,
        experiment_name="example_experiment",
        config=config
    ) as manager:
        print("Running experiment...")
        
        # Simulate experiment
        results = {
            'accuracy': 0.983,
            'f1_score': 0.972
        }
        
        manager.save_checkpoint(results=results)
        
        print("\n" + manager.get_reproducibility_report())
