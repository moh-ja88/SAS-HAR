"""
Experiment artifact management for SAS-HAR.

Provides utilities for saving and loading experiment artifacts including
models, configs, metrics, and reproducibility information.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn


@dataclass
class ExperimentArtifact:
    """
    Container for experiment artifacts.
    
    Attributes:
        name: Experiment name
        timestamp: Creation timestamp
        config: Experiment configuration
        model_path: Path to saved model
        metrics: Dictionary of metrics
        seed: Random seed used
        tags: List of tags for organization
        notes: Free-form notes
        git_commit: Git commit hash
        reproducibility_path: Path to reproducibility info
    """
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    git_commit: Optional[str] = None
    reproducibility_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentArtifact':
        """Create from dictionary."""
        return cls(**d)


class ArtifactManager:
    """
    Manages experiment artifacts with versioning and organization.
    
    Example:
        >>> manager = ArtifactManager('experiments/')
        >>> artifact = manager.create_experiment(
        ...     name='baseline_v1',
        ...     config={'model': {'hidden_dim': 256}},
        ...     tags=['baseline', 'v1']
        ... )
        >>> manager.save_model(artifact, model, optimizer)
        >>> manager.save_metrics(artifact, {'val_f1': 0.95})
        >>> manager.finalize(artifact)
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize artifact manager.
        
        Args:
            base_dir: Base directory for storing artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        seed: Optional[int] = None
    ) -> ExperimentArtifact:
        """
        Create a new experiment artifact container.
        
        Args:
            name: Experiment name
            config: Configuration dictionary
            tags: List of tags
            notes: Experiment notes
            seed: Random seed
        
        Returns:
            ExperimentArtifact instance
        """
        # Get git commit if available
        git_commit = self._get_git_commit()
        
        artifact = ExperimentArtifact(
            name=name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            seed=seed,
            git_commit=git_commit
        )
        
        # Create experiment directory
        exp_dir = self._get_experiment_dir(artifact)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        return artifact
    
    def save_model(
        self,
        artifact: ExperimentArtifact,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        filename: str = 'model.pt'
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            artifact: Experiment artifact
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Metrics dictionary
            filename: Checkpoint filename
        
        Returns:
            Path to saved model
        """
        exp_dir = self._get_experiment_dir(artifact)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_name': model.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        filepath = exp_dir / filename
        torch.save(checkpoint, filepath)
        
        artifact.model_path = str(filepath)
        
        return filepath
    
    def save_config(
        self,
        artifact: ExperimentArtifact,
        config: Dict[str, Any],
        filename: str = 'config.json'
    ) -> Path:
        """
        Save configuration as JSON.
        
        Args:
            artifact: Experiment artifact
            config: Configuration dictionary
            filename: Config filename
        
        Returns:
            Path to saved config
        """
        exp_dir = self._get_experiment_dir(artifact)
        filepath = exp_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        artifact.config = config
        
        return filepath
    
    def save_metrics(
        self,
        artifact: ExperimentArtifact,
        metrics: Dict[str, Any],
        filename: str = 'metrics.json'
    ) -> Path:
        """
        Save metrics as JSON.
        
        Args:
            artifact: Experiment artifact
            metrics: Metrics dictionary
            filename: Metrics filename
        
        Returns:
            Path to saved metrics
        """
        exp_dir = self._get_experiment_dir(artifact)
        filepath = exp_dir / filename
        
        # Load existing metrics if file exists
        if filepath.exists():
            with open(filepath, 'r') as f:
                existing = json.load(f)
            if isinstance(existing.get('history'), list):
                history = existing['history']
            else:
                history = []
        else:
            history = []
        
        # Add new metrics with timestamp
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        history.append(metrics_entry)
        
        # Save
        output = {
            'latest': metrics,
            'history': history
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        artifact.metrics = metrics
        
        return filepath
    
    def save_reproducibility(
        self,
        artifact: ExperimentArtifact,
        reproducibility_info: Dict[str, Any],
        filename: str = 'reproducibility.json'
    ) -> Path:
        """
        Save reproducibility information.
        
        Args:
            artifact: Experiment artifact
            reproducibility_info: Reproducibility info dictionary
            filename: Filename
        
        Returns:
            Path to saved file
        """
        exp_dir = self._get_experiment_dir(artifact)
        filepath = exp_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(reproducibility_info, f, indent=2, default=str)
        
        artifact.reproducibility_path = str(filepath)
        
        return filepath
    
    def finalize(
        self,
        artifact: ExperimentArtifact,
        filename: str = 'experiment.json'
    ) -> Path:
        """
        Finalize and save experiment metadata.
        
        Args:
            artifact: Experiment artifact
            filename: Metadata filename
        
        Returns:
            Path to saved metadata
        """
        exp_dir = self._get_experiment_dir(artifact)
        filepath = exp_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(artifact.to_dict(), f, indent=2, default=str)
        
        return filepath
    
    def load_experiment(self, exp_dir: Union[str, Path]) -> ExperimentArtifact:
        """
        Load experiment from directory.
        
        Args:
            exp_dir: Experiment directory
        
        Returns:
            ExperimentArtifact instance
        """
        exp_dir = Path(exp_dir)
        metadata_path = exp_dir / 'experiment.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            return ExperimentArtifact.from_dict(data)
        
        # Create from directory contents
        return ExperimentArtifact(
            name=exp_dir.name,
            timestamp=datetime.fromtimestamp(exp_dir.stat().st_mtime).isoformat()
        )
    
    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None
    ) -> List[ExperimentArtifact]:
        """
        List all experiments, optionally filtered.
        
        Args:
            tags: Filter by tags (experiment must have ALL tags)
            name_pattern: Filter by name pattern (regex)
        
        Returns:
            List of ExperimentArtifact instances
        """
        experiments = []
        
        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            try:
                artifact = self.load_experiment(exp_dir)
                
                # Filter by tags
                if tags is not None:
                    if not all(tag in artifact.tags for tag in tags):
                        continue
                
                # Filter by name pattern
                if name_pattern is not None:
                    import re
                    if not re.search(name_pattern, artifact.name):
                        continue
                
                experiments.append(artifact)
            except Exception:
                continue
        
        return sorted(experiments, key=lambda x: x.timestamp, reverse=True)
    
    def delete_experiment(self, artifact: ExperimentArtifact) -> None:
        """
        Delete an experiment.
        
        Args:
            artifact: Experiment to delete
        """
        exp_dir = self._get_experiment_dir(artifact)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
    
    def _get_experiment_dir(self, artifact: ExperimentArtifact) -> Path:
        """Get experiment directory path."""
        # Create safe directory name
        safe_name = "".join(
            c if c.isalnum() or c in '-_' else '_' for c in artifact.name
        )
        timestamp_short = artifact.timestamp.replace(':', '-').replace('.', '-')[:19]
        dir_name = f"{safe_name}_{timestamp_short}"
        
        return self.base_dir / dir_name
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None


def save_experiment(
    save_dir: Union[str, Path],
    name: str,
    model: nn.Module,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    seed: Optional[int] = None,
    tags: Optional[List[str]] = None,
    notes: str = "",
    reproducibility_info: Optional[Dict[str, Any]] = None
) -> ExperimentArtifact:
    """
    Convenience function to save a complete experiment.
    
    Args:
        save_dir: Directory to save experiment
        name: Experiment name
        model: Trained model
        config: Experiment configuration
        metrics: Final metrics
        optimizer: Optimizer state (optional)
        seed: Random seed used
        tags: Experiment tags
        notes: Experiment notes
        reproducibility_info: Reproducibility information
    
    Returns:
        ExperimentArtifact instance
    
    Example:
        >>> artifact = save_experiment(
        ...     'experiments/', 'baseline_v1', model,
        ...     config={'lr': 1e-4}, metrics={'val_f1': 0.95}
        ... )
    """
    manager = ArtifactManager(save_dir)
    
    artifact = manager.create_experiment(
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        seed=seed
    )
    
    manager.save_config(artifact, config)
    manager.save_model(artifact, model, optimizer, metrics=metrics)
    manager.save_metrics(artifact, metrics)
    
    if reproducibility_info is not None:
        manager.save_reproducibility(artifact, reproducibility_info)
    
    manager.finalize(artifact)
    
    return artifact


def load_experiment(
    exp_dir: Union[str, Path],
    model: Optional[nn.Module] = None,
    load_optimizer: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to load a complete experiment.
    
    Args:
        exp_dir: Experiment directory
        model: Model to load weights into
        load_optimizer: Whether to load optimizer state
    
    Returns:
        Dictionary with experiment data
    
    Example:
        >>> data = load_experiment('experiments/baseline_v1', model)
        >>> print(data['metrics'])
    """
    exp_dir = Path(exp_dir)
    
    result = {}
    
    # Load metadata
    metadata_path = exp_dir / 'experiment.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            result['artifact'] = ExperimentArtifact.from_dict(json.load(f))
    
    # Load config
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            result['config'] = json.load(f)
    
    # Load metrics
    metrics_path = exp_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            result['metrics_data'] = json.load(f)
            result['metrics'] = result['metrics_data'].get('latest', {})
    
    # Load model
    model_path = exp_dir / 'model.pt'
    if model_path.exists() and model is not None:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        result['checkpoint'] = checkpoint
    
    return result


if __name__ == "__main__":
    # Test artifact management
    print("Testing artifact management...")
    
    import tempfile
    
    # Create test model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ArtifactManager(tmpdir)
        
        # Create experiment
        artifact = manager.create_experiment(
            name='test_experiment',
            config={'model': {'hidden_dim': 256}},
            tags=['test', 'v1'],
            seed=42
        )
        print(f"✓ Created experiment: {artifact.name}")
        
        # Save model
        model_path = manager.save_model(
            artifact, model, optimizer,
            epoch=10, metrics={'val_f1': 0.95}
        )
        print(f"✓ Saved model to: {model_path}")
        
        # Save metrics
        metrics_path = manager.save_metrics(artifact, {'val_f1': 0.95, 'val_acc': 0.97})
        print(f"✓ Saved metrics to: {metrics_path}")
        
        # Finalize
        final_path = manager.finalize(artifact)
        print(f"✓ Finalized experiment: {final_path}")
        
        # List experiments
        experiments = manager.list_experiments()
        print(f"✓ Found {len(experiments)} experiments")
        
        # Test convenience function
        artifact2 = save_experiment(
            tmpdir, 'quick_save', model,
            config={'lr': 1e-4}, metrics={'test': 0.9},
            seed=42
        )
        print(f"✓ Quick saved experiment: {artifact2.name}")
    
    print("\n✓ All artifact tests passed!")
