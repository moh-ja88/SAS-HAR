"""
Reproducibility utilities for SAS-HAR experiments.

Ensures reproducible results across runs by controlling all sources
of randomness.
"""

import os
import random
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import numpy as np
import torch


# Global seed tracker
_CURRENT_SEED: Optional[int] = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for all random number generators.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic mode (slower but reproducible)
    
    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations are reproducible
    """
    global _CURRENT_SEED
    _CURRENT_SEED = seed
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Set environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed() -> Optional[int]:
    """
    Get the current random seed.
    
    Returns:
        Current seed value, or None if not set
    """
    return _CURRENT_SEED


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get comprehensive reproducibility information.
    
    Returns:
        Dictionary containing all reproducibility-relevant information
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'seed': _CURRENT_SEED,
        'python_version': _get_python_version(),
        'packages': _get_package_versions(),
        'cuda': _get_cuda_info(),
        'hardware': _get_hardware_info(),
        'environment': _get_environment_vars(),
        'git': _get_git_info(),
    }
    return info


def save_reproducibility_info(
    save_dir: Union[str, Path],
    filename: str = 'reproducibility_info.json',
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save reproducibility information to a JSON file.
    
    Args:
        save_dir: Directory to save the file
        filename: Name of the file
        config: Optional config dictionary to include
    
    Returns:
        Path to the saved file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    info = get_reproducibility_info()
    
    if config is not None:
        info['config'] = config
        info['config_hash'] = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
    
    filepath = save_dir / filename
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    return filepath


def load_reproducibility_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load reproducibility information from a JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary containing reproducibility information
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        return json.load(f)


def verify_reproducibility(
    filepath: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    strict: bool = False
) -> Dict[str, bool]:
    """
    Verify current environment against saved reproducibility info.
    
    Args:
        filepath: Path to saved reproducibility info
        config: Current config to verify against
        strict: If True, all checks must pass; if False, warnings only
    
    Returns:
        Dictionary of check results
    """
    saved = load_reproducibility_info(filepath)
    current = get_reproducibility_info()
    
    results = {}
    
    # Check seed
    results['seed_match'] = saved.get('seed') == current.get('seed')
    
    # Check Python version
    results['python_version_match'] = (
        saved.get('python_version') == current.get('python_version')
    )
    
    # Check critical packages
    saved_pkgs = saved.get('packages', {})
    current_pkgs = current.get('packages', {})
    critical_packages = ['torch', 'numpy']
    
    pkg_matches = {}
    for pkg in critical_packages:
        pkg_matches[pkg] = saved_pkgs.get(pkg) == current_pkgs.get(pkg)
    results['package_versions_match'] = all(pkg_matches.values())
    results['package_details'] = pkg_matches
    
    # Check CUDA
    saved_cuda = saved.get('cuda', {})
    current_cuda = current.get('cuda', {})
    results['cuda_match'] = (
        saved_cuda.get('version') == current_cuda.get('version') and
        saved_cuda.get('device_name') == current_cuda.get('device_name')
    )
    
    # Check config hash if provided
    if config is not None and 'config_hash' in saved:
        current_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        results['config_match'] = saved['config_hash'] == current_hash
    
    if strict:
        failures = [k for k, v in results.items() if isinstance(v, bool) and not v]
        if failures:
            raise RuntimeError(
                f"Reproducibility verification failed: {failures}"
            )
    
    return results


# Helper functions

def _get_python_version() -> str:
    """Get Python version string."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    
    package_names = [
        'torch', 'numpy', 'scipy', 'scikit-learn', 'pandas',
        'tqdm', 'tensorboard', 'omegaconf', 'hydra-core'
    ]
    
    for pkg in package_names:
        try:
            # Handle different package names
            import_name = pkg.replace('-', '_')
            if pkg == 'scikit-learn':
                import_name = 'sklearn'
            elif pkg == 'hydra-core':
                import_name = 'hydra'
            
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            packages[pkg] = version
        except ImportError:
            pass
    
    return packages


def _get_cuda_info() -> Dict[str, Any]:
    """Get CUDA information."""
    info = {
        'available': torch.cuda.is_available(),
        'version': None,
        'device_count': 0,
        'device_name': None,
        'capability': None,
    }
    
    if torch.cuda.is_available():
        info['version'] = torch.version.cuda
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['capability'] = torch.cuda.get_device_capability(0)
    
    return info


def _get_hardware_info() -> Dict[str, str]:
    """Get hardware information."""
    import platform
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
    }
    
    # Try to get CPU info
    try:
        import cpuinfo
        info['cpu'] = cpuinfo.get_cpu_info()['brand_raw']
    except ImportError:
        info['cpu'] = platform.processor()
    
    # Get memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory_gb'] = f"{mem.total / (1024**3):.1f}"
    except ImportError:
        pass
    
    return info


def _get_environment_vars() -> Dict[str, str]:
    """Get relevant environment variables."""
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'PYTHONHASHSEED',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'TORCH_HOME',
    ]
    
    return {var: os.environ.get(var, '') for var in env_vars}


def _get_git_info() -> Optional[Dict[str, str]]:
    """Get git repository information."""
    try:
        import subprocess
        
        # Check if we're in a git repo
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        info = {}
        
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['commit_hash'] = result.stdout.strip()
        
        # Get branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
        
        # Get status (dirty or clean)
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['dirty'] = bool(result.stdout.strip())
        
        # Get remote URL
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['remote'] = result.stdout.strip()
        
        return info
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


if __name__ == "__main__":
    # Test reproducibility utilities
    print("Testing reproducibility utilities...")
    
    # Set seed
    set_seed(42)
    print(f"Seed set to: {get_seed()}")
    
    # Test random numbers are reproducible
    import random
    vals1 = [random.random() for _ in range(5)]
    
    set_seed(42)
    vals2 = [random.random() for _ in range(5)]
    
    assert vals1 == vals2, "Random values should be identical with same seed"
    print("✓ Random reproducibility verified")
    
    # Get reproducibility info
    info = get_reproducibility_info()
    print(f"\nReproducibility info keys: {list(info.keys())}")
    print(f"Python version: {info['python_version']}")
    print(f"CUDA available: {info['cuda']['available']}")
    if info['cuda']['available']:
        print(f"CUDA device: {info['cuda']['device_name']}")
    
    print("\n✓ All reproducibility tests passed!")
