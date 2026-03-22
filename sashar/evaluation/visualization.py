"""
Visualization Utilities for HAR Research

Provides publication-quality visualization tools for:
- Confusion matrices
- Training curves
- Segmentation boundary visualization
- t-SNE/UMAP embeddings
- Benchmark comparison charts
- Attention maps
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

try:
    import umap as umap_module  # type: ignore
    _has_umap = True
except ImportError:
    _has_umap = False
    umap_module = None

try:
    import seaborn as sns  # type: ignore
    _has_seaborn = True
except ImportError:
    _has_seaborn = False
    sns = None


# Publication-quality settings
PAPER_SETTINGS: dict[str, Any] = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color palette for papers
COLORS: dict[str, str] = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'error': '#F44336',
    'neutral': '#607D8B',
    'background': '#FAFAFA',
}

# Standard method colors for benchmark comparisons
METHOD_COLORS: dict[str, str] = {
    'SAS-HAR (Ours)': '#2E86AB',
    'TCBL (Ours)': '#A23B72',
    'Sliding Window': '#E8E8E8',
    'Adaptive Window': '#BDBDBD',
    'DeepConvLSTM': '#FFB74D',
    'TinyHAR': '#81C784',
    'AttendDiscriminate': '#64B5F6',
}


def setup_paper_style() -> None:
    """Apply publication-quality matplotlib settings."""
    plt.rcParams.update(PAPER_SETTINGS)
    if _has_seaborn and sns is not None:
        sns.set_palette("colorblind")


def plot_confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: list[str] | None = None,
    normalize: bool = True,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix',
    save_path: str | Path | None = None,
    show_values: bool = True,
    show_colorbar: bool = True,
) -> Figure:
    """
    Plot a publication-quality confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: Whether to normalize by row (recall)
        figsize: Figure size
        cmap: Colormap name
        title: Plot title
        save_path: Path to save figure (None = don't save)
        show_values: Whether to show values in cells
        show_colorbar: Whether to show colorbar
    
    Returns:
        Matplotlib figure
    """
    setup_paper_style()
    
    cm_matrix = confusion_matrix(y_true, y_pred)
    
    if normalize:
        row_sums = cm_matrix.sum(axis=1, keepdims=True) + 1
        cm_norm = cm_matrix.astype('float') / row_sums
        fmt = '.2f'
        vmin, vmax = 0.0, 1.0
    else:
        cm_norm = cm_matrix
        fmt = 'd'
        vmin, vmax = 0.0, float(cm_matrix.max())
    
    n_classes = cm_norm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    
    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va='bottom')
    
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    ax.set_xticklabels(class_names, rotation=45, ha='right', rotation_mode='anchor')
    
    if show_values:
        threshold = (vmax + vmin) / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_norm[i, j]
                color = 'white' if val > threshold else 'black'
                text = f'{val:{fmt}}'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    metrics: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str = 'Training Progress',
    save_path: str | Path | None = None,
    show_legend: bool = True,
    show_grid: bool = True,
) -> Figure:
    """
    Plot training curves for loss and metrics.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        metrics: List of metrics to plot (default: all available)
        figsize: Figure size
        title: Overall title
        save_path: Path to save figure
        show_legend: Whether to show legend
        show_grid: Whether to show grid
    
    Returns:
        Matplotlib figure
    """
    setup_paper_style()
    
    if metrics is None:
        all_keys = list(history.keys())
        metrics = []
        for k in all_keys:
            m = k.replace('train_', '').replace('val_', '')
            if m not in metrics:
                metrics.append(m)
        metrics = [m for m in metrics if f'train_{m}' in history or f'val_{m}' in history]
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No metrics found', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
    axes_flat = axes[0]
    
    for ax, metric in zip(axes_flat, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        epochs: list[int] = []
        if train_key in history and history[train_key]:
            epochs = list(range(1, len(history[train_key]) + 1))
            ax.plot(epochs, history[train_key], color=COLORS['primary'], label='Train', linewidth=1.5)
        
        if val_key in history and history[val_key]:
            if not epochs:
                epochs = list(range(1, len(history[val_key]) + 1))
            ax.plot(epochs, history[val_key], color=COLORS['secondary'], label='Validation', linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        if show_legend:
            ax.legend(loc='best')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_segmentation_boundaries(
    sensor_data: NDArray[np.float32],
    true_boundaries: NDArray[np.float32],
    pred_boundaries: NDArray[np.float32],
    activity_labels: NDArray[np.int64] | None = None,
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (14, 6),
    title: str = 'Segmentation Boundaries',
    save_path: str | Path | None = None,
) -> Figure:
    """
    Plot segmentation boundaries on sensor data.
    
    Args:
        sensor_data: Sensor data [C, T] or [T]
        true_boundaries: Ground truth boundaries [T]
        pred_boundaries: Predicted boundaries [T]
        activity_labels: Optional activity labels [T]
        class_names: Optional class names
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    setup_paper_style()
    
    if sensor_data.ndim == 1:
        sensor_data = sensor_data[np.newaxis, :]
    
    n_channels, T = sensor_data.shape
    
    fig, axes = plt.subplots(n_channels + 1, 1, figsize=figsize, sharex=True, squeeze=False)
    axes_flat = axes.flatten()
    
    # Plot sensor channels
    for i, ax in enumerate(axes_flat[:n_channels]):
        ax.plot(sensor_data[i], color=COLORS['primary'], alpha=0.7, linewidth=0.8)
        ax.set_ylabel(f'Ch {i}')
        ax.set_xlim(0, T)
        
        # Mark true boundaries
        true_idx = np.where(true_boundaries > 0.5)[0]
        ax.scatter(true_idx, sensor_data[i, true_idx], color=COLORS['error'], s=20, marker='|', zorder=5)
        
        # Mark predicted boundaries
        pred_idx = np.where(pred_boundaries > 0.5)[0]
        ax.scatter(pred_idx, sensor_data[i, pred_idx], color=COLORS['success'], s=20, marker='|', zorder=5, alpha=0.7)
    
    # Plot boundary scores
    ax_bound = axes_flat[-1]
    ax_bound.plot(true_boundaries, color=COLORS['error'], label='Ground Truth', linewidth=1.5)
    ax_bound.plot(pred_boundaries, color=COLORS['success'], label='Predicted', linewidth=1.5, linestyle='--')
    ax_bound.set_ylabel('Boundary Score')
    ax_bound.set_xlabel('Time Step')
    ax_bound.legend(loc='upper right')
    ax_bound.set_xlim(0, T)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_embeddings(
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    class_names: list[str] | None = None,
    method: str = 'tsne',
    figsize: tuple[float, float] = (10, 8),
    title: str = 'Feature Embeddings',
    save_path: str | Path | None = None,
    perplexity: int = 30,
    n_neighbors: int = 15,
) -> Figure:
    """
    Plot 2D embeddings using t-SNE or UMAP.
    
    Args:
        features: Feature vectors [N, D]
        labels: Class labels [N]
        class_names: Names for each class
        method: 'tsne' or 'umap'
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        perplexity: t-SNE perplexity
        n_neighbors: UMAP n_neighbors
    
    Returns:
        Matplotlib figure
    """
    setup_paper_style()
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
    else:
        class_names = [class_names[i] if i < len(class_names) else f'Class {i}' for i in unique_labels]
    
    # Compute embeddings
    if method == 'umap' and _has_umap and umap_module is not None:
        reducer = umap_module.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(features)
    else:
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, metric='cosine')
        embeddings_2d = reducer.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use colormap
    cmap_colors = plt.cm.get_cmap('tab20', n_classes)
    
    for i, (label, name) in enumerate(zip(unique_labels, class_names)):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap_colors(i)],
            label=name,
            alpha=0.6,
            s=20,
            edgecolors='white',
            linewidths=0.5
        )
    
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_benchmark_comparison(
    results: dict[str, dict[str, float]],
    metric: str = 'f1_score',
    figsize: tuple[float, float] = (10, 6),
    title: str = 'Method Comparison',
    save_path: str | Path | None = None,
    sort_by: str = 'performance',
    show_values: bool = True,
) -> Figure:
    """
    Plot benchmark comparison across methods.
    
    Args:
        results: Dict of method -> metric -> value
        metric: Which metric to plot
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        sort_by: 'performance' or 'name'
        show_values: Whether to show value labels
    
    Returns:
        Matplotlib figure
    """
    setup_paper_style()
    
    methods = list(results.keys())
    values = [results[m].get(metric, 0.0) for m in methods]
    
    # Sort
    if sort_by == 'performance':
        sorted_pairs = sorted(zip(methods, values), key=lambda x: x[1], reverse=True)
        if sorted_pairs:
            methods, values = zip(*sorted_pairs)
            methods, values = list(methods), list(values)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [METHOD_COLORS.get(m, COLORS['neutral']) for m in methods]
    
    bars = ax.barh(methods, values, color=colors, edgecolor='white')
    
    if show_values:
        max_val = max(values) if values else 1.0
        for bar, val in zip(bars, values):
            ax.text(val + 0.01 * max_val, bar.get_y() + bar.get_height() / 2,
                   f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_class_distribution(
    labels: NDArray[np.int64],
    class_names: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str = 'Class Distribution',
    save_path: str | Path | None = None,
    show_percentage: bool = True,
) -> Figure:
    """
    Plot class distribution as a bar chart.
    
    Args:
        labels: Class labels
        class_names: Names for each class
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        show_percentage: Whether to show percentage labels
    
    Returns:
        Matplotlib figure
    """
    setup_paper_style()
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
    else:
        class_names = [class_names[i] if i < len(class_names) else f'Class {i}' for i in unique_labels]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use colormap for bars
    cmap = plt.cm.get_cmap('tab20', min(n_classes, 20))
    colors = [cmap(i % 20) for i in range(n_classes)]
    
    bars = ax.bar(range(n_classes), counts, color=colors, edgecolor='white')
    
    if show_percentage:
        total = float(counts.sum())
        for bar, count in zip(bars, counts):
            percentage = count / total * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Activity Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_timing_comparison(
    timing_results: dict[str, dict[str, float]],
    metric: str = 'inference_time_ms',
    figsize: tuple[float, float] = (10, 6),
    title: str = 'Inference Time Comparison',
    save_path: str | Path | None = None,
    log_scale: bool = False,
) -> Figure:
    """
    Plot timing/performance comparison across methods.
    
    Args:
        timing_results: Dict of method -> metric -> value
        metric: Which timing metric to plot
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        log_scale: Whether to use log scale
    
    Returns:
        Matplotlib figure
    """
    setup_paper_style()
    
    methods = list(timing_results.keys())
    values = [timing_results[m].get(metric, 0.0) for m in methods]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [METHOD_COLORS.get(m, COLORS['neutral']) for i, m in enumerate(methods)]
    
    bars = ax.barh(methods, values, color=colors, edgecolor='white')
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
               f'{val:.2f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    if log_scale:
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_all_paper_figures(
    output_dir: str | Path,
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    history: dict[str, list[float]],
    features: NDArray[np.float32],
    class_names: list[str],
    benchmark_results: dict[str, dict[str, float]],
) -> None:
    """
    Generate all standard paper figures and save to output directory.
    
    Args:
        output_dir: Directory to save figures
        y_true: Ground truth labels
        y_pred: Predicted labels
        history: Training history
        features: Feature vectors for embedding
        class_names: Class names
        benchmark_results: Benchmark comparison data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title='SAS-HAR Confusion Matrix',
        save_path=output_path / 'confusion_matrix.pdf'
    )
    
    # Training curves
    plot_training_curves(
        history,
        title='SAS-HAR Training Progress',
        save_path=output_path / 'training_curves.pdf'
    )
    
    # t-SNE embeddings
    plot_embeddings(
        features, y_true, class_names,
        method='tsne',
        title='SAS-HAR Feature Embeddings (t-SNE)',
        save_path=output_path / 'embeddings_tsne.pdf'
    )
    
    # Class distribution
    plot_class_distribution(
        y_true, class_names,
        title='Class Distribution',
        save_path=output_path / 'class_distribution.pdf'
    )
    
    # Benchmark comparison
    plot_benchmark_comparison(
        benchmark_results,
        title='Method Comparison',
        save_path=output_path / 'benchmark_comparison.pdf'
    )
    
    plt.close('all')
    print(f"All figures saved to {output_path}")
