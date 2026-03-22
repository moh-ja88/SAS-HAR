"""
Baseline methods for HAR temporal segmentation.

Implements:
- Fixed Sliding Window (standard approach)
- Adaptive Sliding Window (energy-based)
- Similarity-Based Segmentation (statistical)
- Deep Similarity Segmentation (CNN-based)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
from scipy import stats
from scipy.signal import find_peaks


class FixedSlidingWindow:
    """
    Standard fixed-size sliding window baseline.
    
    Most common approach in HAR: use fixed window size with fixed stride.
    No adaptive segmentation.
    """
    
    def __init__(
        self,
        window_size: int = 128,
        stride: int = 64,
        threshold: float = 0.5
    ):
        """
        Args:
            window_size: Window size in samples
            stride: Stride between windows
            threshold: Not used for fixed window
        """
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold
        self.name = "FixedSlidingWindow"
    
    def segment(
        self,
        signal: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Create fixed windows from signal.
        
        Args:
            signal: Input signal [C, T] or [T]
        
        Returns:
            segments: List of (start, end) tuples
            boundary_scores: Boundary probability scores [T]
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        T = signal.shape[1]
        
        # Create fixed windows
        segments = []
        for i in range(0, T - self.window_size + 1, self.stride):
            segments.append((i, i + self.window_size))
        
        # Boundary scores are 0 everywhere (no boundaries detected)
        boundary_scores = np.zeros(T)
        
        return segments, boundary_scores
    
    def get_boundary_predictions(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """Get binary boundary predictions (always 0 for fixed window)."""
        T = signal.shape[-1] if signal.ndim > 1 else len(signal)
        return np.zeros(T)


class AdaptiveSlidingWindow:
    """
    Energy-based adaptive sliding window.
    
    Based on Noor et al. (2017): "Adaptive Sliding Window Segmentation
    for Physical Activity Recognition"
    
    Key idea: Adjust window size based on signal energy/variability.
    """
    
    def __init__(
        self,
        min_window: int = 64,
        max_window: int = 512,
        base_window: int = 128,
        energy_threshold: float = 0.5,
        adaptation_factor: float = 0.5
    ):
        """
        Args:
            min_window: Minimum window size
            max_window: Maximum window size
            base_window: Base window size
            energy_threshold: Threshold for energy-based adaptation
            adaptation_factor: Factor for window size adjustment
        """
        self.min_window = min_window
        self.max_window = max_window
        self.base_window = base_window
        self.energy_threshold = energy_threshold
        self.adaptation_factor = adaptation_factor
        self.name = "AdaptiveSlidingWindow"
    
    def _compute_energy(self, signal: np.ndarray) -> float:
        """Compute signal energy (variance-based)."""
        return np.var(signal)
    
    def _compute_activity_intensity(self, signal: np.ndarray) -> float:
        """Compute activity intensity metric."""
        # Use signal magnitude area (SMA)
        if signal.ndim == 1:
            return np.sum(np.abs(signal)) / len(signal)
        else:
            return np.sum(np.abs(signal)) / signal.size
    
    def segment(
        self,
        signal: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Create adaptive windows based on signal characteristics.
        
        Args:
            signal: Input signal [C, T] or [T]
        
        Returns:
            segments: List of (start, end) tuples
            boundary_scores: Boundary probability scores [T]
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        C, T = signal.shape
        segments = []
        boundary_scores = np.zeros(T)
        
        i = 0
        while i < T - self.min_window:
            # Compute local energy
            local_signal = signal[:, i:i+self.base_window]
            energy = self._compute_energy(local_signal)
            intensity = self._compute_activity_intensity(local_signal)
            
            # Adapt window size
            if intensity > self.energy_threshold:
                # High intensity - use smaller window
                window_size = max(
                    self.min_window,
                    int(self.base_window * (1 - self.adaptation_factor))
                )
            else:
                # Low intensity - use larger window
                window_size = min(
                    self.max_window,
                    int(self.base_window * (1 + self.adaptation_factor))
                )
            
            end = min(i + window_size, T)
            segments.append((i, end))
            
            # Mark boundary at window transition
            if i > 0:
                boundary_scores[i] = 1.0
            
            i += window_size // 2  # 50% overlap
        
        return segments, boundary_scores
    
    def get_boundary_predictions(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """Get binary boundary predictions."""
        _, boundary_scores = self.segment(signal)
        return (boundary_scores > 0.5).astype(int)


class SimilaritySegmentation:
    """
    Statistical similarity-based segmentation.
    
    Based on Baraka et al. (2023): "Similarity Segmentation Approach
    for Sensor-Based Activity Recognition"
    
    Key idea: Detect boundaries where similarity between consecutive
    windows drops below a threshold.
    """
    
    def __init__(
        self,
        window_size: int = 64,
        stride: int = 32,
        similarity_threshold: float = 0.7,
        feature_type: str = 'statistical',
        metric: str = 'cosine'
    ):
        """
        Args:
            window_size: Window size for comparison
            stride: Stride between windows
            similarity_threshold: Threshold for boundary detection
            feature_type: 'statistical' or 'raw'
            metric: 'cosine', 'euclidean', or 'correlation'
        """
        self.window_size = window_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.feature_type = feature_type
        self.metric = metric
        self.name = "SimilaritySegmentation"
    
    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract features from a window."""
        if self.feature_type == 'raw':
            return window.flatten()
        
        # Statistical features
        features = []
        
        if window.ndim == 1:
            window = window.reshape(1, -1)
        
        for channel in window:
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.max(channel),
                np.min(channel),
                stats.skew(channel),
                stats.kurtosis(channel),
                np.sqrt(np.mean(channel ** 2)),  # RMS
            ])
        
        return np.array(features)
    
    def _compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """Compute similarity between two feature vectors."""
        if self.metric == 'cosine':
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(features1, features2) / (norm1 * norm2)
        
        elif self.metric == 'euclidean':
            dist = np.linalg.norm(features1 - features2)
            return 1.0 / (1.0 + dist)
        
        elif self.metric == 'correlation':
            if np.std(features1) == 0 or np.std(features2) == 0:
                return 0.0
            return np.corrcoef(features1, features2)[0, 1]
        
        return 0.0
    
    def segment(
        self,
        signal: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Segment based on similarity between consecutive windows.
        
        Args:
            signal: Input signal [C, T] or [T]
        
        Returns:
            segments: List of (start, end) tuples
            boundary_scores: Boundary probability scores [T]
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        C, T = signal.shape
        boundary_scores = np.zeros(T)
        
        # Compute similarity for each position
        similarities = []
        
        for i in range(0, T - 2 * self.window_size, self.stride):
            window1 = signal[:, i:i+self.window_size]
            window2 = signal[:, i+self.stride:i+self.stride+self.window_size]
            
            features1 = self._extract_features(window1)
            features2 = self._extract_features(window2)
            
            similarity = self._compute_similarity(features1, features2)
            similarities.append((i + self.stride, similarity))
        
        # Convert similarity to boundary scores (low similarity = high boundary)
        for pos, sim in similarities:
            if pos < T:
                boundary_scores[pos] = 1.0 - sim
        
        # Detect segments based on boundaries
        boundaries = np.where(boundary_scores > (1 - self.similarity_threshold))[0]
        
        segments = []
        prev = 0
        for b in sorted(boundaries):
            if b - prev >= self.min_window:
                segments.append((prev, b))
                prev = b
        
        if prev < T:
            segments.append((prev, T))
        
        return segments, boundary_scores
    
    def get_boundary_predictions(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """Get boundary probability scores."""
        _, boundary_scores = self.segment(signal)
        return boundary_scores


class DeepSimilaritySegmentation(nn.Module):
    """
    Deep learning-based similarity segmentation.
    
    Based on Baraka et al. (2024): "Deep Similarity Segmentation Model
    for Sensor-Based Activity Recognition"
    
    Uses CNN to learn feature representations, then computes
    similarity in learned feature space.
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        hidden_dim: int = 64,
        window_size: int = 64,
        stride: int = 32,
        similarity_threshold: float = 0.7
    ):
        """
        Args:
            input_channels: Number of sensor channels
            hidden_dim: Hidden dimension for CNN
            window_size: Window size for comparison
            stride: Stride between windows
            similarity_threshold: Threshold for boundary detection
        """
        super().__init__()
        
        self.window_size = window_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.name = "DeepSimilaritySegmentation"
        
        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature representation."""
        return self.encoder(x).squeeze(-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: Input signal [B, C, T]
        
        Returns:
            boundary_scores: Boundary probabilities [B, T']
        """
        B, C, T = x.shape
        
        # Extract overlapping windows
        windows = []
        positions = []
        
        for i in range(0, T - self.window_size, self.stride):
            windows.append(x[:, :, i:i+self.window_size])
            positions.append(i + self.window_size // 2)
        
        if len(windows) == 0:
            return torch.zeros(B, T, device=x.device)
        
        windows = torch.stack(windows, dim=1)  # [B, N, C, W]
        N = windows.shape[1]
        
        # Encode all windows
        windows_flat = windows.view(B * N, C, self.window_size)
        features = self.encode(windows_flat)  # [B*N, D]
        features = features.view(B, N, -1)  # [B, N, D]
        
        # Compute similarity between consecutive windows
        boundary_scores = torch.zeros(B, T, device=x.device)
        
        for i in range(N - 1):
            feat_pair = torch.cat([features[:, i], features[:, i+1]], dim=-1)
            sim = self.similarity_net(feat_pair).squeeze(-1)  # [B]
            
            pos = positions[i+1]
            if pos < T:
                boundary_scores[:, pos] = 1.0 - sim
        
        return boundary_scores
    
    def segment(
        self,
        signal: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Segment signal using deep similarity.
        
        Args:
            signal: Input signal [C, T] or [T]
        
        Returns:
            segments: List of (start, end) tuples
            boundary_scores: Boundary probability scores [T]
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        # Convert to tensor
        x = torch.from_numpy(signal).float().unsqueeze(0)
        
        with torch.no_grad():
            boundary_scores = self.forward(x).squeeze(0).numpy()
        
        # Detect segments
        boundaries = np.where(boundary_scores > (1 - self.similarity_threshold))[0]
        
        segments = []
        prev = 0
        for b in sorted(boundaries):
            if b - prev >= self.window_size // 2:
                segments.append((prev, b))
                prev = b
        
        T = signal.shape[1]
        if prev < T:
            segments.append((prev, T))
        
        return segments, boundary_scores
    
    def get_boundary_predictions(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """Get boundary probability scores."""
        _, boundary_scores = self.segment(signal)
        return boundary_scores


# Registry of baseline methods
BASELINE_REGISTRY = {
    'fixed_window': FixedSlidingWindow,
    'adaptive_window': AdaptiveSlidingWindow,
    'similarity': SimilaritySegmentation,
    'deep_similarity': DeepSimilaritySegmentation
}


def get_baseline(name: str, **kwargs):
    """
    Get a baseline method by name.
    
    Args:
        name: Baseline name ('fixed_window', 'adaptive_window', 'similarity', 'deep_similarity')
        **kwargs: Arguments to pass to baseline constructor
    
    Returns:
        Baseline instance
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    
    return BASELINE_REGISTRY[name](**kwargs)
