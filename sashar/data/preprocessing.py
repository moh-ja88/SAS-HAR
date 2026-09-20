"""
Preprocessing Pipelines for HAR Data

This module provides standard preprocessing protocols for HAR sensor data,
ensuring reproducibility and consistency across experiments.

Standard Pipeline:
1. Resampling (to target sampling rate)
2. Filtering (low-pass, high-pass, band-pass)
3. Normalization (z-score, min-max)
4. Missing value handling
5. Segmentation/windowing
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from enum import Enum


class NormalizationType(Enum):
    """Normalization methods."""
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    NONE = "none"


class FilterType(Enum):
    """Filter types."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    NONE = "none"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Resampling
    target_sampling_rate: Optional[float] = None
    
    # Filtering
    filter_type: FilterType = FilterType.LOWPASS
    filter_cutoff: float = 20.0  # Hz
    filter_order: int = 4
    notch_freq: Optional[float] = None  # For notch filter
    
    # Normalization
    normalization: NormalizationType = NormalizationType.Z_SCORE
    
    # Missing values
    handle_missing: str = "interpolate"  # interpolate, forward_fill, drop
    
    # Windowing
    window_size: int = 128
    stride: int = 64
    
    # Augmentation (for training)
    augment: bool = False
    jitter_std: float = 0.01
    scale_range: Tuple[float, float] = (0.9, 1.1)


class PreprocessingPipeline:
    """
    Standard preprocessing pipeline for HAR sensor data.
    
    Example:
        >>> config = PreprocessingConfig(
        ...     target_sampling_rate=50.0,
        ...     filter_cutoff=20.0,
        ...     normalization=NormalizationType.Z_SCORE
        ... )
        >>> pipeline = PreprocessingPipeline(config)
        >>> processed = pipeline.transform(raw_data, sampling_rate=100.0)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._normalization_params = {}
        
    def fit(self, data: np.ndarray) -> 'PreprocessingPipeline':
        """
        Fit preprocessing parameters on training data.
        
        Args:
            data: Training data [N, C, T] or [C, T]
            
        Returns:
            self (for chaining)
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        
        # Fit normalization parameters
        if self.config.normalization == NormalizationType.Z_SCORE:
            self._normalization_params['mean'] = np.mean(data, axis=(0, 2), keepdims=True)
            self._normalization_params['std'] = np.std(data, axis=(0, 2), keepdims=True) + 1e-8
            
        elif self.config.normalization == NormalizationType.MIN_MAX:
            self._normalization_params['min'] = np.min(data, axis=(0, 2), keepdims=True)
            self._normalization_params['max'] = np.max(data, axis=(0, 2), keepdims=True)
            
        elif self.config.normalization == NormalizationType.ROBUST:
            self._normalization_params['median'] = np.median(data, axis=(0, 2), keepdims=True)
            q75, q25 = np.percentile(data, [75, 25], axis=(0, 2))
            self._normalization_params['iqr'] = (q75 - q25)[np.newaxis, ..., np.newaxis] + 1e-8
        
        return self
    
    def transform(self, data: np.ndarray, sampling_rate: Optional[float] = None) -> np.ndarray:
        """
        Apply preprocessing pipeline to data.
        
        Args:
            data: Input data [N, C, T] or [C, T]
            sampling_rate: Original sampling rate (for resampling)
            
        Returns:
            Preprocessed data
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 1. Handle missing values
        data = self._handle_missing(data)
        
        # 2. Resample if needed
        if self.config.target_sampling_rate is not None and sampling_rate is not None:
            data = self._resample(data, sampling_rate, self.config.target_sampling_rate)
        
        # 3. Filter
        if self.config.filter_type != FilterType.NONE:
            fs = self.config.target_sampling_rate or sampling_rate or 50.0
            data = self._filter(data, fs)
        
        # 4. Normalize
        if self.config.normalization != NormalizationType.NONE:
            data = self._normalize(data)
        
        if squeeze_output:
            data = data.squeeze(0)
        
        return data
    
    def fit_transform(self, data: np.ndarray, sampling_rate: Optional[float] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data, sampling_rate)
    
    def _handle_missing(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values (NaN, Inf)."""
        # Replace inf with nan
        data = np.where(np.isinf(data), np.nan, data)
        
        if self.config.handle_missing == "interpolate":
            # Linear interpolation for each channel
            for n in range(data.shape[0]):
                for c in range(data.shape[1]):
                    mask = ~np.isnan(data[n, c])
                    if mask.sum() > 1:  # Need at least 2 points
                        x_valid = np.where(mask)[0]
                        f = interp1d(x_valid, data[n, c, mask], 
                                   kind='linear', fill_value='extrapolate')
                        data[n, c] = f(np.arange(len(data[n, c])))
                    else:
                        data[n, c] = 0.0  # Fallback
                        
        elif self.config.handle_missing == "forward_fill":
            # Forward fill
            for n in range(data.shape[0]):
                for c in range(data.shape[1]):
                    mask = np.isnan(data[n, c])
                    if mask.any():
                        last_valid = 0
                        for t in range(len(data[n, c])):
                            if not mask[t]:
                                last_valid = data[n, c, t]
                            else:
                                data[n, c, t] = last_valid
                                
        elif self.config.handle_missing == "zero":
            data = np.nan_to_num(data, nan=0.0)
        
        return data
    
    def _resample(self, data: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
        """Resample data to target sampling rate."""
        if orig_sr == target_sr:
            return data
        
        num_samples = int(data.shape[2] * target_sr / orig_sr)
        
        resampled = np.zeros((data.shape[0], data.shape[1], num_samples))
        
        for n in range(data.shape[0]):
            for c in range(data.shape[1]):
                resampled[n, c] = signal.resample(data[n, c], num_samples)
        
        return resampled
    
    def _filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Apply filter to data."""
        nyquist = fs / 2
        
        if self.config.filter_type == FilterType.LOWPASS:
            if self.config.filter_cutoff >= nyquist:
                return data
            b, a = signal.butter(
                self.config.filter_order,
                self.config.filter_cutoff / nyquist,
                btype='low'
            )
            
        elif self.config.filter_type == FilterType.HIGHPASS:
            b, a = signal.butter(
                self.config.filter_order,
                self.config.filter_cutoff / nyquist,
                btype='high'
            )
            
        elif self.config.filter_type == FilterType.BANDPASS:
            if isinstance(self.config.filter_cutoff, (list, tuple)):
                low, high = self.config.filter_cutoff
            else:
                low = self.config.filter_cutoff * 0.5
                high = self.config.filter_cutoff
            b, a = signal.butter(
                self.config.filter_order,
                [low / nyquist, high / nyquist],
                btype='band'
            )
            
        elif self.config.filter_type == FilterType.NOTCH:
            if self.config.notch_freq is None:
                return data
            b, a = signal.iirnotch(
                self.config.notch_freq / nyquist,
                Q=30
            )
        else:
            return data
        
        # Apply filter (zero-phase)
        filtered = np.zeros_like(data)
        for n in range(data.shape[0]):
            for c in range(data.shape[1]):
                filtered[n, c] = signal.filtfilt(b, a, data[n, c])
        
        return filtered
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.config.normalization == NormalizationType.Z_SCORE:
            mean = self._normalization_params.get('mean', 0)
            std = self._normalization_params.get('std', 1)
            return (data - mean) / std
        
        elif self.config.normalization == NormalizationType.MIN_MAX:
            min_val = self._normalization_params.get('min', 0)
            max_val = self._normalization_params.get('max', 1)
            return (data - min_val) / (max_val - min_val + 1e-8)
        
        elif self.config.normalization == NormalizationType.ROBUST:
            median = self._normalization_params.get('median', 0)
            iqr = self._normalization_params.get('iqr', 1)
            return (data - median) / iqr
        
        return data


class StandardProtocols:
    """
    Standard preprocessing protocols for common HAR datasets.
    
    Each protocol follows published research papers and best practices.
    """
    
    @staticmethod
    def uci_har() -> PreprocessingConfig:
        """
        UCI-HAR dataset preprocessing protocol.
        
        Based on:
        - Anguita et al. (2013): "A Public Domain Dataset for HAR"
        - 50Hz sampling, low-pass filtering at 20Hz
        """
        return PreprocessingConfig(
            target_sampling_rate=50.0,
            filter_type=FilterType.LOWPASS,
            filter_cutoff=20.0,
            filter_order=3,
            normalization=NormalizationType.Z_SCORE,
            handle_missing="interpolate",
            window_size=128,
            stride=64
        )
    
    @staticmethod
    def wisdm() -> PreprocessingConfig:
        """
        WISDM dataset preprocessing protocol.
        
        Based on:
        - Kwapisz et al. (2011): "Activity Recognition using Cell Phone Accelerometers"
        - 20Hz sampling, simple noise filtering
        """
        return PreprocessingConfig(
            target_sampling_rate=20.0,
            filter_type=FilterType.LOWPASS,
            filter_cutoff=10.0,
            filter_order=2,
            normalization=NormalizationType.Z_SCORE,
            handle_missing="interpolate",
            window_size=200,  # 10 seconds
            stride=100
        )
    
    @staticmethod
    def pamap2() -> PreprocessingConfig:
        """
        PAMAP2 dataset preprocessing protocol.
        
        Based on:
        - Reiss & Stricker (2012): "Introducing a New Benchmarked Dataset"
        - 100Hz sampling, 10-fold cross-validation protocol
        """
        return PreprocessingConfig(
            target_sampling_rate=100.0,
            filter_type=FilterType.BANDPASS,
            filter_cutoff=(0.5, 25.0),  # Bandpass 0.5-25Hz
            filter_order=4,
            normalization=NormalizationType.Z_SCORE,
            handle_missing="interpolate",  # PAMAP2 has missing values
            window_size=512,  # ~5 seconds
            stride=256
        )
    
    @staticmethod
    def opportunity() -> PreprocessingConfig:
        """
        Opportunity dataset preprocessing protocol.
        
        Based on:
        - Chavarriaga et al. (2013): "The Opportunity Challenge"
        - 30Hz sampling, multiple sensor modalities
        """
        return PreprocessingConfig(
            target_sampling_rate=30.0,
            filter_type=FilterType.LOWPASS,
            filter_cutoff=15.0,
            filter_order=4,
            normalization=NormalizationType.Z_SCORE,
            handle_missing="forward_fill",
            window_size=128,
            stride=64
        )
    
    @staticmethod
    def realworld() -> PreprocessingConfig:
        """
        RealWorld HAR dataset preprocessing protocol.
        
        Based on:
        - Sztyler & Stuckenschmidt (2016): "On-body localization of wearable devices"
        - 50Hz sampling, 7 body positions
        """
        return PreprocessingConfig(
            target_sampling_rate=50.0,
            filter_type=FilterType.LOWPASS,
            filter_cutoff=20.0,
            filter_order=4,
            normalization=NormalizationType.Z_SCORE,
            handle_missing="interpolate",
            window_size=128,
            stride=64
        )
    
    @staticmethod
    def get_protocol(dataset_name: str) -> PreprocessingConfig:
        """Get standard protocol for dataset."""
        protocols = {
            'uci_har': StandardProtocols.uci_har,
            'wisdm': StandardProtocols.wisdm,
            'pamap2': StandardProtocols.pamap2,
            'opportunity': StandardProtocols.opportunity,
            'realworld': StandardProtocols.realworld,
        }
        
        if dataset_name.lower() not in protocols:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(protocols.keys())}")
        
        return protocols[dataset_name.lower()]()


def create_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from continuous data.
    
    Args:
        data: Continuous data [C, T] or [N, C, T]
        labels: Labels for each time step [T] or [N, T]
        window_size: Window size in samples
        stride: Stride between windows
        
    Returns:
        windows: [M, C, window_size]
        window_labels: [M]
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        labels = labels[np.newaxis, ...]
    
    all_windows = []
    all_labels = []
    
    for n in range(data.shape[0]):
        T = data.shape[2]
        
        for i in range(0, T - window_size + 1, stride):
            window = data[n, :, i:i + window_size]
            
            # Majority vote for window label
            window_label_counts = np.bincount(
                labels[n, i:i + window_size].astype(int)
            )
            window_label = np.argmax(window_label_counts)
            
            all_windows.append(window)
            all_labels.append(window_label)
    
    return np.array(all_windows), np.array(all_labels)


def compute_activity_boundaries(
    labels: np.ndarray,
    min_gap: int = 10
) -> np.ndarray:
    """
    Compute boundary positions from activity labels.
    
    Args:
        labels: Activity labels [T]
        min_gap: Minimum gap between boundaries
        
    Returns:
        Binary boundary array [T]
    """
    boundaries = np.zeros_like(labels, dtype=float)
    
    # Find transitions
    transitions = np.where(labels[1:] != labels[:-1])[0] + 1
    boundaries[transitions] = 1.0
    
    # Filter out boundaries too close together
    if min_gap > 0 and len(transitions) > 1:
        filtered = [transitions[0]]
        for t in transitions[1:]:
            if t - filtered[-1] >= min_gap:
                filtered.append(t)
        boundaries[:] = 0.0
        boundaries[filtered] = 1.0
    
    return boundaries


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    data = np.random.randn(6, 1000)  # 6 channels, 1000 samples
    labels = np.zeros(1000, dtype=int)
    labels[300:600] = 1
    labels[600:] = 2
    
    # Apply standard UCI-HAR protocol
    config = StandardProtocols.uci_har()
    pipeline = PreprocessingPipeline(config)
    
    processed = pipeline.fit_transform(data, sampling_rate=100.0)
    print(f"Processed shape: {processed.shape}")
    
    # Create windows
    windows, window_labels = create_windows(
        processed, labels, 
        window_size=config.window_size,
        stride=config.stride
    )
    print(f"Windows shape: {windows.shape}")
    print(f"Window labels shape: {window_labels.shape}")
    
    # Compute boundaries
    boundaries = compute_activity_boundaries(labels)
    print(f"Number of boundaries: {boundaries.sum()}")
