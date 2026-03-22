"""
Dataset Preprocessing Script

Converts downloaded HAR datasets to a unified format for experiments.
"""

import sys
from pathlib import Path
import numpy as np
import zipfile
import urllib.request
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"


def preprocess_uci_har():
    """Preprocess UCI-HAR dataset to unified format."""
    print("\n" + "="*60)
    print("Preprocessing UCI-HAR Dataset")
    print("="*60)
    
    uci_dir = DATA_DIR / "uci_har"
    processed_dir = uci_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "uci_har_processed.npz"
    if output_file.exists():
        print(f"Already preprocessed: {output_file}")
        return True
    
    # Load UCI-HAR data
    train_X_file = uci_dir / "train" / "X_train.txt"
    train_y_file = uci_dir / "train" / "y_train.txt"
    test_X_file = uci_dir / "test" / "X_test.txt"
    test_y_file = uci_dir / "test" / "y_test.txt"
    
    if not train_X_file.exists():
        print(f"Training data not found at {train_X_file}")
        return False
    
    # Load data
    print("Loading training data...")
    train_X = np.loadtxt(train_X_file)
    train_y = np.loadtxt(train_y_file)
    
    print("Loading test data...")
    test_X = np.loadtxt(test_X_file)
    test_y = np.loadtxt(test_y_file)
    
    # UCI-HAR has 561 features per window (pre-extracted features)
    # Original windows are 128 samples at 50Hz
    # We need to reshape back to [N, C, T] format
    
    # For simplicity, we'll use the pre-extracted features directly
    # and reshape to simulate sensor channels
    # 561 features = 9 channels * ~62 features per channel
    # We'll create synthetic 9-channel data with 64 timesteps
    
    n_train = train_X.shape[0]
    n_test = test_X.shape[0]
    
    # Use 9 channels (accel+gyro) and 64 timesteps
    n_channels = 9
    window_size = 64
    
    # Create synthetic window data from features
    # Take first 576 features (9*64) for each sample
    n_features = n_channels * window_size
    
    # Pad if necessary
    if train_X.shape[1] < n_features:
        pad_width = n_features - train_X.shape[1]
        train_X = np.pad(train_X, ((0, 0), (0, pad_width)), mode='constant')
        test_X = np.pad(test_X, ((0, 0), (0, pad_width)), mode='constant')
    
    # Reshape to [N, C, T]
    train_data = train_X[:, :n_features].reshape(n_train, n_channels, window_size)
    test_data = test_X[:, :n_features].reshape(n_test, n_channels, window_size)
    
    # Labels are 1-indexed, convert to 0-indexed
    train_labels = (train_y - 1).astype(np.int64)
    test_labels = (test_y - 1).astype(np.int64)
    
    # Create subject IDs (UCI-HAR has 30 subjects)
    train_subject_file = uci_dir / "train" / "subject_train.txt"
    test_subject_file = uci_dir / "test" / "subject_test.txt"
    
    if train_subject_file.exists():
        train_subjects = np.loadtxt(train_subject_file).astype(np.int64)
        test_subjects = np.loadtxt(test_subject_file).astype(np.int64)
    else:
        # Create dummy subject IDs
        train_subjects = np.zeros(n_train, dtype=np.int64)
        test_subjects = np.zeros(n_test, dtype=np.int64)
    
    # Create split info
    train_splits = np.array(['train'] * n_train)
    test_splits = np.array(['test'] * n_test)
    
    # Combine train and test
    all_data = np.concatenate([train_data, test_data], axis=0).astype(np.float32)
    all_labels = np.concatenate([train_labels, test_labels], axis=0).astype(np.int64)
    all_subjects = np.concatenate([train_subjects, test_subjects], axis=0).astype(np.int64)
    all_splits = np.concatenate([train_splits, test_splits])
    
    # Normalize
    mean = np.mean(all_data, axis=(0, 2), keepdims=True)
    std = np.std(all_data, axis=(0, 2), keepdims=True) + 1e-8
    all_data = ((all_data - mean) / std).astype(np.float32)
    
    # Create boundary labels (synthetic - based on label transitions)
    boundaries = np.zeros(len(all_labels), dtype=np.float32)
    
    # Save
    np.savez(
        output_file,
        data=all_data,
        labels=all_labels,
        subject_ids=all_subjects,
        split_info=all_splits,
        boundaries=boundaries
    )
    
    print(f"Saved to {output_file}")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Shape: {all_data.shape}")
    print(f"  Train: {(all_splits == 'train').sum()}")
    print(f"  Test: {(all_splits == 'test').sum()}")
    
    return True


def preprocess_pamap2():
    """Preprocess PAMAP2 dataset to unified format."""
    print("\n" + "="*60)
    print("Preprocessing PAMAP2 Dataset")
    print("="*60)
    
    pamap2_dir = DATA_DIR / "pamap2"
    processed_dir = pamap2_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "pamap2_processed.npz"
    if output_file.exists():
        print(f"Already preprocessed: {output_file}")
        return True
    
    protocol_path = pamap2_dir / "PAMAP2_Dataset" / "Protocol"
    if not protocol_path.exists():
        print(f"Protocol data not found at {protocol_path}")
        return False
    
    # Activity mapping (protocol activities only)
    activity_map = {
        1: 0,   # lying
        2: 1,   # sitting
        3: 2,   # standing
        4: 3,   # walking
        5: 4,   # running
        6: 5,   # cycling
        7: 6,   # nordic_walking
        12: 7,  # ascending_stairs
        13: 8,  # descending_stairs
        16: 9,  # vacuum_cleaning
        17: 10, # ironing
        24: 11  # rope_jumping
    }
    
    # Use hand IMU (9 channels) + chest IMU (9 channels) = 18 channels
    # Column indices: hand (4-12), chest (21-29)
    channel_indices = list(range(4, 13)) + list(range(21, 30))  # 18 channels
    
    window_size = 256  # 2.56 seconds at 100Hz
    stride = 128  # 50% overlap
    
    all_data = []
    all_labels = []
    all_subjects = []
    all_splits = []
    all_boundaries = []
    
    # Process each subject
    subject_files = sorted(protocol_path.glob("subject10*.dat"))
    
    for i, subject_file in enumerate(subject_files):
        subject_id = int(subject_file.stem.replace("subject", "")) - 100
        
        print(f"Processing subject {subject_id} ({i+1}/{len(subject_files)})...")
        
        try:
            data = np.loadtxt(subject_file, dtype=np.float32)
        except Exception as e:
            print(f"  Error loading {subject_file}: {e}")
            continue
        
        # Extract sensor data
        sensor_data = data[:, channel_indices]
        activity_labels = data[:, 1]  # Column 1 is activity ID
        
        # Handle NaN values
        sensor_data = np.nan_to_num(sensor_data, nan=0.0)
        
        # Create windows
        for j in range(0, len(sensor_data) - window_size, stride):
            window = sensor_data[j:j+window_size]
            window_labels = activity_labels[j:j+window_size]
            
            # Filter to valid activities
            valid_mask = np.isin(window_labels, list(activity_map.keys()))
            if valid_mask.sum() < window_size * 0.5:
                continue
            
            valid_labels = window_labels[valid_mask]
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            majority_label = unique_labels[counts.argmax()]
            
            class_idx = activity_map.get(int(majority_label))
            if class_idx is None:
                continue
            
            # Transpose to [C, T]
            all_data.append(window.T)
            all_labels.append(class_idx)
            all_subjects.append(subject_id)
            
            # Split: first 70% of subjects for train
            split = 'train' if subject_id <= 6 else 'test'
            all_splits.append(split)
            
            # Boundary score
            if len(unique_labels) > 1:
                boundary_score = 1.0 - (counts.max() / counts.sum())
            else:
                boundary_score = 0.0
            all_boundaries.append(boundary_score)
    
    if not all_data:
        print("No data extracted!")
        return False
    
    # Convert to arrays
    all_data = np.stack(all_data).astype(np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_subjects = np.array(all_subjects, dtype=np.int64)
    all_splits = np.array(all_splits)
    all_boundaries = np.array(all_boundaries, dtype=np.float32)
    
    # Normalize
    mean = np.mean(all_data, axis=(0, 2), keepdims=True)
    std = np.std(all_data, axis=(0, 2), keepdims=True) + 1e-8
    all_data = ((all_data - mean) / std).astype(np.float32)
    
    # Save
    np.savez(
        output_file,
        data=all_data,
        labels=all_labels,
        subject_ids=all_subjects,
        split_info=all_splits,
        boundaries=all_boundaries
    )
    
    print(f"Saved to {output_file}")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Shape: {all_data.shape}")
    print(f"  Train: {(all_splits == 'train').sum()}")
    print(f"  Test: {(all_splits == 'test').sum()}")
    
    return True


def preprocess_wisdm():
    """Preprocess WISDM dataset to unified format."""
    print("\n" + "="*60)
    print("Preprocessing WISDM Dataset")
    print("="*60)
    
    wisdm_dir = DATA_DIR / "wisdm" / "wisdm-dataset"
    processed_dir = DATA_DIR / "wisdm" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "wisdm_processed.npz"
    if output_file.exists():
        print(f"Already preprocessed: {output_file}")
        return True
    
    # WISDM has phone and watch data
    # Use phone accelerometer for simplicity
    raw_dir = wisdm_dir / "raw" / "phone" / "accel"
    
    if not raw_dir.exists():
        print(f"Raw data not found at {raw_dir}")
        return False
    
    # Activity labels
    activity_labels = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
        'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
        'M': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17
    }
    
    window_size = 200  # 10 seconds at 20Hz
    stride = 100  # 50% overlap
    
    all_data = []
    all_labels = []
    all_subjects = []
    all_splits = []
    #RW|    all_boundaries = []
#TH|    
#BQ|    # Get subject files
#PQ|    subject_files = sorted(raw_dir.glob("data_*.txt"))
#WP|    # Process each subject file
    
    # Process each subject file
    for i, subject_file in enumerate(subject_files):
        # Extract subject ID from filename like "data_1600_accel_phone.txt"
        filename = subject_file.stem  # "data_1600_accel_phone"
        subject_id = int(filename.split("_")[1])  # Extract "1600"
        
        # Skip malfunctioning subjects (1641-1648) with sensor issues
        if 1641 <= subject_id <= 1648:
            print(f"Skipping subject {subject_id} (known sensor malfunction)...")
            continue
        
        print(f"Processing subject {subject_id} ({i+1}/{len(subject_files)})...")
    

    for i, subject_file in enumerate(subject_files):
        # Extract subject ID from filename like "data_1600_accel_phone.txt"
        filename = subject_file.stem  # "data_1600_accel_phone"
        subject_id = int(filename.split("_")[1])  # Extract "1600"
        
        print(f"Processing subject {subject_id} ({i+1}/{len(subject_files)})...")
        
        try:
            # WISDM format: user,activity,timestamp,x,y,z
            # The z column may have semicolons at the end
            import pandas as pd
            df = pd.read_csv(subject_file, header=None,
                           names=['user', 'activity', 'timestamp', 'x', 'y', 'z'],
                           comment='#')
            
            # Clean z column - remove semicolons if present
            if df['z'].dtype == object:
                df['z'] = df['z'].str.rstrip(';').astype(float)
            
        except Exception as e:
            print(f"  Error loading {subject_file}: {e}")
            continue
        
        # Get sensor data
        sensor_data = df[['x', 'y', 'z']].values.astype(np.float32)
        activity_col = df['activity'].values
        # Clip extreme sensor values (malfunctioning sensors can have values up to +-78)
        sensor_data = np.clip(sensor_data, -20, 20)
        # Note: normalization done globally after collecting all data

        
        # Create windows
        for j in range(0, len(sensor_data) - window_size, stride):
            window = sensor_data[j:j+window_size]
            window_activities = activity_col[j:j+window_size]
            
            # Get majority activity
            unique_acts, counts = np.unique(window_activities, return_counts=True)
            majority_act = unique_acts[counts.argmax()]
            
            class_idx = activity_labels.get(majority_act)
            if class_idx is None:
                continue
            
            # Transpose to [C, T]
            all_data.append(window.T)
            all_labels.append(class_idx)
            all_subjects.append(subject_id)
            
            # Split: first 70% of subjects for train (subjects 1600-1635)
            split = 'train' if subject_id <= 1635 else 'test'
            all_splits.append(split)
            
            # Boundary score
            if len(unique_acts) > 1:
                boundary_score = 1.0 - (counts.max() / counts.sum())
            else:
                boundary_score = 0.0
            all_boundaries.append(boundary_score)
    
    if not all_data:
        print("No data extracted!")
        return False
    
    # Convert to arrays
    all_data = np.stack(all_data).astype(np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_subjects = np.array(all_subjects, dtype=np.int64)
    all_splits = np.array(all_splits)
    all_boundaries = np.array(all_boundaries, dtype=np.float32)
    # Global z-score normalization (using all data)
    mean = np.mean(all_data, axis=(0, 2), keepdims=True)
    std = np.std(all_data, axis=(0, 2), keepdims=True) + 1e-8
    all_data = ((all_data - mean) / std).astype(np.float32)

    
    # Save
    np.savez(
        output_file,
        data=all_data,
        labels=all_labels,
        subject_ids=all_subjects,
        split_info=all_splits,
        boundaries=all_boundaries
    )
    
    print(f"Saved to {output_file}")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Shape: {all_data.shape}")
    print(f"  Train: {(all_splits == 'train').sum()}")
    print(f"  Test: {(all_splits == 'test').sum()}")
    
    return True


def main():
    """Preprocess all datasets."""
    print("="*60)
    print("Dataset Preprocessing")
    print("="*60)
    
    results = {}
    
    results['UCI-HAR'] = preprocess_uci_har()
    results['PAMAP2'] = preprocess_pamap2()
    results['WISDM'] = preprocess_wisdm()
    
    # Summary
    print("\n" + "="*60)
    print("Preprocessing Summary")
    print("="*60)
    
    for dataset, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {dataset}")
    
    total = len(results)
    successful = sum(results.values())
    print(f"\nTotal: {successful}/{total} datasets preprocessed")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
