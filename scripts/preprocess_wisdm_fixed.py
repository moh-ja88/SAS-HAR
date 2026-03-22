"""
Fixed WISDM preprocessing script with robust normalization.

This script addresses the sensor malfunction issues identified in subjects 1641-1648.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"


def preprocess_wisdm_fixed(
    clip_value: float = 20.0,
    exclude_malfunctioning: bool = True,
    use_global_norm: bool = True
):
    """
    Preprocess WISDM dataset with robust normalization.
    
    Args:
        clip_value: Clip raw sensor values to [-clip_value, +clip_value] before normalization.
                   Default 20.0 is reasonable for accelerometer data in m/s².
        exclude_malfunctioning: If True, exclude subjects 1641-1648 which have sensor malfunctions.
        use_global_norm: If True, normalize after combining all subjects (recommended).
                        If False, use per-subject normalization (original behavior).
    """
    print("\n" + "="*60)
    print("Preprocessing WISDM Dataset (FIXED)")
    print("="*60)
    print(f"  clip_value: {clip_value}")
    print(f"  exclude_malfunctioning: {exclude_malfunctioning}")
    print(f"  use_global_norm: {use_global_norm}")
    
    wisdm_dir = DATA_DIR / "wisdm" / "wisdm-dataset"
    processed_dir = DATA_DIR / "wisdm" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Use different output file name to avoid overwriting
    output_file = processed_dir / "wisdm_processed_fixed.npz"
    
    # Subjects with known sensor malfunctions
    MALFUNCTIONING_SUBJECTS = {1641, 1642, 1643, 1644, 1645, 1646, 1648}
    
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
    all_boundaries = []
    
    # Process each subject file
    subject_files = sorted(raw_dir.glob("data_*.txt"))
    
    skipped_subjects = []
    
    for i, subject_file in enumerate(subject_files):
        # Extract subject ID from filename like "data_1600_accel_phone.txt"
        filename = subject_file.stem
        subject_id = int(filename.split("_")[1])
        
        # Skip malfunctioning subjects if requested
        if exclude_malfunctioning and subject_id in MALFUNCTIONING_SUBJECTS:
            print(f"  Skipping subject {subject_id} (sensor malfunction)")
            skipped_subjects.append(subject_id)
            continue
        
        print(f"Processing subject {subject_id} ({i+1}/{len(subject_files)})...")
        
        try:
            # WISDM format: user,activity,timestamp,x,y,z
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
        
        # Clip outliers before any normalization
        if clip_value is not None:
            sensor_data = np.clip(sensor_data, -clip_value, clip_value)
        
        # Per-subject normalization (only if not using global norm)
        if not use_global_norm:
            sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
        
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
            
            # Split: first 70% of subjects for train
            # Adjust split point if excluding malfunctioning subjects
            if exclude_malfunctioning:
                # With malfunctioning subjects excluded, use 1620 as split point
                # This gives roughly 60% train, 40% test
                split = 'train' if subject_id <= 1640 else 'test'
            else:
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
    
    print(f"\nSkipped {len(skipped_subjects)} malfunctioning subjects: {skipped_subjects}")
    
    # Convert to arrays
    all_data = np.stack(all_data).astype(np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_subjects = np.array(all_subjects, dtype=np.int64)
    all_splits = np.array(all_splits)
    all_boundaries = np.array(all_boundaries, dtype=np.float32)
    
    # Global normalization (recommended)
    if use_global_norm:
        print("\nApplying global normalization...")
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
    
    print(f"\nSaved to {output_file}")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Shape: {all_data.shape}")
    print(f"  Train: {(all_splits == 'train').sum()}")
    print(f"  Test: {(all_splits == 'test').sum()}")
    print(f"  Data range: [{all_data.min():.4f}, {all_data.max():.4f}]")
    print(f"  Data mean: {all_data.mean():.4f}, std: {all_data.std():.4f}")
    
    # Print class distribution
    train_labels = all_labels[all_splits == 'train']
    test_labels = all_labels[all_splits == 'test']
    
    print(f"\n  Train class distribution:")
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    for u, c in zip(train_unique, train_counts):
        print(f"    Class {u}: {c} ({100*c/len(train_labels):.1f}%)")
    
    print(f"\n  Test class distribution:")
    test_unique, test_counts = np.unique(test_labels, return_counts=True)
    for u, c in zip(test_unique, test_counts):
        print(f"    Class {u}: {c} ({100*c/len(test_labels):.1f}%)")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed WISDM preprocessing")
    parser.add_argument("--clip-value", type=float, default=20.0,
                       help="Clip raw sensor values to this range (default: 20)")
    parser.add_argument("--include-malfunctioning", action="store_true",
                       help="Include subjects with sensor malfunctions (1641-1648)")
    parser.add_argument("--per-subject-norm", action="store_true",
                       help="Use per-subject normalization instead of global")
    
    args = parser.parse_args()
    
    success = preprocess_wisdm_fixed(
        clip_value=args.clip_value,
        exclude_malfunctioning=not args.include_malfunctioning,
        use_global_norm=not args.per_subject_norm
    )
    
    sys.exit(0 if success else 1)
