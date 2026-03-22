"""
Dataset Download Script

Downloads and verifies all HAR datasets for experiments.
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"


def download_uci_har():
    """Download UCI-HAR dataset."""
    print("\n" + "="*60)
    print("Downloading UCI-HAR Dataset")
    print("="*60)
    
    uci_dir = DATA_DIR / "uci_har"
    uci_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    train_file = uci_dir / "train" / "X_train.txt"
    if train_file.exists():
        print(f"UCI-HAR already downloaded at {uci_dir}")
        return True
    
    # Download URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = uci_dir / "uci_har.zip"
    
    try:
        print(f"Downloading from {url}...")
        
        # Use requests-like headers to avoid 403
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(zip_path, 'wb') as f:
                f.write(response.read())
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(uci_dir)
        
        # Move files to expected locations
        extracted = uci_dir / "UCI HAR Dataset"
        if extracted.exists():
            if (extracted / "train").exists():
                shutil.move(str(extracted / "train"), str(uci_dir / "train"))
            if (extracted / "test").exists():
                shutil.move(str(extracted / "test"), str(uci_dir / "test"))
            shutil.rmtree(extracted)
        
        zip_path.unlink()
        
        # Verify
        if (uci_dir / "train" / "X_train.txt").exists():
            print(f"[OK] UCI-HAR downloaded successfully!")
            return True
        else:
            print("[FAIL] Download completed but files not found in expected location")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error downloading UCI-HAR: {e}")
        return False


def download_wisdm():
    """Download WISDM dataset."""
    print("\n" + "="*60)
    print("Downloading WISDM Dataset")
    print("="*60)
    
    wisdm_dir = DATA_DIR / "wisdm"
    wisdm_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists - look for raw data files
    raw_dir = wisdm_dir / "wisdm-dataset" / "raw"
    if raw_dir.exists() and list(raw_dir.glob("*.txt")):
        print(f"WISDM already downloaded at {wisdm_dir}")
        return True
    
    # Download URL (correct URL from UCI)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
    zip_path = wisdm_dir / "wisdm.zip"
    
    try:
        print(f"Downloading from {url}...")
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(zip_path, 'wb') as f:
                f.write(response.read())
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(wisdm_dir)
        
        zip_path.unlink()
        
        # Verify
        raw_dir = wisdm_dir / "wisdm-dataset" / "raw"
        if raw_dir.exists() and list(raw_dir.glob("*.txt")):
            print(f"[OK] WISDM downloaded successfully!")
            return True
        else:
            print("[FAIL] Download completed but files not found in expected location")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error downloading WISDM: {e}")
        return False

def download_pamap2():
    """Download PAMAP2 dataset."""
    print("\n" + "="*60)
    print("Downloading PAMAP2 Dataset")
    print("="*60)
    
    pamap2_dir = DATA_DIR / "pamap2"
    pamap2_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    protocol_path = pamap2_dir / "PAMAP2_Dataset" / "Protocol"
    if protocol_path.exists() and list(protocol_path.glob("*.dat")):
        print(f"PAMAP2 already downloaded at {pamap2_dir}")
        return True
    
    # Download URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    zip_path = pamap2_dir / "pamap2.zip"
    
    try:
        print(f"Downloading from {url}...")
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(zip_path, 'wb') as f:
                f.write(response.read())
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pamap2_dir)
        
        zip_path.unlink()
        
        # Verify
        protocol_path = pamap2_dir / "PAMAP2_Dataset" / "Protocol"
        if protocol_path.exists() and list(protocol_path.glob("*.dat")):
            print(f"[OK] PAMAP2 downloaded successfully!")
            return True
        else:
            print("[FAIL] Download completed but files not found in expected location")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error downloading PAMAP2: {e}")
        return False


def verify_opportunity():
    """Verify Opportunity dataset exists."""
    print("\n" + "="*60)
    print("Verifying Opportunity Dataset")
    print("="*60)
    
    opp_dir = DATA_DIR / "opportunity"
    processed_dir = opp_dir / "processed"
    
    if processed_dir.exists():
        # Check for .npz or .pt files
        npz_files = list(processed_dir.glob("*.npz"))
        pt_files = list(processed_dir.glob("*.pt"))
        files = npz_files + pt_files
        if files:
            print(f"[OK] Opportunity dataset found with {len(files)} processed files")
            return True
    
    print("[FAIL] Opportunity dataset not found")
    return False

def main():
    """Download all datasets."""
    print("="*60)
    print("HAR Dataset Downloader")
    print("="*60)
    
    results = {}
    
    # Verify Opportunity (already downloaded in previous session)
    results['opportunity'] = verify_opportunity()
    
    # Download other datasets
    results['uci_har'] = download_uci_har()
    results['wisdm'] = download_wisdm()
    results['pamap2'] = download_pamap2()
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    for dataset, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {dataset}: {'Available' if success else 'Failed'}")
    
    total = len(results)
    successful = sum(results.values())
    print(f"\nTotal: {successful}/{total} datasets available")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
