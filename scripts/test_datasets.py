"""
Test dataset loaders.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_uci_har():
    """Test UCI-HAR dataset loader."""
    print("\nTesting UCI-HAR...")
    try:
        from sashar.data.uci_har import UCIHardataset
        ds = UCIHardataset(root='data/uci_har', split='train', download=False)
        item = ds[0]
        print(f"  Train samples: {len(ds)}")
        print(f"  Sample shape: {item['data'].shape}")
        print(f"  Num classes: {ds.NUM_CLASSES}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pamap2():
    """Test PAMAP2 dataset loader."""
    print("\nTesting PAMAP2...")
    try:
        from sashar.data.pamap2 import PAMAP2Dataset
        ds = PAMAP2Dataset(root='data/pamap2', split='train', download=False)
        item = ds[0]
        print(f"  Train samples: {len(ds)}")
        print(f"  Sample shape: {item['data'].shape}")
        print(f"  Num classes: {ds.NUM_CLASSES}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wisdm():
    """Test WISDM dataset loader."""
    print("\nTesting WISDM...")
    try:
        from sashar.data.wisdm import WISDMDataset
        # WISDM needs different path structure
        ds = WISDMDataset(root='data/wisdm/wisdm-dataset', split='train', download=False)
        item = ds[0]
        print(f"  Train samples: {len(ds)}")
        print(f"  Sample shape: {item['data'].shape}")
        print(f"  Num classes: {ds.NUM_CLASSES}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opportunity():
    """Test Opportunity dataset loader."""
    print("\nTesting Opportunity...")
    try:
        from sashar.data.opportunity import OpportunityDataset
        ds = OpportunityDataset(root='data/opportunity', split='train')
        item = ds[0]
        print(f"  Train samples: {len(ds)}")
        print(f"  Sample shape: {item['data'].shape}")
        print(f"  Num classes: {ds.NUM_CLASSES}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing Dataset Loaders")
    print("="*60)
    
    results = {
        'UCI-HAR': test_uci_har(),
        'PAMAP2': test_pamap2(),
        'WISDM': test_wisdm(),
        'Opportunity': test_opportunity(),
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"{name}: {status}")
    
    total = len(results)
    successful = sum(results.values())
    print(f"\nTotal: {successful}/{total} datasets working")
