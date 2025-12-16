"""
Simple format checker for Option B (no dependencies).
Just inspects .npz files to verify structure.
"""
import os
import numpy as np


def check_calibration_cache():
    """Check if calibration cache has per_model_predictions"""
    print("="*80)
    print("Checking calibration cache format")
    print("="*80)
    
    cache_file = "uq_benchmark_results/cache/breastmnist_calib_results.npz"
    
    if not os.path.exists(cache_file):
        print(f"✗ File not found: {cache_file}")
        return None
    
    data = np.load(cache_file)
    print(f"Keys: {list(data.keys())}")
    
    if 'per_model_predictions' in data:
        shape = data['per_model_predictions'].shape
        print(f"per_model_predictions shape: {shape}")
        print(f"Expected: [num_models, num_samples, num_classes]")
        
        if len(shape) == 3:
            print("✓ Correct 3D shape")
            return True
        else:
            print("✗ Wrong shape")
            return False
    else:
        print("✗ per_model_predictions key not found")
        return False


def check_gps_cache_format():
    """Check GPS augmentation cache format"""
    print("\n" + "="*80)
    print("Checking GPS augmentation cache format")
    print("="*80)
    
    gps_dir = "benchmarks/medMNIST/Data/gps_augment/224*224/breastmnist_calibration_set"
    
    if not os.path.exists(gps_dir):
        print(f"✗ Directory not found: {gps_dir}")
        return None
    
    npz_files = sorted([f for f in os.listdir(gps_dir) if f.endswith('.npz')])
    
    if not npz_files:
        print("✗ No .npz files found")
        return None
    
    print(f"Found {len(npz_files)} .npz files")
    
    # Check first file
    first_file = os.path.join(gps_dir, npz_files[0])
    data = np.load(first_file)
    
    print(f"\nFirst file: {npz_files[0]}")
    print(f"Keys: {list(data.keys())}")
    
    version = data.get('version', 1)
    preds = data['predictions']
    
    print(f"Version: {version}")
    print(f"Predictions shape: {preds.shape}")
    
    if version == 2:
        if len(preds.shape) == 3:
            print(f"Expected: [num_models, num_samples, num_classes]")
            num_models = data.get('num_models', 'unknown')
            print(f"num_models: {num_models}")
            print("✓ v2 format with per-model predictions")
            return True
        else:
            print("✗ v2 version but wrong shape")
            return False
    else:
        if len(preds.shape) == 2:
            print(f"Expected: [num_samples, num_classes]")
            print("✓ v1 format (old ensemble predictions)")
            print("⚠ GPS cache needs regeneration for Option B")
            return False
        else:
            print("✗ v1 version but wrong shape")
            return False


def check_storage_increase():
    """Estimate storage increase for v2"""
    print("\n" + "="*80)
    print("Storage impact analysis")
    print("="*80)
    
    gps_dir = "benchmarks/medMNIST/Data/gps_augment/224*224/breastmnist_calibration_set"
    
    if not os.path.exists(gps_dir):
        print("⊘ Cannot analyze - directory not found")
        return None
    
    npz_files = [f for f in os.listdir(gps_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print("⊘ Cannot analyze - no files found")
        return None
    
    total_size = sum(os.path.getsize(os.path.join(gps_dir, f)) for f in npz_files)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"Current total size: {total_size_mb:.2f} MB for {len(npz_files)} files")
    print(f"Average per file: {total_size_mb / len(npz_files):.2f} MB")
    
    # Check if v2
    first_file = os.path.join(gps_dir, npz_files[0])
    data = np.load(first_file)
    version = data.get('version', 1)
    
    if version == 2:
        print(f"\n✓ Already using v2 format")
        num_models = data.get('num_models', 5)
        print(f"  Storage includes {num_models} models")
    else:
        print(f"\n⚠ Still using v1 format")
        print(f"  Expected increase with 5 models: ~{total_size_mb * 5:.2f} MB")
    
    return True


def main():
    print("Option B Format Verification")
    print("="*80)
    print("Checking if GPS calibration caching uses per-model predictions\n")
    
    results = {
        "calibration_cache": check_calibration_cache(),
        "gps_cache_format": check_gps_cache_format(),
        "storage_analysis": check_storage_increase()
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{name:25s}: {status}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if results['gps_cache_format'] is False:
        print("• GPS cache uses old format (v1). To use Option B:")
        print("  1. Delete old GPS cache: rm -rf benchmarks/medMNIST/Data/gps_augment/224*224/breastmnist_calibration_set/*.npz")
        print("  2. Regenerate with new code: python save_gps_policies.py --flag breastmnist")
    elif results['gps_cache_format'] is True:
        print("• GPS cache already uses Option B (v2) format")
        print("• Search will use avg(std(models)) computation")
    else:
        print("• GPS cache not found - generate with: python save_gps_policies.py --flag breastmnist")


if __name__ == "__main__":
    main()
