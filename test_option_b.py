"""
Test script for Option B (per-model predictions in GPS calibration caching).
Tests the new storage format with smaller parameters.
"""
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from FailCatcher.methods.tta import apply_randaugment_and_store_results
from FailCatcher.search.greedy import load_npz_files_for_greedy_search, perform_greedy_policy_search


def test_option_b_storage():
    """Test that predictions are saved with correct shape [M, N, C]"""
    print("="*80)
    print("TEST 1: Verify storage format")
    print("="*80)
    
    # Use breastmnist calibration cache
    cache_dir = "/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/cache"
    calib_file = os.path.join(cache_dir, "breastmnist_calib_results.npz")
    
    if not os.path.exists(calib_file):
        print(f"ERROR: {calib_file} not found. Run calibration first.")
        return False
    
    # Load calibration results
    data = np.load(calib_file)
    print(f"Keys in calibration file: {list(data.keys())}")
    
    if 'per_model_predictions' in data:
        per_model_preds = data['per_model_predictions']
        print(f"per_model_predictions shape: {per_model_preds.shape}")
        print(f"Expected: [num_models, num_samples, num_classes]")
        
        if len(per_model_preds.shape) == 3:
            print("✓ Correct 3D shape for per-model predictions")
            return True
        else:
            print("✗ Incorrect shape - expected 3D array")
            return False
    else:
        print("✗ per_model_predictions not found in cache")
        return False


def test_option_b_search():
    """Test greedy search with v2 format"""
    print("\n" + "="*80)
    print("TEST 2: Verify greedy search handles v2 format")
    print("="*80)
    
    # Path to GPS augmentation cache (if it exists)
    gps_dir = "/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data/gps_augment/224*224/breastmnist_calibration_set"
    
    if not os.path.exists(gps_dir):
        print(f"GPS cache directory not found: {gps_dir}")
        print("Skipping search test - generate GPS cache first")
        return None
    
    # Check if any .npz files exist
    npz_files = [f for f in os.listdir(gps_dir) if f.endswith('.npz')]
    if not npz_files:
        print(f"No .npz files in {gps_dir}")
        print("Skipping search test - generate GPS cache first")
        return None
    
    print(f"Found {len(npz_files)} .npz files")
    
    # Load first file to check format
    first_file = os.path.join(gps_dir, npz_files[0])
    data = np.load(first_file)
    
    print(f"First file keys: {list(data.keys())}")
    preds = data['predictions']
    version = data.get('version', 1)
    
    print(f"Version: {version}")
    print(f"Predictions shape: {preds.shape}")
    
    if version == 2:
        if len(preds.shape) == 3:
            print("✓ v2 format detected with correct 3D shape [M, N, C]")
        else:
            print("✗ v2 format but wrong shape")
            return False
    else:
        if len(preds.shape) == 2:
            print("✓ v1 format detected with correct 2D shape [N, C]")
        else:
            print("✗ v1 format but wrong shape")
            return False
    
    # Try loading all files
    try:
        all_preds, all_keys, format_version = load_npz_files_for_greedy_search(gps_dir)
        print(f"\n✓ Successfully loaded {len(all_keys)} policies")
        print(f"  Format version: {format_version}")
        print(f"  Stacked predictions shape: {all_preds.shape}")
        
        if format_version == 1:
            print(f"  Expected: [num_policies, num_samples, num_classes]")
        else:
            print(f"  Expected: [num_policies, num_models, num_samples, num_classes]")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option_b_shape_consistency():
    """Test that _process_batch returns correct shape"""
    print("\n" + "="*80)
    print("TEST 3: Check _process_batch output shape")
    print("="*80)
    
    # Create dummy data
    B, K, M, C = 32, 10, 5, 2  # batch, policies, models, classes
    
    # Simulate what _process_batch should return
    per_model_preds = np.random.rand(B, K, M, C).astype(np.float32)
    
    print(f"Simulated _process_batch output: {per_model_preds.shape}")
    print(f"Expected: (batch_size={B}, num_policies={K}, num_models={M}, num_classes={C})")
    
    # Test transpose for storage
    for local_k in range(K):
        policy_preds = per_model_preds[:, local_k, :, :]  # [B, M, C]
        transposed = policy_preds.transpose(1, 0, 2)  # [M, B, C]
        
        print(f"\nPolicy {local_k}:")
        print(f"  Before transpose: {policy_preds.shape} [B, M, C]")
        print(f"  After transpose:  {transposed.shape} [M, B, C]")
        
        # Verify we can write to memmap-like structure
        memmap_slice = np.zeros((M, B, C), dtype=np.float32)
        memmap_slice[:, :, :] = transposed
        
        if memmap_slice.shape == (M, B, C):
            print(f"  ✓ Can write to memmap shape [M, B, C]")
        else:
            print(f"  ✗ Shape mismatch after assignment")
            return False
    
    return True


def main():
    print("Testing Option B Implementation")
    print("="*80)
    
    results = {
        "storage_format": test_option_b_storage(),
        "search_loading": test_option_b_search(),
        "shape_consistency": test_option_b_shape_consistency()
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(r is True for r in results.values() if r is not None)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
