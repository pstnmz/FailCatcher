"""
Check calibration properly using the EXACT same flow as run_medmnist_benchmark.py
"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load cached results from benchmark
cache_dir = project_root / 'uq_benchmark_results' / 'cache'
test_cache = np.load(cache_dir / 'breastmnist_resnet18_test_results.npz')
calib_cache = np.load(cache_dir / 'breastmnist_resnet18_calib_results.npz')

# Test set
y_true = test_cache['y_true']
y_scores = test_cache['y_scores']  # [N, C] where C=2 for binary
y_pred = test_cache['y_pred']
correct_idx = test_cache['correct_idx']
incorrect_idx = test_cache['incorrect_idx']

# Calib set
y_true_calib = calib_cache['y_true']
y_scores_calib = calib_cache['y_scores']  # [N_calib, C]

print("="*60)
print("CACHED BENCHMARK DATA ANALYSIS")
print("="*60)

print(f"\nTest set: {len(y_true)} samples")
print(f"  Accuracy: {len(correct_idx) / len(y_true):.4f}")
print(f"  y_scores shape: {y_scores.shape}")
print(f"  y_true unique: {np.unique(y_true)}")

print(f"\nCalib set: {len(y_true_calib)} samples")
print(f"  y_scores shape: {y_scores_calib.shape}")
print(f"  Label distribution: {np.bincount(y_true_calib.astype(int))}")

# Check test predictions
print("\n" + "="*60)
print("UNCALIBRATED PREDICTIONS (from cache)")
print("="*60)

test_pos_prob = y_scores[:, 1]  # Positive class probability
print(f"\nPositive class probability (y_scores[:, 1]):")
print(f"  Min: {test_pos_prob.min():.4f}, Max: {test_pos_prob.max():.4f}, Mean: {test_pos_prob.mean():.4f}")

# Check distribution
bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for low, high in bins:
    count = np.sum((test_pos_prob >= low) & (test_pos_prob < high))
    print(f"  {low:.1f}-{high:.1f}: {count}/{len(test_pos_prob)} ({count/len(test_pos_prob)*100:.1f}%)")

# Compute uncertainty (distance to hard labels)
from FailCatcher.methods.distance import distance_to_hard_labels_computation
uncalib_uncertainty = distance_to_hard_labels_computation(y_scores)
print(f"\nUncalibrated uncertainty (DHL):")
print(f"  Min: {uncalib_uncertainty.min():.4f}, Max: {uncalib_uncertainty.max():.4f}, Mean: {uncalib_uncertainty.mean():.4f}")

# Check uncertainty for correct vs incorrect
uncalib_unc_correct = uncalib_uncertainty[correct_idx]
uncalib_unc_incorrect = uncalib_uncertainty[incorrect_idx]
print(f"  Correct samples: mean={uncalib_unc_correct.mean():.4f}, std={uncalib_unc_correct.std():.4f}")
print(f"  Incorrect samples: mean={uncalib_unc_incorrect.mean():.4f}, std={uncalib_unc_incorrect.std():.4f}")

# Now apply Platt calibration
print("\n" + "="*60)
print("APPLYING PLATT CALIBRATION (same as benchmark)")
print("="*60)

from FailCatcher.methods.distance import posthoc_calibration
from FailCatcher.core.utils import apply_calibration

calibrated_probs_calib, platt_model = posthoc_calibration(y_scores_calib, y_true_calib, 'platt')
print(f"\nPlatt model trained on {len(y_true_calib)} calibration samples")
print(f"  Coefficients: {platt_model.coef_}")
print(f"  Intercept: {platt_model.intercept_}")

# Apply to test set
calibrated_test_probs = apply_calibration(y_scores, platt_model, 'platt')
print(f"\nCalibrated test predictions:")
print(f"  Output shape: {calibrated_test_probs.shape if hasattr(calibrated_test_probs, 'shape') else type(calibrated_test_probs)}")
print(f"  Min: {calibrated_test_probs.min():.4f}, Max: {calibrated_test_probs.max():.4f}, Mean: {calibrated_test_probs.mean():.4f}")

# Distribution
for low, high in bins:
    count = np.sum((calibrated_test_probs >= low) & (calibrated_test_probs < high))
    print(f"  {low:.1f}-{high:.1f}: {count}/{len(calibrated_test_probs)} ({count/len(calibrated_test_probs)*100:.1f}%)")

# Compute calibrated uncertainty
# For binary, calibrated_test_probs is 1D (positive class probability)
# Need to convert to 2D for distance_to_hard_labels_computation
if calibrated_test_probs.ndim == 1:
    # Convert to 2-class format
    calibrated_scores_2d = np.column_stack([1 - calibrated_test_probs, calibrated_test_probs])
else:
    calibrated_scores_2d = calibrated_test_probs

calib_uncertainty = distance_to_hard_labels_computation(calibrated_scores_2d)
print(f"\nCalibrated uncertainty (DHL):")
print(f"  Min: {calib_uncertainty.min():.4f}, Max: {calib_uncertainty.max():.4f}, Mean: {calib_uncertainty.mean():.4f}")

calib_unc_correct = calib_uncertainty[correct_idx]
calib_unc_incorrect = calib_uncertainty[incorrect_idx]
print(f"  Correct samples: mean={calib_unc_correct.mean():.4f}, std={calib_unc_correct.std():.4f}")
print(f"  Incorrect samples: mean={calib_unc_incorrect.mean():.4f}, std={calib_unc_incorrect.std():.4f}")

# Compute AUROC for failure detection
from sklearn.metrics import roc_auc_score
is_incorrect = np.zeros(len(y_true), dtype=bool)
is_incorrect[incorrect_idx] = True

try:
    auroc_uncalib = roc_auc_score(is_incorrect, uncalib_uncertainty)
    auroc_calib = roc_auc_score(is_incorrect, calib_uncertainty)
    print(f"\nAUROC for failure detection:")
    print(f"  Uncalibrated MSR: {auroc_uncalib:.4f}")
    print(f"  Calibrated (Platt): {auroc_calib:.4f}")
    print(f"  Difference: {auroc_calib - auroc_uncalib:+.4f}")
except Exception as e:
    print(f"\nCould not compute AUROC: {e}")

print("\n" + "="*60)
print("COMPARISON WITH BENCHMARK RESULTS")
print("="*60)

# Load benchmark results
import json
results_files = sorted((project_root / 'uq_benchmark_results').glob('uq_benchmark_breastmnist_resnet18_*.json'))
if results_files:
    latest_result = results_files[-1]
    with open(latest_result) as f:
        bench_results = json.load(f)
    
    print(f"\nBenchmark file: {latest_result.name}")
    if 'MSR' in bench_results['methods']:
        print(f"  MSR AUROC: {bench_results['methods']['MSR']['auroc_f']:.4f}")
    if 'MSR_calibrated' in bench_results['methods']:
        print(f"  MSR_calibrated AUROC: {bench_results['methods']['MSR_calibrated']['auroc_f']:.4f}")
    
    print(f"\nDoes our analysis match the benchmark?")
    if 'MSR' in bench_results['methods']:
        match_msr = abs(auroc_uncalib - bench_results['methods']['MSR']['auroc_f']) < 0.01
        print(f"  MSR: {auroc_uncalib:.4f} vs {bench_results['methods']['MSR']['auroc_f']:.4f} - {'✓ MATCH' if match_msr else '✗ MISMATCH'}")
    if 'MSR_calibrated' in bench_results['methods']:
        match_calib = abs(auroc_calib - bench_results['methods']['MSR_calibrated']['auroc_f']) < 0.01
        print(f"  Calibrated: {auroc_calib:.4f} vs {bench_results['methods']['MSR_calibrated']['auroc_f']:.4f} - {'✓ MATCH' if match_calib else '✗ MISMATCH'}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
