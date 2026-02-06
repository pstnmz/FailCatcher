#!/usr/bin/env python3
"""
Script to recompute Mean_Aggregation and Mean_Aggregation_Ensemble in NPZ files.

Mean_Aggregation: For each fold, z-score normalize confidence scores from per_fold methods
(MSR, MSR_calibrated, MLS, GPS, KNN_Raw, MCDropout), then take the mean.

Mean_Aggregation_Ensemble: Z-score normalize ensemble-level confidence scores from
(MSR, MSR_calibrated, MLS, GPS, KNN_Raw, MCDropout, Ensembling), then take the mean.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import warnings

def z_score_normalize(scores: np.ndarray) -> np.ndarray:
    """Z-score normalize an array (mean=0, std=1)."""
    mean = np.mean(scores)
    std = np.std(scores)
    if std < 1e-10:  # Avoid division by zero
        warnings.warn(f"Standard deviation is near zero ({std}), returning zeros")
        return np.zeros_like(scores)
    return (scores - mean) / std

def compute_mean_aggregation_per_fold(data: dict, method_names: List[str], n_folds: int) -> Optional[np.ndarray]:
    """
    Compute Mean_Aggregation for per-fold data.
    
    Args:
        data: Dictionary containing NPZ data
        method_names: List of method names to aggregate (without _per_fold suffix)
        n_folds: Number of folds
    
    Returns:
        Array of shape (n_folds, n_samples) or None if data is missing
    """
    per_fold_keys = [f"{method}_per_fold" for method in method_names]
    
    # Check which keys exist
    available_keys = [key for key in per_fold_keys if key in data]
    
    if len(available_keys) == 0:
        return None
    
    # Get the first available array to determine shape
    first_array = data[available_keys[0]]
    if first_array.ndim == 1:
        # Single fold case
        n_samples = len(first_array)
        result = np.zeros((1, n_samples))
    else:
        n_folds, n_samples = first_array.shape
        result = np.zeros((n_folds, n_samples))
    
    # For each fold, z-score normalize and average
    for fold_idx in range(n_folds):
        z_scored_arrays = []
        
        for key in available_keys:
            scores = data[key]
            if scores.ndim == 1:
                fold_scores = scores
            else:
                fold_scores = scores[fold_idx]
            
            # Z-score normalize
            z_scored = z_score_normalize(fold_scores)
            z_scored_arrays.append(z_scored)
        
        # Mean of z-scored arrays
        result[fold_idx] = np.mean(z_scored_arrays, axis=0)
    
    return result

def compute_mean_aggregation_ensemble(data: dict, method_names: List[str], include_ensembling: bool = True) -> Optional[np.ndarray]:
    """
    Compute Mean_Aggregation_Ensemble for ensemble-level data.
    
    Args:
        data: Dictionary containing NPZ data
        method_names: List of method names to aggregate (without _ensemble suffix)
        include_ensembling: Whether to include the 'Ensembling' method
    
    Returns:
        Array of shape (n_samples,) or None if data is missing
    """
    ensemble_keys = [f"{method}_ensemble" for method in method_names]
    if include_ensembling and 'Ensembling' in data:
        ensemble_keys.append('Ensembling')
    
    # Check which keys exist
    available_keys = [key for key in ensemble_keys if key in data]
    
    if len(available_keys) == 0:
        return None
    
    # Z-score normalize each method and then average
    z_scored_arrays = []
    
    for key in available_keys:
        scores = data[key]
        z_scored = z_score_normalize(scores)
        z_scored_arrays.append(z_scored)
    
    # Mean of z-scored arrays
    result = np.mean(z_scored_arrays, axis=0)
    
    return result

def recompute_aggregations_in_npz(npz_path: Path, method_names: List[str]) -> bool:
    """
    Recompute Mean_Aggregation and Mean_Aggregation_Ensemble in an NPZ file.
    
    Args:
        npz_path: Path to the NPZ file
        method_names: List of method names to include in aggregation
    
    Returns:
        True if file was modified, False otherwise
    """
    # Load data
    data = dict(np.load(npz_path, allow_pickle=True))
    
    modified = False
    
    # Determine number of folds from any per_fold array
    n_folds = 5  # Default
    for method in method_names:
        key = f"{method}_per_fold"
        if key in data:
            if data[key].ndim > 1:
                n_folds = data[key].shape[0]
            break
    
    # Compute Mean_Aggregation
    mean_agg = compute_mean_aggregation_per_fold(data, method_names, n_folds)
    if mean_agg is not None:
        old_exists = 'Mean_Aggregation' in data
        if old_exists:
            old_value = data['Mean_Aggregation']
            if not np.array_equal(old_value, mean_agg):
                data['Mean_Aggregation'] = mean_agg
                modified = True
        else:
            data['Mean_Aggregation'] = mean_agg
            modified = True
    
    # Compute Mean_Aggregation_Ensemble
    mean_agg_ens = compute_mean_aggregation_ensemble(data, method_names, include_ensembling=True)
    if mean_agg_ens is not None:
        old_exists = 'Mean_Aggregation_Ensemble' in data
        if old_exists:
            old_value = data['Mean_Aggregation_Ensemble']
            if not np.array_equal(old_value, mean_agg_ens):
                data['Mean_Aggregation_Ensemble'] = mean_agg_ens
                modified = True
        else:
            data['Mean_Aggregation_Ensemble'] = mean_agg_ens
            modified = True
    
    # Save if modified
    if modified:
        np.savez(npz_path, **data)
        return True
    
    return False

def process_directory(directory: Path, method_names: List[str]) -> None:
    """Process all NPZ files in a directory."""
    npz_files = list(directory.glob('all_metrics_*.npz'))
    
    print(f"\nFound {len(npz_files)} NPZ files to process")
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for npz_file in npz_files:
        try:
            was_modified = recompute_aggregations_in_npz(npz_file, method_names)
            if was_modified:
                print(f"✓ Updated: {npz_file.name}")
                updated_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"✗ Error processing {npz_file.name}: {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped (no changes): {skipped_count}")
    print(f"  Errors: {error_count}")

def main():
    # Configuration
    corruption_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/corruption_shifts')
    
    # Methods to include in aggregation (without suffix)
    method_names = ['MSR', 'MSR_calibrated', 'MLS', 'GPS', 'KNN_Raw', 'MCDropout']
    
    print("=" * 80)
    print("Recomputing Mean_Aggregation and Mean_Aggregation_Ensemble")
    print("=" * 80)
    print(f"Directory: {corruption_dir}")
    print(f"Methods: {', '.join(method_names)}")
    
    process_directory(corruption_dir, method_names)
    
    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
