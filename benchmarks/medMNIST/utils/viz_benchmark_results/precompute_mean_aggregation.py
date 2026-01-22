"""
Pre-compute Mean Aggregation scores by:
1. Computing z-score + aggregation of UQ scores in NPZ files
2. Computing AUROC_f and AUGRC metrics using cached correct/incorrect indices
3. Saving aggregated scores to NPZ and metrics to JSON
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from FailCatcher.evaluation.evaluation import compute_augrc


def aggregate_uq_scores(npz_data, aggregation='mean', use_ensemble=False):
    """
    Aggregate UQ scores using z-score normalization.
    
    Args:
        npz_data: Loaded NPZ data
        aggregation: 'mean', 'min', 'max', or 'vote'
        use_ensemble: If True, use ensemble scores; if False, use per_fold scores
    
    Returns:
        np.ndarray: Aggregated scores (shape depends on use_ensemble)
    """
    # Get method keys based on mode
    if use_ensemble:
        # Ensemble: use base methods (not _per_fold, not _ensemble suffix)
        # Include Ensembling for ensemble mode
        method_keys = [k for k in npz_data.keys() 
                      if not k.endswith('_per_fold') 
                      and not k.endswith('_ensemble')
                      and k not in ['TTA', 'Mean_Aggregation', 'Mean_Aggregation_Ensemble']]
    else:
        # Per-fold: use _per_fold methods
        method_keys = [k for k in npz_data.keys() 
                      if k.endswith('_per_fold')
                      and k not in ['TTA_per_fold', 'Ensembling_per_fold', 'Mean_Aggregation', 'Mean_Aggregation_Ensemble']]
    
    if not method_keys:
        return None
    
    if use_ensemble:
        # Ensemble mode: single set of scores
        normalized_arrays = []
        for method_name in method_keys:
            scores = npz_data[method_name]
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            if std_val > 0:
                z_score = (scores - mean_val) / std_val
                normalized_arrays.append(z_score)
        
        if not normalized_arrays:
            return None
        
        stacked = np.stack(normalized_arrays, axis=0)
        
        if aggregation == 'mean':
            return np.mean(stacked, axis=0)
        elif aggregation == 'min':
            return np.min(stacked, axis=0)
        elif aggregation == 'max':
            return np.max(stacked, axis=0)
        elif aggregation == 'vote':
            return np.sum(stacked > 0, axis=0) / len(normalized_arrays)
    else:
        # Per-fold mode: iterate through folds
        num_folds = len(npz_data[method_keys[0]])
        aggregated_folds = []
        
        for fold_idx in range(num_folds):
            normalized_arrays = []
            for method_name in method_keys:
                scores = npz_data[method_name][fold_idx]
                mean_val = np.mean(scores)
                std_val = np.std(scores)
                if std_val > 0:
                    z_score = (scores - mean_val) / std_val
                    normalized_arrays.append(z_score)
            
            if not normalized_arrays:
                continue
            
            stacked = np.stack(normalized_arrays, axis=0)
            
            if aggregation == 'mean':
                fold_agg = np.mean(stacked, axis=0)
            elif aggregation == 'min':
                fold_agg = np.min(stacked, axis=0)
            elif aggregation == 'max':
                fold_agg = np.max(stacked, axis=0)
            elif aggregation == 'vote':
                fold_agg = np.sum(stacked > 0, axis=0) / len(normalized_arrays)
            
            aggregated_folds.append(fold_agg)
        
        if not aggregated_folds:
            return None
        
        return np.array(aggregated_folds)


def process_npz_file(npz_path, aggregation='mean', force=False):
    """
    Add Mean_Aggregation and Mean_Aggregation_Ensemble to NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        aggregation: Aggregation strategy
        force: Force recompute
    
    Returns:
        bool: True if updated
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Check if already exists
    if not force and 'Mean_Aggregation' in data and 'Mean_Aggregation_Ensemble' in data:
        return False
    
    updated = False
    new_data = {k: data[k] for k in data.keys()}
    
    # Compute per-fold aggregation
    if force or 'Mean_Aggregation' not in data:
        aggregated_per_fold = aggregate_uq_scores(data, aggregation=aggregation, use_ensemble=False)
        if aggregated_per_fold is not None:
            new_data['Mean_Aggregation'] = aggregated_per_fold
            updated = True
    
    # Compute ensemble aggregation
    if force or 'Mean_Aggregation_Ensemble' not in data:
        aggregated_ensemble = aggregate_uq_scores(data, aggregation=aggregation, use_ensemble=True)
        if aggregated_ensemble is not None:
            new_data['Mean_Aggregation_Ensemble'] = aggregated_ensemble
            updated = True
    
    if updated:
        np.savez(npz_path, **new_data)
    
    return updated


def compute_metrics_from_npz(npz_path, cache_dir, use_ensemble=False):
    """
    Compute AUROC_f and AUGRC from NPZ using cached indices.
    
    Args:
        npz_path: Path to NPZ file with Mean_Aggregation
        cache_dir: Path to cache directory
        use_ensemble: Use ensemble or per-fold
    
    Returns:
        dict: {'auroc_f_mean': float, 'augrc_mean': float} or None
    """
    # Load NPZ data
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # Get aggregated scores
    if use_ensemble:
        if 'Mean_Aggregation_Ensemble' not in npz_data:
            return None
        aggregated_scores = npz_data['Mean_Aggregation_Ensemble']
    else:
        if 'Mean_Aggregation' not in npz_data:
            return None
        aggregated_scores = npz_data['Mean_Aggregation']
    
    # Determine cache file name (remove timestamp from npz_name)
    npz_name = npz_path.name.replace('all_metrics_', '').replace('.npz', '')
    # Remove timestamp pattern (_YYYYMMDD_HHMMSS)
    import re
    npz_name = re.sub(r'_\d{8}_\d{6}$', '', npz_name)
    
    # Try different cache patterns
    parent_dir = npz_path.parent.name
    
    if parent_dir == 'new_class_shifts':
        cache_pattern = f"{npz_name}_new_class_shift_test_results.npz"
    elif parent_dir == 'corruption_shifts':
        # Remove corrupt_severity3_test and replace with corrupt3
        npz_name_clean = npz_name.replace('_corrupt_severity3_test', '')
        # Try both patterns
        cache_pattern_v1 = f"{npz_name_clean}_corrupt3_test_results.npz"
        cache_pattern_v2 = f"{npz_name_clean}_corrupt3_test_test_results.npz"
        if (cache_dir / cache_pattern_v2).exists():
            cache_pattern = cache_pattern_v2
        else:
            cache_pattern = cache_pattern_v1
    else:  # id or population_shifts
        cache_pattern_v1 = f"{npz_name}_test_results.npz"
        cache_pattern_v2 = f"{npz_name}_test_test_results.npz"
        if (cache_dir / cache_pattern_v2).exists():
            cache_pattern = cache_pattern_v2
        else:
            cache_pattern = cache_pattern_v1
    
    cache_path = cache_dir / cache_pattern
    if not cache_path.exists():
        return None
    
    # Load cache data
    cache_data = np.load(cache_path, allow_pickle=True)
    
    if use_ensemble:
        # Ensemble mode: use ensemble indices
        if 'correct_idx' in cache_data and 'incorrect_idx' in cache_data:
            correct_idx = np.asarray(cache_data['correct_idx'], dtype=int)
            incorrect_idx = np.asarray(cache_data['incorrect_idx'], dtype=int)
        else:
            return None
        
        n_samples = len(correct_idx) + len(incorrect_idx)
        failure_labels = np.zeros(n_samples)
        failure_labels[incorrect_idx] = 1
        
        if len(failure_labels) != len(aggregated_scores):
            return None
        
        # Compute AUROC_f
        auroc_f = roc_auc_score(failure_labels, aggregated_scores)
        
        # Compute AUGRC
        if 'y_pred' in cache_data and 'y_true' in cache_data:
            predictions = cache_data['y_pred']
            labels = cache_data['y_true']
            augrc, _ = compute_augrc(aggregated_scores, predictions, labels,
                                    correct_idx=correct_idx, incorrect_idx=incorrect_idx)
        else:
            augrc = np.nan
        
        return {
            'auroc_f_mean': float(auroc_f),
            'augrc_mean': float(augrc) if not np.isnan(augrc) else None
        }
    else:
        # Per-fold mode
        if 'per_fold_correct_idx' not in cache_data:
            return None
        
        per_fold_correct = [np.asarray(idx, dtype=int) for idx in cache_data['per_fold_correct_idx']]
        per_fold_incorrect = [np.asarray(idx, dtype=int) for idx in cache_data['per_fold_incorrect_idx']]
        
        num_folds = len(per_fold_correct)
        if len(aggregated_scores) != num_folds:
            return None
        
        # Get per-fold predictions for AUGRC
        if 'per_fold_predictions' in cache_data:
            per_fold_predictions = cache_data['per_fold_predictions']
            y_true = cache_data['y_true']
        else:
            per_fold_predictions = None
        
        auroc_scores = []
        augrc_scores = []
        
        for fold_idx in range(num_folds):
            correct_idx = per_fold_correct[fold_idx]
            incorrect_idx = per_fold_incorrect[fold_idx]
            fold_scores = aggregated_scores[fold_idx]
            
            n_samples = len(correct_idx) + len(incorrect_idx)
            failure_labels = np.zeros(n_samples)
            failure_labels[incorrect_idx] = 1
            
            if len(failure_labels) != len(fold_scores):
                continue
            
            # AUROC_f
            auroc_f = roc_auc_score(failure_labels, fold_scores)
            auroc_scores.append(auroc_f)
            
            # AUGRC
            if per_fold_predictions is not None:
                fold_predictions = per_fold_predictions[fold_idx]
                augrc, _ = compute_augrc(fold_scores, fold_predictions, y_true,
                                        correct_idx=correct_idx, incorrect_idx=incorrect_idx)
                augrc_scores.append(augrc)
        
        if not auroc_scores:
            return None
        
        return {
            'auroc_f_mean': float(np.mean(auroc_scores)),
            'augrc_mean': float(np.mean(augrc_scores)) if augrc_scores else None
        }


def process_json_file(json_path, results_dir, cache_dir, force=False):
    """
    Add Mean_Aggregation metrics to JSON file.
    
    Args:
        json_path: Path to JSON file
        results_dir: Results directory
        cache_dir: Cache directory
        force: Force recompute
    
    Returns:
        bool: True if updated
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if already exists and has actual values (not just empty dicts)
    methods = data.get('methods', {})
    has_mean_agg = 'Mean_Aggregation' in methods and methods['Mean_Aggregation']
    has_mean_agg_ens = 'Mean_Aggregation_Ensemble' in methods and methods['Mean_Aggregation_Ensemble']
    
    if not force and has_mean_agg and has_mean_agg_ens:
        return False
    
    # Get corresponding NPZ file
    npz_filename = json_path.name.replace('uq_benchmark_', 'all_metrics_')
    npz_filename = npz_filename.replace('.json', '.npz')
    npz_path = json_path.parent / npz_filename
    
    if not npz_path.exists():
        return False
    
    # Compute metrics for both modes
    updated = False
    
    # Per-fold metrics
    metrics_per_fold = compute_metrics_from_npz(npz_path, cache_dir, use_ensemble=False)
    if metrics_per_fold:
        data['methods']['Mean_Aggregation'] = metrics_per_fold
        updated = True
    
    # Ensemble metrics
    metrics_ensemble = compute_metrics_from_npz(npz_path, cache_dir, use_ensemble=True)
    if metrics_ensemble:
        data['methods']['Mean_Aggregation_Ensemble'] = metrics_ensemble
        updated = True
    
    if updated:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return updated


def main(force=False, aggregation='mean'):
    """Main function."""
    print("=" * 80)
    print("Pre-computing Mean Aggregation Scores")
    print(f"Aggregation: {aggregation.upper()}")
    print(f"Force recompute: {force}")
    print("=" * 80)
    
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    cache_dir = results_dir / 'cache'
    
    # Process NPZ files
    print("\n" + "=" * 80)
    print("Processing NPZ files...")
    print("=" * 80)
    
    subdirs = ['id', 'corruption_shifts', 'population_shifts', 'new_class_shifts']
    all_npz_files = []
    for subdir in subdirs:
        subdir_path = results_dir / subdir
        if subdir_path.exists():
            npz_files = sorted(subdir_path.glob('all_metrics_*.npz'))
            all_npz_files.extend(npz_files)
    
    if all_npz_files:
        updated_count = 0
        for npz_file in tqdm(all_npz_files, desc="Processing NPZ files"):
            try:
                if process_npz_file(npz_file, aggregation=aggregation, force=force):
                    updated_count += 1
            except Exception as e:
                print(f"\n❌ Error processing {npz_file.name}: {e}")
        
        print(f"\n✓ Updated {updated_count}/{len(all_npz_files)} NPZ files")
    
    # Process JSON files
    print("\n" + "=" * 80)
    print("Processing JSON files...")
    print("=" * 80)
    
    all_json_files = []
    for subdir in subdirs:
        subdir_path = results_dir / subdir
        if subdir_path.exists():
            json_files = sorted(subdir_path.glob('uq_benchmark_*.json'))
            all_json_files.extend(json_files)
    
    if all_json_files:
        updated_count = 0
        for json_file in tqdm(all_json_files, desc="Processing JSON files"):
            try:
                if process_json_file(json_file, results_dir, cache_dir, force=force):
                    updated_count += 1
            except Exception as e:
                print(f"\n❌ Error processing {json_file.name}: {e}")
        
        print(f"\n✓ Updated {updated_count}/{len(all_json_files)} JSON files")
    
    print("\n" + "=" * 80)
    print("✓ Pre-computation complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-compute Mean Aggregation scores')
    parser.add_argument('--force', action='store_true',
                       help='Force recompute even if Mean_Aggregation already exists')
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'min', 'max', 'vote'],
                       help='Aggregation strategy')
    
    args = parser.parse_args()
    
    main(force=args.force, aggregation=args.aggregation)

