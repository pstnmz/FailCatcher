"""
Update Mean_Aggregation metrics in JSON files for POPULATION SHIFTS.
Uses FailCatcher evaluation functions with correct indices from cache files.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add FailCatcher to path
failcatcher_path = Path(__file__).resolve().parents[4] / 'FailCatcher'
sys.path.insert(0, str(failcatcher_path))
from evaluation.evaluation import compute_all_metrics_per_fold, compute_all_metrics

def find_cache_file(json_path: Path, cache_dir: Path) -> Path:
    """Find the corresponding cache file for a JSON file."""
    # Parse JSON filename to extract info
    filename = json_path.stem  # Remove .json
    filename = filename.replace('uq_benchmark_', '')
    
    # For population_shifts: {dataset}_{model}_{setup}_{date}_{time}
    # Cache: {dataset}_{model}_{setup}_test_results.npz (if setup exists)
    #        {dataset}_{model}_test_results.npz (if no setup)
    
    parts = filename.split('_')
    
    # Remove timestamp: last TWO parts (date YYYYMMDD and time HHMMSS)
    # Check if last parts look like timestamp (8 digits and 6 digits)
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 6 and parts[-2].isdigit() and len(parts[-2]) == 8:
        parts = parts[:-2]  # Remove both date and time
    
    # Rejoin and add _test_results.npz
    filename_no_timestamp = '_'.join(parts)
    cache_name = filename_no_timestamp + '_test_results.npz'
    
    cache_file = cache_dir / cache_name
    return cache_file

def update_json_file(json_path: Path, npz_path: Path, cache_dir: Path) -> bool:
    """Update Mean_Aggregation metrics in a JSON file using evaluation functions."""
    
    # Load NPZ data with aggregation scores
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # Check if Mean_Aggregation exists
    if 'Mean_Aggregation' not in npz_data and 'Mean_Aggregation_Ensemble' not in npz_data:
        return False
    
    # Load cache file with predictions and indices
    cache_file = find_cache_file(json_path, cache_dir)
    
    if not cache_file.exists():
        print(f"  ⚠ Warning: Cache file not found: {cache_file.name}")
        return False
    
    cache_data = np.load(cache_file, allow_pickle=True)
    
    # Get predictions and labels
    y_true = cache_data['y_true']
    y_pred = cache_data['y_pred']
    per_fold_predictions = cache_data['per_fold_predictions']
    
    # Get indices
    per_fold_correct_idx = cache_data['per_fold_correct_idx']
    per_fold_incorrect_idx = cache_data['per_fold_incorrect_idx']
    ensemble_correct_idx = cache_data['correct_idx']
    ensemble_incorrect_idx = cache_data['incorrect_idx']
    
    # Load JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    modified = False
    
    # Get Mean_Aggregation scores
    mean_agg_per_fold = npz_data.get('Mean_Aggregation')
    mean_agg_ensemble = npz_data.get('Mean_Aggregation_Ensemble')
    
    # Compute metrics for Mean_Aggregation (per-fold, WITHOUT ensemble_uncertainties)
    if mean_agg_per_fold is not None:
        results_perfold = compute_all_metrics_per_fold(
            uncertainties_per_fold=mean_agg_per_fold,
            predictions=y_pred,
            labels=y_true,
            predictions_per_fold=per_fold_predictions,
            ensemble_uncertainties=None,  # Will use mean of per-fold for ensemble AUROC
            per_fold_correct_idx=per_fold_correct_idx,
            per_fold_incorrect_idx=per_fold_incorrect_idx,
            ensemble_correct_idx=ensemble_correct_idx,  # Needed for ensemble AUROC computation
            ensemble_incorrect_idx=ensemble_incorrect_idx  # Needed for ensemble AUROC computation
        )
        
        # Compute standard deviations across folds
        per_fold_auroc_f = [fold['auroc_f'] for fold in results_perfold['per_fold_metrics']]
        per_fold_aurc = [fold['aurc'] for fold in results_perfold['per_fold_metrics']]
        per_fold_augrc = [fold['augrc'] for fold in results_perfold['per_fold_metrics']]
        auroc_f_std = float(np.std(per_fold_auroc_f))
        aurc_std = float(np.std(per_fold_aurc))
        augrc_std = float(np.std(per_fold_augrc))
        
        json_data['methods']['Mean_Aggregation'] = {
            'auroc_f': results_perfold['auroc_f'],  # AUROC of mean(per-fold scores)
            'auroc_f_mean': results_perfold['auroc_f_mean'],  # Mean of per-fold AUROC-F
            'auroc_f_std': auroc_f_std,  # Std of per-fold AUROC-F
            'aurc': results_perfold['aurc'],  # AURC of mean(per-fold scores)
            'aurc_mean': results_perfold['aurc_mean'],  # Mean of per-fold AURC
            'aurc_std': aurc_std,  # Std of per-fold AURC
            'augrc': results_perfold['augrc'],  # AUGRC of mean(per-fold scores)
            'augrc_mean': results_perfold['augrc_mean'],  # Mean of per-fold AUGRC
            'augrc_std': augrc_std  # Std of per-fold AUGRC
        }
        if 'per_fold_metrics' in results_perfold:
            json_data['methods']['Mean_Aggregation']['per_fold_metrics'] = results_perfold['per_fold_metrics']
        modified = True
    
    # Compute metrics for Mean_Aggregation_Ensemble (TRUE ensemble with Ensembling method included)
    if mean_agg_ensemble is not None:
        # Use compute_all_metrics directly for ensemble (not per-fold)
        ensemble_metrics = compute_all_metrics(
            uncertainties=mean_agg_ensemble,  # TRUE ensemble uncertainties (includes Ensembling)
            predictions=y_pred,
            labels=y_true,
            correct_idx=ensemble_correct_idx,
            incorrect_idx=ensemble_incorrect_idx
        )
        
        json_data['methods']['Mean_Aggregation_Ensemble'] = {
            'auroc_f': ensemble_metrics['auroc_f'],  # TRUE ensemble AUROC-F (with Ensembling)
            'aurc': ensemble_metrics['aurc'],
            'augrc': ensemble_metrics['augrc']
        }
        modified = True
    
    # Save updated JSON
    if modified:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        return True
    
    return False

def process_directory(directory: Path, cache_dir: Path) -> None:
    """Process all JSON files in a directory."""
    json_files = list(directory.glob('uq_benchmark_*.json'))
    
    print(f"\nFound {len(json_files)} JSON files to process")
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            # Find corresponding NPZ file
            npz_name = json_file.name.replace('uq_benchmark_', 'all_metrics_')
            npz_file = directory / npz_name.replace('.json', '.npz')
            
            if not npz_file.exists():
                print(f"  ⚠ NPZ file not found for {json_file.name}")
                skipped_count += 1
                continue
            
            was_modified = update_json_file(json_file, npz_file, cache_dir)
            
            if was_modified:
                print(f"✓ Updated: {json_file.name}")
                updated_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    print(f"\nSummary:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped (no changes or missing files): {skipped_count}")
    print(f"  Errors: {error_count}")

def main():
    ps_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/full_results/population_shifts')
    cache_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/cache')
    
    print("=" * 80)
    print("Updating Mean_Aggregation Metrics in JSON Files - POPULATION SHIFTS")
    print("=" * 80)
    print(f"Directory: {ps_dir}")
    print(f"Cache: {cache_dir}")
    
    process_directory(ps_dir, cache_dir)
    
    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
