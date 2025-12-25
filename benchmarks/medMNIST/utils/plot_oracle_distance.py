"""
Plot Normalized Gap to Oracle (Ngto) for each CSF and setup.

Ngto measures the fraction of avoidable silent-failure risk that remains relative 
to an oracle (perfect) scoring function. It ranges from 0 (oracle CSF) to 1 (random CSF).

Formula:
    AUGRC = (1 - AUROC-f) × acc × (1 - acc) + 0.5(1 - acc)²
    Oracle AUGRC (A*): AUROC-f = 1 → A* = 0.5(1 - acc)²
    Random AUGRC (A^rand): AUROC-f = 0.5 → A^rand = 0.5(1 - acc)
    Ngto = (A - A*) / (A^rand - A*)
"""

import json
import numpy as np
import sys
from pathlib import Path
import importlib.util

# Import compute_augrc dynamically to avoid torch dependency at module level
def _get_compute_augrc():
    """Lazy load compute_augrc function to avoid importing torch at module level."""
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    eval_module_path = workspace_root / 'FailCatcher' / 'evaluation' / 'evaluation.py'
    
    spec = importlib.util.spec_from_file_location("evaluation_module", eval_module_path)
    evaluation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation_module)
    return evaluation_module.compute_augrc

# Cache the function after first load
_compute_augrc_cached = None

def get_compute_augrc():
    global _compute_augrc_cached
    if _compute_augrc_cached is None:
        _compute_augrc_cached = _get_compute_augrc()
    return _compute_augrc_cached
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def compute_augrc(auroc_f, accuracy):
    """
    Compute AUGRC given AUROC-f and accuracy.
    
    Args:
        auroc_f: AUROC for failure detection (between 0 and 1)
        accuracy: Model accuracy (between 0 and 1)
    
    Returns:
        float: AUGRC = (1 - AUROC-f) × acc × (1 - acc) + 0.5(1 - acc)²
    """
    return (1.0 - auroc_f) * accuracy * (1.0 - accuracy) + 0.5 * (1.0 - accuracy) ** 2


def compute_oracle_augrc(accuracy):
    """
    Compute oracle AUGRC (AUROC-f = 1).
    
    Args:
        accuracy: Model accuracy (between 0 and 1)
    
    Returns:
        float: A* = 0.5 * (1 - accuracy)^2
    """
    return 0.5 * (1.0 - accuracy) ** 2


def compute_random_augrc(accuracy):
    """
    Compute random AUGRC (AUROC-f = 0.5).
    
    Args:
        accuracy: Model accuracy (between 0 and 1)
    
    Returns:
        float: A^rand = 0.5 * (1 - accuracy)
    """
    return 0.5 * (1.0 - accuracy)


def compute_ngto(auroc_f, accuracy):
    """
    Compute Normalized Gap to Oracle (Ngto).
    
    Args:
        auroc_f: AUROC for failure detection
        accuracy: Model accuracy
    
    Returns:
        float: Ngto = (A - A*) / (A^rand - A*), ranges from 0 (oracle) to 1 (random)
    """
    A = compute_augrc(auroc_f, accuracy)
    A_star = compute_oracle_augrc(accuracy)
    A_rand = compute_random_augrc(accuracy)
    
    denominator = A_rand - A_star
    if denominator == 0 or denominator < 1e-10:
        return 0.0
    
    ngto = (A - A_star) / denominator
    # Clamp to [0, 1] range
    return np.clip(ngto, 0.0, 1.0)


def load_ngto_data(results_dir, dataset_name, model_name='resnet18', setup='standard'):
    """
    Load per-fold and ensemble Ngto values for all CSFs from JSON files.
    
    Returns:
        dict: {
            'per_fold': {method_name: [ngto_fold0, ngto_fold1, ...]},
            'ensemble': {method_name: ngto_ensemble}
        }
    """
    results_dir = Path(results_dir)
    
    # Construct JSON filename pattern - using uq_benchmark format
    if setup == 'standard':
        pattern = f"uq_benchmark_{dataset_name}_{model_name}_*.json"
    else:
        pattern = f"uq_benchmark_{dataset_name}_{model_name}_{setup}_*.json"
    
    all_json_files = list(results_dir.glob(pattern))
    
    # Filter for standard setup (exclude DA, DO, DADO in filename)
    if setup == 'standard':
        json_files = [f for f in all_json_files if not any(
            f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO']
        )]
    else:
        json_files = all_json_files
    
    if not json_files:
        return {'per_fold': {}, 'ensemble': {}}
    
    # Use most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Use the top-level test_accuracy (consistent across all methods)
        # This is the ACCURACY (not balanced_accuracy) used for oracle_augrc computation
        test_accuracy = data.get('test_accuracy', np.nan)
        
        ngto_data = {'per_fold': {}, 'ensemble': {}}
        methods_dict = data.get('methods', {})
        
        for method_name, method_data in methods_dict.items():
            # Per-fold Ngto - CRITICAL: use each fold's accuracy for oracle/random, not ensemble accuracy
            # Each fold has its own test set with potentially different accuracy
            per_fold_metrics = method_data.get('per_fold_metrics', [])
            if per_fold_metrics:
                per_fold_ngto = []
                for i, fold in enumerate(per_fold_metrics):
                    augrc = fold.get('augrc', np.nan)  # A (actual)
                    fold_accuracy = fold.get('accuracy', np.nan)  # This fold's accuracy on its test set
                    
                    if not np.isnan(augrc) and not np.isnan(fold_accuracy):
                        # Use this fold's accuracy for oracle and random (not ensemble accuracy!)
                        oracle_augrc = compute_oracle_augrc(fold_accuracy)  # A* = 0.5(1-acc)^2
                        A_rand = compute_random_augrc(fold_accuracy)  # A^rand = 0.5(1-acc)
                        # Ngto = (A - A*) / (A^rand - A*)
                        ngto = (augrc - oracle_augrc) / (A_rand - oracle_augrc) if (A_rand - oracle_augrc) != 0 else np.nan
                        per_fold_ngto.append(ngto)
                    else:
                        per_fold_ngto.append(np.nan)
                
                ngto_data['per_fold'][method_name] = np.array(per_fold_ngto)
            
            # Ensemble Ngto - use test_accuracy for oracle/random
            augrc_ens = method_data.get('augrc', np.nan)  # A (actual)
            if not np.isnan(augrc_ens) and not np.isnan(test_accuracy):
                # Use consistent test_accuracy for oracle and random
                oracle_augrc = compute_oracle_augrc(test_accuracy)  # A* = 0.5(1-acc)^2
                A_rand = compute_random_augrc(test_accuracy)  # A^rand = 0.5(1-acc)
                # Ngto = (A - A*) / (A^rand - A*)
                ngto = (augrc_ens - oracle_augrc) / (A_rand - oracle_augrc) if (A_rand - oracle_augrc) != 0 else np.nan
                ngto_data['ensemble'][method_name] = ngto
        
        # Compute Mean_Aggregation (z-scored average of specific methods)
        # Load NPZ file to get raw uncertainties for z-score computation
        workspace_root = results_dir.parent.parent
        results_dir_base = workspace_root / 'uq_benchmark_results'
        
        # Parse dataset_name and setup to find NPZ file
        if setup == 'standard':
            npz_pattern = f"all_metrics_{dataset_name}_{model_name}_*.npz"
        else:
            npz_pattern = f"all_metrics_{dataset_name}_{model_name}_{setup}_*.npz"
        
        npz_search_dir = results_dir_base / 'id_results'
        all_npz_files = list(npz_search_dir.glob(npz_pattern))
        
        # Filter for standard setup
        if setup == 'standard':
            npz_files = [f for f in all_npz_files if not any(
                f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO']
            )]
        else:
            npz_files = all_npz_files
        
        if npz_files:
            npz_file = max(npz_files, key=lambda p: p.stat().st_mtime)
            
            try:
                npz_data = np.load(npz_file, allow_pickle=True)
                
                # Methods to aggregate (exclude TTA and Ensembling)
                methods_to_use = ['MSR', 'MSR_calibrated', 'MLS', 'GPS', 'KNN_Raw', 'MCDropout']
                per_fold_keys = [f'{m}_per_fold' for m in methods_to_use]
                
                # Check which keys exist (MCDropout only in DO/DADO)
                available_keys = [k for k in per_fold_keys if k in npz_data.keys()]
                
                if available_keys:
                    # Get number of folds
                    num_folds = len(npz_data[available_keys[0]])
                    
                    # Get cached test results for labels
                    cache_dir = results_dir_base / 'cache'
                    if setup == 'standard':
                        cache_pattern = f"{dataset_name}_{model_name}_test_results.npz"
                    else:
                        cache_pattern = f"{dataset_name}_{model_name}_{setup}_test_results.npz"
                    
                    cache_file_path = cache_dir / cache_pattern
                    if cache_file_path.exists():
                        cache_data = np.load(cache_file_path, allow_pickle=True)
                        per_fold_predictions = cache_data['per_fold_predictions']
                        y_true = cache_data['y_true']
                        
                        # Compute AUGRC for each fold's aggregated uncertainties
                        agg_per_fold_augrc = []
                        agg_per_fold_accuracy = []
                        
                        for fold_idx in range(num_folds):
                            # Z-score normalize each method for this fold
                            normalized_arrays = []
                            
                            for key in available_keys:
                                uncertainties = npz_data[key][fold_idx]
                                mean_val = np.mean(uncertainties)
                                std_val = np.std(uncertainties)
                                
                                if std_val > 0:
                                    z_score = (uncertainties - mean_val) / std_val
                                    normalized_arrays.append(z_score)
                            
                            if normalized_arrays:
                                # Mean aggregation
                                stacked = np.stack(normalized_arrays, axis=0)
                                aggregated = np.mean(stacked, axis=0)
                                
                                # Get predictions and compute accuracy for this fold
                                predictions = per_fold_predictions[fold_idx]
                                labels = y_true
                                fold_accuracy = np.mean(predictions == labels)
                                agg_per_fold_accuracy.append(fold_accuracy)
                                
                                # Compute AUGRC for this fold
                                compute_augrc = get_compute_augrc()
                                augrc_value, _ = compute_augrc(aggregated, predictions, labels)
                                agg_per_fold_augrc.append(augrc_value)
                        
                        # Compute per-fold Ngto for Mean_Aggregation
                        if agg_per_fold_augrc:
                            mean_agg_per_fold_ngto = []
                            for augrc, acc in zip(agg_per_fold_augrc, agg_per_fold_accuracy):
                                oracle_augrc = compute_oracle_augrc(acc)
                                A_rand = compute_random_augrc(acc)
                                ngto = (augrc - oracle_augrc) / (A_rand - oracle_augrc) if (A_rand - oracle_augrc) != 0 else np.nan
                                mean_agg_per_fold_ngto.append(ngto)
                            
                            ngto_data['per_fold']['Mean_Aggregation'] = np.array(mean_agg_per_fold_ngto)
                        
                        # Ensemble Mean_Aggregation - z-score and average ensemble uncertainties
                        ensemble_keys = [f'{m}_ensemble' for m in methods_to_use]
                        available_ensemble_keys = [k for k in ensemble_keys if k in npz_data.keys()]
                        
                        if available_ensemble_keys and cache_file_path.exists():
                            # Z-score normalize each method's ensemble uncertainties
                            normalized_ensemble = []
                            for key in available_ensemble_keys:
                                uncertainties = npz_data[key]
                                mean_val = np.mean(uncertainties)
                                std_val = np.std(uncertainties)
                                
                                if std_val > 0:
                                    z_score = (uncertainties - mean_val) / std_val
                                    normalized_ensemble.append(z_score)
                            
                            if normalized_ensemble:
                                # Mean aggregation
                                stacked = np.stack(normalized_ensemble, axis=0)
                                aggregated_ensemble = np.mean(stacked, axis=0)
                                
                                # Get ensemble predictions and labels
                                cache_data = np.load(cache_file_path, allow_pickle=True)
                                ensemble_predictions = cache_data['y_pred']  # Ensemble predictions
                                y_true = cache_data['y_true']
                                
                                # Compute AUGRC for ensemble
                                compute_augrc_func = get_compute_augrc()
                                ensemble_augrc, _ = compute_augrc_func(aggregated_ensemble, ensemble_predictions, y_true)
                                
                                # Convert to Ngto using test_accuracy (ensemble accuracy)
                                oracle_augrc = compute_oracle_augrc(test_accuracy)
                                A_rand = compute_random_augrc(test_accuracy)
                                ensemble_ngto = (ensemble_augrc - oracle_augrc) / (A_rand - oracle_augrc) if (A_rand - oracle_augrc) != 0 else np.nan
                                ngto_data['ensemble']['Mean_Aggregation'] = ensemble_ngto
                            
            except Exception as e:
                print(f"      Warning: Could not compute Mean_Aggregation for {dataset_name} {setup}: {e}")
        
        return ngto_data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {'per_fold': {}, 'ensemble': {}}


def plot_ngto_for_dataset(dataset_name, model_names, results_dir, ax):
    """
    Create Ngto histogram for one dataset with both ResNet18 and ViT models.
    
    Shows per-fold Ngto distribution as histograms and ensemble Ngto as diamond markers.
    
    Args:
        dataset_name: Base dataset name (e.g., 'breastmnist')
        model_names: List of model names ['resnet18', 'vit_b_16']
        results_dir: Path to in_distribution results directory
        ax: Matplotlib axis to plot on
    """
    setups = ['standard', 'DA', 'DO', 'DADO']
    # Methods sorted alphabetically - MCDropout only in DO/DADO, Mean_Aggregation at end
    methods = ['Ensembling', 'GPS', 'KNN_Raw', 'MLS', 'MSR', 'MSR_calibrated', 'TTA', 'Mean_Aggregation']
    methods_with_dropout = ['Ensembling', 'GPS', 'KNN_Raw', 'MCDropout', 'MLS', 'MSR', 'MSR_calibrated', 'TTA', 'Mean_Aggregation']
    
    # Define colors for each method (consistent with other plots)
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_with_dropout)))
    method_colors = {method: colors[i] for i, method in enumerate(methods_with_dropout)}
    
    # Set background
    ax.set_facecolor('#F8F9FA')
    
    # Add dataset name inside plot (top right)
    ax.text(0.98, 0.95, dataset_name.upper(), transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='gray', alpha=0.9, linewidth=1.5))
    
    # Collect all data
    all_data = []
    bar_width = 0.09
    
    x_offset = 0
    for model_idx, model_name in enumerate(model_names):
        for setup_idx, setup in enumerate(setups):
            # Load Ngto data
            ngto_data = load_ngto_data(results_dir, dataset_name, model_name, setup)
            
            # Select methods based on setup
            current_methods = methods_with_dropout if setup in ['DO', 'DADO'] else methods
            
            # Debug: print methods being used
            if model_idx == 0 and setup_idx == 0:
                print(f"  Setup {setup}: Iterating over {len(current_methods)} methods: {current_methods}")
            
            for method_idx, method in enumerate(current_methods):
                # Get per-fold Ngto
                per_fold_ngto = ngto_data['per_fold'].get(method, None)
                ensemble_ngto = ngto_data['ensemble'].get(method, None)
                
                # Debug
                if method_idx == 0 and setup_idx == 0 and model_idx == 0:
                    print(f"      Method {method}: per_fold={per_fold_ngto}, ensemble={ensemble_ngto}")
                
                # Special handling for Ensembling: only ensemble point (no per-fold bars)
                if method == 'Ensembling':
                    if ensemble_ngto is not None and not np.isnan(ensemble_ngto):
                        x_pos = x_offset + method_idx * bar_width
                        all_data.append({
                            'x_pos': x_pos,
                            'per_fold_ngto': np.array([]),  # Empty for Ensembling
                            'ensemble_ngto': ensemble_ngto,
                            'method': method,
                            'model': model_name,
                            'setup': setup,
                            'color': method_colors[method]
                        })
                    continue
                
                if per_fold_ngto is None or len(per_fold_ngto) == 0:
                    continue
                
                # Filter out NaN values
                valid_ngto = per_fold_ngto[~np.isnan(per_fold_ngto)]
                if len(valid_ngto) == 0:
                    continue
                
                x_pos = x_offset + method_idx * bar_width
                
                all_data.append({
                    'x_pos': x_pos,
                    'per_fold_ngto': valid_ngto,
                    'ensemble_ngto': ensemble_ngto,
                    'method': method,
                    'model': model_name,
                    'setup': setup,
                    'color': method_colors[method]
                })
            
            # Add spacing between setups
            x_offset += len(current_methods) * bar_width + 0.15
        
        # Add larger spacing between models
        x_offset += 0.25
    
    print(f"    Collected {len(all_data)} data points for plotting")
    
    # Debug: Check if Mean_Aggregation is in all_data
    mean_agg_entries = [d for d in all_data if d['method'] == 'Mean_Aggregation']
    ensembling_entries = [d for d in all_data if d['method'] == 'Ensembling']
    print(f"    Mean_Aggregation entries: {len(mean_agg_entries)}")
    print(f"    Ensembling entries: {len(ensembling_entries)}")
    if mean_agg_entries:
        first_ma = mean_agg_entries[0]
        print(f"    First Mean_Aggregation: model={first_ma['model']}, setup={first_ma['setup']}, "
              f"per_fold_len={len(first_ma['per_fold_ngto'])}, ensemble={first_ma['ensemble_ngto']:.4f}")
    
    # Plot bars: per-fold Ngto as boxplots/histograms
    for data in all_data:
        x_pos = data['x_pos']
        per_fold_ngto = data['per_fold_ngto']
        ensemble_ngto = data['ensemble_ngto']
        color = data['color']
        method = data['method']
        
        # Debug first few
        if len(all_data) <= 5 or all_data.index(data) < 2:
            if len(per_fold_ngto) > 0:
                print(f"      Plotting {method} at x={x_pos:.2f}, Ngto range=[{np.min(per_fold_ngto):.3f}, {np.max(per_fold_ngto):.3f}]")
            else:
                print(f"      Plotting {method} at x={x_pos:.2f}, Ngto range=[N/A]")
        
        # Create violin-like representation: show min, max, median
        # Skip bars for Ensembling (only show diamond)
        if len(per_fold_ngto) > 0 and method != 'Ensembling':
            min_val = np.min(per_fold_ngto)
            max_val = np.max(per_fold_ngto)
            median_val = np.median(per_fold_ngto)
            
            # Bar from min to max (fold variability)
            height = max_val - min_val
            ax.bar(x_pos, height, width=bar_width * 0.8, bottom=min_val,
                   color=color, edgecolor='gray', linewidth=1.0, alpha=0.3, zorder=2)
            
            # Median line
            ax.hlines(median_val, x_pos - bar_width * 0.4, x_pos + bar_width * 0.4,
                     colors='black', linewidth=2.0, zorder=3)
        
        # Plot ensemble marker for ALL methods (including Ensembling)
        if ensemble_ngto is not None and not np.isnan(ensemble_ngto):
            if method == 'Mean_Aggregation':
                # Lightning bolt marker for Mean_Aggregation (same as radar plots)
                ax.scatter(x_pos, ensemble_ngto, s=300, marker='$\u26A1$',
                          color=color, edgecolors='black', linewidths=1.5, zorder=4)
            else:
                # Diamond for other methods (including Ensembling)
                ax.plot(x_pos, ensemble_ngto, marker='D', markersize=8,
                       color=color, markeredgecolor='black', markeredgewidth=1.5, zorder=4)
    
    # Formatting
    # Determine y-axis range based on actual data
    all_values = []
    for data in all_data:
        all_values.extend(data['per_fold_ngto'])
        if data['ensemble_ngto'] is not None and not np.isnan(data['ensemble_ngto']):
            all_values.append(data['ensemble_ngto'])
    
    if all_values:
        max_ngto = max(all_values)
        y_max = max(1.1, max_ngto + 0.05)  # At least 1.1, or higher if needed
    else:
        y_max = 1.1
    
    ax.set_ylim(-0.05, y_max)
    ax.set_ylabel('Ngto (0=Oracle, 1=Random)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_xlim(-0.2, x_offset - 0.25)
    ax.set_xticks([])
    
    # Add horizontal reference lines
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Oracle (Ngto=0)')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random (Ngto=1)')
    
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=1.2, color='#BCC1C7', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', labelsize=12, width=2, length=6)


def main():
    """Generate Ngto plots for all ID datasets."""
    
    print("="*80)
    print("Ngto (Normalized Gap to Oracle) Plot Generator")
    print("="*80)
    
    # Setup paths - use uq_benchmark_results/id_results where the AUROC data is
    script_dir = Path(__file__).parent
    uq_results_dir = script_dir.parent.parent.parent / 'uq_benchmark_results' / 'id_results'
    output_dir = script_dir / 'oracle_plots'
    output_dir.mkdir(exist_ok=True)
    
    # ID datasets (excluding pathmnist)
    datasets = ['tissuemnist', 'dermamnist-e-id', 'breastmnist', 'pneumoniamnist', 
                'octmnist', 'organamnist', 'bloodmnist']
    
    # Models
    models = ['resnet18', 'vit_b_16']
    
    print()
    print("="*80)
    print(f"Creating Ngto plot with all {len(datasets)} datasets")
    print("="*80)
    
    # Create single figure with all datasets
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(24, 6*n_datasets), facecolor='white')
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Normalized Gap to Oracle (Ngto) - All ID Datasets', 
                 fontsize=28, fontweight='bold', y=0.995)
    
    # Plot each dataset
    for idx, dataset in enumerate(datasets):
        print(f"  Plotting {dataset}...")
        plot_ngto_for_dataset(dataset, models, uq_results_dir, axes[idx])
    
    # Add model and setup labels at the top
    ax_top = axes[0]
    setups = ['standard', 'DA', 'DO', 'DADO']
    methods = ['Ensembling', 'GPS', 'KNN_Raw', 'MLS', 'MSR', 'MSR_calibrated', 'TTA', 'Mean_Aggregation']
    methods_with_dropout = ['Ensembling', 'GPS', 'KNN_Raw', 'MCDropout', 'MLS', 'MSR', 'MSR_calibrated', 'TTA', 'Mean_Aggregation']
    bar_width = 0.09
    
    x_offset = 0
    for model_idx, model_name in enumerate(models):
        # Calculate model width
        model_width = 0
        for setup in setups:
            current_methods = methods_with_dropout if setup in ['DO', 'DADO'] else methods
            model_width += len(current_methods) * bar_width + 0.15
        model_width -= 0.15
        
        model_center = x_offset + model_width / 2
        # Convert to normalized coordinates
        xlim_range = ax_top.get_xlim()
        model_center_norm = (model_center - xlim_range[0]) / (xlim_range[1] - xlim_range[0])
        
        ax_top.text(model_center_norm, 1.18, model_name.replace('_', ' ').upper(), ha='center', va='bottom',
                   fontsize=16, fontweight='bold', transform=ax_top.transAxes,
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', edgecolor='darkblue', 
                            alpha=0.7, linewidth=2))
        
        for setup_idx, setup in enumerate(setups):
            current_methods = methods_with_dropout if setup in ['DO', 'DADO'] else methods
            setup_x_start = x_offset
            setup_width = len(current_methods) * bar_width
            center_x = setup_x_start + setup_width / 2
            
            # Convert to axis coordinates
            xlim_range = ax_top.get_xlim()
            center_x_norm = (center_x - xlim_range[0]) / (xlim_range[1] - xlim_range[0])
            
            ax_top.text(center_x_norm, 1.10, setup.upper(), ha='center', va='bottom', 
                       fontsize=12, fontweight='bold', transform=ax_top.transAxes,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8))
            
            x_offset += len(current_methods) * bar_width + 0.15
        
        x_offset += 0.25
    
    # Add legend - one line with darker colors matching the markers
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_with_dropout)))
    method_colors = {method: colors[i] for i, method in enumerate(methods_with_dropout)}
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=method_colors[method], 
                     edgecolor='black', linewidth=1.0, alpha=1.0)  # Fully opaque, darker
        for method in methods_with_dropout
    ]
    legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                                     markeredgecolor='black', markersize=8, 
                                     label='Ensemble Ngto'))
    
    # Place legend below first graph (tissuemnist)
    axes[0].legend(legend_elements, methods_with_dropout + ['Ensemble'], 
                   loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                   fontsize=12, ncol=len(methods_with_dropout)+1,  # One line
                   framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.992])
    plt.subplots_adjust(hspace=0.35)
    
    # Save plot
    output_path = output_dir / 'ngto_all_datasets.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print()
    print(f"✓ Saved {output_path}")
    plt.close(fig)
    
    print()
    print("="*80)
    print("✓ Ngto plot generated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
