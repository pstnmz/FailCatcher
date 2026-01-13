"""
Plot AUGRC values for each CSF and setup, showing oracle and random boundaries.

Shows actual AUGRC values with reference to:
- Oracle AUGRC (A*): AUROC-f = 1 → A* = 0.5(1 - acc)² (best, lowest)
- Random AUGRC (A^rand): AUROC-f = 0.5 → A^rand = 0.5(1 - acc) (worst, highest)

For each method, displays:
- Box spanning from oracle to random AUGRC (theoretical bounds)
- Horizontal line at mean per-fold AUGRC
- Diamond/lightning marker at ensemble AUGRC
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


def compute_augrc_from_auroc(auroc_f, accuracy):
    """
    Compute AUGRC given AUROC-f and accuracy (wrapper for clarity).
    
    Args:
        auroc_f: AUROC for failure detection
        accuracy: Model accuracy
    
    Returns:
        float: AUGRC value
    """
    return compute_augrc(auroc_f, accuracy)


def load_augrc_data(results_dir, dataset_name, model_name='resnet18', setup='standard'):
    """
    Load per-fold and ensemble AUGRC values for all CSFs from JSON files.
    Also computes oracle and random AUGRC boundaries based on test accuracy.
    
    Returns:
        dict: {
            'per_fold': {method_name: [augrc_fold0, augrc_fold1, ...]},
            'ensemble': {method_name: augrc_ensemble},
            'oracle_augrc': float (A* = 0.5(1-acc)²),
            'random_augrc': float (A^rand = 0.5(1-acc)),
            'test_accuracy': float
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
        
        # Compute mean per-fold accuracy from methods' per-fold metrics
        # (uq_benchmark files don't have top-level per_fold_metrics)
        methods_dict = data.get('methods', {})
        per_fold_accuracies = []
        
        # Get per-fold accuracies from the first method that has them
        for method_name, method_data in methods_dict.items():
            per_fold_metrics = method_data.get('per_fold_metrics', [])
            if per_fold_metrics and len(per_fold_metrics) > 0:
                # Extract accuracy from each fold
                for fold in per_fold_metrics:
                    acc = fold.get('accuracy', np.nan)
                    if not np.isnan(acc):
                        per_fold_accuracies.append(acc)
                break  # Use first method with data
        
        # Compute mean accuracy
        if per_fold_accuracies:
            # Assuming all methods have same number of folds, compute mean across folds
            num_folds = len(per_fold_accuracies)
            mean_per_fold_accuracy = np.mean(per_fold_accuracies)
        else:
            # Fallback to ensemble accuracy if no per-fold data
            mean_per_fold_accuracy = data.get('test_accuracy', np.nan)
        
        # Compute oracle and random AUGRC bounds using mean per-fold accuracy
        oracle_augrc = compute_oracle_augrc(mean_per_fold_accuracy)
        random_augrc = compute_random_augrc(mean_per_fold_accuracy)
        
        augrc_data = {
            'per_fold': {}, 
            'ensemble': {},
            'oracle_augrc': oracle_augrc,
            'random_augrc': random_augrc,
            'test_accuracy': mean_per_fold_accuracy
        }

        
        for method_name, method_data in methods_dict.items():
            # Per-fold AUGRC - use each fold's AUGRC directly from JSON
            per_fold_metrics = method_data.get('per_fold_metrics', [])
            if per_fold_metrics:
                per_fold_augrc = []
                for i, fold in enumerate(per_fold_metrics):
                    augrc = fold.get('augrc', np.nan)
                    if not np.isnan(augrc):
                        per_fold_augrc.append(augrc)
                    else:
                        per_fold_augrc.append(np.nan)
                
                augrc_data['per_fold'][method_name] = np.array(per_fold_augrc)
            
            # Ensemble AUGRC - use ensemble AUGRC from JSON
            augrc_ens = method_data.get('augrc', np.nan)
            if not np.isnan(augrc_ens):
                augrc_data['ensemble'][method_name] = augrc_ens
        
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
                                
                                # Get predictions for this fold
                                predictions = per_fold_predictions[fold_idx]
                                labels = y_true
                                
                                # Compute AUGRC for this fold
                                compute_augrc_func = get_compute_augrc()
                                from FailCatcher.evaluation.evaluation import compute_auroc
                                
                                # Compute correct/incorrect indices for this fold
                                fold_correct = np.where(predictions == labels)[0]
                                fold_incorrect = np.where(predictions != labels)[0]
                                
                                if len(fold_correct) > 0 and len(fold_incorrect) > 0:
                                    auroc_f = compute_auroc(aggregated, fold_correct, fold_incorrect)
                                    # Compute fold accuracy
                                    fold_acc = len(fold_correct) / len(predictions)
                                    # Compute AUGRC
                                    fold_augrc = compute_augrc_from_auroc(auroc_f, fold_acc)
                                    agg_per_fold_augrc.append(fold_augrc)
                                else:
                                    agg_per_fold_augrc.append(np.nan)
                        
                        # Store per-fold AUGRC for Mean_Aggregation
                        if agg_per_fold_augrc:
                            augrc_data['per_fold']['Mean_Aggregation'] = np.array(agg_per_fold_augrc)
                        
                        # Ensemble Mean_Aggregation - compute AUROC_f from ensemble uncertainties
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
                                
                                # Compute AUROC_f for ensemble
                                from FailCatcher.evaluation.evaluation import compute_auroc
                                ensemble_correct = np.where(ensemble_predictions == y_true)[0]
                                ensemble_incorrect = np.where(ensemble_predictions != y_true)[0]
                                
                                if len(ensemble_correct) > 0 and len(ensemble_incorrect) > 0:
                                    auroc_f_ens = compute_auroc(aggregated_ensemble, ensemble_correct, ensemble_incorrect)
                                    # Compute ensemble accuracy
                                    ens_acc = len(ensemble_correct) / len(ensemble_predictions)
                                    # Compute ensemble AUGRC
                                    ensemble_augrc = compute_augrc_from_auroc(auroc_f_ens, ens_acc)
                                    augrc_data['ensemble']['Mean_Aggregation'] = ensemble_augrc
                            
            except Exception as e:
                print(f"      Warning: Could not compute Mean_Aggregation for {dataset_name} {setup}: {e}")
        
        return augrc_data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {'per_fold': {}, 'ensemble': {}, 'oracle_augrc': np.nan, 'random_augrc': np.nan, 'test_accuracy': np.nan}


def plot_augrc_for_dataset(dataset_name, model_names, results_dir, ax):
    """
    Create AUGRC plot for one dataset with both ResNet18 and ViT models.
    
    Shows boxes spanning oracle to random AUGRC, with mean per-fold AUGRC as line
    and ensemble AUGRC as diamond/lightning markers.
    
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
            # Load AUGRC data
            augrc_data = load_augrc_data(results_dir, dataset_name, model_name, setup)
            oracle_augrc = augrc_data['oracle_augrc']
            random_augrc = augrc_data['random_augrc']
            
            # Select methods based on setup
            current_methods = methods_with_dropout if setup in ['DO', 'DADO'] else methods
            
            for method_idx, method in enumerate(current_methods):
                # Get per-fold AUGRC
                per_fold_augrc = augrc_data['per_fold'].get(method, None)
                ensemble_augrc = augrc_data['ensemble'].get(method, None)
                
                # Special handling for Ensembling: only ensemble point (no per-fold bars)
                if method == 'Ensembling':
                    if ensemble_augrc is not None and not np.isnan(ensemble_augrc):
                        x_pos = x_offset + method_idx * bar_width
                        all_data.append({
                            'x_pos': x_pos,
                            'per_fold_augrc': np.array([]),  # Empty for Ensembling
                            'ensemble_augrc': ensemble_augrc,
                            'method': method,
                            'model': model_name,
                            'setup': setup,
                            'color': method_colors[method],
                            'oracle_augrc': oracle_augrc,
                            'random_augrc': random_augrc
                        })
                    continue
                
                if per_fold_augrc is None or len(per_fold_augrc) == 0:
                    continue
                
                # Filter out NaN values
                valid_augrc = per_fold_augrc[~np.isnan(per_fold_augrc)]
                if len(valid_augrc) == 0:
                    continue
                
                x_pos = x_offset + method_idx * bar_width
                
                all_data.append({
                    'x_pos': x_pos,
                    'per_fold_augrc': valid_augrc,
                    'ensemble_augrc': ensemble_augrc,
                    'method': method,
                    'model': model_name,
                    'setup': setup,
                    'color': method_colors[method],
                    'oracle_augrc': oracle_augrc,
                    'random_augrc': random_augrc
                })
            
            # Add spacing between setups
            x_offset += len(current_methods) * bar_width + 0.15
        
        # Add larger spacing between models
        x_offset += 0.25
    
    # Plot boxes and markers
    for data in all_data:
        x_pos = data['x_pos']
        per_fold_augrc = data['per_fold_augrc']
        ensemble_augrc = data['ensemble_augrc']
        color = data['color']
        method = data['method']
        oracle_augrc = data['oracle_augrc']
        random_augrc = data['random_augrc']
        
        # Draw box from oracle (bottom, best) to random (top, worst) AUGRC
        if not np.isnan(oracle_augrc) and not np.isnan(random_augrc):
            box_height = random_augrc - oracle_augrc
            ax.bar(x_pos, box_height, width=bar_width * 0.8, bottom=oracle_augrc,
                   color=color, edgecolor='gray', linewidth=1.0, alpha=0.15, zorder=1)
        
        # Plot mean per-fold AUGRC as horizontal line
        if len(per_fold_augrc) > 0 and method != 'Ensembling':
            mean_augrc = np.mean(per_fold_augrc)
            ax.hlines(mean_augrc, x_pos - bar_width * 0.4, x_pos + bar_width * 0.4,
                     colors='black', linewidth=2.5, zorder=3)
        
        # Plot ensemble marker for ALL methods (including Ensembling)
        if ensemble_augrc is not None and not np.isnan(ensemble_augrc):
            if method == 'Mean_Aggregation':
                # Lightning bolt marker for Mean_Aggregation
                ax.scatter(x_pos, ensemble_augrc, s=300, marker='$\u26A1$',
                          color=color, edgecolors='black', linewidths=1.5, zorder=4)
            else:
                # Diamond for other methods (including Ensembling)
                ax.plot(x_pos, ensemble_augrc, marker='D', markersize=8,
                       color=color, markeredgecolor='black', markeredgewidth=1.5, zorder=4)
    
    # Formatting
    # Determine y-axis range based on actual data (bars span oracle to random)
    all_values = []
    oracle_values = []
    random_values = []
    for data in all_data:
        all_values.extend(data['per_fold_augrc'])
        if data['ensemble_augrc'] is not None and not np.isnan(data['ensemble_augrc']):
            all_values.append(data['ensemble_augrc'])
        if not np.isnan(data['oracle_augrc']):
            oracle_values.append(data['oracle_augrc'])
        if not np.isnan(data['random_augrc']):
            random_values.append(data['random_augrc'])
    
    # Set y-axis limits based on min/max of oracle and random bounds
    if oracle_values and random_values:
        min_oracle = min(oracle_values)
        max_random = max(random_values)
        # Add 5% padding above and below
        y_range = max_random - min_oracle
        y_min = max(0, min_oracle - 0.05 * y_range)
        y_max = max_random + 0.05 * y_range
    else:
        y_min = 0
        y_max = 0.15
    
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('AUGRC (lower is better)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_xlim(-0.2, x_offset - 0.25)
    ax.set_xticks([])
    
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=1.2, color='#BCC1C7', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', labelsize=12, width=2, length=6)


def main():
    """Generate AUGRC plots for all ID datasets."""
    
    print("="*80)
    print("AUGRC Plot Generator (Oracle vs Random Boundaries)")
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
    print(f"Creating AUGRC plot with all {len(datasets)} datasets")
    print("="*80)
    
    # Create single figure with all datasets
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(24, 6*n_datasets), facecolor='white')
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('AUGRC with Oracle and Random Boundaries - All ID Datasets', 
                 fontsize=28, fontweight='bold', y=0.995)
    
    # Plot each dataset
    for idx, dataset in enumerate(datasets):
        print(f"  Plotting {dataset}...")
        plot_augrc_for_dataset(dataset, models, uq_results_dir, axes[idx])
    
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
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='lightgray', 
                     edgecolor='gray', linewidth=1.0, alpha=0.15,
                     label='Oracle→Random bounds'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2.5, 
                                     label='Mean per-fold AUGRC'))
    legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                                     markeredgecolor='black', markersize=8, 
                                     label='Ensemble AUGRC'))
    
    # Place legend below first graph (tissuemnist)
    axes[0].legend(legend_elements, methods_with_dropout + ['Oracle→Random', 'Mean per-fold', 'Ensemble'], 
                   loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                   fontsize=12, ncol=len(methods_with_dropout)+3,  # One line
                   framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.992])
    plt.subplots_adjust(hspace=0.35)
    
    # Save plot
    output_path = output_dir / 'augrc_oracle_bounds_all_datasets.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print()
    print(f"✓ Saved {output_path}")
    plt.close(fig)
    
    print()
    print("="*80)
    print("✓ AUGRC plot generated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
