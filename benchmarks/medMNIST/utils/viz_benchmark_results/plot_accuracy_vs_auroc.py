"""
Generate scatter plot showing relationship between balanced accuracy and AUROC_f.

For each fold, configuration, dataset, and method, plot:
- X-axis: Balanced accuracy of that fold
- Y-axis: AUROC_f of the method on that fold

Includes:
- Per-fold methods: GPS, TTA, KNN, MCD, MLS, MSR, MSR-S, Mean Agg (8 methods × 5 folds)
- Ensemble method: DE (1 point per dataset-config combination)
- All configurations: standard, DA, DO, DADO (4 configs)
- All backbones: resnet18, vit_b_16 (2 backbones)
- All shifts: in_distribution, corruption (standard datasets), population, new_class
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_per_fold_data(cache_file):
    """
    Load per-fold data from cache file.
    
    Returns:
        dict with keys:
            - per_fold_predictions: list of arrays (5 folds)
            - per_fold_correct_idx: list of arrays (5 folds)
            - per_fold_incorrect_idx: list of arrays (5 folds)
            - y_true: array of true labels
            - method_scores: dict of method_name -> per_fold scores
    """
    try:
        data = np.load(cache_file, allow_pickle=True)
        
        result = {
            'y_true': data['y_true'],
            'per_fold_predictions': data.get('per_fold_predictions', None),
            'per_fold_correct_idx': [np.asarray(idx, dtype=int) for idx in data.get('per_fold_correct_idx', [])],
            'per_fold_incorrect_idx': [np.asarray(idx, dtype=int) for idx in data.get('per_fold_incorrect_idx', [])],
            'correct_idx': data.get('correct_idx', None),
            'incorrect_idx': data.get('incorrect_idx', None),
        }
        
        return result
    except Exception as e:
        print(f"Failed to load {cache_file}: {e}")
        return None


def load_method_scores(metrics_file):
    """
    Load method scores from all_metrics npz file.
    
    Returns:
        dict of method_name -> per_fold scores (or ensemble scores)
    """
    try:
        data = np.load(metrics_file, allow_pickle=True)
        
        method_scores = {}
        for key in data.keys():
            if key.endswith('_per_fold'):
                method_name = key.replace('_per_fold', '')
                method_scores[method_name] = data[key]
            elif not key.endswith('_ensemble'):
                method_scores[key] = data[key]
        
        return method_scores
    except Exception as e:
        print(f"Failed to load {metrics_file}: {e}")
        return {}


def compute_balanced_accuracy_per_fold(predictions, y_true):
    """Compute balanced accuracy for predictions."""
    return balanced_accuracy_score(y_true, predictions)


def compute_auroc_f_per_fold(method_scores, correct_idx, incorrect_idx):
    """Compute AUROC_f for a method on a specific fold."""
    n_samples = len(correct_idx) + len(incorrect_idx)
    failure_labels = np.zeros(n_samples)
    failure_labels[incorrect_idx] = 1
    
    if len(method_scores) != n_samples:
        return np.nan
    
    try:
        return roc_auc_score(failure_labels, method_scores)
    except Exception:
        return np.nan


def parse_metrics_filename(filename):
    """
    Parse metrics filename to extract dataset, model, config.
    
    Examples:
        - all_metrics_breastmnist_resnet18_20251219_091146.npz -> (breastmnist, resnet18, standard)
        - all_metrics_breastmnist_resnet18_DA_20251219_091146.npz -> (breastmnist, resnet18, DA)
        - all_metrics_breastmnist_vit_b_16_20251219_091146.npz -> (breastmnist, vit_b_16, standard)
    """
    # Remove prefix and timestamp suffix
    name = filename.replace('all_metrics_', '').replace('.npz', '')
    
    # Remove timestamp (8 digits_6 digits at the end)
    import re
    name = re.sub(r'_\d{8}_\d{6}$', '', name)
    
    # Parse dataset, model, config
    parts = name.split('_')
    
    # Find model name (resnet18 or vit_b_16)
    if 'vit' in parts:
        model_idx = parts.index('vit')
        model = 'vit_b_16'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+3:]  # After 'vit', 'b', '16'
    elif 'resnet18' in parts:
        model_idx = parts.index('resnet18')
        model = 'resnet18'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+1:]  # After 'resnet18'
    else:
        return None, None, None
    
    # Determine config
    if config_parts and config_parts[0] in ['DA', 'DO', 'DADO']:
        config = config_parts[0]
    else:
        config = 'standard'
    
    return dataset, model, config


def parse_cache_filename(filename):
    """
    Parse cache filename to extract dataset, model, config, and shift info.
    
    Examples:
        - breastmnist_resnet18_test_results.npz -> (breastmnist, resnet18, standard, in_distribution)
        - breastmnist_resnet18_DA_test_results.npz -> (breastmnist, resnet18, DA, in_distribution)
        - breastmnist_resnet18_corrupt3_test_results.npz -> (breastmnist, resnet18, standard, corruption)
        - breastmnist_resnet18_DA_corrupt3_test_results.npz -> (breastmnist, resnet18, DA, corruption)
        - breastmnist_resnet18_new_class_shift_test_results.npz -> (breastmnist, resnet18, standard, new_class)
    """
    name = filename.replace('_test_results.npz', '').replace('_test_test_results.npz', '')
    
    # Check for shift types
    if 'new_class_shift' in name:
        shift = 'new_class'
        name = name.replace('_new_class_shift', '')
    elif 'corrupt3' in name:
        shift = 'corruption'
        name = name.replace('_corrupt3', '')
    else:
        shift = 'in_distribution'
    
    # Parse dataset, model, config
    parts = name.split('_')
    
    # Find model name (resnet18 or vit_b_16)
    if 'vit' in parts:
        model_idx = parts.index('vit') 
        model = 'vit_b_16'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+3:]  # After 'vit', 'b', '16'
    elif 'resnet18' in parts:
        model_idx = parts.index('resnet18')
        model = 'resnet18'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+1:]  # After 'resnet18'
    else:
        return None, None, None, None
    
    # Determine config
    if config_parts and config_parts[0] in ['DA', 'DO', 'DADO']:
        config = config_parts[0]
    else:
        config = 'standard'
    
    return dataset, model, config, shift


def collect_all_data_points(results_dir):
    """
    Collect all (balanced_accuracy, auroc_f) pairs across all methods, folds, configs, and shifts.
    
    Returns:
        dict: {
            'per_fold': list of (acc, auroc_f, method, dataset, model, config, shift, fold) tuples
            'ensemble': list of (acc, auroc_f, method, dataset, model, config, shift) tuples
        }
    """
    results_dir = Path(results_dir)
    cache_dir = results_dir / 'cache'
    
    per_fold_data = []
    ensemble_data = []
    
    # Define shifts to search (exclude new_class as balanced accuracy doesn't make sense for unseen classes)
    shifts_dirs = {
        'in_distribution': results_dir / 'id_results',
        'corruption': results_dir / 'corruption_shifts',
        'population': results_dir / 'population_shifts'
    }
    
    # Methods to include (per-fold) - only those that actually exist in the data files
    # MCDropout only exists in DO and DADO setups
    per_fold_methods = ['GPS', 'TTA', 'KNN_Raw', 'MCDropout', 'MLS', 'MSR', 'MSR_calibrated']
    
    for shift_name, shift_dir in shifts_dirs.items():
        if not shift_dir.exists():
            print(f"Skipping {shift_name}: directory not found")
            continue
        
        print(f"\nProcessing {shift_name}...")
        
        # Find all metrics files in this shift directory
        metrics_files = list(shift_dir.glob('all_metrics_*.npz'))
        print(f"  Found {len(metrics_files)} metrics files")
        
        for metrics_file in metrics_files:
            # Parse filename to get dataset, model, config
            dataset, model, config = parse_metrics_filename(metrics_file.name)
            if dataset is None:
                continue
            
            shift = shift_name
            
            # Find corresponding cache file
            if shift_name == 'in_distribution':
                if config == 'standard':
                    cache_patterns = [
                        f"{dataset}_{model}_test_test_results.npz",
                        f"{dataset}_{model}_test_results.npz"
                    ]
                else:
                    cache_patterns = [
                        f"{dataset}_{model}_{config}_test_test_results.npz",
                        f"{dataset}_{model}_{config}_test_results.npz"
                    ]
            elif shift_name == 'corruption':
                if config == 'standard':
                    cache_patterns = [
                        f"{dataset}_{model}_corrupt3_test_test_results.npz",
                        f"{dataset}_{model}_corrupt3_test_results.npz"
                    ]
                else:
                    cache_patterns = [
                        f"{dataset}_{model}_{config}_corrupt3_test_test_results.npz",
                        f"{dataset}_{model}_{config}_corrupt3_test_results.npz"
                    ]
            elif shift_name == 'new_class':
                if config == 'standard':
                    cache_patterns = [
                        f"{dataset}_{model}_new_class_shift_test_results.npz"
                    ]
                else:
                    cache_patterns = [
                        f"{dataset}_{model}_{config}_new_class_shift_test_results.npz"
                    ]
            else:  # population
                if config == 'standard':
                    cache_patterns = [
                        f"{dataset}_{model}_test_test_results.npz",
                        f"{dataset}_{model}_test_results.npz"
                    ]
                else:
                    cache_patterns = [
                        f"{dataset}_{model}_{config}_test_test_results.npz",
                        f"{dataset}_{model}_{config}_test_results.npz"
                    ]
            
            cache_file = None
            for pattern in cache_patterns:
                candidate = cache_dir / pattern
                if candidate.exists():
                    cache_file = candidate
                    break
            
            if cache_file is None:
                continue
            
            # Load data
            cache_data = load_per_fold_data(cache_file)
            method_scores = load_method_scores(metrics_file)
            
            if cache_data is None or not method_scores:
                continue
            
            # Process per-fold methods
            if cache_data['per_fold_predictions'] is not None:
                num_folds = len(cache_data['per_fold_predictions'])
                
                for fold_idx in range(num_folds):
                    # Compute balanced accuracy for this fold
                    predictions = cache_data['per_fold_predictions'][fold_idx]
                    y_true = cache_data['y_true']
                    fold_acc = compute_balanced_accuracy_per_fold(predictions, y_true)
                    
                    # Get correct/incorrect indices for this fold
                    correct_idx = cache_data['per_fold_correct_idx'][fold_idx]
                    incorrect_idx = cache_data['per_fold_incorrect_idx'][fold_idx]
                    
                    # Collect scores for Mean_Aggregation computation
                    mean_agg_methods = ['GPS', 'KNN_Raw', 'MLS', 'MSR', 'MSR_calibrated', 'MCDropout']
                    mean_agg_scores = []
                    
                    # Process each method
                    for method in per_fold_methods:
                        if method not in method_scores:
                            continue
                        
                        method_fold_scores = method_scores[method][fold_idx]
                        auroc_f = compute_auroc_f_per_fold(method_fold_scores, correct_idx, incorrect_idx)
                        
                        if not np.isnan(auroc_f):
                            per_fold_data.append({
                                'balanced_accuracy': fold_acc,
                                'auroc_f': auroc_f,
                                'method': method,
                                'dataset': dataset,
                                'model': model,
                                'config': config,
                                'shift': shift_name,
                                'fold': fold_idx
                            })
                        
                        # Collect for Mean_Aggregation if applicable
                        if method in mean_agg_methods:
                            mean_val = np.mean(method_fold_scores)
                            std_val = np.std(method_fold_scores)
                            if std_val > 0:
                                z_score = (method_fold_scores - mean_val) / std_val
                                mean_agg_scores.append(z_score)
                    
                    # Compute Mean_Aggregation if we have at least 3 methods
                    if len(mean_agg_scores) >= 3:
                        aggregated = np.mean(np.stack(mean_agg_scores, axis=0), axis=0)
                        mean_agg_auroc = compute_auroc_f_per_fold(aggregated, correct_idx, incorrect_idx)
                        
                        if not np.isnan(mean_agg_auroc):
                            per_fold_data.append({
                                'balanced_accuracy': fold_acc,
                                'auroc_f': mean_agg_auroc,
                                'method': 'Mean_Aggregation',
                                'dataset': dataset,
                                'model': model,
                                'config': config,
                                'shift': shift_name,
                                'fold': fold_idx
                            })
            
            # Process ensemble method (Ensembling/DE)
            if 'Ensembling' in method_scores and cache_data['correct_idx'] is not None:
                # Get ensemble predictions (use y_pred from cache if available)
                if 'y_pred' in np.load(cache_file, allow_pickle=True):
                    ensemble_pred = np.load(cache_file, allow_pickle=True)['y_pred']
                    ensemble_acc = compute_balanced_accuracy_per_fold(ensemble_pred, cache_data['y_true'])
                else:
                    # Fall back to mean of per-fold accuracies
                    if cache_data['per_fold_predictions'] is not None:
                        fold_accs = []
                        for fold_idx in range(len(cache_data['per_fold_predictions'])):
                            predictions = cache_data['per_fold_predictions'][fold_idx]
                            fold_acc = compute_balanced_accuracy_per_fold(predictions, cache_data['y_true'])
                            fold_accs.append(fold_acc)
                        ensemble_acc = np.mean(fold_accs)
                    else:
                        continue
                
                ensemble_scores = method_scores['Ensembling']
                correct_idx = cache_data['correct_idx']
                incorrect_idx = cache_data['incorrect_idx']
                
                auroc_f = compute_auroc_f_per_fold(ensemble_scores, correct_idx, incorrect_idx)
                
                if not np.isnan(auroc_f):
                    ensemble_data.append({
                        'balanced_accuracy': ensemble_acc,
                        'auroc_f': auroc_f,
                        'method': 'DE',
                        'dataset': dataset,
                        'model': model,
                        'config': config,
                        'shift': shift_name
                    })
    
    print(f"\n{'='*80}")
    print(f"Data collection complete:")
    print(f"  Per-fold points: {len(per_fold_data)}")
    print(f"  Ensemble points: {len(ensemble_data)}")
    print(f"  Total points: {len(per_fold_data) + len(ensemble_data)}")
    print(f"{'='*80}")
    
    return {'per_fold': per_fold_data, 'ensemble': ensemble_data}


def create_scatter_plot(data, output_dir):
    """
    Create scatter plot of balanced accuracy vs AUROC_f.
    
    Args:
        data: dict with 'per_fold' and 'ensemble' lists
        output_dir: directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Combine all data
    all_data = data['per_fold'] + data['ensemble']
    
    # Get methods with ORIGINAL names for color assignment (to match radar plots)
    methods_original = sorted(set(d['method'] for d in all_data))
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_original)))
    method_colors_original = dict(zip(methods_original, colors))
    
    # Now rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'MCDropout': 'MCD',
        'Mean_Aggregation': 'Mean Agg'
    }
    
    for d in all_data:
        if d['method'] in name_mapping:
            d['method'] = name_mapping[d['method']]
    
    # Create color map with display names
    method_colors = {}
    for orig_name, color in method_colors_original.items():
        display_name = name_mapping.get(orig_name, orig_name)
        method_colors[display_name] = color
    
    # Extract unique methods for plotting (now with display names)
    methods = sorted(set(d['method'] for d in all_data))
    
    # Plot each method
    for method in methods:
        method_data = [d for d in all_data if d['method'] == method]
        x = [d['balanced_accuracy'] for d in method_data]
        y = [d['auroc_f'] for d in method_data]
        
        # Use different markers for different method types
        if method == 'Mean Agg':
            # Lightning bolt marker matching radar plot style
            ax.scatter(x, y, s=150, marker='$⚡$', c=[method_colors[method]],
                      label='Mean Agg', alpha=0.8, edgecolors='black', linewidths=0.3)
            continue
        elif method == 'DE':
            marker = 's'  # Square for ensemble
            alpha = 0.7
            size = 50
            edgecolor = 'white'
            linewidth = 0.5
        else:
            marker = 'o'  # Circle for others
            alpha = 0.7
            size = 20
            edgecolor = 'white'
            linewidth = 0.5
        
        ax.scatter(x, y, c=[method_colors[method]], marker=marker, s=size, 
                  alpha=alpha, label=method, edgecolors=edgecolor, linewidths=linewidth)
    
    # Add diagonal reference line (perfect correlation)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=1, label='y=x reference')
    
    ax.set_xlabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUROC (Failure Detection)', fontsize=14, fontweight='bold')
    ax.set_title('Relationship between Model Accuracy and Failure Detection Performance\nAcross All Methods, Configurations, and Shifts', 
                fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper left', ncol=2, fontsize=10, frameon=True, framealpha=0.9, markerscale=2.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    output_path = output_dir / 'accuracy_vs_auroc_scatter.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Scatter plot saved to {output_path}")
    
    plt.close(fig)
    
    # Also create separate plots by shift type
    create_scatter_by_shift(data, output_dir)


def create_scatter_by_shift(data, output_dir):
    """Create separate scatter plots for each shift type."""
    all_data = data['per_fold'] + data['ensemble']
    
    # Get methods with ORIGINAL names for color assignment (to match radar plots)
    methods_original = sorted(set(d['method'] for d in all_data))
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_original)))
    method_colors_original = dict(zip(methods_original, colors))
    
    # Now rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'MCDropout': 'MCD',
        'Mean_Aggregation': 'Mean Agg'
    }
    
    for d in all_data:
        if d['method'] in name_mapping:
            d['method'] = name_mapping[d['method']]
    
    # Create color map with display names
    method_colors = {}
    for orig_name, color in method_colors_original.items():
        display_name = name_mapping.get(orig_name, orig_name)
        method_colors[display_name] = color
    
    # Custom order for shifts: in_distribution first, then corruption, then population
    shift_order = ['in_distribution', 'corruption', 'population']
    shifts = [s for s in shift_order if s in set(d['shift'] for d in all_data)]
    
    # Create figure with custom layout: 2 top, legend bottom left, population bottom right
    fig = plt.figure(figsize=(16, 12))
    
    # Top row: 2 subplots
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    # Bottom right: population subplot
    ax3 = plt.subplot(2, 2, 4)
    
    axes_list = [ax1, ax2, ax3]
    
    # Extract unique methods for plotting (now with display names)
    methods = sorted(set(d['method'] for d in all_data))
    

    for idx, shift in enumerate(shifts):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]
        shift_data = [d for d in all_data if d['shift'] == shift]
        
        for method in methods:
            method_data = [d for d in shift_data if d['method'] == method]
            if not method_data:
                continue
            
            x = [d['balanced_accuracy'] for d in method_data]
            y = [d['auroc_f'] for d in method_data]
            
            # Use different markers for different method types
            if method == 'Mean Agg':
                # Lightning bolt marker matching radar plot style
                ax.scatter(x, y, s=150, marker='$⚡$', c=[method_colors[method]],
                          label='Mean Agg', alpha=0.8, edgecolors='black', linewidths=0.3)
                continue
            elif method == 'DE':
                marker = 's'  # Square for ensemble
                alpha = 0.7
                size = 50
                edgecolor = 'white'
                linewidth = 0.5
            else:
                marker = 'o'  # Circle for others
                alpha = 0.7
                size = 20
                edgecolor = 'white'
                linewidth = 0.5
            
            ax.scatter(x, y, c=[method_colors[method]], marker=marker, s=size,
                      alpha=alpha, label=method, edgecolors=edgecolor, linewidths=linewidth)
        
        # Add diagonal reference
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=1)
        
        ax.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUROC (Failure Detection)', fontsize=12, fontweight='bold')
        
        # Add "Shift" suffix for corruption and population
        title = shift.replace("_", " ").title()
        if shift in ['corruption', 'population']:
            title += ' Shifts'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add shared legend in bottom left position
    # Collect handles and labels from first subplot
    handles, labels = axes_list[0].get_legend_handles_labels()
    # Create invisible subplot for legend in bottom left
    ax_legend = plt.subplot(2, 2, 3)
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=2, fontsize=12, 
                    frameon=True, framealpha=0.95, title='Methods', title_fontsize=14, markerscale=2.0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'accuracy_vs_auroc_by_shift.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Shift-wise scatter plot saved to {output_path}")
    
    plt.close(fig)


def main():
    """Main function."""
    print("=" * 80)
    print("Balanced Accuracy vs AUROC_f Scatter Plot Generator")
    print("=" * 80)
    
    # Get workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    
    print(f"Workspace root: {workspace_root}")
    print(f"Results directory: {results_dir}")
    
    # Create output directory
    output_dir = script_dir / 'scatter_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Collect all data points
    data = collect_all_data_points(results_dir)
    
    # Create scatter plots
    create_scatter_plot(data, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
