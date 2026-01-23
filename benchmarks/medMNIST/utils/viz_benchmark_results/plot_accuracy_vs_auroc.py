"""
Generate scatter plot showing relationship between balanced accuracy and AUROC_f.

For each fold, configuration, dataset, and method, plot:
- X-axis: Balanced accuracy of that fold
- Y-axis: AUROC_f of the method on that fold

Loads pre-computed results from:
- comprehensive_evaluation_results/*.json (classification metrics including balanced accuracy)
- uq_benchmark_results/all_metrics_*.npz (AUROC_f values)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import warnings
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pandas as pd
warnings.filterwarnings('ignore')


def load_comprehensive_metrics(json_file):
    """
    Load comprehensive metrics from JSON file.
    
    Returns:
        dict with balanced accuracy per fold and ensemble
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle two different JSON structures:
        # 1. Standard format: {'per_fold_metrics': [...], 'ensemble_metrics': {...}}
        # 2. Corruption format: {'per_fold': [...], 'ensemble': {...}}
        
        per_fold_acc = []
        
        # Try standard format first
        if 'per_fold_metrics' in data:
            for fold_data in data['per_fold_metrics']:
                if 'balanced_accuracy' in fold_data:
                    per_fold_acc.append(fold_data['balanced_accuracy'])
        # Try corruption format
        elif 'per_fold' in data:
            for fold_data in data['per_fold']:
                if 'balanced_accuracy' in fold_data:
                    per_fold_acc.append(fold_data['balanced_accuracy'])
        
        # Extract ensemble balanced accuracy
        ensemble_acc = None
        if 'ensemble_metrics' in data and 'balanced_accuracy' in data['ensemble_metrics']:
            ensemble_acc = data['ensemble_metrics']['balanced_accuracy']
        elif 'ensemble' in data and 'balanced_accuracy' in data['ensemble']:
            ensemble_acc = data['ensemble']['balanced_accuracy']
        
        return {
            'per_fold_acc': per_fold_acc,
            'ensemble_acc': ensemble_acc
        }
    except Exception as e:
        print(f"Failed to load {json_file}: {e}")
        return None


def compute_auroc_f_from_scores(scores, y_true, y_pred):
    """
    Compute AUROC_f from uncertainty scores and predictions.
    
    Args:
        scores: uncertainty scores (higher = more uncertain)
        y_true: true labels
        y_pred: predicted labels
    
    Returns:
        AUROC_f value
    """
    # Create failure labels (1 = incorrect, 0 = correct)
    failures = (y_true != y_pred).astype(int)
    
    try:
        return roc_auc_score(failures, scores)
    except:
        return np.nan


def find_cache_file(cache_dir, dataset, model, config, shift_name):
    """Find the test_test cache file for a given configuration."""
    if shift_name == 'in_distribution':
        if config == 'standard':
            patterns = [f"{dataset}_{model}_test_test_results.npz", f"{dataset}_{model}_test_results.npz"]
        else:
            patterns = [f"{dataset}_{model}_{config}_test_test_results.npz", f"{dataset}_{model}_{config}_test_results.npz"]
    elif shift_name == 'corruption':
        if config == 'standard':
            patterns = [f"{dataset}_{model}_corrupt3_test_test_results.npz", f"{dataset}_{model}_corrupt3_test_results.npz"]
        else:
            patterns = [f"{dataset}_{model}_{config}_corrupt3_test_test_results.npz", f"{dataset}_{model}_{config}_corrupt3_test_results.npz"]
    else:  # population
        if config == 'standard':
            patterns = [f"{dataset}_{model}_test_test_results.npz", f"{dataset}_{model}_test_results.npz"]
        else:
            patterns = [f"{dataset}_{model}_{config}_test_test_results.npz", f"{dataset}_{model}_{config}_test_results.npz"]
    
    for pattern in patterns:
        cache_file = cache_dir / pattern
        if cache_file.exists():
            return cache_file
    return None


def load_cache_data(cache_file):
    """Load y_true, y_pred, and per_fold_predictions from cache file."""
    try:
        data = np.load(cache_file, allow_pickle=True)
        return {
            'y_true': data['y_true'],
            'y_pred': data.get('y_pred', None),
            'per_fold_predictions': data.get('per_fold_predictions', None)
        }
    except:
        return None


def load_uq_metrics(json_file):
    """
    Load UQ metrics from JSON file.
    Also loads Mean_Aggregation data from corresponding NPZ file.
    
    Returns:
        dict of method_name -> {per_fold: [...], ensemble: scalar}
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        methods_data = {}
        
        # Process methods (note: key is 'methods' not 'results')
        if 'methods' in data:
            for method_name, method_results in data['methods'].items():
                methods_data[method_name] = {}
                
                # Get per-fold AUROC_f from per_fold_metrics array
                if 'per_fold_metrics' in method_results:
                    per_fold_auroc = [fold['auroc_f'] for fold in method_results['per_fold_metrics']]
                    methods_data[method_name]['per_fold'] = per_fold_auroc
                
                # Get ensemble AUROC_f (top-level auroc_f is the ensemble result)
                if 'auroc_f' in method_results:
                    methods_data[method_name]['ensemble'] = method_results['auroc_f']
        
        # Load Mean_Aggregation data from NPZ file
        if 'metrics_file' in data:
            npz_file = data['metrics_file']
            if Path(npz_file).exists():
                try:
                    npz_data = np.load(npz_file, allow_pickle=True)
                    
                    # Mean_Aggregation: per-fold scores (shape: [n_folds, n_samples])
                    if 'Mean_Aggregation' in npz_data:
                        mean_agg_scores = npz_data['Mean_Aggregation']
                        # Compute AUROC_f per fold
                        methods_data['Mean_Aggregation'] = {'per_fold': []}
                        
                    # Mean_Aggregation_Ensemble: ensemble scores (shape: [n_samples])
                    if 'Mean_Aggregation_Ensemble' in npz_data:
                        methods_data['Mean_Aggregation_Ensemble'] = {'ensemble': None}
                        # Will compute AUROC_f when we have correct/incorrect indices
                        
                except Exception as e:
                    print(f"Warning: Could not load NPZ file {npz_file}: {e}")
        
        return methods_data
    except Exception as e:
        print(f"Failed to load {json_file}: {e}")
        return {}


def parse_filename_to_key(filename):
    """
    Parse filename to extract (dataset, model, config).
    
    Args:
        filename: filename to parse (JSON format)
    
    Returns:
        tuple: (dataset, model, config)
    """
    import re
    
    # Handle different filename patterns:
    # 1. comprehensive_metrics_bloodmnist_resnet18_DADO.json
    # 2. uq_benchmark_bloodmnist_resnet18_20260116_095229.json
    # 3. bloodmnist_resnet18_DADO_severity3.json (corruption)
    
    if filename.startswith('comprehensive_metrics_'):
        name = filename.replace('comprehensive_metrics_', '').replace('.json', '')
    elif filename.startswith('uq_benchmark_'):
        name = filename.replace('uq_benchmark_', '').replace('.json', '')
        # Remove timestamp
        name = re.sub(r'_\d{8}_\d{6}$', '', name)
    else:
        # Corruption format: {dataset}_{model}_{config}_severity3.json
        name = filename.replace('.json', '').replace('_severity3', '')
    
    # Parse parts
    parts = name.split('_')
    
    # Find model name
    if 'vit' in parts:
        model_idx = parts.index('vit')
        model = 'vit_b_16'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+3:]
    elif 'resnet18' in parts:
        model_idx = parts.index('resnet18')
        model = 'resnet18'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+1:]
    else:
        return None
    
    # Determine config
    if config_parts and config_parts[0] in ['DA', 'DO', 'DADO']:
        config = config_parts[0]
    else:
        config = 'standard'
    
    return (dataset, model, config)


def collect_all_data_points(workspace_root):
    """
    Collect all (balanced_accuracy, auroc_f) pairs from pre-computed JSON results.
    
    Returns:
        dict: {
            'per_fold': list of data dicts
            'ensemble': list of data dicts
        }
    """
    comprehensive_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results'
    uq_results_dir = workspace_root / 'uq_benchmark_results'
    
    per_fold_data = []
    ensemble_data = []
    
    # Shifts to process
    shift_dirs = {
        'in_distribution': ('in_distribution', 'in_distribution'),
        'corruption': ('corruption_shifts', 'corruption_shifts'),
        'population': ('population_shift', 'population_shifts')
    }
    
    for shift_name, (comprehensive_subdir, uq_subdir) in shift_dirs.items():
        comprehensive_shift_dir = comprehensive_dir / comprehensive_subdir
        uq_shift_dir = uq_results_dir / uq_subdir
        
        if not comprehensive_shift_dir.exists() or not uq_shift_dir.exists():
            print(f"Skipping {shift_name}: directories not found")
            continue
        
        print(f"\nProcessing {shift_name}...")
        
        # Find all comprehensive metrics JSON files
        # Different naming patterns for different shifts:
        # - in_distribution & population: comprehensive_metrics_*.json
        # - corruption: {dataset}_{model}_{config}_severity3.json
        if shift_name == 'corruption':
            json_files = list(comprehensive_shift_dir.glob('*_severity3.json'))
        else:
            json_files = list(comprehensive_shift_dir.glob('comprehensive_metrics_*.json'))
        print(f"  Found {len(json_files)} classification JSON files")
        
        for json_file in json_files:
            # Parse filename
            key = parse_filename_to_key(json_file.name)
            if key is None:
                continue
            
            dataset, model, config = key
            
            # Handle dataset name mismatches between comprehensive and UQ files
            # (comprehensive uses different names than UQ for population shifts)
            dataset_mapping = {
                'amos22': 'amos2022',
                'dermamnist-e-ood': 'dermamnist-e-external'
            }
            uq_dataset = dataset_mapping.get(dataset, dataset)
            
            # Load classification metrics
            class_metrics = load_comprehensive_metrics(json_file)
            if class_metrics is None or not class_metrics['per_fold_acc']:
                continue
            
            # Find corresponding UQ metrics JSON file
            if shift_name == 'corruption':
                # Corruption files have special naming: uq_benchmark_{dataset}_{model}_{config}_corrupt_severity3_test_*.json
                if config != 'standard':
                    uq_json_files = list(uq_shift_dir.glob(f'uq_benchmark_{uq_dataset}_{model}_{config}_corrupt_severity3_*.json'))
                else:
                    uq_json_files = list(uq_shift_dir.glob(f'uq_benchmark_{uq_dataset}_{model}_corrupt_severity3_*.json'))
            else:
                # Standard naming for in_distribution and population
                uq_json_files = list(uq_shift_dir.glob(f'uq_benchmark_{uq_dataset}_{model}_*.json'))
            
            if shift_name != 'corruption':
                if config != 'standard':
                    uq_json_files = [f for f in uq_json_files if f'{model}_{config}_' in f.name]
                else:
                    uq_json_files = [f for f in uq_json_files if not any(c in f.name for c in ['_DA_', '_DO_', '_DADO_'])]
            
            if not uq_json_files:
                continue
            
            # Use the most recent file
            uq_json_file = sorted(uq_json_files)[-1]
            
            # Load UQ metrics
            uq_metrics = load_uq_metrics(uq_json_file)
            if not uq_metrics:
                continue
            
            # Load cache file for computing Mean_Aggregation AUROC_f
            # Use mapped dataset name for cache files (for population shifts)
            cache_dataset = uq_dataset if shift_name == 'population' else dataset
            cache_file = find_cache_file(uq_shift_dir.parent / 'cache', cache_dataset, model, config, shift_name)
            if cache_file and cache_file.exists():
                cache_data = load_cache_data(cache_file)
                
                # Compute Mean_Aggregation AUROC_f if available
                if 'Mean_Aggregation' in uq_metrics and cache_data is not None:
                    # Find the corresponding all_metrics NPZ file
                    npz_pattern = uq_json_file.name.replace('uq_benchmark_', 'all_metrics_').replace('.json', '.npz')
                    npz_file = uq_shift_dir / npz_pattern
                    
                    if npz_file.exists():
                        try:
                            npz_data = np.load(npz_file, allow_pickle=True)
                            if 'Mean_Aggregation' in npz_data and cache_data['per_fold_predictions'] is not None:
                                mean_agg_scores_all_folds = npz_data['Mean_Aggregation']  # shape: (n_folds, n_samples)
                                
                                # Compute per-fold AUROC_f
                                per_fold_auroc = []
                                for fold_idx in range(len(cache_data['per_fold_predictions'])):
                                    y_true = cache_data['y_true']
                                    y_pred = cache_data['per_fold_predictions'][fold_idx]
                                    scores = mean_agg_scores_all_folds[fold_idx]
                                    auroc = compute_auroc_f_from_scores(scores, y_true, y_pred)
                                    per_fold_auroc.append(auroc)
                                
                                uq_metrics['Mean_Aggregation']['per_fold'] = per_fold_auroc
                            
                            if 'Mean_Aggregation_Ensemble' in npz_data and cache_data['y_pred'] is not None:
                                ensemble_scores = npz_data['Mean_Aggregation_Ensemble']  # shape: (n_samples,)
                                y_true = cache_data['y_true']
                                y_pred = cache_data['y_pred']
                                auroc = compute_auroc_f_from_scores(ensemble_scores, y_true, y_pred)
                                uq_metrics['Mean_Aggregation_Ensemble']['ensemble'] = auroc
                        except Exception as e:
                            pass
            
            # Process per-fold data
            num_folds = len(class_metrics['per_fold_acc'])
            
            for method_name, method_data in uq_metrics.items():
                if 'per_fold' not in method_data:
                    continue
                
                auroc_per_fold = method_data['per_fold']
                
                # Match folds
                for fold_idx in range(min(num_folds, len(auroc_per_fold))):
                    fold_acc = class_metrics['per_fold_acc'][fold_idx]
                    fold_auroc = auroc_per_fold[fold_idx]
                    
                    if not np.isnan(fold_acc) and not np.isnan(fold_auroc):
                        per_fold_data.append({
                            'balanced_accuracy': fold_acc,
                            'auroc_f': fold_auroc,
                            'method': method_name,
                            'dataset': dataset,
                            'model': model,
                            'config': config,
                            'shift': shift_name,
                            'fold': fold_idx
                        })
            
            # Process ensemble data (use ensemble classification accuracy and ensemble UQ score)
            if class_metrics['ensemble_acc'] is not None:
                for method_name, method_data in uq_metrics.items():
                    if 'ensemble' in method_data:
                        ensemble_acc = class_metrics['ensemble_acc']
                        ensemble_auroc = method_data['ensemble']
                        
                        if not np.isnan(ensemble_acc) and not np.isnan(ensemble_auroc):
                            ensemble_data.append({
                                'balanced_accuracy': ensemble_acc,
                                'auroc_f': ensemble_auroc,
                                'method': method_name,
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
    fig, ax = plt.subplots(figsize=(8, 8))
    
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
    
    # Extract unique methods - reorder for display but keep color mapping consistent
    all_methods = set(d['method'] for d in all_data)
    mean_agg_methods = ['Mean Agg', 'Mean_Aggregation_Ensemble']
    regular_methods = sorted([m for m in all_methods if m not in mean_agg_methods])
    methods = regular_methods + [m for m in mean_agg_methods if m in all_methods]
    
    # Plot each method
    for method in methods:
        method_data = [d for d in all_data if d['method'] == method]
        x = [d['balanced_accuracy'] for d in method_data]
        y = [d['auroc_f'] for d in method_data]
        
        # Use different markers for different method types
        if method == 'Mean Agg':
            # Red star for Mean Aggregation
            ax.scatter(x, y, s=150, marker='*', c='red',
                      label='Mean Agg', alpha=0.8, edgecolors='darkred', linewidths=0.5)
            continue
        elif method == 'Mean_Aggregation_Ensemble':
            # Lightning bolt for Mean Aggregation + Ensemble
            ax.scatter(x, y, s=150, marker='$⚡$', c='orange',
                      label='Mean Agg+Ens', alpha=0.8, edgecolors='black', linewidths=0.3)
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
    #ax.set_title('Relationship between Model Accuracy and Failure Detection Performance\nAcross All Methods, Configurations, and Shifts', 
     #           fontsize=16, fontweight='bold', pad=20)
    
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
    
    # Extract unique methods - reorder for display but keep color mapping consistent
    all_methods = set(d['method'] for d in all_data)
    mean_agg_methods = ['Mean Agg', 'Mean_Aggregation_Ensemble']
    regular_methods = sorted([m for m in all_methods if m not in mean_agg_methods])
    methods = regular_methods + [m for m in mean_agg_methods if m in all_methods]

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
                # Red star for Mean Aggregation
                ax.scatter(x, y, s=150, marker='*', c='red',
                          label='Mean Agg', alpha=0.8, edgecolors='darkred', linewidths=0.5)
                continue
            elif method == 'Mean_Aggregation_Ensemble':
                # Lightning bolt for Mean Aggregation + Ensemble
                ax.scatter(x, y, s=150, marker='$⚡$', c='orange',
                          label='Mean Agg+Ens', alpha=0.8, edgecolors='black', linewidths=0.3)
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


def create_per_method_correlation_plots(data, output_dir):
    """
    Create a single large figure with 3 shift types stacked vertically.
    Each shift has a 2-row × 5-column grid of method subplots.
    
    Args:
        data: dict with 'per_fold' and 'ensemble' lists
        output_dir: directory to save plots
    """
    all_data = data['per_fold'] + data['ensemble']
    
    # Rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'MCDropout': 'MCD',
        'Mean_Aggregation': 'Mean Agg'
    }
    
    for d in all_data:
        if d['method'] in name_mapping:
            d['method'] = name_mapping[d['method']]
    
    # Define shift order and titles
    shift_info = [
        ('in_distribution', 'In-Distribution'),
        ('corruption', 'Corruption Shifts'),
        ('population', 'Population Shifts')
    ]
    
    # Get all methods across all shifts
    all_methods = set(d['method'] for d in all_data)
    
    # Create colors based on ALPHABETICAL order (for consistency with other plots)
    methods_for_colors = sorted(all_methods)
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_for_colors)))
    method_colors = dict(zip(methods_for_colors, colors))
    
    # Reorder methods for display (Mean Agg methods at end) but keep color mapping
    mean_agg_methods = ['Mean Agg', 'Mean_Aggregation_Ensemble']
    regular_methods = sorted([m for m in all_methods if m not in mean_agg_methods])
    methods = regular_methods + [m for m in mean_agg_methods if m in all_methods]
    
    n_methods = len(methods)
    n_cols = 5
    n_rows_per_shift = (n_methods + n_cols - 1) // n_cols
    n_shifts = len(shift_info)
    
    # Create large figure with nested GridSpec for custom spacing
    fig = plt.figure(figsize=(20, 4*n_rows_per_shift*n_shifts))
    
    import matplotlib.gridspec as gridspec
    # Outer GridSpec: one row per shift group with larger gaps between them
    outer_gs = gridspec.GridSpec(n_shifts, 1, figure=fig, hspace=0.28)
    
    # Create axes array manually
    total_rows = n_rows_per_shift * n_shifts
    axes = np.empty((total_rows, n_cols), dtype=object)
    
    # For each shift, create an inner GridSpec with tight spacing
    for shift_idx in range(n_shifts):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows_per_shift, n_cols, 
            subplot_spec=outer_gs[shift_idx],
            hspace=0.21, wspace=0.3
        )
        
        # Fill in the axes
        for row_in_shift in range(n_rows_per_shift):
            for col in range(n_cols):
                row = shift_idx * n_rows_per_shift + row_in_shift
                axes[row, col] = fig.add_subplot(inner_gs[row_in_shift, col])
    
    for shift_idx, (shift_type, shift_title) in enumerate(shift_info):
        shift_data = [d for d in all_data if d['shift'] == shift_type]
        
        if not shift_data:
            continue
        
        # Get methods available for this shift
        shift_methods = [m for m in methods if any(d['method'] == m for d in shift_data)]
        
        for method_idx, method in enumerate(shift_methods):
            row_in_shift = method_idx // n_cols
            col = method_idx % n_cols
            # Calculate absolute row index
            row = shift_idx * n_rows_per_shift + row_in_shift
            ax = axes[row, col]
            
            # Get data for this method
            method_data = [d for d in shift_data if d['method'] == method]
            
            x = np.array([d['balanced_accuracy'] for d in method_data])
            y = np.array([d['auroc_f'] for d in method_data])
            
            # Plot scatter with special handling for Mean Agg methods
            if method == 'Mean Agg':
                ax.scatter(x, y, s=150, marker='*', c='red',
                          alpha=0.8, edgecolors='darkred', linewidths=0.5)
            elif method == 'Mean_Aggregation_Ensemble':
                ax.scatter(x, y, s=150, marker='$⚡$', c='orange',
                          alpha=0.8, edgecolors='black', linewidths=0.3)
            elif method == 'DE':
                ax.scatter(x, y, c=[method_colors[method]], marker='s', s=100,
                          alpha=0.7, edgecolors='white', linewidths=1)
            else:
                ax.scatter(x, y, c=[method_colors[method]], marker='o', s=50,
                          alpha=0.7, edgecolors='white', linewidths=0.5)
            
            # Compute and display correlation
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                
                # Add trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)
                
                # Add correlation text
                ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Add x=y diagonal line
            ax.plot([0.4, 1.0], [0.4, 1.0], 'k--', alpha=0.3, linewidth=1, zorder=0)
            
            # Only show labels on specific subplots to reduce clutter
            # X-label: only on row 1 (second row of first shift) - leftmost subplot
            if row in [1, 3, 5]:# and col == 0:
                ax.set_xlabel('Balanced Accuracy', fontsize=11, fontweight='bold')
            else:
                ax.set_xlabel('')
            
            # Y-label: only on first subplot of each pair of rows (row 0, 2, 4)
            if col == 0:# and row_in_shift == 0:
                ax.set_ylabel('AUROC_f', fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            # Add shift label to first subplot of each shift
            if method_idx == 2:
                ax.text(0.5, 1.15, shift_title, transform=ax.transAxes,
                       fontsize=16, fontweight='bold', ha='center')
            
            ax.set_title(method, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set consistent limits
            ax.set_xlim([0.4, 1.0])
            ax.set_ylim([0.4, 1.0])
        
        # Hide unused subplots in this shift's grid
        n_used = len(shift_methods)
        for idx in range(n_used, n_rows_per_shift * n_cols):
            row_in_shift = idx // n_cols
            col = idx % n_cols
            row = shift_idx * n_rows_per_shift + row_in_shift
            axes[row, col].axis('off')
    
    # fig.suptitle('Method-Specific Accuracy vs AUROC_f Correlations Across All Shift Types',
    #             fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    # Note: hspace is already set in GridSpec
    
    # Save merged figure
    output_path = output_dir / 'per_method_correlation_all_shifts.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Merged per-method correlation plot saved to {output_path}")
    
    plt.close(fig)


def create_method_correlation_pairplots(data, output_dir):
    """
    Create pairplots showing correlations between methods for each shift type.
    Each pairplot shows how different methods' AUROC_f scores correlate with each other.
    
    Note: Separates per-fold and ensemble results since they represent different test cases.
    
    Args:
        data: dict with 'per_fold' and 'ensemble' lists
        output_dir: directory to save plots
    """
    # Rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'MCDropout': 'MCD',
        'Mean_Aggregation': 'Mean Agg'
    }
    
    # Define ensemble methods (these should only correlate with other ensemble results)
    ensemble_methods = ['DE', 'Mean_Aggregation_Ensemble']
    
    # Process each shift type separately
    shift_info = [
        ('in_distribution', 'In-Distribution'),
        ('corruption', 'Corruption Shifts'),
        ('population', 'Population Shifts')
    ]
    
    for shift_type, shift_title in shift_info:
        # Process per-fold data (exclude ensemble methods)
        per_fold_data = [d for d in data['per_fold'] if d['shift'] == shift_type]
        
        # Rename methods
        for d in per_fold_data:
            if d['method'] in name_mapping:
                d['method'] = name_mapping[d['method']]
        
        if not per_fold_data:
            continue
        
        # Create a DataFrame where each row is a unique test case (dataset, config, fold)
        # and each column is a method's AUROC_f score
        test_cases = defaultdict(dict)
        for d in per_fold_data:
            # Exclude ensemble methods from per-fold data
            if d['method'] in ensemble_methods:
                continue
            key = (d['dataset'], d['config'], d.get('fold', 'NA'))
            test_cases[key][d['method']] = d['auroc_f']
        
        # Convert to DataFrame
        rows = []
        for (dataset, config, fold), method_scores in test_cases.items():
            row = {'dataset': dataset, 'config': config, 'fold': fold}
            row.update(method_scores)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Get method columns (exclude metadata columns and ensemble methods)
        method_cols = [col for col in df.columns 
                      if col not in ['dataset', 'config', 'fold'] 
                      and col not in ensemble_methods]
        
        # Only keep methods that have data for at least some test cases
        method_cols = [col for col in method_cols if df[col].notna().sum() > 0]
        
        # Sort methods alphabetically (to match color scheme)
        method_cols = sorted(method_cols)
        
        print(f"\n{shift_title}: Creating pairplot with {len(method_cols)} per-fold methods and {len(df)} test cases")
        
        # Create pairplot
        if len(method_cols) < 2:
            print(f"  Skipping {shift_title} - need at least 2 methods")
            continue
        
        # Set up the plot style
        sns.set_style("whitegrid")
        
        # Create pairplot with smaller markers and transparency
        g = sns.pairplot(
            df[method_cols],
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'none'},
            diag_kws={'linewidth': 2},
            corner=False
        )
        
        # Add title
        g.fig.suptitle(f'{shift_title}: Per-Fold Method AUROC_f Correlations', 
                      fontsize=16, fontweight='bold', y=1.0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'method_correlation_pairplot_{shift_type}.png'
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Pairplot saved to {output_path}")
        
        plt.close()


def create_sample_level_pairplots(workspace_root, output_dir):
    """
    Create pairplots showing correlations between methods at the sample level.
    Each point is an individual sample's uncertainty score across all methods.
    
    Args:
        workspace_root: path to workspace root
        output_dir: directory to save plots
    """
    # Rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'MCDropout': 'MCD',
        'Mean_Aggregation': 'Mean Agg'
    }
    
    # Define ensemble methods to exclude
    ensemble_methods = ['DE', 'Mean_Aggregation_Ensemble']
    
    # Process each shift type
    shift_info = [
        ('in_distribution', 'In-Distribution'),
        ('corruption_shifts', 'Corruption Shifts'),
        ('population_shifts', 'Population Shifts')
    ]
    
    for shift_dir_name, shift_title in shift_info:
        print(f"\n{shift_title}: Loading sample-level scores...")
        
        uq_shift_dir = workspace_root / 'uq_benchmark_results' / shift_dir_name
        
        if not uq_shift_dir.exists():
            print(f"  Directory not found: {uq_shift_dir}")
            continue
        
        # Collect all sample-level scores across all NPZ files
        # Structure: {(dataset, config, fold, sample_idx): {method: score}}
        sample_scores = defaultdict(dict)
        
        # Find all NPZ files
        npz_files = list(uq_shift_dir.glob('all_metrics_*.npz'))
        
        for npz_file in npz_files:
            # Parse filename to get dataset, model, config
            filename = npz_file.name
            if filename.startswith('all_metrics_'):
                name = filename.replace('all_metrics_', '').replace('.json', '').replace('.npz', '')
                
                # Remove timestamp and severity patterns
                import re
                name = re.sub(r'_\d{8}_\d{6}$', '', name)
                name = name.replace('_corrupt3', '')
                name = name.replace('_corrupt_severity3', '')
                
                parts = name.split('_')
                
                # Find model
                if 'resnet18' in parts:
                    model_idx = parts.index('resnet18')
                    model = 'resnet18'
                elif 'vit' in parts and 'b' in parts and '16' in parts:
                    model_idx = parts.index('vit')
                    model = 'vit_b_16'
                else:
                    continue
                
                dataset = '_'.join(parts[:model_idx])
                config_parts = parts[model_idx+1:] if model == 'resnet18' else parts[model_idx+3:]
                config = config_parts[0] if config_parts and config_parts[0] in ['DA', 'DO', 'DADO'] else 'standard'
                
                # Load NPZ data
                try:
                    npz_data = np.load(npz_file, allow_pickle=True)
                    
                    # Get cache file for y_true
                    cache_dir = uq_shift_dir.parent / 'cache'
                    shift_name_short = shift_dir_name.replace('_shifts', '')
                    cache_file = find_cache_file(cache_dir, dataset, model, config, shift_name_short)
                    
                    if not cache_file or not cache_file.exists():
                        continue
                    
                    cache_data = load_cache_data(cache_file)
                    if cache_data is None or cache_data['per_fold_predictions'] is None:
                        continue
                    
                    n_folds = len(cache_data['per_fold_predictions'])
                    n_samples = len(cache_data['y_true'])
                    
                    # Process each method in NPZ file
                    for method_name in npz_data.keys():
                        # Only keep _per_fold variants
                        if not method_name.endswith('_per_fold'):
                            continue
                        
                        # Remove _per_fold suffix for display name
                        base_method_name = method_name.replace('_per_fold', '')
                        
                        if base_method_name in ['Mean_Aggregation_Ensemble', 'DE']:  # Skip ensemble
                            continue
                        
                        display_name = name_mapping.get(base_method_name, base_method_name)
                        if display_name in ensemble_methods:
                            continue
                        
                        scores = npz_data[method_name]
                        
                        # Check shape
                        if scores.ndim == 2:  # (n_folds, n_samples)
                            for fold_idx in range(n_folds):
                                fold_scores = scores[fold_idx]
                                for sample_idx in range(len(fold_scores)):
                                    key = (dataset, config, fold_idx, sample_idx)
                                    sample_scores[key][display_name] = fold_scores[sample_idx]
                        elif scores.ndim == 1:  # (n_samples,) - method that doesn't use folds
                            for fold_idx in range(n_folds):
                                for sample_idx in range(len(scores)):
                                    key = (dataset, config, fold_idx, sample_idx)
                                    sample_scores[key][display_name] = scores[sample_idx]
                
                except Exception as e:
                    print(f"  Warning: Failed to process {npz_file.name}: {e}")
                    continue
        
        if not sample_scores:
            print(f"  No sample scores found")
            continue
        
        # Convert to DataFrame
        rows = []
        for (dataset, config, fold, sample_idx), method_dict in sample_scores.items():
            row = {'dataset': dataset, 'config': config, 'fold': fold, 'sample_idx': sample_idx}
            row.update(method_dict)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Get method columns
        method_cols = [col for col in df.columns 
                      if col not in ['dataset', 'config', 'fold', 'sample_idx']]
        
        # Only keep methods with sufficient data
        method_cols = [col for col in method_cols if df[col].notna().sum() > 100]
        
        # Sort alphabetically
        method_cols = sorted(method_cols)
        
        # Remove samples with missing values in any method
        df_complete = df[method_cols].dropna()
        
        print(f"  Found {len(df_complete)} complete samples across {len(method_cols)} methods")
        
        if len(method_cols) < 2 or len(df_complete) < 10:
            print(f"  Skipping - insufficient data")
            continue
        
        # Subsample if too many points (for visualization performance)
        if len(df_complete) > 5000:
            print(f"  Subsampling {len(df_complete)} -> 5000 samples for visualization")
            df_complete = df_complete.sample(n=5000, random_state=42)
        
        # Create pairplot
        sns.set_style("whitegrid")
        
        print(f"  Creating pairplot...")
        g = sns.pairplot(
            df_complete,
            diag_kind='kde',
            plot_kws={'alpha': 0.3, 's': 5, 'edgecolor': 'none'},
            diag_kws={'linewidth': 2},
            corner=False
        )
        
        # Add title
        g.fig.suptitle(f'{shift_title}: Sample-Level Method Score Correlations\n({len(df_complete)} samples)', 
                      fontsize=16, fontweight='bold', y=1.0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'sample_level_pairplot_{shift_dir_name}.png'
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Sample-level pairplot saved to {output_path}")
        
        plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("Balanced Accuracy vs AUROC_f Scatter Plot Generator")
    print("=" * 80)
    
    # Get workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent
    
    print(f"Workspace root: {workspace_root}")
    
    # Create output directory
    output_dir = script_dir / 'scatter_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Collect all data points from pre-computed results
    data = collect_all_data_points(workspace_root)
    
    # Create scatter plots
    create_scatter_plot(data, output_dir)
    
    # Create per-method correlation plots
    create_per_method_correlation_plots(data, output_dir)
    
    # Create method correlation pairplots
    create_method_correlation_pairplots(data, output_dir)
    
    # Create sample-level pairplots
    create_sample_level_pairplots(workspace_root, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
