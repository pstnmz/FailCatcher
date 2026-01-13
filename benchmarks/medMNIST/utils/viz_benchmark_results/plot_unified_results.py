"""
Unified visualization of UQ benchmark results across all shift types.

Creates 3 figures:
1. AUROC_f Radar Plots: 3 rows (ID, CS, PS/NCS) × 2 columns (ResNet18, ViT)
2. AUGRC Radar Plots: Same layout as AUROC_f
3. Heatmaps: 3 rows (ID, CS, PS/NCS) × 2 columns (AUROC_f, AUGRC)
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import functions from existing scripts
from benchmarks.medMNIST.utils.viz_benchmark_results.generate_radar_plots import create_radar_plot_on_axis, parse_results_directory, compute_mean_aggregation_metric
from plot_ensemble_vs_mean_heatmap import compute_differences, prepare_heatmap_data


def load_and_parse_results(results_dirs, metric='auroc_f'):
    """
    Load and parse results from multiple directories.
    
    Args:
        results_dirs: Dict with keys 'id', 'corruption', 'population', 'new_class'
        metric: Metric to extract ('auroc_f' or 'augrc')
    
    Returns:
        Dict with keys 'id', 'corruption', 'population' (merged pop + new_class)
    """
    parsed = {}
    
    # Load ID and corruption shifts normally
    for shift_type in ['id', 'corruption']:
        dir_path = results_dirs.get(shift_type)
        if dir_path is not None and dir_path.exists():
            parsed[shift_type] = parse_results_directory(dir_path, metric=metric)
        else:
            parsed[shift_type] = {}
    
    # Load and merge population and new_class shifts
    pop_results = {}
    new_class_results = {}
    
    if results_dirs.get('population') is not None and results_dirs['population'].exists():
        pop_results = parse_results_directory(results_dirs['population'], metric=metric)
    
    if results_dirs.get('new_class') is not None and results_dirs['new_class'].exists():
        new_class_results = parse_results_directory(results_dirs['new_class'], metric=metric)
    
    # Merge population and new_class (prefix new_class datasets with "new_class_")
    merged = {}
    for model in ['resnet18', 'vit_b_16']:
        merged[model] = {}
        # Add population shift datasets
        if model in pop_results:
            for dataset in pop_results[model]:
                merged[model][dataset] = pop_results[model][dataset]
        # Add new class shift datasets (prefix with "new_class_")
        if model in new_class_results:
            for dataset in new_class_results[model]:
                new_dataset_name = f"new_class_{dataset}"
                merged[model][new_dataset_name] = new_class_results[model][dataset]
    
    parsed['population'] = merged
    
    return parsed


def load_heatmap_data(results_dirs):
    """
    Load and prepare heatmap data from all shift directories.
    
    Returns:
        Dict with keys 'id', 'corruption', 'population' containing matrices and metadata
    """
    heatmap_data = {}
    
    # Process ID and corruption shifts normally
    for shift_type in ['id', 'corruption']:
        dir_path = results_dirs.get(shift_type)
        if dir_path is None or not dir_path.exists():
            continue
            
        all_data = _load_json_files(dir_path)
        if all_data:
            heatmap_data[shift_type] = _process_heatmap_data(all_data)
    
    # Merge population and new_class shifts
    all_data_merged = {}
    
    # Load population shift data
    if results_dirs.get('population') is not None and results_dirs['population'].exists():
        all_data_merged.update(_load_json_files(results_dirs['population']))
    
    # Load new class shift data (they will have same keys, just merged)
    if results_dirs.get('new_class') is not None and results_dirs['new_class'].exists():
        all_data_merged.update(_load_json_files(results_dirs['new_class']))
    
    if all_data_merged:
        heatmap_data['population'] = _process_heatmap_data(all_data_merged)
    
    return heatmap_data


def _load_json_files(dir_path):
    """Load all JSON files from a directory."""
    json_files = list(dir_path.glob('uq_benchmark_*.json'))
    all_data = {}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Parse filename: uq_benchmark_{dataset}_{model}_{setup}_{timestamp}.json
            filename = json_file.stem
            parts = filename.replace('uq_benchmark_', '').rsplit('_', 1)
            
            if len(parts) == 2:
                prefix, timestamp = parts
                prefix_parts = prefix.split('_')
                
                if len(prefix_parts) >= 2:
                    setup = prefix_parts[-1]
                    model = prefix_parts[-2]
                    dataset = '_'.join(prefix_parts[:-2])
                    key = f"{dataset}_{model}_{setup}"
                    all_data[key] = data
    
    return all_data


def _process_heatmap_data(all_data):
    """Process loaded JSON data into heatmap matrices."""
    # Compute differences
    auroc_diffs, augrc_diffs = compute_differences(all_data)
    
    # Prepare matrices
    auroc_matrix, methods, display_names = prepare_heatmap_data(auroc_diffs)
    augrc_matrix, _, _ = prepare_heatmap_data(augrc_diffs)
    
    # Sort columns
    setup_order = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}
    
    def get_sort_key(name):
        clean_name = name.replace('_corrupt_severity3_test', '').replace('_test', '').replace('_external', '')
        setup = 'standard'
        for setup_name in ['DADO', 'DO', 'DA']:
            if setup_name in clean_name:
                setup = setup_name
                clean_name = clean_name.replace('_' + setup_name, '')
                break
        
        model = 'unknown'
        if '_vit_b_16' in clean_name:
            model = 'vit_b_16'
            clean_name = clean_name.replace('_vit_b_16', '')
        elif '_resnet18' in clean_name:
            model = 'resnet18'
            clean_name = clean_name.replace('_resnet18', '')
        
        dataset = clean_name
        return (dataset, model, setup_order.get(setup, 99))
    
    sorted_indices = sorted(range(len(display_names)), key=lambda i: get_sort_key(display_names[i]))
    display_names = [display_names[i] for i in sorted_indices]
    auroc_matrix = auroc_matrix[:, sorted_indices]
    augrc_matrix = augrc_matrix[:, sorted_indices]
    
    # Rename methods
    methods_display = [m.replace('MSR_calibrated', 'MSR-S')
                        .replace('KNN_Raw', 'KNN')
                        .replace('Ensembling', 'DE')
                        .replace('MCDropout', 'MCD') 
                       for m in methods]
    
    return {
        'auroc_matrix': auroc_matrix,
        'augrc_matrix': augrc_matrix,
        'methods': methods_display,
        'display_names': display_names
    }


def create_radar_figure(results_auroc, results_augrc, metric='auroc_f', aggregation='mean', 
                        results_dir=None, comp_eval_dirs=None):
    """
    Create unified radar plot figure for one metric (AUROC_f or AUGRC).
    
    Layout: 3 rows (ID, CS, PS/NCS) × 2 columns (ResNet18, ViT)
    
    Args:
        results_auroc: Dict with keys 'id', 'corruption', 'population'
        results_augrc: Dict with keys 'id', 'corruption', 'population'
        metric: 'auroc_f' or 'augrc'
        aggregation: Aggregation strategy
        results_dir: Main results directory
        comp_eval_dirs: Dict of comprehensive evaluation directories
        
    Returns:
        fig: matplotlib figure
    """
    results_map = results_auroc if metric == 'auroc_f' else results_augrc
    
    # Create figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(16, 21))
    gs = fig.add_gridspec(3, 2, hspace=0.15, wspace=0.30,
                         left=0.10, right=0.90, top=0.96, bottom=0.04)
    
    # Create all radar axes
    radar_axes = []
    for row in range(3):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col], projection='polar')
            radar_axes.append(ax)
    
    # Define shift types and their settings
    shift_configs = [
        ('id', 'in_distribution', 'In Distribution', 0),
        ('corruption', 'corruption_shifts', 'Corruption Shifts', 1),
        ('population', 'population_shift', 'Population / New Class Shift', 2)
    ]
    
    model_names = ['resnet18', 'vit_b_16']
    
    all_handles = []
    all_labels = []
    
    print(f"\nGenerating {metric.upper()} radar plots...")
    
    for shift_key, shift_name, shift_label, row in shift_configs:
        results = results_map.get(shift_key, {})
        
        if not results:
            print(f"  ⚠ No results for {shift_key}")
            continue
        
        # Get comprehensive evaluation directory
        comp_eval_dir = comp_eval_dirs.get(shift_key) if comp_eval_dirs else None
        
        # Add shift type label centered between the two columns
        if row == 0:
            fig.text(0.5, 0.967, shift_label.upper(), ha='center', va='center', 
                    fontsize=15, fontweight='bold')
        elif row == 1:
            fig.text(0.5, 0.646, shift_label.upper(), ha='center', va='center', 
                    fontsize=15, fontweight='bold')
        else:  # row == 2
            shift_label = 'POPULATION /\n NEW CLASS SHIFTS'
            fig.text(0.5, 0.319, shift_label, ha='center', va='center', 
                    fontsize=15, fontweight='bold')
        
        for col, model_name in enumerate(model_names):
            ax_idx = row * 2 + col
            ax = radar_axes[ax_idx]
            
            if model_name not in results:
                print(f"  ⚠ No {model_name} results for {shift_key}")
                continue
            
            model_results = results[model_name]
            
            # Generate radar plot
            handles, labels = create_radar_plot_on_axis(
                ax, model_results, model_name,
                results_dir=results_dir, runs_dir=comp_eval_dir,
                metric=metric, aggregation=aggregation, shift=shift_name
            )
            
            # Collect legend info from first subplot
            if ax_idx == 0 and handles and labels:
                all_handles = handles
                all_labels = labels
            
            # Add subplot title
            model_display = 'RESNET18' if model_name == 'resnet18' else 'VIT_B_16'
            metric_display = 'AUROC F' if metric == 'auroc_f' else 'AUGRC'
            if model_name == 'resnet18':
                ax.set_title(f'{model_display}\nPer-fold {aggregation.capitalize()}\n{metric_display}',
                            fontsize=14, fontweight='bold', y=0.95, x=-0.15, ha='left')
            else:
                ax.set_title(f'{model_display}\nPer-fold {aggregation.capitalize()}\n{metric_display}',
                            fontsize=14, fontweight='bold', pad=0.95, x=1.15, ha='right')
    
    # Add legend
    if all_handles and all_labels:
        # Move "Mean_Aggregation" to bottom of legend
        if 'Mean_Aggregation' in all_labels:
            idx = all_labels.index('Mean_Aggregation')
            all_labels.append(all_labels.pop(idx))
            all_handles.append(all_handles.pop(idx))
        
        # Rename method labels
        all_labels = [label.replace('KNN_Raw', 'KNN')
                           .replace('Ensembling', 'DE')
                           .replace('MCDropout', 'MCD')
                           .replace('MSR_calibrated', 'MSR-S')
                           .replace('Mean_Aggregation', 'Mean Agg')
                      for label in all_labels]
        
        fig.legend(all_handles, all_labels, loc='center', ncol=2,
                  fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.70))
    
    return fig


def create_radar_figure_alt_layout(results_auroc, results_augrc, metric='auroc_f', aggregation='mean', 
                                    results_dir=None, comp_eval_dirs=None):
    """
    Create alternate unified radar plot figure for one metric (AUROC_f or AUGRC).
    
    Layout: 2 rows (ResNet18, ViT) × 3 columns (ID, CS, PS/NCS)
    Radars are 10% smaller, with model names on the left of each row.
    
    Args:
        results_auroc: Dict with keys 'id', 'corruption', 'population'
        results_augrc: Dict with keys 'id', 'corruption', 'population'
        metric: 'auroc_f' or 'augrc'
        aggregation: Aggregation strategy
        results_dir: Main results directory
        comp_eval_dirs: Dict of comprehensive evaluation directories
        
    Returns:
        fig: matplotlib figure
    """
    results_map = results_auroc if metric == 'auroc_f' else results_augrc
    
    # Create figure with 2 rows and 3 columns - wider to accommodate 3 columns
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.25,
                         left=0.14, right=0.94, top=0.92, bottom=0.08)
    
    # Create all radar axes with reduced radius (90% of original)
    radar_axes = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col], projection='polar')
            radar_axes.append(ax)
    
    # Define shift types and their settings
    shift_configs = [
        ('id', 'in_distribution', 'ID', 0),
        ('corruption', 'corruption_shifts', 'CS', 1),
        ('population', 'population_shift', 'PS/NCS', 2)
    ]
    
    model_names = ['resnet18', 'vit_b_16']
    model_display_names = {
        'resnet18': 'RESNET18',
        'vit_b_16': 'ViT-B/16'
    }
    
    all_handles = []
    all_labels = []
    
    print(f"\nGenerating {metric.upper()} radar plots (alternate layout)...")
    
    # Add column titles (shift types)
    for shift_key, shift_name, shift_label, col in shift_configs:
        if col == 0:
            x_pos = 0.145
        elif col ==1:
            x_pos = 0.43
        else:
            x_pos = 0.74
        fig.text(x_pos, 0.945, shift_label, ha='center', va='center', 
                fontsize=16, fontweight='bold')
    
    # Add row labels (model names) - positioned at bottom of each row, aligned to the right
    metric_display = 'AUROC_f' if metric == 'auroc_f' else 'AUGRC'
    for row, model_name in enumerate(model_names):
        # Position at bottom of row (row 0 = top, row 1 = bottom)
        y_pos = 0.08 + (1 - row) * (0.84 / 2)  # Bottom of each row
        model_display = model_display_names[model_name]
        # Align to right side, position below first AUROC F title
        x_pos = 0.4  # Aligned with first column
        fig.text(x_pos, y_pos, f'{model_display}\nPer-fold {aggregation.capitalize()}\n{metric_display}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=0)
    
    for row, model_name in enumerate(model_names):
        for shift_key, shift_name, shift_label, col in shift_configs:
            results = results_map.get(shift_key, {})
            
            if not results:
                print(f"  ⚠ No results for {shift_key}")
                continue
            
            # Get comprehensive evaluation directory
            comp_eval_dir = comp_eval_dirs.get(shift_key) if comp_eval_dirs else None
            
            ax_idx = row * 3 + col
            ax = radar_axes[ax_idx]
            
            if model_name not in results:
                print(f"  ⚠ No {model_name} results for {shift_key}")
                continue
            
            model_results = results[model_name]
            
            # Generate radar plot
            handles, labels = create_radar_plot_on_axis(
                ax, model_results, model_name,
                results_dir=results_dir, runs_dir=comp_eval_dir,
                metric=metric, aggregation=aggregation, shift=shift_name
            )
            
            # Collect legend info from first subplot
            if ax_idx == 0 and handles and labels:
                all_handles = handles
                all_labels = labels
            
            # Remove y-axis TITLE for CS and PS/NCS columns (col > 0)
            if col > 0:
                ax.set_ylabel('')
            
            # Scale down the radius by 10% (set ylim to 90% of original)
            if metric == 'auroc_f':
                y_min, y_max = 0.4, 1.0
                # Keep same range but display will be smaller due to figure layout
            else:  # augrc
                current_ylim = ax.get_ylim()
                y_min, y_max = current_ylim
    
    # Add legend at the bottom
    if all_handles and all_labels:
        # Move "Mean_Aggregation" and "Mean_Aggregation_Ensemble" to end
        special_methods = ['Mean_Aggregation', 'Mean_Aggregation_Ensemble']
        for special in special_methods:
            if special in all_labels:
                idx = all_labels.index(special)
                all_labels.append(all_labels.pop(idx))
                all_handles.append(all_handles.pop(idx))
        
        # Rename method labels
        all_labels = [label.replace('KNN_Raw', 'KNN')
                           .replace('Ensembling', 'DE')
                           .replace('MCDropout', 'MCD')
                           .replace('MSR_calibrated', 'MSR-S')
                           .replace('Mean_Aggregation_Ensemble', 'Mean Agg + Ens')
                           .replace('Mean_Aggregation', 'Mean Agg')
                      for label in all_labels]
        
        fig.legend(all_handles, all_labels, loc='lower center', ncol=5,
                  fontsize=11, frameon=True, bbox_to_anchor=(0.54, -0.02))
    
    return fig


def create_heatmap_figure(heatmap_data, results_dir, aggregation='mean'):
    """
    Create unified heatmap figure.
    
    Layout: 6 rows stacked vertically
    - Top 3: AUROC_f (ID, CS, PS/NCS)
    - Bottom 3: AUGRC (ID, CS, PS/NCS)
    Methods as rows, setups as columns
    
    Args:
        heatmap_data: Dict with keys 'id', 'corruption', 'population'
        results_dir: Main results directory
        aggregation: Aggregation strategy
        
    Returns:
        fig: matplotlib figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nGenerating heatmaps...")
    
    # Define shift configurations
    shift_configs = [
        ('id', 'IN DISTRIBUTION', 'in_distribution'),
        ('corruption', 'CORRUPTION SHIFTS', 'corruption_shifts'),
        ('population', 'POPULATION / NEW CLASS SHIFTS', 'population_shift')
    ]
    
    # Create figure with 6 rows (3 for AUROC_f, 3 for AUGRC)
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(6, 1, hspace=0.15, 
                         left=0.08, right=0.92, top=0.96, bottom=0.04)
    
    # Color scale ranges
    vmin_auroc, vmax_auroc = -0.2, 0.2
    vmin_augrc, vmax_augrc = -0.1, 0.1
    
    all_axes = []
    
    # Generate AUROC_f heatmaps (top 3 rows)
    for row_idx, (shift_key, shift_label, shift_name) in enumerate(shift_configs):
        data = heatmap_data.get(shift_key)
        
        if data is None:
            print(f"  ⚠ No heatmap data for {shift_key}")
            continue
        
        auroc_matrix = data['auroc_matrix']
        methods = data['methods']
        display_names = data['display_names']
        
        # Add Mean_Aggregation row
        auroc_agg_row = compute_mean_agg_row(
            results_dir, display_names, shift_name, 'auroc_f', aggregation
        )
        
        auroc_matrix_with_agg = np.vstack([auroc_matrix, auroc_agg_row])
        methods_with_agg = methods + ['⚡ Mean Agg']
        
        # Create heatmap
        ax = fig.add_subplot(gs[row_idx, 0])
        all_axes.append(ax)
        
        sns.heatmap(auroc_matrix_with_agg,
                    xticklabels=display_names,
                    yticklabels=methods_with_agg,
                    cmap='RdBu_r',
                    center=0,
                    vmin=vmin_auroc,
                    vmax=vmax_auroc,
                    annot=False,
                    cbar=False,
                    ax=ax)
        
        ax.set_title(f'{shift_label} - AUROC_f', fontsize=12, fontweight='bold', pad=5)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # Only show x-tick labels on bottom AUROC_f heatmap
        if row_idx < 2:
            ax.set_xticklabels([])
    
    # Generate AUGRC heatmaps (bottom 3 rows)
    for row_idx, (shift_key, shift_label, shift_name) in enumerate(shift_configs):
        data = heatmap_data.get(shift_key)
        
        if data is None:
            continue
        
        augrc_matrix = data['augrc_matrix']
        methods = data['methods']
        display_names = data['display_names']
        
        # Add Mean_Aggregation row
        augrc_agg_row = compute_mean_agg_row(
            results_dir, display_names, shift_name, 'augrc', aggregation
        )
        
        augrc_matrix_with_agg = np.vstack([augrc_matrix, augrc_agg_row])
        methods_with_agg = methods + ['⚡ Mean Agg']
        
        # Create heatmap
        ax = fig.add_subplot(gs[row_idx + 3, 0])
        all_axes.append(ax)
        
        sns.heatmap(augrc_matrix_with_agg,
                    xticklabels=display_names,
                    yticklabels=methods_with_agg,
                    cmap='RdBu_r',
                    center=0,
                    vmin=vmin_augrc,
                    vmax=vmax_augrc,
                    annot=False,
                    cbar=False,
                    ax=ax)
        
        ax.set_title(f'{shift_label} - AUGRC', fontsize=12, fontweight='bold', pad=5)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # Only show x-tick labels on bottom AUGRC heatmap
        if row_idx < 2:
            ax.set_xticklabels([])
    
    # Add colorbars
    from matplotlib.cm import RdBu_r
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    
    # AUROC_f colorbar (shared for top 3 heatmaps)
    cbar_ax1 = fig.add_axes([0.93, 0.66, 0.015, 0.28])
    norm_auroc = Normalize(vmin=vmin_auroc, vmax=vmax_auroc)
    cbar1 = mpl.colorbar.ColorbarBase(cbar_ax1, cmap=RdBu_r, norm=norm_auroc, orientation='vertical')
    cbar1.set_label('ΔAUROC_f', fontsize=12, fontweight='bold')
    
    # AUGRC colorbar (shared for bottom 3 heatmaps)
    cbar_ax2 = fig.add_axes([0.93, 0.08, 0.015, 0.28])
    norm_augrc = Normalize(vmin=vmin_augrc, vmax=vmax_augrc)
    cbar2 = mpl.colorbar.ColorbarBase(cbar_ax2, cmap=RdBu_r, norm=norm_augrc, orientation='vertical')
    cbar2.set_label('ΔAUGRC', fontsize=12, fontweight='bold')
    
    # Add main title
    fig.suptitle(f'Ensemble vs Per-Fold Differences ({aggregation.capitalize()})', 
                fontsize=18, fontweight='bold', y=0.99)
    
    return fig

def compute_mean_agg_row(results_dir, display_names, shift_name, metric, aggregation):
    """Compute Mean_Aggregation row for heatmap."""
    agg_row_list = []
    
    for display_name in display_names:
        # Parse display_name
        clean_name = display_name.replace('_corrupt_severity3_test', '').replace('_test', '').replace('_external', '')
        
        setup = 'standard'
        for setup_name in ['DADO', 'DO', 'DA']:
            if setup_name in clean_name:
                setup = setup_name
                clean_name = clean_name.replace('_' + setup_name, '')
                break
        
        model = 'unknown'
        if '_vit_b_16' in clean_name:
            model = 'vit_b_16'
            clean_name = clean_name.replace('_vit_b_16', '')
        elif '_resnet18' in clean_name:
            model = 'resnet18'
            clean_name = clean_name.replace('_resnet18', '')
        
        dataset = clean_name
        dataset_key = dataset if setup == 'standard' else f"{dataset}_{setup}"
        
        # Compute differences
        ensemble_val = compute_mean_aggregation_metric(
            results_dir, dataset_key, model, 
            metric=metric, aggregation=aggregation, 
            shift=shift_name, use_ensemble=True
        )
        perfold_val = compute_mean_aggregation_metric(
            results_dir, dataset_key, model, 
            metric=metric, aggregation=aggregation, 
            shift=shift_name, use_ensemble=False
        )
        
        diff = ensemble_val - perfold_val if not np.isnan(ensemble_val) and not np.isnan(perfold_val) else np.nan
        agg_row_list.append(diff)
    
    return np.array(agg_row_list).reshape(1, -1)


def parse_display_names(display_names):
    """Parse display names into setup, model, dataset lists."""
    setups = []
    models = []
    datasets = []
    setup_names = ['DA', 'DO', 'DADO']
    model_names = ['resnet18', 'vit_b_16']
    
    for name in display_names:
        clean_name = name.replace('_corrupt_severity3_test', '').replace('_test', '').replace('_external', '')
        
        setup = 'standard'
        col_without_setup = clean_name
        for setup_name in setup_names:
            if clean_name.endswith('_' + setup_name):
                setup = setup_name
                col_without_setup = clean_name[:-len(setup_name)-1]
                break
        
        model = None
        dataset = None
        for model_name in model_names:
            if col_without_setup.endswith('_' + model_name):
                model = model_name
                dataset = col_without_setup[:-len(model_name)-1]
                break
        
        if model is None:
            parts = col_without_setup.split('_')
            model = parts[-1] if len(parts) > 0 else ''
            dataset = '_'.join(parts[:-1]) if len(parts) > 1 else ''
        
        setups.append(setup)
        models.append(model)
        datasets.append(dataset)
    
    return setups, models, datasets


def add_hierarchical_labels(ax, ticks, setups, models, datasets):
    """Add model and dataset labels below x-axis."""
    # Group by dataset and model
    groups_detailed = []
    i = 0
    while i < len(datasets):
        curr_dataset = datasets[i]
        curr_model = models[i]
        start = i
        while i < len(datasets) and datasets[i] == curr_dataset and models[i] == curr_model:
            i += 1
        groups_detailed.append((start, i-1, curr_dataset, curr_model))
    
    # Add model names
    for (s, e, dname, mname) in groups_detailed:
        if not mname:
            continue
        center = (ticks[s] + ticks[e]) / 2.0
        ax.text(center, -0.25, mname, transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontsize=9, fontweight='normal')
    
    # Group by dataset only
    dataset_groups = []
    i = 0
    while i < len(datasets):
        curr_dataset = datasets[i]
        start = i
        while i < len(datasets) and datasets[i] == curr_dataset:
            i += 1
        dataset_groups.append((start, i-1, curr_dataset))
    
    # Add dataset names
    for (s, e, dname) in dataset_groups:
        if not dname:
            continue
        center = (ticks[s] + ticks[e]) / 2.0
        ax.text(center, -0.38, dname, transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontsize=9, fontweight='bold')


def main(aggregation='mean'):
    """Main function to generate unified figures."""
    
    print("=" * 80)
    print("Unified Results Visualization")
    print(f"Aggregation: {aggregation.upper()}")
    print("=" * 80)
    
    # Get paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    
    # Define all results directories
    results_dirs = {
        'id': results_dir / 'id_results',
        'corruption': results_dir / 'corruption_shifts',
        'population': results_dir / 'population_shifts',
        'new_class': results_dir / 'new_class_shifts'
    }
    
    # Merge population and new_class for visualization
    # (they will be combined in the same row as PS/NCS)
    
    # Define comprehensive evaluation directories
    comp_eval_base = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results'
    comp_eval_dirs = {
        'id': comp_eval_base / 'in_distribution',
        'corruption': comp_eval_base / 'corruption_shifts',
        'population': comp_eval_base / 'population_shifts'
    }
    
    print(f"Workspace root: {workspace_root}")
    print(f"Results directory: {results_dir}")
    
    # Create output directory
    output_dir = results_dir / 'figures' / 'unified'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. Load all results
    # ==========================================
    print("\n" + "=" * 80)
    print("Loading results...")
    print("=" * 80)
    
    results_auroc = load_and_parse_results(results_dirs, metric='auroc_f')
    results_augrc = load_and_parse_results(results_dirs, metric='augrc')
    
    # Merge population and new_class results
    for metric_results in [results_auroc, results_augrc]:
        if 'population' in metric_results and 'new_class' in metric_results:
            pop_res = metric_results['population']
            new_res = metric_results['new_class']
            
            # Merge both into 'population' key
            merged = {}
            for model in ['resnet18', 'vit_b_16']:
                merged[model] = {}
                if model in pop_res:
                    merged[model].update(pop_res[model])
                if model in new_res:
                    merged[model].update(new_res[model])
            
            metric_results['population'] = merged
    
    # Load heatmap data
    heatmap_data = load_heatmap_data(results_dirs)
    
    # Merge population and new_class heatmap data
    if 'population' in heatmap_data and 'new_class' in heatmap_data:
        pop_data = heatmap_data['population']
        new_data = heatmap_data['new_class']
        
        # Concatenate matrices and names
        pop_data['auroc_matrix'] = np.hstack([pop_data['auroc_matrix'], new_data['auroc_matrix']])
        pop_data['augrc_matrix'] = np.hstack([pop_data['augrc_matrix'], new_data['augrc_matrix']])
        pop_data['display_names'] = pop_data['display_names'] + new_data['display_names']
    
    # ==========================================
    # 2. Create AUROC_f radar figure
    # ==========================================
    # print("\n" + "=" * 80)
    # print("Creating AUROC_f radar figure...")
    # print("=" * 80)
    
    # fig_auroc = create_radar_figure(
    #     results_auroc, results_augrc, 
    #     metric='auroc_f', aggregation=aggregation,
    #     results_dir=results_dir, comp_eval_dirs=comp_eval_dirs
    # )
    
    # output_path = output_dir / f'unified_auroc_f_radars_{aggregation}.png'
    # fig_auroc.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"\n✓ Saved to {output_path}")
    # plt.close(fig_auroc)
    
    # ==========================================
    # 3. Create AUGRC radar figure
    # ==========================================
    # print("\n" + "=" * 80)
    # print("Creating AUGRC radar figure...")
    # print("=" * 80)
    
    # fig_augrc = create_radar_figure(
    #     results_auroc, results_augrc,
    #     metric='augrc', aggregation=aggregation,
    #     results_dir=results_dir, comp_eval_dirs=comp_eval_dirs
    # )
    
    # output_path = output_dir / f'unified_augrc_radars_{aggregation}.png'
    # fig_augrc.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"\n✓ Saved to {output_path}")
    # plt.close(fig_augrc)
    
    # ==========================================
    # 2b. Create alternate layout AUROC_f radar figure
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating AUROC_f radar figure (alternate layout)...")
    print("=" * 80)
    
    fig_auroc_alt = create_radar_figure_alt_layout(
        results_auroc, results_augrc, 
        metric='auroc_f', aggregation=aggregation,
        results_dir=results_dir, comp_eval_dirs=comp_eval_dirs
    )
    
    output_path = output_dir / f'unified_auroc_f_radars_{aggregation}_alt.png'
    fig_auroc_alt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    plt.close(fig_auroc_alt)
    
    # ==========================================
    # 3b. Create alternate layout AUGRC radar figure
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating AUGRC radar figure (alternate layout)...")
    print("=" * 80)
    
    fig_augrc_alt = create_radar_figure_alt_layout(
        results_auroc, results_augrc,
        metric='augrc', aggregation=aggregation,
        results_dir=results_dir, comp_eval_dirs=comp_eval_dirs
    )
    
    output_path = output_dir / f'unified_augrc_radars_{aggregation}_alt.png'
    fig_augrc_alt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    plt.close(fig_augrc_alt)

    # ==========================================
    # 4. Create heatmap figure
    # ==========================================
    # print("\n" + "=" * 80)
    # print("Creating heatmap figure...")
    # print("=" * 80)
    
    # fig_heatmap = create_heatmap_figure(
    #     heatmap_data, results_dir, aggregation=aggregation
    # )
    
    # output_path = output_dir / f'unified_heatmaps_{aggregation}.png'
    # fig_heatmap.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"\n✓ Saved to {output_path}")
    # plt.close(fig_heatmap)
    
    print("\n" + "=" * 80)
    print("✓ All figures saved!")
    print("=" * 80)


if __name__ == '__main__':
    aggregation = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    
    if aggregation not in ['mean', 'min', 'max', 'vote']:
        print(f"Unknown aggregation: {aggregation}")
        print("Usage: python plot_unified_results.py [mean|min|max|vote]")
        sys.exit(1)
    
    main(aggregation=aggregation)
