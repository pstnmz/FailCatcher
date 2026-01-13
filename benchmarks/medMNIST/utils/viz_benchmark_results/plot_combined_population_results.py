"""
Combined visualization of population shift and new class shift UQ benchmark results.

Creates a single figure with:
- Top: 2x2 radar plots (ResNet18/ViT x AUROC_f/AUGRC)
- Bottom: 2 stacked heatmaps showing ensemble vs mean per-fold differences

Population shifts include: AMOS, dermamnist-e-external, pathmnist
New class shifts include: AMOS new classes (OOD detection)
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


def create_combined_figure(results_dir, pop_results_dir, new_class_results_dir, comp_eval_dir, aggregation='mean'):
    """
    Create a combined figure with radar plots on top and heatmaps below.
    
    Args:
        results_dir: Main results directory
        pop_results_dir: Population shift results directory with JSON files
        new_class_results_dir: New class shift results directory with JSON files
        comp_eval_dir: Comprehensive evaluation directory
        aggregation: Aggregation strategy for radar plots
    
    Returns:
        fig: matplotlib figure
    """
    # Parse results for radar plots from both directories
    results_auroc_pop = parse_results_directory(pop_results_dir, metric='auroc_f')
    results_augrc_pop = parse_results_directory(pop_results_dir, metric='augrc')
    results_auroc_new = parse_results_directory(new_class_results_dir, metric='auroc_f')
    results_augrc_new = parse_results_directory(new_class_results_dir, metric='augrc')
    
    # Merge results - add new_class as additional dataset
    def merge_results(pop_res, new_res):
        merged = {}
        for model in ['resnet18', 'vit_b_16']:
            merged[model] = {}
            # Add population shift datasets
            if model in pop_res:
                for dataset in pop_res[model]:
                    merged[model][dataset] = pop_res[model][dataset]
            # Add new class shift datasets (prefix with "new_class_")
            if model in new_res:
                for dataset in new_res[model]:
                    # Rename amos2022 to new_class_amos2022 to distinguish
                    new_dataset_name = f"new_class_{dataset}"
                    merged[model][new_dataset_name] = new_res[model][dataset]
        return merged
    
    results_auroc = merge_results(results_auroc_pop, results_auroc_new)
    results_augrc = merge_results(results_augrc_pop, results_augrc_new)
    
    # Load data for heatmaps from both directories
    all_data = {}
    
    # Load population shift data
    json_files_pop = list(pop_results_dir.glob('uq_benchmark_*.json'))
    for json_file in json_files_pop:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Parse filename: uq_benchmark_{dataset}_{model}_{setup}_{timestamp}.json
            filename = json_file.stem  # Remove .json
            parts = filename.replace('uq_benchmark_', '').rsplit('_', 1)  # Split off timestamp
            
            if len(parts) == 2:
                prefix, timestamp = parts
                # Now split prefix into dataset, model, setup
                prefix_parts = prefix.split('_')
                
                # Last 2 parts are model and setup
                if len(prefix_parts) >= 2:
                    setup = prefix_parts[-1]
                    model = prefix_parts[-2]
                    dataset = '_'.join(prefix_parts[:-2])
                    key = f"{dataset}_{model}_{setup}"
                    all_data[key] = data
    
    # Load new class shift data
    json_files_new = list(new_class_results_dir.glob('uq_benchmark_*.json'))
    for json_file in json_files_new:
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
                    # Prefix with "new_class_" to distinguish
                    key = f"new_class_{dataset}_{model}_{setup}"
                    all_data[key] = data
    
    # Compute differences for heatmaps
    auroc_diffs, augrc_diffs = compute_differences(all_data)
    
    # Prepare heatmap data
    auroc_matrix, methods, display_names = prepare_heatmap_data(auroc_diffs)
    augrc_matrix, _, _ = prepare_heatmap_data(augrc_diffs)
    
    # Sort columns by dataset, model, then setup order (standard, DA, DO, DADO)
    setup_order = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}
    
    def get_sort_key(name):
        # Remove common suffixes
        clean_name = name.replace('_population_shift_test', '').replace('_test', '')
        
        # Parse setup
        setup = 'standard'
        for setup_name in ['DADO', 'DO', 'DA']:  # Check longest first
            if setup_name in clean_name:
                setup = setup_name
                clean_name = clean_name.replace('_' + setup_name, '')
                break
        
        # Parse model
        model = 'unknown'
        if '_vit_b_16' in clean_name:
            model = 'vit_b_16'
            clean_name = clean_name.replace('_vit_b_16', '')
        elif '_resnet18' in clean_name:
            model = 'resnet18'
            clean_name = clean_name.replace('_resnet18', '')
        
        # Remaining is dataset
        dataset = clean_name
        
        return (dataset, model, setup_order.get(setup, 99))
    
    # Create sorted indices
    sorted_indices = sorted(range(len(display_names)), key=lambda i: get_sort_key(display_names[i]))
    
    # Reorder display_names and matrix columns
    display_names = [display_names[i] for i in sorted_indices]
    auroc_matrix = auroc_matrix[:, sorted_indices]
    augrc_matrix = augrc_matrix[:, sorted_indices]
    
    # Create aggregated row for heatmaps - compute actual Mean_Aggregation differences
    print("\nComputing Mean_Aggregation differences for heatmaps...")
    auroc_agg_row_list = []
    augrc_agg_row_list = []
    
    for display_name in display_names:
        # Parse display_name to get dataset, model, setup
        clean_name = display_name.replace('_population_shift_test', '').replace('_test', '')
        
        # Determine shift type based on display_name
        if 'new_class_' in display_name:
            shift = 'new_class_shift'
        else:
            shift = 'population_shift'
        
        # Parse setup
        setup = 'standard'
        for setup_name in ['DADO', 'DO', 'DA']:
            if setup_name in clean_name:
                setup = setup_name
                clean_name = clean_name.replace('_' + setup_name, '')
                break
        
        # Parse model
        if '_vit_b_16' in clean_name:
            model = 'vit_b_16'
            clean_name = clean_name.replace('_vit_b_16', '')
        elif '_resnet18' in clean_name:
            model = 'resnet18'
            clean_name = clean_name.replace('_resnet18', '')
        else:
            model = 'resnet18'
        
        dataset = clean_name
        dataset_key = f"{dataset}_{setup}"
        
        # Compute ensemble-based Mean_Aggregation (using ensemble scores + ensemble correct/incorrect)
        ensemble_auroc = compute_mean_aggregation_metric(
            results_dir, dataset_key, model, metric='auroc_f', 
            aggregation='mean', shift=shift, use_ensemble=True
        )
        
        # Compute per-fold-based Mean_Aggregation (using per-fold scores + per-fold correct/incorrect)
        perfold_auroc = compute_mean_aggregation_metric(
            results_dir, dataset_key, model, metric='auroc_f',
            aggregation='mean', shift=shift, use_ensemble=False
        )
        
        # Difference: ensemble - perfold
        auroc_diff = ensemble_auroc - perfold_auroc if not np.isnan(ensemble_auroc) and not np.isnan(perfold_auroc) else np.nan
        auroc_agg_row_list.append(auroc_diff)
        
        # Same for AUGRC
        ensemble_augrc = compute_mean_aggregation_metric(
            results_dir, dataset_key, model, metric='augrc',
            aggregation='mean', shift=shift, use_ensemble=True
        )
        
        perfold_augrc = compute_mean_aggregation_metric(
            results_dir, dataset_key, model, metric='augrc',
            aggregation='mean', shift=shift, use_ensemble=False
        )
        
        augrc_diff = ensemble_augrc - perfold_augrc if not np.isnan(ensemble_augrc) and not np.isnan(perfold_augrc) else np.nan
        augrc_agg_row_list.append(augrc_diff)
    
    auroc_agg_row = np.array(auroc_agg_row_list).reshape(1, -1)
    augrc_agg_row = np.array(augrc_agg_row_list).reshape(1, -1)
    
    auroc_matrix_with_agg = np.vstack([auroc_matrix, auroc_agg_row])
    augrc_matrix_with_agg = np.vstack([augrc_matrix, augrc_agg_row])
    
    # Rename method labels for display
    methods_display = [m.replace('MSR_calibrated', 'MSR-S')
                        .replace('KNN_Raw', 'KNN')
                        .replace('Ensembling', 'DE')
                        .replace('MCDropout', 'MCD') 
                       for m in methods]
    methods_with_agg = methods_display + ['⚡ Mean Agg']
    
    # Create figure with GridSpec for complex layout
    # Smaller figure since only 3 datasets
    fig = plt.figure(figsize=(16, 21))
    
    # Define grid: 4 rows (radar top half, heatmap1, heatmap2)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.85, 0.90, 0.30, 0.30], 
                         hspace=0.05, wspace=0.45,
                         left=0.10, right=0.90, top=0.96, bottom=0.08)
    
    # Create radar plot axes (top 2 rows, 2 columns)
    radar_axes = [
        fig.add_subplot(gs[0, 0], projection='polar'),  # Top-left: ResNet18 AUROC_f
        fig.add_subplot(gs[0, 1], projection='polar'),  # Top-right: ViT AUROC_f
        fig.add_subplot(gs[1, 0], projection='polar'),  # Bottom-left: ResNet18 AUGRC
        fig.add_subplot(gs[1, 1], projection='polar'),  # Bottom-right: ViT AUGRC
    ]
    
    # Create heatmap axes (bottom 2 rows, spanning both columns)
    heatmap_ax1 = fig.add_subplot(gs[2, :])  # AUROC_f heatmap
    heatmap_ax2 = fig.add_subplot(gs[3, :])  # AUGRC heatmap
    
    # ========================
    # Generate radar plots
    # ========================
    model_names = ['resnet18', 'vit_b_16']
    metrics = ['auroc_f', 'augrc']
    results_map = {'auroc_f': results_auroc, 'augrc': results_augrc}
    
    all_handles = []
    all_labels = []
    
    print("\nGenerating radar plots...")
    for idx, (row, metric) in enumerate([(0, 'auroc_f'), (1, 'augrc')]):
        for col, model_name in enumerate(model_names):
            ax_idx = row * 2 + col
            ax = radar_axes[ax_idx]
            
            results = results_map[metric]
            if not results or model_name not in results:
                continue
            
            model_results = results[model_name]
            
            # Generate radar plot
            handles, labels = create_radar_plot_on_axis(
                ax, model_results, model_name,
                results_dir=results_dir, runs_dir=comp_eval_dir,
                metric=metric, aggregation=aggregation, shift='population_shift'
            )
            
            # Collect legend info from first subplot
            if ax_idx == 0:
                all_handles = handles
                all_labels = labels
            
            # Add subplot title
            model_display = 'RESNET18' if model_name == 'resnet18' else 'VIT_B_16'
            metric_display = 'AUROC F' if metric == 'auroc_f' else 'AUGRC'
            if model_name == 'resnet18':
               ax.set_title(f'{model_display}\nPer-fold {aggregation.capitalize()} {metric_display}',
                            fontsize=14, fontweight='bold', pad=30, x=-0.1, ha='left')
            else:
                ax.set_title(f'{model_display}\nPer-fold {aggregation.capitalize()} {metric_display}',
                             fontsize=14, fontweight='bold', pad=30, x=1.1, ha='right')
    
    # Move "Mean_Aggregation" to bottom of legend
    if 'Mean_Aggregation' in all_labels:
        idx = all_labels.index('Mean_Aggregation')
        all_labels.append(all_labels.pop(idx))
        all_handles.append(all_handles.pop(idx))
    
    # Rename method labels for radar plots
    all_labels = [label.replace('KNN_Raw', 'KNN')
                       .replace('Ensembling', 'DE')
                       .replace('MCDropout', 'MCD')
                       .replace('MSR_calibrated', 'MSR-S')
                       .replace('Mean_Aggregation', 'Mean Agg')
                  for label in all_labels]
    
    # Add legend for radar plots - centered between the 4 radars
    fig.legend(all_handles, all_labels, loc='center', ncol=2,
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.66))
    
    # ========================
    # Generate heatmaps
    # ========================
    print("\nGenerating heatmaps...")
    
    # Use separate color scales for each metric (AUGRC has smaller typical range)
    vmin_auroc, vmax_auroc = -0.2, 0.2
    vmin_augrc, vmax_augrc = -0.1, 0.1
    
    # AUROC_f heatmap
    sns.heatmap(auroc_matrix_with_agg,
                xticklabels=[],
                yticklabels=methods_with_agg,
                cmap='RdBu_r',
                center=0,
                vmin=vmin_auroc,
                vmax=vmax_auroc,
                annot=False,
                cbar=False,
                ax=heatmap_ax1)
    
    heatmap_ax1.set_title('AUROC_f: Ensemble vs Mean Per-Fold', 
                         fontsize=12, fontweight='bold', pad=5)
    heatmap_ax1.set_ylabel('')
    plt.setp(heatmap_ax1.get_yticklabels(), rotation=0)
    
    # AUGRC heatmap
    sns.heatmap(augrc_matrix_with_agg,
                xticklabels=display_names,
                yticklabels=methods_with_agg,
                cmap='RdBu_r',
                center=0,
                vmin=vmin_augrc,
                vmax=vmax_augrc,
                annot=False,
                cbar=False,
                ax=heatmap_ax2)
    
    heatmap_ax2.set_title('AUGRC: Ensemble vs Mean Per-Fold',
                         fontsize=12, fontweight='bold', pad=5)
    heatmap_ax2.set_ylabel('')

    # Parse display names for multi-line x labels
    setups = []
    models = []
    datasets = []
    setup_names = ['DA', 'DO', 'DADO']
    model_names = ['resnet18', 'vit_b_16']
    
    for name in display_names:
        # Remove common suffixes if present
        clean_name = name.replace('_population_shift_test', '').replace('_test', '')
        
        # Try to identify setup, model, dataset properly
        setup = 'standard'
        model = None
        dataset = None
        
        # Check if ends with setup name
        col_without_setup = clean_name
        for setup_name in setup_names:
            if clean_name.endswith('_' + setup_name):
                setup = setup_name
                col_without_setup = clean_name[:-len(setup_name)-1]
                break
        
        # Now find model in col_without_setup
        for model_name in model_names:
            if col_without_setup.endswith('_' + model_name):
                model = model_name
                dataset = col_without_setup[:-len(model_name)-1]
                break
        
        # Fallback if model not found
        if model is None:
            parts = col_without_setup.split('_')
            model = parts[-1] if len(parts) > 0 else ''
            dataset = '_'.join(parts[:-1]) if len(parts) > 1 else ''
        
        setups.append(setup)
        models.append(model)
        datasets.append(dataset)

    # Set xticklabels on bottom axis to just setup names
    heatmap_ax2.set_xticklabels(setups, rotation=45, ha='right', fontsize=8)
    plt.setp(heatmap_ax2.get_yticklabels(), rotation=0)

    # Get tick positions - use explicit array for robustness
    ticks = np.arange(len(display_names)) + 0.5  # seaborn centers ticks at +0.5
    
    # Group by dataset and model to add model and dataset labels
    # Build groups: (start_idx, end_idx, dataset, model)
    groups_detailed = []
    i = 0
    while i < len(display_names):
        curr_dataset = datasets[i]
        curr_model = models[i]
        start = i
        # Find end of this dataset+model group
        while i < len(display_names) and datasets[i] == curr_dataset and models[i] == curr_model:
            i += 1
        groups_detailed.append((start, i-1, curr_dataset, curr_model))
    
    # Add model names below ticks (spanning 4 setups typically)
    for (s, e, dname, mname) in groups_detailed:
        if not mname:
            continue
        center = (ticks[s] + ticks[e]) / 2.0
        heatmap_ax2.text(center, -0.25, mname, transform=heatmap_ax2.get_xaxis_transform(), 
                 ha='center', va='top', fontsize=9, fontweight='normal')
    
    # Add dataset names below model names (centered across all models for that dataset)
    # Group by dataset only
    dataset_groups = []
    i = 0
    while i < len(datasets):
        curr_dataset = datasets[i]
        start = i
        while i < len(datasets) and datasets[i] == curr_dataset:
            i += 1
        dataset_groups.append((start, i-1, curr_dataset))
    
    for (s, e, dname) in dataset_groups:
        if not dname:
            continue
        center = (ticks[s] + ticks[e]) / 2.0
        heatmap_ax2.text(center, -0.38, dname, transform=heatmap_ax2.get_xaxis_transform(), 
                 ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Add separate colorbars for each heatmap
    # AUROC_f colorbar (aligned with top heatmap)
    cbar_ax1 = fig.add_axes([0.92, 0.21, 0.01, 0.09])  # [left, bottom, width, height]
    mappable1 = heatmap_ax1.collections[0]
    cbar1 = fig.colorbar(mappable1, cax=cbar_ax1, orientation='vertical')
    cbar1.set_label('ΔAUROC_f', fontsize=9)
    
    # AUGRC colorbar (aligned with bottom heatmap)
    cbar_ax2 = fig.add_axes([0.92, 0.09, 0.01, 0.09])  # [left, bottom, width, height]
    mappable2 = heatmap_ax2.collections[0]
    cbar2 = fig.colorbar(mappable2, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('ΔAUGRC', fontsize=9)
    
    # Add main title
    fig.suptitle('CSF Performances - Population & New Class Shift', 
                fontsize=20, fontweight='bold', y=0.99)
    
    return fig


def main(aggregation='mean'):
    """Main function to generate combined figure."""
    
    print("=" * 80)
    print("Combined Population Shift & New Class Shift Results Visualization")
    print(f"Aggregation: {aggregation.upper()}")
    print("=" * 80)
    
    # Get paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    pop_results_dir = results_dir / 'population_shifts'
    new_class_results_dir = results_dir / 'new_class_shifts'
    comp_eval_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results' / 'population_shift'
    
    print(f"Workspace root: {workspace_root}")
    print(f"Population shift results: {pop_results_dir}")
    print(f"New class shift results: {new_class_results_dir}")
    print(f"Comprehensive eval: {comp_eval_dir}")
    
    # Create output directory
    output_dir = results_dir / 'figures' / 'combined_population'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate combined figure
    fig = create_combined_figure(results_dir, pop_results_dir, new_class_results_dir, comp_eval_dir, aggregation)
    
    # Save
    output_path = output_dir / f'combined_results_{aggregation}_population_and_new_class_shift.png'
    fig.savefig(output_path, dpi=300)
    print(f"\n✓ Saved to {output_path}")
    
    plt.close(fig)
    
    print("\n" + "=" * 80)
    print("✓ Done!")
    print("=" * 80)


if __name__ == '__main__':
    aggregation = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    
    if aggregation not in ['mean', 'min', 'max', 'vote']:
        print(f"Unknown aggregation: {aggregation}")
        print("Usage: python plot_combined_population_results.py [mean|min|max|vote]")
        sys.exit(1)
    
    main(aggregation=aggregation)
