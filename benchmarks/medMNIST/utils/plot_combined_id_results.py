"""
Combined visualization of in-distribution UQ benchmark results.

Creates a single figure with:
- Top: 2x2 radar plots (ResNet18/ViT x AUROC_f/AUGRC)
- Bottom: 2 stacked heatmaps showing ensemble vs mean per-fold differences
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
from generate_radar_plots import create_radar_plot_on_axis, parse_results_directory
from plot_ensemble_vs_mean_heatmap import compute_differences, prepare_heatmap_data


def create_combined_figure(results_dir, id_results_dir, comp_eval_dir, aggregation='mean', shift='in_distribution'):
    """
    Create a combined figure with radar plots on top and heatmaps below.
    
    Args:
        results_dir: Main results directory
        id_results_dir: ID results directory with JSON files
        comp_eval_dir: Comprehensive evaluation directory
        aggregation: Aggregation strategy for radar plots
        shift: Shift type - 'in_distribution' or 'corruption_shifts'
    
    Returns:
        fig: matplotlib figure
    """
    # Parse results for radar plots
    results_auroc = parse_results_directory(id_results_dir, metric='auroc_f')
    results_augrc = parse_results_directory(id_results_dir, metric='augrc')
    
    # Load data for heatmaps (same logic as load_comprehensive_results in standalone script)
    json_files = list(id_results_dir.glob('uq_benchmark_*.json'))
    all_data = {}
    for json_file in json_files:
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
    
    # Compute differences for heatmaps
    auroc_diffs, augrc_diffs = compute_differences(all_data)
    
    # Prepare heatmap data
    auroc_matrix, methods, display_names = prepare_heatmap_data(auroc_diffs)
    augrc_matrix, _, _ = prepare_heatmap_data(augrc_diffs)
    
    # Sort columns by dataset, model, then setup order (standard, DA, DO, DADO)
    setup_order = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}
    
    def get_sort_key(name):
        # Remove suffixes
        clean_name = name.replace('_corrupt_severity3_test', '').replace('_test', '')
        
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
    
    # Create aggregated row for heatmaps
    agg_methods = ['MSR', 'MSR_calibrated', 'MLS', 'MCDropout', 'KNN_Raw', 'GPS']
    agg_methods_idx = [i for i, m in enumerate(methods) if m in agg_methods]
    auroc_agg_row = np.nanmean(auroc_matrix[agg_methods_idx, :], axis=0, keepdims=True)
    augrc_agg_row = np.nanmean(augrc_matrix[agg_methods_idx, :], axis=0, keepdims=True)
    
    auroc_matrix_with_agg = np.vstack([auroc_matrix, auroc_agg_row])
    augrc_matrix_with_agg = np.vstack([augrc_matrix, augrc_agg_row])
    methods_with_agg = methods + ['⚡ Mean Aggregation']
    
    # Create figure with GridSpec for complex layout
    fig = plt.figure(figsize=(19, 23))
    
    # Define grid: 3 rows (radar top half, heatmap1, heatmap2)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.85, 0.90, 0.25, 0.25], 
                         hspace=0.05, wspace=0.45,
                         left=0.10, right=0.90, top=0.96, bottom=0.06)
    
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
                metric=metric, aggregation=aggregation, shift=shift
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
    
    # Add legend for radar plots - centered between the 4 radars
    fig.legend(all_handles, all_labels, loc='center', ncol=2,
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.615))
    
    # ========================
    # Generate heatmaps
    # ========================
    print("\nGenerating heatmaps...")
    
    # Fixed colormap range for consistency across ID and corruption shifts
    vmin, vmax = -0.2, 0.2
    
    # AUROC_f heatmap
    sns.heatmap(auroc_matrix_with_agg,
                xticklabels=[],
                yticklabels=methods_with_agg,
                cmap='RdBu_r',
                center=0,
                vmin=vmin,
                vmax=vmax,
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
                vmin=vmin,
                vmax=vmax,
                annot=False,
                cbar=False,
                ax=heatmap_ax2)
    
    heatmap_ax2.set_title('AUGRC: Ensemble vs Mean Per-Fold',
                         fontsize=12, fontweight='bold', pad=5)
    heatmap_ax2.set_ylabel('')

    # Parse display names for multi-line x labels (exact copy from standalone script)
    setups = []
    models = []
    datasets = []
    setup_names = ['DA', 'DO', 'DADO']
    model_names = ['resnet18', 'vit_b_16']
    
    for name in display_names:
        # Remove common suffixes if present
        clean_name = name.replace('_corrupt_severity3_test', '').replace('_test', '')
        
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
    
    # Add colorbar for heatmaps
    cbar_ax = fig.add_axes([0.92, 0.07, 0.01, 0.18])  # Smaller and positioned to match heatmaps
    mappable = heatmap_ax2.collections[0]
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Difference (Ensemble - Mean Per-Fold)', fontsize=9)
    
    # Add main title
    title_shift = 'Corruption Shifts' if shift == 'corruption_shifts' else 'In Distribution'
    fig.suptitle(f'CSF Performances - {title_shift}', 
                fontsize=20, fontweight='bold', y=0.99)
    
    return fig


def main(aggregation='mean', shift='in_distribution'):
    """Main function to generate combined figure."""
    
    print("=" * 80)
    print("Combined Results Visualization")
    print(f"Aggregation: {aggregation.upper()}")
    print(f"Shift: {shift}")
    print("=" * 80)
    
    # Get paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    
    # Route to correct directories based on shift type
    if shift == 'corruption_shifts':
        id_results_dir = results_dir / 'corruption_shifts'
        comp_eval_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results' / 'corruption_shifts'
        output_subdir = 'combined_corruption'
    else:
        id_results_dir = results_dir / 'id_results'
        comp_eval_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results' / 'in_distribution'
        output_subdir = 'combined_id'
    
    print(f"Workspace root: {workspace_root}")
    print(f"Results directory: {id_results_dir}")
    print(f"Comprehensive eval: {comp_eval_dir}")
    
    # Create output directory
    output_dir = results_dir / 'figures' / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate combined figure
    fig = create_combined_figure(results_dir, id_results_dir, comp_eval_dir, aggregation, shift)
    
    # Save
    output_path = output_dir / f'combined_results_{aggregation}_{shift}.png'
    fig.savefig(output_path, dpi=300)
    print(f"\n✓ Saved to {output_path}")
    
    plt.close(fig)
    
    print("\n" + "=" * 80)
    print("✓ Done!")
    print("=" * 80)


if __name__ == '__main__':
    aggregation = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    shift = sys.argv[2] if len(sys.argv) > 2 else 'in_distribution'
    
    if aggregation not in ['mean', 'min', 'max', 'vote']:
        print(f"Unknown aggregation: {aggregation}")
        print("Usage: python plot_combined_id_results.py [mean|min|max|vote] [in_distribution|corruption_shifts]")
        sys.exit(1)
    
    if shift not in ['in_distribution', 'corruption_shifts']:
        print(f"Unknown shift type: {shift}")
        print("Usage: python plot_combined_id_results.py [mean|min|max|vote] [in_distribution|corruption_shifts]")
        sys.exit(1)
    
    main(aggregation=aggregation, shift=shift)
