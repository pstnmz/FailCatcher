import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_comprehensive_results(base_path):
    """Load all UQ benchmark results for ID datasets."""
    results = {}
    
    # Scan the directory for all JSON files
    json_files = sorted(base_path.glob('uq_benchmark_*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    for filepath in json_files:
        # Parse filename: uq_benchmark_{dataset}_{model}_{setup}_{timestamp}.json
        filename = filepath.stem  # Remove .json
        parts = filename.replace('uq_benchmark_', '').rsplit('_', 1)  # Split off timestamp
        
        if len(parts) == 2:
            prefix, timestamp = parts
            # Now split prefix into dataset, model, setup
            prefix_parts = prefix.split('_')
            
            # Handle cases like dermamnist-e-id (contains hyphens)
            # Last 2 parts are model and setup
            if len(prefix_parts) >= 2:
                setup = prefix_parts[-1]
                model = prefix_parts[-2]
                dataset = '_'.join(prefix_parts[:-2])
                
                # Load the data
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    key = f"{dataset}_{model}_{setup}"
                    results[key] = data
                    print(f"  Loaded: {key}")
            else:
                print(f"  Skipped (unexpected format): {filename}")
        else:
            print(f"  Skipped (no timestamp): {filename}")
    
    return results

def compute_differences(results):
    """
    Compute difference between ensemble and mean per-fold for AUROC_f and AUGRC.
    Returns two dictionaries: auroc_f_diff and augrc_diff
    Each has structure: {method: {column_key: difference_value}}
    """
    auroc_f_diff = {}
    augrc_diff = {}
    
    # All possible methods (CSFs)
    all_methods = set()
    
    # First pass: collect all methods (exclude Ensembling)
    for key, data in results.items():
        if 'methods' in data:
            for method in data['methods'].keys():
                if method != 'Ensembling':  # Skip Ensembling (no per-fold version)
                    all_methods.add(method)
    
    # Initialize dictionaries
    for method in all_methods:
        auroc_f_diff[method] = {}
        augrc_diff[method] = {}
    
    # Second pass: compute differences
    for key, data in results.items():
        if 'methods' not in data:
            continue
        
        for method in all_methods:
            if method not in data['methods']:
                continue
            
            method_data = data['methods'][method]
            
            # Get ensemble values
            ensemble_auroc_f = method_data.get('auroc_f')
            ensemble_augrc = method_data.get('augrc')
            
            # Get per-fold values
            per_fold_metrics = method_data.get('per_fold_metrics', [])
            
            if per_fold_metrics and len(per_fold_metrics) > 0:
                # Compute mean per-fold
                per_fold_auroc_f = [fold.get('auroc_f') for fold in per_fold_metrics if fold.get('auroc_f') is not None]
                per_fold_augrc = [fold.get('augrc') for fold in per_fold_metrics if fold.get('augrc') is not None]
                
                if per_fold_auroc_f and ensemble_auroc_f is not None:
                    mean_fold_auroc_f = np.mean(per_fold_auroc_f)
                    auroc_f_diff[method][key] = ensemble_auroc_f - mean_fold_auroc_f
                
                if per_fold_augrc and ensemble_augrc is not None:
                    mean_fold_augrc = np.mean(per_fold_augrc)
                    augrc_diff[method][key] = ensemble_augrc - mean_fold_augrc
    
    return auroc_f_diff, augrc_diff

def prepare_heatmap_data(data_dict):
    """Prepare data for heatmap plotting."""
    methods = sorted(data_dict.keys())
    # Remove timestamps from column names (keep only dataset_model_setup)
    columns_with_keys = sorted(set(col for method_data in data_dict.values() for col in method_data.keys()))
    # Create mapping: original_key -> display_name (without timestamp)
    key_to_display = {}
    for col in columns_with_keys:
        # Split by underscores and remove last part if it looks like a timestamp
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
            key_to_display[col] = parts[0]  # Remove timestamp
        else:
            key_to_display[col] = col
    
    # Get unique display names (some may map to same display name)
    display_names = []
    columns = []
    seen_display = set()
    for col in columns_with_keys:
        display = key_to_display[col]
        if display not in seen_display:
            display_names.append(display)
            columns.append(col)
            seen_display.add(display)
    
    # Create matrix
    matrix = np.full((len(methods), len(columns)), np.nan)
    
    for i, method in enumerate(methods):
        for j, col in enumerate(columns):
            if col in data_dict[method]:
                matrix[i, j] = data_dict[method][col]
    
    return matrix, methods, display_names

def create_combined_heatmap(auroc_f_diff, augrc_diff, output_path):
    """Create a combined figure with both heatmaps stacked vertically."""
    # Prepare data for both heatmaps
    auroc_matrix, methods, display_names = prepare_heatmap_data(auroc_f_diff)
    augrc_matrix, _, _ = prepare_heatmap_data(augrc_diff)
    
    # Add aggregated row (mean across MSR, MSR_calibrated, MLS, MCDropout, KNN_Raw, GPS)
    aggregation_methods = ['MSR', 'MSR_calibrated', 'MLS', 'MCDropout', 'KNN_Raw', 'GPS']
    
    # Find indices of aggregation methods
    agg_indices = [i for i, m in enumerate(methods) if m in aggregation_methods]
    
    # Compute aggregated row (mean across selected methods)
    auroc_agg_row = np.nanmean(auroc_matrix[agg_indices, :], axis=0).reshape(1, -1)
    augrc_agg_row = np.nanmean(augrc_matrix[agg_indices, :], axis=0).reshape(1, -1)
    
    # Append aggregated row
    auroc_matrix_with_agg = np.vstack([auroc_matrix, auroc_agg_row])
    augrc_matrix_with_agg = np.vstack([augrc_matrix, augrc_agg_row])
    methods_with_agg = methods + ['⚡ Mean Aggregation']

    # Create figure with two subplots stacked vertically
    # Increased height and width to prevent overlap
    row_height = 0.4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(22, len(display_names) * 0.35), 
                                                    len(methods_with_agg) * row_height * 2.5))
    
    # Find common vmax for symmetric colormap across both plots
    abs_max = max(np.nanmax(np.abs(auroc_matrix_with_agg)), np.nanmax(np.abs(augrc_matrix_with_agg)))
    vmin, vmax = -abs_max, abs_max
    
    # AUROC_f heatmap (top) — draw without a colorbar (we will add a single shared colorbar)
    sns.heatmap(auroc_matrix_with_agg, 
                xticklabels=[],  # No x labels on top plot
                yticklabels=methods_with_agg,
                cmap='RdBu_r',
                center=0,
                vmin=vmin,
                vmax=vmax,
                annot=False,
                fmt='.3f',
                cbar=False,
                ax=ax1)
    
    ax1.set_title('AUROC_f: Ensemble vs Mean Per-Fold', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('')
    ax1.set_ylabel('', fontsize=11, fontweight='bold')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # AUGRC heatmap (bottom) — no colorbar
    sns.heatmap(augrc_matrix_with_agg, 
                xticklabels=display_names, 
                yticklabels=methods_with_agg,
                cmap='RdBu_r',
                center=0,
                vmin=vmin,
                vmax=vmax,
                annot=False,
                fmt='.3f',
                cbar=False,
                ax=ax2)
    
    ax2.set_title('AUGRC: Ensemble vs Mean Per-Fold', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('', fontsize=11, fontweight='bold')
    ax2.set_ylabel('', fontsize=11, fontweight='bold')

    # Parse display names to get setup, model, dataset for multi-line x labels
    setups = []
    models = []
    datasets = []
    setup_names = ['DA', 'DO', 'DADO']
    model_names = ['resnet18', 'vit_b_16']
    
    for name in display_names:
        # Try to identify setup, model, dataset properly
        setup = 'standard'
        model = None
        dataset = None
        
        # Check if ends with setup name
        col_without_setup = name
        for setup_name in setup_names:
            if name.endswith('_' + setup_name):
                setup = setup_name
                col_without_setup = name[:-len(setup_name)-1]
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
    ax2.set_xticklabels(setups, rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0)

    # Get tick positions
    ticks = ax2.get_xticks()
    if len(ticks) == 0:
        ticks = np.arange(len(display_names))
    
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
        ax2.text(center, -0.20, mname, transform=ax2.get_xaxis_transform(), 
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
        ax2.text(center, -0.38, dname, transform=ax2.get_xaxis_transform(), 
                 ha='center', va='top', fontsize=9, fontweight='bold')

    # Adjust layout to make room for labels and colorbar
    # Don't use tight_layout as it interferes with colorbar positioning
    plt.subplots_adjust(bottom=0.18, right=0.82, top=0.95, left=0.08)
    
    # Add a single shared colorbar on the right for both heatmaps
    # Create a new axes for the colorbar manually
    cbar_ax = fig.add_axes([0.85, 0.18, 0.015, 0.77])  # [left, bottom, width, height] - thinner colorbar
    mappable = ax2.collections[0]
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Difference (Ensemble - Mean Per-Fold)')
    
    # Save
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined heatmap to {output_path}")
    
    return fig

def main():
    # Paths
    base_path = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/id_results')
    output_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/figures/ensemble_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading comprehensive evaluation results...")
    results = load_comprehensive_results(base_path)
    print(f"Loaded {len(results)} dataset/model/setup combinations")
    
    # Compute differences
    print("\nComputing ensemble vs mean per-fold differences...")
    auroc_f_diff, augrc_diff = compute_differences(results)
    
    # Print summary statistics
    print("\nAUROC_f Differences (Ensemble - Mean Per-Fold):")
    for method in sorted(auroc_f_diff.keys()):
        values = list(auroc_f_diff[method].values())
        if values:
            print(f"  {method:25s}: mean={np.mean(values):+.4f}, std={np.std(values):.4f}, "
                  f"min={np.min(values):+.4f}, max={np.max(values):+.4f}")
    
    print("\nAUGRC Differences (Ensemble - Mean Per-Fold):")
    for method in sorted(augrc_diff.keys()):
        values = list(augrc_diff[method].values())
        if values:
            print(f"  {method:25s}: mean={np.mean(values):+.4f}, std={np.std(values):.4f}, "
                  f"min={np.min(values):+.4f}, max={np.max(values):+.4f}")
    
    # Create combined heatmap
    print("\nGenerating combined heatmap...")
    
    # Combined heatmap output
    combined_output = output_dir / 'ensemble_vs_mean_combined_heatmap.png'
    create_combined_heatmap(auroc_f_diff, augrc_diff, combined_output)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
