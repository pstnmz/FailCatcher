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
    
    # First pass: collect all methods
    for key, data in results.items():
        if 'methods' in data:
            for method in data['methods'].keys():
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

def create_heatmap(data_dict, title, output_path):
    """Create a heatmap from the difference dictionary."""
    # Convert to DataFrame format
    methods = sorted(data_dict.keys())
    columns = sorted(set(col for method_data in data_dict.values() for col in method_data.keys()))
    
    # Create matrix
    matrix = np.full((len(methods), len(columns)), np.nan)
    
    for i, method in enumerate(methods):
        for j, col in enumerate(columns):
            if col in data_dict[method]:
                matrix[i, j] = data_dict[method][col]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(columns) * 0.5), max(8, len(methods) * 0.4)))
    
    # Find vmax for symmetric colormap
    abs_max = np.nanmax(np.abs(matrix))
    vmin, vmax = -abs_max, abs_max
    
    # Create heatmap
    sns.heatmap(matrix, 
                xticklabels=columns, 
                yticklabels=methods,
                cmap='RdBu_r',  # Red for negative (ensemble worse), Blue for positive (ensemble better)
                center=0,
                vmin=vmin,
                vmax=vmax,
                annot=False,  # Set to True to show values
                fmt='.3f',
                cbar_kws={'label': f'{title} Difference\n(Ensemble - Mean Per-Fold)'},
                ax=ax)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset_Model_Setup', fontsize=11, fontweight='bold')
    ax.set_ylabel('Uncertainty Method (CSF)', fontsize=11, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    
    return fig, ax

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
    
    # Create heatmaps
    print("\nGenerating heatmaps...")
    
    # AUROC_f heatmap
    auroc_output = output_dir / 'ensemble_vs_mean_auroc_f_heatmap.png'
    create_heatmap(auroc_f_diff, 
                   'AUROC_f: Ensemble vs Mean Per-Fold',
                   auroc_output)
    
    # AUGRC heatmap
    augrc_output = output_dir / 'ensemble_vs_mean_augrc_heatmap.png'
    create_heatmap(augrc_diff,
                   'AUGRC: Ensemble vs Mean Per-Fold', 
                   augrc_output)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
