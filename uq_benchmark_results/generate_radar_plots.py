"""
Generate radar plots for UQ benchmark results across datasets and model configurations.

Creates two large radar plots (ResNet18 and ViT) showing mean AUROC_f values
for different UQ methods across all dataset-setup combinations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def parse_results_directory(results_dir='./'):
    """
    Parse all JSON result files in the directory.
    
    Returns:
        dict: Nested dictionary structure:
            {
                'resnet18': {
                    'breastmnist_standard': {
                        'MSR': 0.75,
                        'MSR_platt': 0.82,
                        ...
                    },
                    ...
                },
                'vit_b_16': {
                    ...
                }
            }
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    results_dir = Path(results_dir)
    json_files = list(results_dir.glob('uq_benchmark_*.json'))
    
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            flag = data.get('flag', '')
            model_backbone = data.get('model_backbone', 'unknown')
            setup = data.get('setup', 'standard')
            
            # Create dataset-setup key
            if setup and setup != 'standard':
                dataset_key = f"{flag}_{setup}"
            else:
                dataset_key = f"{flag}_standard"
            
            # Extract method results
            methods_data = data.get('methods', {})
            for method_name, method_results in methods_data.items():
                # Get AUROC_f (use mean if available, otherwise single value)
                if 'auroc_f_mean' in method_results:
                    auroc = method_results['auroc_f_mean']
                elif 'auroc_f' in method_results:
                    auroc = method_results['auroc_f']
                else:
                    continue
                
                results[model_backbone][dataset_key][method_name] = float(auroc)
            
            print(f"  Loaded: {model_backbone} - {dataset_key} ({len(methods_data)} methods)")
        
        except Exception as e:
            print(f"  ⚠️  Failed to parse {json_file.name}: {e}")
    
    return dict(results)


def get_dataset_accuracy(results_dir, dataset_key, model_name='resnet18'):
    """
    Get the test accuracy for a dataset from its JSON file.
    
    Args:
        results_dir: Path to results directory
        dataset_key: Dataset key (e.g., 'breastmnist_standard')
        model_name: Model name to filter by
    
    Returns:
        float: Test accuracy, or 0 if not found
    """
    results_dir = Path(results_dir)
    # Look for JSON files matching this dataset and model
    pattern = f"uq_benchmark_{dataset_key.replace('_standard', '')}_{model_name}_*.json"
    json_files = list(results_dir.glob(pattern))
    
    if not json_files:
        return 0.0
    
    # Use the most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('test_accuracy', 0.0)
    except:
        return 0.0


def create_radar_plot(model_results, model_name, output_path, results_dir=None):
    """
    Create a radar plot for a single model showing all dataset-setup combinations.
    
    Args:
        model_results: dict mapping dataset_key -> method -> auroc
        model_name: Name of the model (e.g., 'resnet18', 'vit_b_16')
        output_path: Path to save the figure
        results_dir: Path to results directory (for sorting by accuracy)
    """
    # Group datasets by family (base dataset name)
    dataset_families = defaultdict(list)
    for dataset_key in model_results.keys():
        # Extract base dataset name (before _setup)
        base_name = dataset_key.rsplit('_', 1)[0] if '_' in dataset_key else dataset_key
        # If it ends with _standard, remove it to get true base name
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        dataset_families[base_name].append(dataset_key)
    
    # Sort setups within each family: standard first, then alphabetically
    setup_order = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}
    for base_name in dataset_families:
        dataset_families[base_name].sort(key=lambda k: (
            setup_order.get(k.split('_')[-1], 99),  # Setup priority
            k  # Then alphabetically
        ))
    
    # Sort families by their standard variant's accuracy (ascending)
    if results_dir and model_name == 'resnet18':
        family_order = []
        for base_name, variants in dataset_families.items():
            # Find the standard variant
            standard_key = f"{base_name}_standard"
            if standard_key not in variants:
                # If no standard, use first variant
                standard_key = variants[0]
            accuracy = get_dataset_accuracy(results_dir, standard_key, 'resnet18')
            family_order.append((accuracy, base_name))
        
        # Sort families by accuracy (ascending)
        family_order.sort()
        
        # Build final dataset list: each family's variants in order
        dataset_keys = []
        for _, base_name in family_order:
            dataset_keys.extend(dataset_families[base_name])
    else:
        # Default: alphabetical by family, then by setup
        dataset_keys = []
        for base_name in sorted(dataset_families.keys()):
            dataset_keys.extend(dataset_families[base_name])
    
    num_datasets = len(dataset_keys)
    
    if num_datasets == 0:
        print(f"No data for {model_name}, skipping radar plot")
        return
    
    # Get all unique methods across datasets
    all_methods = set()
    for dataset_data in model_results.values():
        all_methods.update(dataset_data.keys())
    all_methods = sorted(all_methods)
    
    print(f"\nCreating radar plot for {model_name}")
    print(f"  Datasets: {num_datasets}")
    print(f"  Methods: {len(all_methods)}")
    
    # Set up the angles for radar plot with family clustering
    # Count families and datasets per family
    family_sizes = defaultdict(int)
    for dataset_key in dataset_keys:
        base_name = dataset_key.rsplit('_', 1)[0] if '_' in dataset_key else dataset_key
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        family_sizes[base_name] += 1
    
    num_families = len(family_sizes)
    
    # Calculate spacing: distribute 2π across families and their members
    # Allocate more space to families with more members
    total_members = len(dataset_keys)
    base_angle_per_dataset = 2 * np.pi / total_members
    
    # Reduce spacing within families, add gaps between families
    within_family_factor = 0.6  # Make family members closer (60% of normal spacing)
    between_family_gap = base_angle_per_dataset * 1.5  # Extra gap between families
    
    angles = []
    current_angle = 0
    current_family = None
    
    for dataset_key in dataset_keys:
        base_name = dataset_key.rsplit('_', 1)[0] if '_' in dataset_key else dataset_key
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        
        if current_family is None:
            # First dataset
            current_family = base_name
            angles.append(current_angle)
        elif current_family != base_name:
            # New family - add gap
            current_angle += between_family_gap
            current_family = base_name
            angles.append(current_angle)
        else:
            # Same family - reduced spacing
            current_angle += base_angle_per_dataset * within_family_factor
            angles.append(current_angle)
    
    angles = np.array(angles).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    # Color map for methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_methods)))
    
    # Plot each method (lines only, no fill)
    for method_idx, method_name in enumerate(all_methods):
        values = []
        for dataset_key in dataset_keys:
            auroc = model_results[dataset_key].get(method_name, np.nan)
            values.append(auroc)
        
        # Complete the circle
        values += values[:1]
        
        # Plot lines only (removed fill)
        ax.plot(angles, values, 'o-', linewidth=2.5, label=method_name, 
                color=colors[method_idx], markersize=7, markeredgewidth=1.5,
                markeredgecolor='white')
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dataset_keys, size=9)
    
    # Set y-axis (AUROC range)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], size=10)
    ax.set_ylabel('AUROC_f', size=12, labelpad=30)
    
    # Grid
    ax.grid(True, linewidth=0.5, alpha=0.5)
    
    # Title
    model_display_name = model_name.replace('_', ' ').upper()
    ax.set_title(f'UQ Methods Performance - {model_display_name}\n(Mean AUROC_f across datasets)',
                 size=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def generate_summary_table(results, output_path):
    """
    Generate a summary table showing all results in CSV format.
    
    Args:
        results: Parsed results dictionary
        output_path: Path to save CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Model', 'Dataset', 'Method', 'AUROC_f'])
        
        # Data
        for model_name, model_results in sorted(results.items()):
            for dataset_key, dataset_data in sorted(model_results.items()):
                for method_name, auroc in sorted(dataset_data.items()):
                    writer.writerow([model_name, dataset_key, method_name, f"{auroc:.4f}"])
    
    print(f"\n✓ Summary table saved to {output_path}")


def main():
    """Main function to generate radar plots from benchmark results."""
    
    print("=" * 80)
    print("UQ Benchmark Radar Plot Generator")
    print("=" * 80)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    print(f"Looking for JSON files in: {script_dir}")
    
    # Parse results from the script's directory
    results = parse_results_directory(script_dir)
    
    if not results:
        print("\n⚠️  No results found! Make sure JSON files are in the script directory.")
        return
    
    # Create output directory for plots in the script's directory
    output_dir = script_dir / 'radar_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Generate radar plot for each model
    for model_name, model_results in results.items():
        output_path = output_dir / f'radar_plot_{model_name}.png'
        create_radar_plot(model_results, model_name, output_path, results_dir=script_dir)
    
    # Generate summary table
    summary_path = output_dir / 'results_summary.csv'
    generate_summary_table(results, summary_path)
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
