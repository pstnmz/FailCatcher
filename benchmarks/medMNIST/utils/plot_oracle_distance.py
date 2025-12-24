"""
Plot distance to oracle AUGRC for each method and setup.

For each dataset, creates a grid of subplots (8 methods x 4 setups) showing:
- Full bar height = oracle AUGRC (per-fold mean, or ensemble for Ensembling method)
- Filled portion = actual CSF AUGRC (per-fold mean)
- Percentage displayed inside bar
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def compute_oracle_augrc(accuracy):
    """
    Compute oracle AUGRC - theoretical best AUGRC given accuracy.
    
    Args:
        accuracy: Model accuracy (between 0 and 1)
    
    Returns:
        float: Oracle AUGRC = 0.5 * (1 - accuracy)^2
    """
    return 0.5 * (1.0 - accuracy) ** 2


def get_ensemble_accuracy(runs_dir, dataset_key, model_name='resnet18'):
    """Get ensemble balanced accuracy from runs directory."""
    runs_dir = Path(runs_dir)
    
    # Parse dataset_key
    if '_' in dataset_key and dataset_key.split('_')[-1] in ['standard', 'DA', 'DO', 'DADO']:
        parts = dataset_key.rsplit('_', 1)
        dataset_name = parts[0]
        setup = parts[1]
    else:
        dataset_name = dataset_key
        setup = 'standard'
    
    # Handle dermamnist-e-id -> dermamnist-e mapping
    dataset_name_for_runs = dataset_name.replace('-id', '')
    
    # Try to find metrics_ensemble.json
    dataset_dir = runs_dir / dataset_name_for_runs
    if not dataset_dir.exists():
        return 0.0
    
    # Look for model directories
    model_dirs = list(dataset_dir.glob(f'{model_name}_*'))
    if not model_dirs:
        return 0.0
    
    # Use most recent directory
    model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
    metrics_file = model_dir / 'metrics_ensemble.json'
    
    if not metrics_file.exists():
        return 0.0
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        metrics = data.get('metrics', {})
        return metrics.get('balanced_accuracy', 0.0)
    except:
        return 0.0


def load_per_fold_augrc(results_dir, dataset_key, model_name='resnet18'):
    """
    Load per-fold AUGRC values for all methods from JSON files.
    
    Returns:
        dict: {method_name: per_fold_augrc_array} where per_fold_augrc_array is [fold0, fold1, ...]
    """
    results_dir = Path(results_dir)
    
    # Parse dataset_key
    if '_' in dataset_key and dataset_key.split('_')[-1] in ['standard', 'DA', 'DO', 'DADO']:
        parts = dataset_key.rsplit('_', 1)
        dataset_name = parts[0]
        setup = parts[1]
    else:
        dataset_name = dataset_key
        setup = 'standard'
    
    # Construct JSON filename pattern
    if setup == 'standard':
        pattern = f"uq_benchmark_{dataset_name}_{model_name}_*.json"
    else:
        pattern = f"uq_benchmark_{dataset_name}_{model_name}_{setup}_*.json"
    
    # Search directly in results_dir (which is already id_results from main())
    all_json_files = list(results_dir.glob(pattern))
    
    # Filter for standard setup (exclude DA, DO, DADO in filename)
    if setup == 'standard':
        json_files = [f for f in all_json_files if not any(
            f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO']
        )]
    else:
        json_files = all_json_files
    
    if not json_files:
        return {}
    
    # Use most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract per-fold AUGRC for each method
        per_fold_data = {}
        methods_dict = data.get('methods', {})
        
        for method_name, method_data in methods_dict.items():
            per_fold_metrics = method_data.get('per_fold_metrics', [])
            if per_fold_metrics:
                # Extract AUGRC from each fold
                per_fold_augrc = [fold.get('augrc', np.nan) for fold in per_fold_metrics]
                per_fold_data[method_name] = np.array(per_fold_augrc)
        
        return per_fold_data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {}


def load_per_fold_accuracy(results_dir, dataset_key, model_name='resnet18'):
    """
    Load per-fold accuracy values from JSON files.
    
    Returns:
        array: per_fold_accuracy [fold0, fold1, ...]
    """
    results_dir = Path(results_dir)
    
    # Parse dataset_key
    if '_' in dataset_key and dataset_key.split('_')[-1] in ['standard', 'DA', 'DO', 'DADO']:
        parts = dataset_key.rsplit('_', 1)
        dataset_name = parts[0]
        setup = parts[1]
    else:
        dataset_name = dataset_key
        setup = 'standard'
    
    # Construct JSON filename pattern
    if setup == 'standard':
        pattern = f"uq_benchmark_{dataset_name}_{model_name}_*.json"
    else:
        pattern = f"uq_benchmark_{dataset_name}_{model_name}_{setup}_*.json"
    
    # Search directly in results_dir (which is already id_results from main())
    all_json_files = list(results_dir.glob(pattern))
    
    # Filter for standard setup
    if setup == 'standard':
        json_files = [f for f in all_json_files if not any(
            f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO']
        )]
    else:
        json_files = all_json_files
    
    if not json_files:
        return None
    
    # Use most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get per-fold accuracy from any method (they all have the same folds)
        methods_dict = data.get('methods', {})
        if not methods_dict:
            return None
        
        # Pick first method and extract accuracy from each fold
        first_method = next(iter(methods_dict.values()))
        per_fold_metrics = first_method.get('per_fold_metrics', [])
        
        if per_fold_metrics:
            per_fold_acc = [fold.get('accuracy', np.nan) for fold in per_fold_metrics]
            return np.array(per_fold_acc)
        
        return None
    except Exception as e:
        print(f"Error loading accuracy from {json_file}: {e}")
        return None


def plot_oracle_distance_for_dataset(dataset_name, model_names, results_dir, runs_dir, ax):
    """
    Create oracle distance plot for one dataset with both ResNet18 and ViT models on a given axis.
    
    Args:
        dataset_name: Base dataset name (e.g., 'breastmnist')
        model_names: List of model names ['resnet18', 'vit_b_16']
        results_dir: Path to results directory
        runs_dir: Path to runs directory
        ax: Matplotlib axis to plot on
    """
    setups = ['standard', 'DA', 'DO', 'DADO']
    # Methods sorted alphabetically to match radar plots - MCDropout only in DO/DADO
    methods = ['Ensembling', 'GPS', 'KNN_Raw', 'MLS', 'MSR', 'MSR_calibrated', 'TTA']
    methods_with_dropout = ['Ensembling', 'GPS', 'KNN_Raw', 'MCDropout', 'MLS', 'MSR', 'MSR_calibrated', 'TTA']
    
    # Define colors for each method (same as radar plots - tab20 colormap, sorted order)
    # Use the full method list with MCDropout for consistent coloring
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_with_dropout)))
    method_colors = {method: colors[i] for i, method in enumerate(methods_with_dropout)}
    
    # Set background
    ax.set_facecolor('#F8F9FA')
    ax.set_title(f'{dataset_name.upper()} - Oracle AUGRC Performance', 
                 fontsize=18, fontweight='bold', pad=10)
    
    # Collect all data across models and setups
    all_data = []
    bar_width = 0.09
    
    x_offset = 0
    for model_idx, model_name in enumerate(model_names):
        for setup_idx, setup in enumerate(setups):
            # Construct dataset key
            dataset_key = f'{dataset_name}_{setup}'
            
            # Load per-fold AUGRC data
            per_fold_data = load_per_fold_augrc(results_dir, dataset_key, model_name)
            per_fold_acc = load_per_fold_accuracy(results_dir, dataset_key, model_name)
            ensemble_acc = get_ensemble_accuracy(runs_dir, dataset_key, model_name)
            
            # Select methods based on setup (MCDropout only in DO/DADO)
            current_methods = methods_with_dropout if setup in ['DO', 'DADO'] else methods
            
            for method_idx, method in enumerate(current_methods):
                # Special handling for Ensembling - no per-fold data
                if method == 'Ensembling':
                    # Get Ensembling AUGRC from JSON (single value, not per-fold)
                    if setup == 'standard':
                        pattern = f"uq_benchmark_{dataset_name}_{model_name}_*.json"
                    else:
                        pattern = f"uq_benchmark_{dataset_name}_{model_name}_{setup}_*.json"
                    
                    json_files = list(results_dir.glob(pattern))
                    if setup == 'standard':
                        json_files = [f for f in json_files if not any(f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO'])]
                    
                    if json_files:
                        json_file = max(json_files, key=lambda p: p.stat().st_mtime)
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            if 'Ensembling' in data['methods']:
                                actual_augrc = data['methods']['Ensembling'].get('augrc', 0.0)
                                oracle_augrc = compute_oracle_augrc(ensemble_acc) if ensemble_acc > 0 else 0.0
                                
                                if oracle_augrc > 0 and actual_augrc > 0:
                                    x_pos = x_offset + method_idx * bar_width
                                    all_data.append({
                                        'x_pos': x_pos,
                                        'oracle': oracle_augrc,
                                        'actual': actual_augrc,
                                        'method': method,
                                        'model': model_name,
                                        'setup': setup,
                                        'color': method_colors[method]
                                    })
                        except:
                            pass
                    continue
                
                if method not in per_fold_data:
                    continue
                
                # Get actual AUGRC (mean across folds)
                actual_augrc = np.mean(per_fold_data[method])
                
                # Compute oracle AUGRC
                if method == 'Ensembling':
                    oracle_augrc = compute_oracle_augrc(ensemble_acc) if ensemble_acc > 0 else 0.0
                else:
                    if per_fold_acc is not None and len(per_fold_acc) > 0:
                        mean_acc = np.mean(per_fold_acc)
                        oracle_augrc = compute_oracle_augrc(mean_acc)
                    else:
                        oracle_augrc = 0.0
                
                # Skip if no valid oracle
                if oracle_augrc == 0.0:
                    continue
                
                # Position for this bar
                x_pos = x_offset + method_idx * bar_width
                
                all_data.append({
                    'x_pos': x_pos,
                    'oracle': oracle_augrc,
                    'actual': actual_augrc,
                    'method': method,
                    'model': model_name,
                    'setup': setup,
                    'color': method_colors[method]
                })
            
            # Add small spacing between setups
            x_offset += len(current_methods) * bar_width + 0.15
        
        # Add larger spacing between models
        x_offset += 0.25
    
    # Plot bars: oracle (transparent) from 0.5 to oracle value, actual (solid) from 0.5 to actual value
    for data in all_data:
        x_pos = data['x_pos']
        oracle_val = data['oracle']
        actual_val = data['actual']
        color = data['color']
        
        # Oracle bar from 0.5 to oracle (transparent)
        oracle_height = 0.5 - oracle_val
        ax.bar(x_pos, oracle_height, width=bar_width, bottom=oracle_val,
               color=color, edgecolor='gray', linewidth=1.0, alpha=0.25, zorder=2)
        
        # Actual bar from 0.5 to actual (solid)
        actual_height = 0.5 - actual_val
        ax.bar(x_pos, actual_height, width=bar_width, bottom=actual_val,
               color=color, edgecolor='white', linewidth=1.0, alpha=0.9, zorder=3)
        
        # Add percentage text inside the actual bar (rotated 90 degrees)
        if oracle_val > 0 and actual_height > 0:
            # Percentage: how much of the oracle bar (0.5 to oracle) is filled by actual bar (0.5 to actual)
            pct = ((0.5 - actual_val) / (0.5 - oracle_val)) * 100
            # Position text in middle of actual bar
            text_y = actual_val + actual_height / 2
            if actual_height > 0.02:  # Only show if bar is tall enough
                ax.text(x_pos, text_y, f'{pct:.2f}%', 
                       ha='center', va='center', fontsize=11, fontweight='black', 
                       color='black', rotation=90, zorder=4)
    
    # Formatting with improved aesthetics
    ax.set_yscale('log')  # Apply log scale to better visualize small AUGRC differences
    ax.set_ylim(0.5, 0.0001)  # Start at 0.5 (bottom, worst) and go to small value near 0 (top, best)
    ax.set_ylabel('AUGRC (log scale)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_xlim(-0.2, x_offset - 0.25)
    ax.set_xticks([])  # Remove x-ticks since we have setup labels
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=1.2, color='#BCC1C7', zorder=0, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', labelsize=12, width=2, length=6, which='both')
    ax.invert_yaxis()  # Invert so 0.5 is at bottom and small values at top


def main():
    """Generate combined oracle distance plot for all ID datasets."""
    
    print("="*80)
    print("Oracle AUGRC Distance Plot Generator")
    print("="*80)
    
    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent.parent.parent / 'uq_benchmark_results' / 'id_results'
    runs_dir = script_dir.parent / 'runs'
    output_dir = script_dir / 'oracle_plots'
    output_dir.mkdir(exist_ok=True)
    
    # ID datasets
    datasets = ['tissuemnist', 'dermamnist-e-id', 'breastmnist', 'pneumoniamnist', 
                'octmnist', 'organamnist', 'bloodmnist']
    
    # Models (both on same plot)
    models = ['resnet18', 'vit_b_16']
    
    print()
    print("="*80)
    print(f"Creating combined plot with all 7 datasets")
    print("="*80)
    
    # Create single figure with all datasets vertically stacked
    fig, axes = plt.subplots(7, 1, figsize=(24, 54), facecolor='white')
    fig.suptitle('Oracle AUGRC Distance - All Datasets', fontsize=28, fontweight='bold', y=0.995)
    
    # Plot each dataset on its own axis
    for idx, dataset in enumerate(datasets):
        print(f"  Plotting {dataset}...")
        plot_oracle_distance_for_dataset(dataset, models, results_dir, runs_dir, axes[idx])
    
    # Add model and setup labels once at the top (using first axis for positioning)
    ax_top = axes[0]
    setups = ['standard', 'DA', 'DO', 'DADO']
    methods = ['Ensembling', 'GPS', 'KNN_Raw', 'MLS', 'MSR', 'MSR_calibrated', 'TTA']
    methods_with_dropout = ['Ensembling', 'GPS', 'KNN_Raw', 'MCDropout', 'MLS', 'MSR', 'MSR_calibrated', 'TTA']
    bar_width = 0.09
    
    x_offset = 0
    for model_idx, model_name in enumerate(models):
        # Calculate model center considering variable method counts
        model_width = 0
        for setup in setups:
            current_methods = methods_with_dropout if setup in ['DO', 'DADO'] else methods
            model_width += len(current_methods) * bar_width + 0.15
        model_width -= 0.15  # Remove last spacing
        
        model_center = x_offset + model_width / 2
        ax_top.text(model_center, 1.18, model_name.replace('_', ' ').upper(), ha='center', va='bottom',
                   fontsize=16, fontweight='bold', transform=ax_top.transAxes,
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', edgecolor='darkblue', 
                            alpha=0.7, linewidth=2))
        
        for setup_idx, setup in enumerate(setups):
            current_methods = methods_with_dropout if setup in ['DO', 'DADO'] else methods
            # Calculate relative position for this setup
            setup_x_start = x_offset
            setup_width = len(current_methods) * bar_width
            center_x = setup_x_start + setup_width / 2
            
            # Convert to axis coordinates (0 to 1)
            xlim_range = ax_top.get_xlim()
            center_x_norm = (center_x - xlim_range[0]) / (xlim_range[1] - xlim_range[0])
            
            ax_top.text(center_x_norm, 1.10, setup.upper(), ha='center', va='bottom', 
                       fontsize=12, fontweight='bold', transform=ax_top.transAxes,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8))
            
            x_offset += len(current_methods) * bar_width + 0.15
        
        x_offset += 0.25
    
    # Add legend once at top right of the entire figure
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_with_dropout)))
    method_colors = {method: colors[i] for i, method in enumerate(methods_with_dropout)}
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=method_colors[method], 
                     edgecolor='white', linewidth=1.0, alpha=0.9) 
        for method in methods_with_dropout
    ]
    
    # Position legend at top right corner of the figure
    fig.legend(legend_elements, methods_with_dropout, 
              loc='upper right', bbox_to_anchor=(0.98, 0.99), 
              fontsize=12, ncol=4, 
              framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.992])
    plt.subplots_adjust(hspace=0.35)
    
    # Save combined plot
    output_path = output_dir / 'oracle_distance_all_datasets.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print()
    print(f"✓ Saved {output_path}")
    plt.close(fig)
    
    print()
    print("="*80)
    print("✓ Combined plot generated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
