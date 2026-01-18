"""
Unified visualization of UQ method RANKS across all shift types using AUGRC metric.

Creates radar plots showing method ranks based on AUGRC values (lower is better).
Layout: 2 rows (ResNet18, ViT) × 3 columns (ID, CS, PS/NCS)
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# Import functions from existing scripts
from benchmarks.medMNIST.utils.viz_benchmark_results.generate_radar_plots import parse_results_directory, compute_mean_aggregation_metric
from benchmarks.medMNIST.utils.viz_benchmark_results.plot_unified_results import load_and_parse_results


def compute_ranks_from_results_augrc(results):
    """
    Compute ranks for each method on each dataset based on AUGRC.
    
    Args:
        results: Dict[model][dataset][method] = augrc_value
        
    Returns:
        Dict[model][dataset][method] = rank (1 = best, higher = worse)
    """
    ranked_results = {}
    
    for model in results:
        ranked_results[model] = {}
        
        for dataset in results[model]:
            # Get all method values for this dataset
            method_values = results[model][dataset]
            
            # Filter out NaN values
            valid_methods = {m: v for m, v in method_values.items() if not np.isnan(v)}
            
            if not valid_methods:
                ranked_results[model][dataset] = method_values
                continue
            
            # Rank methods (lower AUGRC = better = lower rank number)
            # Use 'min' method to handle ties (give them the minimum rank)
            values = np.array(list(valid_methods.values()))
            ranks = rankdata(values, method='min')  # Positive: lower values get rank 1
            
            # Reverse ranks so rank 1 is at edge (better = farther from center)
            num_methods = len(valid_methods)
            reversed_ranks = num_methods + 1 - ranks
            
            # Create rank dictionary
            ranked_results[model][dataset] = {}
            for (method, _), rank in zip(valid_methods.items(), reversed_ranks):
                ranked_results[model][dataset][method] = float(rank)
            
            # Add back NaN methods as NaN
            for method, value in method_values.items():
                if np.isnan(value):
                    ranked_results[model][dataset][method] = np.nan
    
    return ranked_results


def create_rank_radar_plot_on_axis(ax, model_results, model_name, metric='rank', 
                                    aggregation='mean', shift='in_distribution'):
    """
    Create radar plot showing method ranks instead of values.
    Adapted from generate_radar_plots.create_radar_plot_on_axis
    """
    # Group datasets by family (base name without _standard, _DA, _DO, _DADO)
    dataset_families = defaultdict(list)
    for ds_key in model_results.keys():
        # Extract base dataset name (before _setup)
        base_name = ds_key.rsplit('_', 1)[0] if '_' in ds_key else ds_key
        # If it ends with _standard, remove it to get true base name
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        dataset_families[base_name].append(ds_key)
    
    # Sort setups within each family: standard first, then alphabetically
    setup_order = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}
    for base_name in dataset_families:
        dataset_families[base_name].sort(key=lambda k: (
            setup_order.get(k.split('_')[-1], 99),  # Setup priority
            k  # Then alphabetically
        ))
    
    # Manual ordering by classification performance (easiest to hardest)
    preferred_order = [
        'tissuemnist', 'dermamnist-e', 'breastmnist', 'pneumoniamnist', 
        'octmnist', 'pathmnist', 'organamnist', 'bloodmnist',
        'dermamnist-e-id', 'dermamnist-e-external', 'amos2022', 'new_class_amos2022'
    ]
    
    # Build dataset list following the preferred order
    dataset_keys = []
    for base_name in preferred_order:
        if base_name in dataset_families:
            dataset_keys.extend(dataset_families[base_name])
    
    # Add any remaining datasets not in the preferred order
    for base_name in sorted(dataset_families.keys()):
        if base_name not in preferred_order:
            dataset_keys.extend(dataset_families[base_name])
    
    num_datasets = len(dataset_keys)
    
    if num_datasets == 0:
        print(f"  No datasets found for {model_name}")
        return None, None
    
    # Get all methods across all datasets
    all_methods = set()
    for dataset in dataset_keys:
        all_methods.update(model_results[dataset].keys())
    all_methods = sorted(all_methods)
    
    print(f"  Plotting {len(all_methods)} methods across {num_datasets} datasets")
    
    # Group datasets by family (base name without _standard, _DA, _DO, _DADO)
    dataset_families = defaultdict(list)
    for ds_key in dataset_keys:
        # Extract base name (remove training setup suffix)
        base_name = ds_key.rsplit('_', 1)[0] if '_' in ds_key else ds_key
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        dataset_families[base_name].append(ds_key)
    
    # Calculate angles for datasets (with grouped spacing)
    angles = []
    dataset_labels = []  # Will store just the setup name (standard, DA, DO, DADO)
    family_angles = []  # Store angles for family name labels
    family_names = []
    
    # Process datasets sequentially, grouping by family on the fly
    processed_families = set()
    current_family_idx = 0
    num_families_total = len(dataset_families)
    angle_per_family = 2 * np.pi / num_families_total
    within_family_factor = 0.6
    
    i = 0
    while i < len(dataset_keys):
        dataset_key = dataset_keys[i]
        # Extract base family name
        base_name = dataset_key.rsplit('_', 1)[0] if '_' in dataset_key else dataset_key
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        
        if base_name not in processed_families:
            processed_families.add(base_name)
            family_datasets = dataset_families[base_name]
            
            family_center = current_family_idx * angle_per_family
            family_size = len(family_datasets)
            
            # Store family info for outer labels
            family_angles.append(family_center)
            family_names.append(base_name)
            
            if family_size == 1:
                angles.append(family_center)
                setup_name = family_datasets[0].split('_')[-1]
                # Replace 'standard' with 'S'
                if setup_name == 'standard':
                    setup_name = 'S'
                dataset_labels.append(setup_name)
            else:
                # Distribute setups within family's allocated space
                family_span = angle_per_family * within_family_factor
                for j, ds_key in enumerate(family_datasets):
                    offset = (j - (family_size - 1) / 2) * (family_span / family_size)
                    angle = (family_center + offset) % (2 * np.pi)  # Ensure positive
                    angles.append(angle)
                    setup_name = ds_key.split('_')[-1]
                    # Replace 'standard' with 'S'
                    if setup_name == 'standard':
                        setup_name = 'S'
                    dataset_labels.append(setup_name)
            
            current_family_idx += 1
            i += len(family_datasets)
        else:
            i += 1
    
    # Rotate dataset positions for population/new_class shifts
    if shift in ['population_shift', 'new_class_shift']:
        rotation_offset = -np.pi / 6
        angles = [(a + rotation_offset) % (2 * np.pi) for a in angles]
        family_angles = [(a + rotation_offset) % (2 * np.pi) for a in family_angles]
    
    # Complete the circle
    angles = angles + [angles[0]]
    
    # Color map for methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_methods)))
    
    # Plot each method
    for method_idx, method_name in enumerate(all_methods):
        values = []
        for dataset_key in dataset_keys:
            rank = model_results[dataset_key].get(method_name, np.nan)
            # Add +1 offset for standard and DA setups (7 methods) to align with DO/DADO (8 methods)
            setup = dataset_key.split('_')[-1]
            if setup in ['standard', 'DA']:
                rank = rank + 1 if not np.isnan(rank) else np.nan
            values.append(rank)
        
        # Complete the circle
        values += values[:1]
        
        # Plot based on method type
        if method_name == 'Mean_Aggregation':
            ax.scatter(angles[:-1], values[:-1], s=270, marker='*', 
                       color='red', label=method_name, alpha=0.6, zorder=99,
                       edgecolors='black', linewidths=0.5)
        elif method_name == 'Mean_Aggregation_Ensemble':
            ax.scatter(angles[:-1], values[:-1], s=300, marker='$\u26A1$', 
                       color=colors[method_idx], label='Mean Agg + Ens',
                       alpha=0.9, zorder=100, edgecolors='black', linewidths=0.5)
        else:
            display_name = method_name
            if method_name == "MSR_calibrated":
                display_name = "MSR-S"
            ax.plot(angles, values, 'o-', linewidth=2, label=display_name, 
                    color=colors[method_idx], markersize=8, markeredgewidth=2,
                    markeredgecolor='white', alpha=0.85)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dataset_labels, size=9, fontweight='medium', 
                       rotation=0, ha='center')
    
    # Set up circular display
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, 2 * np.pi)
    
    # Set y-axis range for ranks (inverted: worst at center, best at edge)
    # Rank 10 (worst) is at center, rank 1 (best) is at edge
    y_min, y_max = 0, 11
    y_ticks = np.arange(1, 11, 1)
    ax.set_yticks(y_ticks)
    # Show radial position (1 at center, 10 at edge)
    ax.set_yticklabels([f'{int(y)}' for y in y_ticks], size=11, fontweight='medium')
    
    ax.set_ylim(y_min, y_max)
    ax.set_rlim(y_min, y_max)
    
    # Set radial axis label position based on shift type
    if shift in ['population_shift', 'new_class_shift']:
        ax.set_rlabel_position(200)
    else:
        ax.set_rlabel_position(240)
    
    # Add dataset family names
    label_position = 12.5  # Beyond the edge
    
    for angle, name in zip(family_angles, family_names):
        # Add subtle background box for family names
        angle_positive = angle % (2 * np.pi)  # Ensure positive angle
        # Remove 'mnist' suffix for cleaner display
        display_name = name.replace('mnist', '')
        
        # Custom angular adjustment for specific datasets to avoid overlap
        custom_angle = angle_positive
        if name == 'bloodmnist':
            # Shift blood label slightly around the circle to avoid overlap with pneumonia
            custom_angle = angle_positive - 0.27
        elif name == 'new_class_amos2022':
            display_name = 'new class\namos2022'
            custom_angle = angle_positive - 0.16
        elif name == 'dermamnist-e-external':
            display_name = 'derma-e\n-external'
            custom_angle = angle_positive - 0.1
        elif name == 'dermamnist-e-id':
            custom_angle = angle_positive + 0.18
        elif name == 'pneumoniamnist':
            custom_angle = angle_positive + 0.25
        elif name == 'amos2022':
            custom_angle = angle_positive - 0.25
        elif name == 'organamnist':
            custom_angle = angle_positive + 0.1
        elif name == 'breastmnist':
            custom_angle = angle_positive + 0.07
        elif name == 'octmnist':
            custom_angle = angle_positive - 0.05
        elif name == 'pathmnist':
            custom_angle = angle_positive - 0.25

        ax.text(custom_angle, label_position, display_name, 
                horizontalalignment='center', verticalalignment='center',
                size=12, fontweight='bold', transform=ax.transData,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    # Set axis label
    ax.set_ylabel('Rank', fontsize=12, fontweight='bold', labelpad=25)
    
    return ax.get_legend_handles_labels()


def create_rank_radar_figure(results_augrc, aggregation='mean', results_dir=None, comp_eval_dirs=None):
    """
    Create unified rank radar plot figure using AUGRC.
    
    Layout: 2 rows (ResNet18, ViT) × 3 columns (ID, CS, PS/NCS)
    
    Args:
        results_augrc: Dict with keys 'id', 'corruption', 'population'
        aggregation: Aggregation strategy
        results_dir: Main results directory
        comp_eval_dirs: Dict of comprehensive evaluation directories
        
    Returns:
        fig: matplotlib figure
    """
    # Add Mean Aggregation methods for each shift type
    for shift_key in ['id', 'corruption', 'population']:
        if shift_key not in results_augrc:
            continue
            
        shift_name = {
            'id': 'in_distribution',
            'corruption': 'corruption_shifts',
            'population': 'population_shift'
        }[shift_key]
        
        for model in ['resnet18', 'vit_b_16']:
            if model not in results_augrc[shift_key]:
                continue
                
            for dataset in list(results_augrc[shift_key][model].keys()):
                # Compute Mean Aggregation (per-fold)
                mean_agg_value = compute_mean_aggregation_metric(
                    results_dir, dataset, model,
                    metric='augrc',
                    aggregation=aggregation,
                    shift=shift_name,
                    use_ensemble=False
                )
                
                # Compute Mean Aggregation Ensemble
                mean_agg_ens_value = compute_mean_aggregation_metric(
                    results_dir, dataset, model,
                    metric='augrc',
                    aggregation=aggregation,
                    shift=shift_name,
                    use_ensemble=True
                )
                
                # Add to results (only if not NaN)
                if not np.isnan(mean_agg_value):
                    results_augrc[shift_key][model][dataset]['Mean_Aggregation'] = mean_agg_value
                if not np.isnan(mean_agg_ens_value):
                    results_augrc[shift_key][model][dataset]['Mean_Aggregation_Ensemble'] = mean_agg_ens_value
        
        # Debug: Print methods found
        if model in results_augrc[shift_key] and results_augrc[shift_key][model]:
            first_dataset = list(results_augrc[shift_key][model].keys())[0]
            methods = list(results_augrc[shift_key][model][first_dataset].keys())
            print(f"  {shift_key} - {model} - Methods: {methods}")
        
        print(f"  Added Mean Aggregation methods for {shift_key}")
    
    # Compute ranks for each shift type
    results_ranks = {}
    for shift_key in ['id', 'corruption', 'population']:
        if shift_key in results_augrc:
            results_ranks[shift_key] = compute_ranks_from_results_augrc(results_augrc[shift_key])
        else:
            results_ranks[shift_key] = {}
    
    # Create figure with 2 rows and 3 columns
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.25,
                         left=0.14, right=0.94, top=0.92, bottom=0.08)
    
    # Create all radar axes
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
    
    print(f"\nGenerating RANK radar plots (AUGRC)...")
    
    # Add column titles (shift types)
    for shift_key, shift_name, shift_label, col in shift_configs:
        if col == 0:
            x_pos = 0.145
        elif col == 1:
            x_pos = 0.43
        else:
            x_pos = 0.74
        fig.text(x_pos, 0.945, shift_label, ha='center', va='center', 
                fontsize=16, fontweight='bold')
    
    # Add row labels (model names)
    for row, model_name in enumerate(model_names):
        y_pos = 0.08 + (1 - row) * (0.84 / 2)
        model_display = model_display_names[model_name]
        x_pos = 0.4
        fig.text(x_pos, y_pos, f'{model_display}\nPer-fold {aggregation.capitalize()}\nRank (AUGRC)', 
                ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=0)
    
    for row, model_name in enumerate(model_names):
        for shift_key, shift_name, shift_label, col in shift_configs:
            results = results_ranks.get(shift_key, {})
            
            if not results:
                print(f"  ⚠ No results for {shift_key}")
                continue
            
            ax_idx = row * 3 + col
            ax = radar_axes[ax_idx]
            
            if model_name not in results:
                print(f"  ⚠ No {model_name} results for {shift_key}")
                continue
            
            model_results = results[model_name]
            
            # Generate radar plot
            handles, labels = create_rank_radar_plot_on_axis(
                ax, model_results, model_name,
                metric='rank', aggregation=aggregation, shift=shift_name
            )
            
            # Collect legend info from first subplot
            if ax_idx == 0 and handles and labels:
                all_handles = handles
                all_labels = labels
            
            # Remove y-axis TITLE for CS and PS/NCS columns (col > 0)
            if col > 0:
                ax.set_ylabel('')
    
    # Add legend at the bottom
    if all_handles and all_labels:
        # Move special methods to end
        special_methods = ['Mean_Aggregation', 'Mean_Aggregation_Ensemble', 'Mean Agg + Ens']
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


def main(aggregation='mean'):
    """
    Main execution function.
    """
    # Define results directories
    base_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results')
    
    results_dirs = {
        'id': base_dir / 'id',
        'corruption': base_dir / 'corruption_shifts',
        'population': base_dir / 'population_shifts',
        'new_class': base_dir / 'new_class_shifts'
    }
    
    comp_eval_dirs = {
        'id': base_dir / 'comprehensive_evaluation',
        'corruption': base_dir / 'comprehensive_evaluation_corruption',
        'population': base_dir / 'comprehensive_evaluation_population'
    }
    
    # Create output directory
    output_dir = base_dir / 'figures' / 'unified'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("UNIFIED RANK RADAR PLOTS (AUGRC)")
    print("=" * 80)
    
    # Load AUGRC results
    print("\nLoading AUGRC results...")
    results_augrc = load_and_parse_results(results_dirs, metric='augrc')
    
    # Merge population and new_class results
    if 'population' in results_augrc and 'new_class' in results_augrc:
        pop_res = results_augrc['population']
        new_res = results_augrc['new_class']
        
        merged = {}
        for model in ['resnet18', 'vit_b_16']:
            merged[model] = {}
            if model in pop_res:
                merged[model].update(pop_res[model])
            if model in new_res:
                merged[model].update(new_res[model])
        
        results_augrc['population'] = merged
    
    # Create rank radar figure
    print("\n" + "=" * 80)
    print("Creating RANK radar figure (AUGRC)...")
    print("=" * 80)
    
    fig_ranks = create_rank_radar_figure(
        results_augrc, aggregation=aggregation,
        results_dir=base_dir, comp_eval_dirs=comp_eval_dirs
    )
    
    # Save figure
    output_path = output_dir / f'unified_ranks_radars_{aggregation}_augrc.png'
    fig_ranks.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main(aggregation='mean')
