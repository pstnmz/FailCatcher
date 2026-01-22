"""
Unified visualization of UQ benchmark results across all shift types.

Creates 3 figures:
1. AUROC_f Radar Plots: 3 rows (ID, CS, PS/NCS) × 2 columns (ResNet18, ViT)
2. AUGRC Radar Plots: Same layout as AUROC_f
3. Heatmaps: 3 rows (ID, CS, PS/NCS) × 2 columns (AUROC_f, AUGRC)
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import functions from existing scripts
from benchmarks.medMNIST.utils.viz_benchmark_results.generate_radar_plots import create_radar_plot_on_axis, parse_results_directory


def compute_polygon_area_polar(angles, radii):
    """
    Compute the area of a polygon in polar coordinates.
    
    For a polygon with vertices at (θ_i, r_i), the area is:
    A = 0.5 * Σ(r_i * r_{i+1} * sin(θ_{i+1} - θ_i))
    
    Args:
        angles: List of angles in radians (should form a closed loop)
        radii: List of radii at each angle
    
    Returns:
        float: Area of the polygon
    """
    if len(angles) != len(radii):
        raise ValueError("Angles and radii must have the same length")
    
    # Ensure the polygon is closed
    if angles[-1] != angles[0]:
        angles = list(angles) + [angles[0]]
        radii = list(radii) + [radii[0]]
    
    area = 0.0
    n = len(angles) - 1  # Exclude the duplicate closing point
    
    for i in range(n):
        theta_diff = angles[i+1] - angles[i]
        # Handle wraparound at 2π
        if theta_diff < -np.pi:
            theta_diff += 2 * np.pi
        elif theta_diff > np.pi:
            theta_diff -= 2 * np.pi
        
        # Area contribution from this segment
        area += 0.5 * radii[i] * radii[i+1] * np.sin(theta_diff)
    
    return abs(area)


def normalize_radar_surface(area, angles, y_min, y_max, metric='auroc_f'):
    """
    Normalize radar surface area to [0, 1] where 1 is perfect performance.
    
    Args:
        area: Computed polygon area
        angles: List of angles (without closing point) - only valid angles for this method
        y_min: Minimum y-axis value on radar
        y_max: Maximum y-axis value on radar  
        metric: 'auroc_f' or 'augrc'
    
    Returns:
        float: Normalized area (1.0 = perfect, 0.0 = worst)
    """
    # For perfect performance:
    # - AUROC_f: r = y_max (1.0) for all points
    # - AUGRC: r = y_max (edge) for all points
    
    if len(angles) < 3:  # Need at least 3 points
        return 0.0
    
    if metric == 'auroc_f':
        perfect_radius = y_max
    else:  # augrc
        # For AUGRC, best performance is at the edge (maximum transformed value)
        perfect_radius = y_max
    
    # Compute perfect area using only the valid angles for this method
    perfect_radii = [perfect_radius] * len(angles)
    perfect_area = compute_polygon_area_polar(angles + [angles[0]], 
                                             perfect_radii + [perfect_radii[0]])
    
    # Normalize
    if perfect_area == 0:
        return 0.0
    
    normalized = area / perfect_area
    
    # Clip to [0, 1] range
    return np.clip(normalized, 0.0, 1.0)


def create_surface_histogram_figure(all_surfaces, aggregation='mean'):
    """
    Create histogram figure showing normalized radar surfaces for all methods.
    
    Layout: 3 rows × 1 column (one per shift, both models per row)
    ResNet18 reversed on right y-axis, ViT normal on left y-axis.
    
    Args:
        all_surfaces: Dict structure:
            {
                'id': {
                    'resnet18': {
                        'auroc_f': {'MSR': 0.85, 'GPS': 0.90, ...},
                        'augrc': {'MSR': 0.82, 'GPS': 0.88, ...}
                    },
                    'vit_b_16': {...}
                },
                'corruption': {...},
                'population': {...}
            }
        aggregation: Aggregation strategy
    
    Returns:
        fig: matplotlib figure
    """
    # Create figure with 3 rows and 1 column (one per shift)
    fig, axes = plt.subplots(3, 1, figsize=(6, 12), gridspec_kw={'hspace': 0.25})
    
    # Define shift types
    shift_configs = [
        ('id', 'In Distribution', 0),
        ('corruption', 'Corruption Shifts', 1),
        ('population', 'Population / New Class Shifts', 2)
    ]
    
    model_names = ['resnet18', 'vit_b_16']
    model_display_names = {
        'resnet18': 'ResNet18',
        'vit_b_16': 'ViT-B/16'
    }
    
    # Define method order for histogram display
    method_display_order = {
        'MSR': 0,
        'MSR_calibrated': 1,
        'MLS': 2,
        'TTA': 3,
        'GPS': 4,
        'MCDropout': 5,
        'KNN_Raw': 6,
        'Ensembling': 7,
        'Mean_Aggregation': 8,
        'Mean_Aggregation_Ensemble': 9
    }
    
    # Get all methods across all shifts
    all_methods_set = set()
    for shift_data in all_surfaces.values():
        for model_data in shift_data.values():
            for metric_data in model_data.values():
                all_methods_set.update(metric_data.keys())
    
    # Sort methods ALPHABETICALLY for consistent color mapping (same as radar plots)
    all_methods_sorted_for_colors = sorted(all_methods_set)
    
    # Create color mapping matching radar plots (tab20 colormap) - ALPHABETICAL ORDER
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, len(all_methods_sorted_for_colors)))
    method_colors = {method: colors_tab20[i] for i, method in enumerate(all_methods_sorted_for_colors)}
    
    # Override Mean_Aggregation to red
    method_colors['Mean_Aggregation'] = 'red'
    
    print("\nGenerating surface histogram figure...")
    
    for shift_idx, (shift_key, shift_label, shift_num) in enumerate(shift_configs):
        shift_data = all_surfaces.get(shift_key, {})
        
        if not shift_data:
            print(f"  ⚠ No surface data for {shift_key}")
            continue
        
        ax = axes[shift_idx]
        
        # Get data for both models
        resnet_data = shift_data.get('resnet18', {})
        vit_data = shift_data.get('vit_b_16', {})
        
        if not resnet_data and not vit_data:
            print(f"  ⚠ No model data for {shift_key}")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Get AUROC surfaces for both models
        resnet_auroc = resnet_data.get('auroc_f', {})
        vit_auroc = vit_data.get('auroc_f', {})
        
        # Get all methods (union of both models)
        all_methods = sorted(set(list(resnet_auroc.keys()) + list(vit_auroc.keys())), 
                           key=lambda m: method_display_order.get(m, 999))
        
        if not all_methods:
            print(f"  ⚠ No methods for {shift_key}")
            continue
        
        # Prepare data for both models
        resnet_values = [resnet_auroc.get(m, 0.0) for m in all_methods]
        vit_values = [vit_auroc.get(m, 0.0) for m in all_methods]
        
        # Get colors for each method
        bar_colors = [method_colors.get(m, 'gray') for m in all_methods]
        
        # Create offset bar positions with reduced spacing
        x = np.arange(len(all_methods)) * 0.7  # Multiply by 0.7 to reduce spacing between bars
        bar_width = 0.35  # Narrower to fit both
        offset = bar_width * 0#0.75  # Larger offset to prevent bars from touching
        
        # Plot ViT on primary (left) y-axis
        bars_vit = ax.bar(x - offset, vit_values, width=bar_width, color=bar_colors, 
                         alpha=0.8, edgecolor='black', linewidth=1, label='ViT')
        
        # Create secondary (right) y-axis for ResNet
        ax2 = ax.twinx()
        bars_resnet = ax2.bar(x + offset, resnet_values, width=bar_width, color=bar_colors,
                             alpha=0.8, edgecolor='black', linewidth=1, label='ResNet18')
        
        # Surfaces are now normalized by oracle (perfect method = 1.0)
        # Set y-axis limits with gap to separate visually
        
        # Format left y-axis (ViT - normal, 0 at bottom)
        ax.set_ylim(0, 1.6)
        ax.set_ylabel('ViT', fontsize=12, fontweight='bold', y=0.25)
        
        # Custom y-ticks for ViT: 0 to 0.8, then blank gap, then nothing (ResNet uses right axis)
        vit_ticks = [0, 0.2, 0.4, 0.6, 0.8]
        vit_labels = ['0.0', '0.2', '0.4', '0.6', '0.8']
        ax.set_yticks(vit_ticks)
        ax.set_yticklabels(vit_labels, fontsize=9)
        
        # Format right y-axis (ResNet - reversed, 0 at top)
        ax2.set_ylim(1.6, 0)  # Reversed
        ax2.set_ylabel('ResNet18', fontsize=12, fontweight='bold', y=0.75, rotation=-90, labelpad=13)
        
        # Custom y-ticks for ResNet: skip 0.8-1.6 range, show 0 to 0.8
        # Since axis is reversed (1.6 to 0), ticks at physical positions 0 to 0.8 correspond to values 1.6 to 0.8
        resnet_ticks = [0, 0.2, 0.4, 0.6, 0.8]  # Physical positions on the reversed axis
        resnet_labels = ['0.0', '0.2', '0.4', '0.6', '0.8']
        ax2.set_yticks(resnet_ticks)
        ax2.set_yticklabels(resnet_labels, fontsize=9)
        
        # Add a visual separator line at 0.8
        ax.axhline(y=0.8, color='lightgray', linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
        
        # Remove x-tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([])
        
        # Grid on primary axis only
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        # Grid on primary axis only
        ax2.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax2.set_axisbelow(True)
        
        # Hide ALL spines from both axes (we'll draw custom ones)
        for spine in ax.spines.values():
            if spine.spine_type in ['top', 'right', 'left']:
                spine.set_visible(False)
        for spine in ax2.spines.values():
            if spine.spine_type in ['bottom', 'right', 'left']:
                spine.set_visible(False)
        
        # Draw custom left spine segment only from 0 to 0.8
        # Use axis transform: x=0 (left edge) in axis coords, y in data coords
        ax.plot([0, 0], [0, 0.8], 
                color='black', linewidth=0.8, clip_on=False, zorder=100, 
                transform=ax.get_yaxis_transform())
        ax2.plot([1, 1], [0, 0.8], 
                color='black', linewidth=0.8, clip_on=False, zorder=100, 
                transform=ax2.get_yaxis_transform())
        
        # Add subtitle for each shift
        ax.set_title(shift_label, fontsize=12, fontweight='bold', pad=10)
    
    # Add main title
    fig.suptitle('Normalized Surface',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


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
            
            # Generate radar plot (uses pre-computed Mean_Aggregation from JSON)
            handles, labels, method_surfaces, method_angles = create_radar_plot_on_axis(
                ax, model_results, model_name,
                runs_dir=comp_eval_dir,
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
        tuple: (fig, surfaces_dict) where surfaces_dict has structure:
            {
                'shift_key': {
                    'model_name': {
                        method_name: surface_area
                    }
                }
            }
    """
    results_map = results_auroc if metric == 'auroc_f' else results_augrc
    
    # Dictionary to store all surface areas
    all_surfaces = {}
    
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
            
            # Generate radar plot (uses pre-computed Mean_Aggregation from JSON)
            handles, labels, method_surfaces, method_angles = create_radar_plot_on_axis(
                ax, model_results, model_name,
                runs_dir=comp_eval_dir,
                metric=metric, aggregation=aggregation, shift=shift_name
            )
            
            # Store method surfaces for histogram generation
            if shift_key not in all_surfaces:
                all_surfaces[shift_key] = {}
            if model_name not in all_surfaces[shift_key]:
                all_surfaces[shift_key][model_name] = method_surfaces
            
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
    
    return fig, all_surfaces


def create_combined_radar_histogram_figure(results_auroc, results_augrc, all_surfaces, 
                                           aggregation='mean', results_dir=None, comp_eval_dirs=None):
    """
    Create combined radar + histogram figure for AUROC_f.
    
    Layout: 3 rows × 3 columns
    - Row 0: ResNet18 radars (ID, CS, PS/NCS)
    - Row 1: Histograms (ID, CS, PS/NCS)
    - Row 2: ViT radars (ID, CS, PS/NCS)
    
    Args:
        results_auroc: Dict with keys 'id', 'corruption', 'population'
        results_augrc: Dict with keys 'id', 'corruption', 'population' (for radar plotting)
        all_surfaces: Surface areas dict from radar computation
        aggregation: Aggregation strategy
        results_dir: Main results directory
        comp_eval_dirs: Dict of comprehensive evaluation directories
        
    Returns:
        fig: matplotlib figure
    """
    metric = 'auroc_f'
    results_map = results_auroc
    
    # Create figure with 3 rows and 3 columns
    fig = plt.figure(figsize=(20, 14))
    # Adjust spacing to have more room between rows
    gs = fig.add_gridspec(3, 3, hspace=0.20, wspace=0.2,
                         left=0.10, right=0.94, top=0.94, bottom=0.06)
    
    # Create radar axes for rows 0 and 2
    radar_axes = {}
    for row in [0, 2]:
        for col in range(3):
            ax = fig.add_subplot(gs[row, col], projection='polar')
            radar_axes[(row, col)] = ax
    
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
    
    print(f"\nGenerating combined radar + histogram figure (AUROC_f)...")
    
    # Add column titles (shift types) at the top
    for shift_key, shift_name, shift_label, col in shift_configs:
        if col == 0:
            x_pos = 0.15
        elif col == 1:
            x_pos = 0.45
        else:
            x_pos = 0.75
        fig.text(x_pos, 0.96, shift_label, ha='center', va='center', 
                fontsize=16, fontweight='bold')
    
    # Add row labels (model names) on the left
    metric_display = 'AUROC_f'
    for row_idx, (row, model_name) in enumerate([(0, 'resnet18'), (2, 'vit_b_16')]):
        y_pos = 0.08 + (2 - row) * (0.88 / 3)  # Evenly spaced across 3 rows
        model_display = model_display_names[model_name]
        fig.text(0.373, y_pos, f'{model_display}\n{metric_display}', 
                ha='center', va='center', fontsize=14, fontweight='bold')#, rotation=90)
    
    # Plot radars for both models
    for model_idx, (radar_row, model_name) in enumerate([(0, 'resnet18'), (2, 'vit_b_16')]):
        for shift_key, shift_name, shift_label, col in shift_configs:
            results = results_map.get(shift_key, {})
            
            if not results or model_name not in results:
                continue
            
            # Get comprehensive evaluation directory
            comp_eval_dir = comp_eval_dirs.get(shift_key) if comp_eval_dirs else None
            
            ax = radar_axes[(radar_row, col)]
            
            # Call create_radar_plot_on_axis (from generate_radar_plots.py)
            # Uses pre-computed Mean_Aggregation from JSON
            handles, labels, method_surfaces, method_angles = create_radar_plot_on_axis(
                ax=ax,
                model_results=results[model_name],
                model_name=model_name,
                runs_dir=comp_eval_dir,
                metric=metric,
                aggregation=aggregation,
                shift=shift_name
            )
            
            # Collect handles and labels for legend
            if handles and labels:
                for h, l in zip(handles, labels):
                    if l not in all_labels:
                        all_handles.append(h)
                        all_labels.append(l)
    
    # Plot histograms in row 1
    # Get method order and colors (same as radar plots)
    method_display_order = {
        'MSR': 0,
        'MSR_calibrated': 1,
        'MLS': 2,
        'TTA': 3,
        'GPS': 4,
        'MCDropout': 5,
        'KNN_Raw': 6,
        'Ensembling': 7,
        'Mean_Aggregation': 8,
        'Mean_Aggregation_Ensemble': 9
    }
    
    # Get all methods across all shifts for color mapping
    all_methods_set = set()
    for shift_data in all_surfaces.values():
        for model_data in shift_data.values():
            all_methods_set.update(model_data.get('auroc_f', {}).keys())
    
    # Sort methods ALPHABETICALLY for consistent color mapping
    all_methods_sorted_for_colors = sorted(all_methods_set)
    
    # Create color mapping matching radar plots
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, len(all_methods_sorted_for_colors)))
    method_colors = {method: colors_tab20[i] for i, method in enumerate(all_methods_sorted_for_colors)}
    
    # Override Mean_Aggregation to red (keep Mean_Aggregation_Ensemble with default yellow from tab20)
    method_colors['Mean_Aggregation'] = 'red'
    
    # Create histogram for each shift (in middle row)
    for shift_idx, (shift_key, shift_name, shift_label, col) in enumerate(shift_configs):
        shift_data = all_surfaces.get(shift_key, {})
        
        if not shift_data:
            continue
        
        # Create regular (non-polar) subplot for histogram
        ax = fig.add_subplot(gs[1, col])
        
        # Get data for both models
        resnet_data = shift_data.get('resnet18', {})
        vit_data = shift_data.get('vit_b_16', {})
        
        # Get AUROC surfaces for both models
        resnet_auroc = resnet_data.get('auroc_f', {})
        vit_auroc = vit_data.get('auroc_f', {})
        
        # Get all methods (union of both models)
        all_methods = sorted(set(list(resnet_auroc.keys()) + list(vit_auroc.keys())), 
                           key=lambda m: method_display_order.get(m, 999))
        
        if not all_methods:
            continue
        
        # Prepare data for both models
        resnet_values = [resnet_auroc.get(m, 0.0) for m in all_methods]
        vit_values = [vit_auroc.get(m, 0.0) for m in all_methods]
        
        # Get colors for each method
        bar_colors = [method_colors.get(m, 'gray') for m in all_methods]
        
        # Create offset bar positions with reduced spacing
        x = np.arange(len(all_methods)) * 0.6  # Multiply by 0.7 to reduce spacing between bars
        bar_width = 0.2
        offset = bar_width * 0
        
        # Plot ViT on primary (left) y-axis
        bars_vit = ax.bar(x - offset, vit_values, width=bar_width, color=bar_colors, 
                         alpha=0.8, edgecolor='black', linewidth=1, label='ViT')
        
        # Create secondary (right) y-axis for ResNet
        ax2 = ax.twinx()
        bars_resnet = ax2.bar(x - offset, resnet_values, width=bar_width, color=bar_colors,
                             alpha=0.8, edgecolor='black', linewidth=1, label='ResNet18')
        
        # Format left y-axis (ViT - normal, 0 at bottom)
        ax.set_ylim(0.3, 1.3)
        ax.set_ylabel('ViT', fontsize=12, fontweight='bold', y=0.25)
        
        # Custom y-ticks for ViT: 0 to 0.8, then blank gap, then nothing (ResNet uses right axis)
        vit_ticks = [0.4, 0.5, 0.6, 0.7, 0.8]
        vit_labels = ['0.4', '0.5', '0.6', '0.7', '0.8']
        ax.set_yticks(vit_ticks)
        ax.set_yticklabels(vit_labels, fontsize=9)
        
        # Format right y-axis (ResNet - reversed, 0 at top)
        ax2.set_ylim(1.3, 0.3)  # Reversed
        ax2.set_ylabel('ResNet18', fontsize=12, fontweight='bold', y=0.75, labelpad=13, rotation=-90)
        
        # Custom y-ticks for ResNet: skip 0.8-1.6 range, show 0 to 0.8
        resnet_ticks = [0.4, 0.5, 0.6, 0.7, 0.8]  # Physical positions on the reversed axis
        resnet_labels = ['0.4', '0.5', '0.6', '0.7', '0.8']
        ax2.set_yticks(resnet_ticks)
        ax2.set_yticklabels(resnet_labels, fontsize=9)
        
        # Add a visual separator line at 0.8
        ax.axhline(y=0.8, color='lightgray', linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
        
        # Remove x-tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([])
        
        # Grid on primary axis only
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax2.set_axisbelow(True)
        
        # Hide ALL spines from both axes (we'll draw custom ones)
        for spine in ax.spines.values():
            if spine.spine_type in ['top', 'right', 'left']:
                spine.set_visible(False)
        for spine in ax2.spines.values():
            if spine.spine_type in ['bottom', 'right', 'left']:
                spine.set_visible(False)
        
        # Draw custom left spine segment only from 0 to 0.8
        ax.plot([0, 0], [0.3, 0.8], 
                color='black', linewidth=0.8, clip_on=False, zorder=100, 
                transform=ax.get_yaxis_transform())
        ax2.plot([1, 1], [0.3, 0.8], 
                color='black', linewidth=0.8, clip_on=False, zorder=100, 
                transform=ax2.get_yaxis_transform())
        
        # Add subtitle for histogram (centered above middle plot only)
        ax.set_title('Normalized Surface', fontsize=12, fontweight='bold', pad=8)
    
    # Add legend at the bottom
    if all_handles and all_labels:
        # Move special methods to end
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
        
        fig.legend(all_handles, all_labels, loc='lower center', ncol=4,
                  fontsize=11, frameon=True, bbox_to_anchor=(0.52, 0.48), framealpha=1)
    
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
    import json
    import os
    
    print("\nGenerating heatmaps...")
    
    # Load pre-computed results from JSON files for Mean_Aggregation
    results = {}
    shift_dirs = ['id', 'corruption_shifts', 'population_shifts', 'new_class_shifts']
    
    for shift_dir in shift_dirs:
        shift_path = os.path.join(results_dir, shift_dir)
        if not os.path.exists(shift_path):
            continue
            
        for json_file in os.listdir(shift_path):
            if not json_file.endswith('.json'):
                continue
                
            json_path = os.path.join(shift_path, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Parse model, dataset, setup from filename
            # Format: {dataset}_{model}_{setup}_results.json or {dataset}_{model}_results.json
            parts = json_file.replace('_results.json', '').split('_')
            
            if 'vit_b_16' in json_file:
                model = 'vit_b_16'
                dataset_parts = json_file.replace('_vit_b_16', '').replace('_results.json', '').split('_')
            elif 'resnet18' in json_file:
                model = 'resnet18'
                dataset_parts = json_file.replace('_resnet18', '').replace('_results.json', '').split('_')
            else:
                continue
            
            # Check for setup (DADO, DO, DA)
            setup = 'standard'
            for setup_name in ['DADO', 'DO', 'DA']:
                if setup_name in dataset_parts:
                    setup = setup_name
                    dataset_parts = [p for p in dataset_parts if p != setup_name]
                    break
            
            dataset = '_'.join(dataset_parts)
            dataset_key = dataset if setup == 'standard' else f"{dataset}_{setup}"
            
            # Initialize nested dicts
            if model not in results:
                results[model] = {}
            if dataset_key not in results[model]:
                results[model][dataset_key] = {}
            
            # Store method results
            for method_name, method_data in data.items():
                if isinstance(method_data, dict) and 'auroc_f' in method_data:
                    results[model][dataset_key][method_name] = method_data
    
    print(f"  Loaded results for {len(results)} model types")
    
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
            results, display_names, shift_name, 'auroc_f', aggregation
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
            results, display_names, shift_name, 'augrc', aggregation
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

def compute_mean_agg_row(results, display_names, shift_name, metric, aggregation):
    """
    Compute Mean_Aggregation row for heatmap using pre-computed values from JSON.
    
    Args:
        results: Dict with pre-loaded results [model][dataset][method] structure
        display_names: List of display names matching heatmap columns
        shift_name: Shift type name
        metric: 'auroc_f' or 'augrc'
        aggregation: Aggregation strategy name
    
    Returns:
        np.array: Row of differences (ensemble - per-fold)
    """
    agg_row_list = []
    metric_key = f'{metric}_mean' if metric == 'augrc' else metric
    
    for display_name in display_names:
        # Parse display_name to extract model and dataset
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
        
        # Load pre-computed values from results dict
        try:
            dataset_results = results.get(model, {}).get(dataset_key, {})
            
            # Get Mean_Aggregation_Ensemble and Mean_Aggregation values
            ensemble_val = dataset_results.get('Mean_Aggregation_Ensemble', {}).get(metric_key, np.nan)
            perfold_val = dataset_results.get('Mean_Aggregation', {}).get(metric_key, np.nan)
            
            diff = ensemble_val - perfold_val if not np.isnan(ensemble_val) and not np.isnan(perfold_val) else np.nan
        except:
            diff = np.nan
        
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
    workspace_root = script_dir.parent.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    
    # Define all results directories
    results_dirs = {
        'id': results_dir / 'id',
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
    
    fig_auroc_alt, surfaces_auroc = create_radar_figure_alt_layout(
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
    
    fig_augrc_alt, surfaces_augrc = create_radar_figure_alt_layout(
        results_auroc, results_augrc,
        metric='augrc', aggregation=aggregation,
        results_dir=results_dir, comp_eval_dirs=comp_eval_dirs
    )
    
    output_path = output_dir / f'unified_augrc_radars_{aggregation}_alt.png'
    fig_augrc_alt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    plt.close(fig_augrc_alt)
    
    # ==========================================
    # 4. Merge AUROC_f and AUGRC surfaces
    # ==========================================
    print("\n" + "=" * 80)
    print("Merging surface data...")
    print("=" * 80)
    
    # Merge surfaces from both metrics
    all_surfaces = {}
    for shift_key in ['id', 'corruption', 'population']:
        all_surfaces[shift_key] = {}
        for model_name in ['resnet18', 'vit_b_16']:
            all_surfaces[shift_key][model_name] = {
                'auroc_f': surfaces_auroc.get(shift_key, {}).get(model_name, {}),
                'augrc': surfaces_augrc.get(shift_key, {}).get(model_name, {})
            }
    
    # ==========================================
    # 5. Create surface histogram figure
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating surface histogram figure...")
    print("=" * 80)
    
    fig_surfaces = create_surface_histogram_figure(all_surfaces, aggregation=aggregation)
    
    output_path = output_dir / f'unified_radar_surfaces_{aggregation}.png'
    fig_surfaces.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    plt.close(fig_surfaces)

    # ==========================================
    # 6. Create combined radar + histogram figure (AUROC_f only)
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating combined radar + histogram figure (AUROC_f)...")
    print("=" * 80)
    
    fig_combined = create_combined_radar_histogram_figure(
        results_auroc, results_augrc, all_surfaces,
        aggregation=aggregation,
        results_dir=results_dir,
        comp_eval_dirs=comp_eval_dirs
    )
    
    output_path = output_dir / f'unified_auroc_f_radars_with_histograms_{aggregation}.png'
    fig_combined.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    plt.close(fig_combined)

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
