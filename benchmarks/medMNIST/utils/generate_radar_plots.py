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
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def parse_results_directory(results_dir='./', metric='auroc_f'):
    """
    Parse all JSON result files in the directory.
    
    Args:
        results_dir: Directory containing JSON result files
        metric: Metric to extract - 'auroc_f' or 'augrc'
    
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
                # Get specified metric (use mean if available, otherwise single value)
                metric_mean_key = f'{metric}_mean'
                if metric_mean_key in method_results:
                    value = method_results[metric_mean_key]
                elif metric in method_results:
                    value = method_results[metric]
                else:
                    continue
                
                results[model_backbone][dataset_key][method_name] = float(value)
            
            print(f"  Loaded: {model_backbone} - {dataset_key} ({len(methods_data)} methods)")
        
        except Exception as e:
            print(f"  ⚠️  Failed to parse {json_file.name}: {e}")
    
    return dict(results)


def get_dataset_accuracy(results_dir, dataset_key, model_name='resnet18'):
    """
    Get the test accuracy for a dataset from its JSON file.
    Look in benchmarks/medMNIST/runs folder for accuracy data.
    
    Args:
        results_dir: Path to results directory (will navigate to runs folder)
        dataset_key: Dataset key (e.g., 'breastmnist_standard', 'breastmnist_DA')
        model_name: Model name to filter by
    
    Returns:
        float: Test accuracy, or 0 if not found
    """
    # Navigate to runs folder from results_dir
    workspace_root = Path(results_dir).parent
    runs_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'runs'
    
    if not runs_dir.exists():
        return 0.0
    
    # Parse dataset_key to extract base name and setup
    # e.g., 'breastmnist_standard' -> ('breastmnist', 'standard')
    # e.g., 'breastmnist_DA' -> ('breastmnist', 'DA')
    if '_' in dataset_key:
        parts = dataset_key.rsplit('_', 1)
        base_name = parts[0]
        setup = parts[1] if parts[1] in ['standard', 'DA', 'DO', 'DADO'] else 'standard'
        # If the split didn't give us a valid setup, treat whole key as base name
        if setup == 'standard' and parts[1] != 'standard':
            base_name = dataset_key
            setup = 'standard'
    else:
        base_name = dataset_key
        setup = 'standard'
    
    # Build pattern based on setup
    if setup == 'standard':
        # For standard setup, filename doesn't have setup suffix
        # Need to filter out files with setup suffixes (DA, DO, DADO)
        pattern = f"uq_benchmark_{base_name}_{model_name}_*.json"
        all_matches = list(runs_dir.glob(pattern))
        # Filter out files with setup suffixes
        json_files = [f for f in all_matches if not any(
            f.stem.endswith(f'_{s}_{ts}') or f.stem.endswith(f'_{s}')
            for s in ['DA', 'DO', 'DADO']
            for ts in [f.stem.split('_')[-1]]  # Last part is timestamp
        )]
        # Simpler filter: exclude if filename contains _DA_, _DO_, or _DADO_ before timestamp
        json_files = [f for f in all_matches if not any(
            f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO']
        )]
    else:
        # For DA/DO/DADO, filename includes setup
        pattern = f"uq_benchmark_{base_name}_{model_name}_{setup}_*.json"
        json_files = list(runs_dir.glob(pattern))
    
    if not json_files:
        return 0.0
    
    # Use the most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('ensemble_balanced_accuracy', 0.0)
    except:
        return 0.0


def get_ensemble_accuracy_from_runs(runs_dir, dataset_key, model_name='resnet18'):
    """
    Get ensemble balanced accuracy from runs/[dataset]/[model]_*/metrics_ensemble.json
    
    Args:
        runs_dir: Path to runs directory (benchmarks/medMNIST/runs)
        dataset_key: Dataset key (e.g., 'tissuemnist_standard', 'tissuemnist_DA')
        model_name: Model name (e.g., 'resnet18', 'vit_b_16')
    
    Returns:
        float: Ensemble balanced accuracy, or 0 if not found
    """
    runs_dir = Path(runs_dir)
    
    # Parse dataset_key to get base name and setup
    if '_' in dataset_key:
        parts = dataset_key.rsplit('_', 1)
        base_name = parts[0]
        setup = parts[1] if parts[1] in ['standard', 'DA', 'DO', 'DADO'] else 'standard'
        if setup == 'standard' and parts[1] != 'standard':
            base_name = dataset_key
            setup = 'standard'
    else:
        base_name = dataset_key
        setup = 'standard'
    
    # Handle dermamnist-e-id -> dermamnist-e mapping
    if 'dermamnist-e-id' in base_name:
        base_name = 'dermamnist-e'
    
    # Path: runs/[dataset]/[model]_*/metrics_ensemble.json
    dataset_dir = runs_dir / base_name
    if not dataset_dir.exists():
        return 0.0
    
    # Map setup to training pattern
    # standard: randaug0 (no dropout)
    # DA: randaug1 (no dropout) 
    # DO: randaug0 + dropout
    # DADO: randaug1 + dropout
    if setup == 'standard':
        pattern = f"{model_name}_*_randaug0_*"
        exclude_pattern = "dropout"
    elif setup == 'DA':
        pattern = f"{model_name}_*_randaug1_*"
        exclude_pattern = "dropout"
    elif setup == 'DO':
        pattern = f"{model_name}_*_randaug0_*dropout*"
        exclude_pattern = None
    elif setup == 'DADO':
        pattern = f"{model_name}_*_randaug1_*dropout*"
        exclude_pattern = None
    else:
        return 0.0
    
    # Find matching directories
    matching_dirs = []
    for dir_path in dataset_dir.glob(pattern):
        if dir_path.is_dir():
            if exclude_pattern and exclude_pattern in dir_path.name:
                continue
            matching_dirs.append(dir_path)
    
    if not matching_dirs:
        return 0.0
    
    # Use most recent directory
    most_recent = max(matching_dirs, key=lambda p: p.stat().st_mtime)
    
    # Load metrics_ensemble.json
    metrics_file = most_recent / 'metrics_ensemble.json'
    if not metrics_file.exists():
        return 0.0
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        # balanced_accuracy is nested under 'metrics'
        metrics = data.get('metrics', {})
        return metrics.get('balanced_accuracy', 0.0)
    except:
        return 0.0


def create_radar_plot(model_results, model_name, output_path, results_dir=None, runs_dir=None, metric='auroc_f'):
    """
    Create a radar plot for a single model showing all dataset-setup combinations.
    
    Args:
        model_results: dict mapping dataset_key -> method -> metric_value
        model_name: Name of the model (e.g., 'resnet18', 'vit_b_16')
        output_path: Path to save the figure
        results_dir: Path to results directory (for computing mean aggregation)
        runs_dir: Path to runs directory (for ensemble balanced accuracy)
        metric: Metric to plot - 'auroc_f' or 'augrc'
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
    
    # Manual ordering by classification performance (easiest to hardest)
    # Order: tissuemnist, dermamnist-e, breastmnist, pneumoniamnist, octmnist, pathmnist, organamnist, bloodmnist
    preferred_order = [
        'tissuemnist', 'dermamnist-e', 'breastmnist', 'pneumoniamnist', 
        'octmnist', 'pathmnist', 'organamnist', 'bloodmnist'
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
        print(f"No data for {model_name}, skipping radar plot")
        return
    
    # Add mean aggregation method for each dataset
    metric_name = metric.upper().replace('_', ' ')
    print(f"  Computing mean aggregation {metric_name} for each dataset...")
    mean_agg_count = 0
    for dataset_key in dataset_keys:
        mean_agg_value = compute_mean_aggregation_metric(results_dir, dataset_key, model_name, metric=metric)
        if not np.isnan(mean_agg_value):
            model_results[dataset_key]['Mean_Aggregation'] = mean_agg_value
            mean_agg_count += 1
    print(f"  Successfully computed Mean_Aggregation for {mean_agg_count}/{len(dataset_keys)} datasets")
    
    # Get all unique methods across datasets
    all_methods = set()
    for dataset_data in model_results.values():
        all_methods.update(dataset_data.keys())
    all_methods = sorted(all_methods)
    
    print(f"\nCreating radar plot for {model_name}")
    print(f"  Datasets: {num_datasets}")
    print(f"  Methods: {len(all_methods)}")
    print(f"  Dataset order: {dataset_keys[:5]}... (showing first 5)")
    print(f"  Dataset families found: {sorted(dataset_families.keys())}")
    print(f"  Preferred order: {preferred_order}")
    
    # Cluster setups from same dataset with smaller angles
    # Allocate space per dataset family, then subdivide for setups
    angles = []
    dataset_labels = []  # Will store just the setup name (standard, DA, DO, DADO)
    family_angles = []  # Store angles for family name labels
    family_names = []
    
    # Use the actual dataset_keys order (which includes all datasets)
    # Group by family on the fly
    processed_families = set()
    current_family_idx = 0
    num_families_total = len(dataset_families)
    angle_per_family = 2 * np.pi / num_families_total
    within_family_factor = 0.6  # Tighter clustering within family
    
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
                dataset_labels.append(setup_name)
            else:
                # Distribute setups within family's allocated space
                family_span = angle_per_family * within_family_factor
                for j, ds_key in enumerate(family_datasets):
                    offset = (j - (family_size - 1) / 2) * (family_span / family_size)
                    angle = (family_center + offset) % (2 * np.pi)  # Ensure positive
                    angles.append(angle)
                    setup_name = ds_key.split('_')[-1]
                    dataset_labels.append(setup_name)
            
            current_family_idx += 1
            i += len(family_datasets)
        else:
            i += 1
    
    # Ensure all angles are positive and within [0, 2π]
    angles = [(a % (2 * np.pi)) for a in angles]
    
    # Store number of datasets before adding closing point
    num_angle_points = len(angles)
    angles = angles + [angles[0]]  # Complete the circle
    
    print(f"  Angle range: {min(angles):.2f} to {max(angles):.2f} radians")
    print(f"  Families: {len(family_names)}")
    print(f"  Angle points (before close): {num_angle_points}, dataset_keys: {num_datasets}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 18), subplot_kw=dict(projection='polar'))
    
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
        
        # Plot Mean_Aggregation with lightning icon, others with lines
        if method_name == 'Mean_Aggregation':
            # Plot lightning bolt markers without connecting lines
            ax.scatter(angles[:-1], values[:-1], s=300, marker='$\u26A1$', 
                       color=colors[method_idx], label=method_name, 
                       zorder=99, alpha=0.9, edgecolors='black', linewidths=0.5)
        else:
            # Plot lines with enhanced styling for better visibility
            ax.plot(angles, values, 'o-', linewidth=3, label=method_name, 
                    color=colors[method_idx], markersize=8, markeredgewidth=2,
                    markeredgecolor='white', alpha=0.85)
    
    # Add ensemble balanced accuracy scatter overlay
    if runs_dir:
        accuracy_values = []
        accuracy_angles = []
        for idx, dataset_key in enumerate(dataset_keys):
            accuracy = get_ensemble_accuracy_from_runs(runs_dir, dataset_key, model_name)
            # Only include if valid (non-zero) accuracy
            if accuracy > 0:
                accuracy_values.append(accuracy)
                accuracy_angles.append(angles[idx])
        
        # Plot as distinct scatter points (only if we have valid data)
        if accuracy_values:
            ax.scatter(accuracy_angles, accuracy_values, s=200, c='red', marker='*', 
                       edgecolors='black', linewidths=2, zorder=100, 
                       label='Ensemble Balanced Accuracy', alpha=1.0)
            print(f"  Added {len(accuracy_values)} accuracy markers")
        else:
            print("  No valid accuracy values found")
    
    # Set labels - setup names only (increased font size for readability)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dataset_labels, size=9, rotation=0, fontweight='medium')
    
    # Add dataset family names (larger, further out, with background)
    for angle, name in zip(family_angles, family_names):
        # Add subtle background box for family names
        # Position at 1.08 (beyond the 1.0 max AUROC but within visible range)
        angle_positive = angle % (2 * np.pi)  # Ensure positive angle
        ax.text(angle_positive, 1.08, name, 
                horizontalalignment='center', verticalalignment='center',
                size=12, fontweight='bold', transform=ax.transData,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    # Ensure full circle is visible - CRITICAL for proper display
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_thetalim(0, 2 * np.pi)  # Force full circle view
    
    # Set y-axis range based on metric
    if metric == 'auroc_f':
        y_min, y_max = 0.4, 1.0
    else:  # augrc - inverted scale (1 to 0)
        y_min, y_max = 1.0, 0.0
    
    ax.set_ylim(y_min, y_max)
    ax.set_rlim(y_min, y_max)  # Also set radial limits explicitly
    
    # Set ticks every 0.1 with better styling
    y_ticks = np.arange(y_min, y_max + 0.05, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], size=11, fontweight='medium')
    metric_label = metric.upper().replace('_', ' ')
    ax.set_ylabel(metric_label, size=13, fontweight='bold', labelpad=35)
    
    # Enhanced grid
    ax.grid(True, linewidth=0.7, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)  # Grid behind data
    
    # Title in bottom right corner (as text annotation)
    model_display_name = model_name.replace('_', ' ').upper()
    metric_display = metric.upper().replace('_', ' ')
    fig.text(0.78, 0.08, f'UQ Methods Performance\n{model_display_name}\n(Mean {metric_display})',
             ha='center', va='center', size=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', 
                      edgecolor='black', alpha=0.9, linewidth=2))
    
    # Legend - positioned on right side with improved styling
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=11,
             frameon=True, fancybox=True, shadow=True, ncol=1)
    
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
        writer.writerow(['Model', 'Dataset', 'Method', 'AUROC_f', 'AUGRC'])
        
        # Data
        for model_name, model_results in sorted(results.items()):
            for dataset_key, dataset_data in sorted(model_results.items()):
                for method_name, auroc in sorted(dataset_data.items()):
                    writer.writerow([model_name, dataset_key, method_name, f"{auroc:.4f}"])
    
    print(f"\n✓ Summary table saved to {output_path}")


def compute_mean_aggregation_metric(results_dir, dataset_key, model_name, metric='auroc_f'):
    """
    Compute specified metric for mean aggregation of z-scored UQ methods.
    
    Args:
        results_dir: Path to results directory
        dataset_key: Dataset key (e.g., 'breastmnist_standard')
        model_name: Model name (e.g., 'resnet18', 'vit_b_16')
        metric: Metric to compute - 'auroc_f' or 'augrc'
    
    Returns:
        float: Metric value for mean aggregation, or NaN if not found
    """
    results_dir = Path(results_dir)
    
    # Parse dataset_key to get base dataset name and setup
    if '_' in dataset_key and dataset_key.split('_')[-1] in ['standard', 'DA', 'DO', 'DADO']:
        parts = dataset_key.rsplit('_', 1)
        dataset_name = parts[0]
        setup = parts[1]
    else:
        dataset_name = dataset_key
        setup = 'standard'
    
    # Construct npz filename pattern
    if setup == 'standard':
        # For standard, exclude DA/DO/DADO in filename
        pattern = f"all_metrics_{dataset_name}_{model_name}_*.npz"
    else:
        pattern = f"all_metrics_{dataset_name}_{model_name}_{setup}_*.npz"
    
    # Search in id_results subdirectory
    search_dir = results_dir / 'id_results'
    
    # Find matching npz files
    all_npz_files = list(search_dir.glob(pattern))
    
    # Filter for standard setup (exclude DA, DO, DADO)
    if '_' in dataset_key and dataset_key.split('_')[-1] == 'standard':
        npz_files = [f for f in all_npz_files if not any(
            f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO']
        )]
    else:
        npz_files = all_npz_files
    if not npz_files:
        return np.nan
    
    # Use most recent file
    npz_file = max(npz_files, key=lambda p: p.stat().st_mtime)
    
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Get all method keys (exclude _per_fold, _ensemble, and TTA)
        method_keys = [k for k in data.keys() if not k.endswith('_per_fold') and not k.endswith('_ensemble') and k != 'TTA']
        
        if not method_keys:
            return np.nan
        
        # Z-score normalize each method and compute mean
        normalized_arrays = []
        for method_name in method_keys:
            uncertainties = data[method_name]
            mean_val = np.mean(uncertainties)
            std_val = np.std(uncertainties)
            
            if std_val > 0:
                z_score = (uncertainties - mean_val) / std_val
                normalized_arrays.append(z_score)
        
        if not normalized_arrays:
            return np.nan
        
        # Mean aggregation
        stacked = np.stack(normalized_arrays, axis=0)
        aggregated = np.mean(stacked, axis=0)
        
        # Get labels from cached test results (not from JSON)
        # Look for the corresponding cache file
        cache_dir = results_dir / 'cache'
        if setup == 'standard':
            cache_pattern = f"{dataset_name}_{model_name}_test_results.npz"
        else:
            cache_pattern = f"{dataset_name}_{model_name}_{setup}_test_results.npz"
        
        cache_file_path = cache_dir / cache_pattern
        
        if not cache_file_path.exists():
            return np.nan
        
        cache_data = np.load(cache_file_path, allow_pickle=True)
        
        # Build failure labels from correct/incorrect indices
        correct_idx = cache_data['correct_idx']
        incorrect_idx = cache_data['incorrect_idx']
        n_samples = len(correct_idx) + len(incorrect_idx)
        
        failure_labels = np.zeros(n_samples)
        failure_labels[incorrect_idx] = 1  # Mark failures as 1
        
        if len(failure_labels) != len(aggregated):
            return np.nan
        
        # Compute specified metric
        if metric == 'auroc_f':
            from sklearn.metrics import roc_auc_score
            result = roc_auc_score(failure_labels, aggregated)
        elif metric == 'augrc':
            # Compute AUGRC: Area Under Gain-Risk Curve
            # Sort by uncertainty (descending)
            sorted_indices = np.argsort(-aggregated)
            sorted_labels = failure_labels[sorted_indices]
            
            # Compute cumulative failure rate vs coverage
            n_total = len(sorted_labels)
            cumulative_failures = np.cumsum(sorted_labels)
            coverage = np.arange(1, n_total + 1) / n_total
            failure_rate = cumulative_failures / np.arange(1, n_total + 1)
            
            # Baseline is random: expected failure rate at each coverage
            baseline_rate = np.mean(failure_labels)
            
            # Gain = baseline - actual failure rate (positive when we catch more failures)
            gain = baseline_rate - failure_rate
            
            # AUGRC: area under the gain curve (use trapezoidal rule)
            result = np.trapz(gain, coverage)
        else:
            return np.nan
        
        return result
        
    except Exception as e:
        import traceback
        print(f"    Warning: Could not compute mean aggregation for {dataset_key}: {e}")
        traceback.print_exc()
        return np.nan


def aggregate_uq_methods(npz_path, aggregation='mean', methods=None, output_path=None):
    """
    Aggregate multiple UQ methods by z-score normalizing each method and combining.
    
    Args:
        npz_path: Path to the npz file containing UQ method uncertainties
        aggregation: Aggregation strategy - 'mean', 'max', 'min', or 'vote'
        methods: List of method names to aggregate (None = all methods)
        output_path: Optional path to save aggregated results
    
    Returns:
        dict: Dictionary with aggregated uncertainties and per-method z-scores
            {
                'aggregated': array of aggregated uncertainties,
                'z_scores': dict of {method_name: z_scored_array},
                'methods_used': list of method names,
                'aggregation': aggregation type
            }
    """
    # Load the npz file
    data = np.load(npz_path, allow_pickle=True)
    
    # Get available methods (exclude '_per_fold' and '_ensemble' suffixes)
    all_keys = list(data.keys())
    method_keys = [k for k in all_keys if not k.endswith('_per_fold') and not k.endswith('_ensemble')]
    
    if methods is None:
        methods = method_keys
    else:
        # Filter to only requested methods that exist
        methods = [m for m in methods if m in method_keys]
    
    if not methods:
        raise ValueError(f"No valid methods found in {npz_path}. Available: {method_keys}")
    
    print(f"\nAggregating UQ methods from: {npz_path}")
    print(f"  Methods to aggregate: {methods}")
    print(f"  Aggregation strategy: {aggregation}")
    
    # Z-score normalize each method
    z_scores = {}
    normalized_arrays = []
    
    for method_name in methods:
        uncertainties = data[method_name]
        
        # Z-score normalization: (x - mean) / std
        mean = np.mean(uncertainties)
        std = np.std(uncertainties)
        
        if std == 0:
            print(f"  Warning: {method_name} has zero std, skipping")
            continue
        
        z_score = (uncertainties - mean) / std
        z_scores[method_name] = z_score
        normalized_arrays.append(z_score)
        
        print(f"  {method_name}: mean={mean:.4f}, std={std:.4f}")
    
    if not normalized_arrays:
        raise ValueError("No valid methods after normalization")
    
    # Stack all z-scored arrays (shape: num_methods x num_samples)
    stacked = np.stack(normalized_arrays, axis=0)
    
    # Aggregate across methods
    if aggregation == 'mean':
        aggregated = np.mean(stacked, axis=0)
    elif aggregation == 'max':
        aggregated = np.max(stacked, axis=0)
    elif aggregation == 'min':
        aggregated = np.min(stacked, axis=0)
    elif aggregation == 'vote':
        # Majority vote: count how many methods have z-score > 0 (above average uncertainty)
        aggregated = np.sum(stacked > 0, axis=0) / len(normalized_arrays)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean', 'max', 'min', or 'vote'")
    
    print(f"  Aggregated shape: {aggregated.shape}")
    print(f"  Aggregated range: [{aggregated.min():.4f}, {aggregated.max():.4f}]")
    
    result = {
        'aggregated': aggregated,
        'z_scores': z_scores,
        'methods_used': [m for m in methods if m in z_scores],
        'aggregation': aggregation
    }
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        np.savez_compressed(
            output_path,
            aggregated=aggregated,
            **z_scores,
            methods_used=result['methods_used'],
            aggregation=aggregation
        )
        print(f"  ✓ Saved to {output_path}")
    
    return result


def main():
    """Main function to generate radar plots from benchmark results."""
    
    print("=" * 80)
    print("UQ Benchmark Radar Plot Generator")
    print("=" * 80)
    
    # Get the workspace root (3 levels up from script: utils -> medMNIST -> benchmarks -> UQ_Toolbox)
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    id_results_dir = results_dir / 'id_results'
    runs_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'runs'
    
    print(f"Workspace root: {workspace_root}")
    print(f"Looking for JSON files in: {id_results_dir}")
    print(f"Looking for accuracy in: {runs_dir}")
    
    # Create output directory for plots in the script's directory
    output_dir = script_dir / 'radar_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots for both AUROC_f and AUGRC
    for metric in ['auroc_f', 'augrc']:
        metric_display = metric.upper().replace('_', ' ')
        print(f"\n{'='*80}")
        print(f"Generating {metric_display} plots...")
        print(f"{'='*80}")
        
        # Parse results for this metric
        results = parse_results_directory(id_results_dir, metric=metric)
        
        if not results:
            print(f"\n⚠️  No {metric_display} results found!")
            continue
        
        # Generate radar plot for each model
        for model_name, model_results in results.items():
            output_path = output_dir / f'radar_plot_{model_name}_{metric}.png'
            create_radar_plot(model_results, model_name, output_path, 
                            results_dir=results_dir, runs_dir=runs_dir, metric=metric)
    
    # Generate summary table
    summary_path = output_dir / 'results_summary.csv'
    generate_summary_table(results, summary_path)
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    import sys
    
    # If first argument is 'aggregate', run aggregation demo
    if len(sys.argv) > 1 and sys.argv[1] == 'aggregate':
        if len(sys.argv) < 3:
            print("Usage: python generate_radar_plots.py aggregate <npz_file> [mean|max|min|vote]")
            print("\nExample:")
            print("  python generate_radar_plots.py aggregate ../../uq_benchmark_results/all_metrics_breastmnist_20251128_192502.npz mean")
            sys.exit(1)
        
        npz_file = sys.argv[2]
        aggregation = sys.argv[3] if len(sys.argv) > 3 else 'mean'
        
        output_path = Path(npz_file).parent / f"aggregated_{aggregation}_{Path(npz_file).stem}.npz"
        
        result = aggregate_uq_methods(npz_file, aggregation=aggregation, output_path=output_path)
        
        print(f"\n{'='*80}")
        print(f"Aggregation complete!")
        print(f"  Methods used: {', '.join(result['methods_used'])}")
        print(f"  Strategy: {aggregation}")
        print(f"  Output saved to: {output_path}")
        print(f"{'='*80}")
    else:
        # Normal radar plot generation
        main()
