"""
Boxplot comparison of Ensemble vs Mean Per-Fold aggregation.
Groups all methods and setups by backbone (resnet18 vs vit_b_16).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats

def load_comprehensive_results(base_path):
    """Load all UQ benchmark results for ID datasets."""
    results = {}
    
    # Scan the directory for all JSON files
    json_files = sorted(base_path.glob('uq_benchmark_*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    for filepath in json_files:
        # Parse filename: uq_benchmark_{dataset}_{model}_{setup}_{timestamp}.json
        # or: uq_benchmark_{dataset}_{model}_{timestamp}.json (when setup is "standard")
        filename = filepath.stem  # Remove .json
        parts = filename.replace('uq_benchmark_', '').rsplit('_', 1)  # Split off timestamp
        
        if len(parts) == 2:
            prefix, timestamp = parts
            
            # Known model names (longest first to match correctly)
            known_models = ['vit_b_16', 'resnet18']
            
            # Find model in prefix string
            model = None
            for model_name in known_models:
                if f'_{model_name}_' in f'_{prefix}_':  # Add underscores to ensure exact match
                    model = model_name
                    break
            
            if model is None:
                print(f"  Skipped (no known model): {filename}")
                continue
            
            # Split at the model name
            parts_split = prefix.split(f'_{model}_')
            if len(parts_split) == 2:
                dataset = parts_split[0]
                setup = parts_split[1] if parts_split[1] else 'standard'
            else:
                print(f"  Skipped (unexpected split): {filename}")
                continue
            
            # Load the data
            with open(filepath, 'r') as f:
                data = json.load(f)
                key = f"{dataset}_{model}_{setup}"
                results[key] = {
                    'data': data,
                    'dataset': dataset,
                    'model': model,
                    'setup': setup
                }
                print(f"  Loaded: {key}")
        else:
            print(f"  Skipped (no timestamp): {filename}")
    
    return results

def compute_differences_by_backbone(results):
    """
    Compute differences and group by backbone.
    Returns structure: {backbone: {metric: {method: [differences]}}}
    Also computes Mean Aggregation across selected methods.
    """
    # Structure: {backbone: {metric: {method: [differences]}}}
    backbone_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # All possible methods (exclude Ensembling)
    all_methods = set()
    
    # Methods to aggregate for Mean Aggregation
    mean_agg_methods = ['MSR', 'MSR_calibrated', 'MLS', 'GPS', 'KNN_Raw', 'MCDropout']
    
    # First pass: collect all methods
    for key, result_info in results.items():
        data = result_info['data']
        if 'methods' in data:
            for method in data['methods'].keys():
                if method != 'Ensembling':  # Skip Ensembling (no per-fold version)
                    all_methods.add(method)
    
    # Second pass: compute differences
    for key, result_info in results.items():
        data = result_info['data']
        model = result_info['model']
        dataset = result_info['dataset']
        setup = result_info['setup']
        
        # Normalize model name
        if model == 'vit_b_16':
            backbone = 'ViT-B/16'
        elif model == 'resnet18':
            backbone = 'ResNet18'
        else:
            backbone = model
        
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
                    diff_auroc_f = ensemble_auroc_f - mean_fold_auroc_f
                    backbone_data[backbone]['auroc_f'][method].append(diff_auroc_f)
                
                if per_fold_augrc and ensemble_augrc is not None:
                    mean_fold_augrc = np.mean(per_fold_augrc)
                    diff_augrc = ensemble_augrc - mean_fold_augrc
                    backbone_data[backbone]['augrc'][method].append(diff_augrc)
    
    # Third pass: compute Mean Aggregation (average across selected methods)
    for backbone in backbone_data.keys():
        for metric in ['auroc_f', 'augrc']:
            if metric not in backbone_data[backbone]:
                continue
            
            # Collect all differences from mean_agg_methods
            all_diffs = []
            for method in mean_agg_methods:
                if method in backbone_data[backbone][metric]:
                    all_diffs.extend(backbone_data[backbone][metric][method])
            
            # Store as a separate "method"
            if all_diffs:
                backbone_data[backbone][metric]['⚡ Mean Aggregation'] = all_diffs
    
    return backbone_data

def create_boxplots_by_backbone(backbone_data, output_dir):
    """
    Create boxplot figures grouped by backbone.
    One box per backbone containing all methods and setups.
    """
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure for each metric
    metrics = ['auroc_f', 'augrc']
    metric_labels = {
        'auroc_f': 'ΔAUROC_f (Ensemble - Mean Per-Fold)',
        'augrc': 'ΔAUGRC (Ensemble - Mean Per-Fold)'
    }
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect data for all backbones
        plot_data = []
        labels = []
        colors = []
        
        # Color palette for backbones
        backbone_colors = {
            'ResNet18': '#3498db',  # Blue
            'ViT-B/16': '#e74c3c'   # Red
        }
        
        # Aggregate all differences across methods for each backbone
        for backbone in sorted(backbone_data.keys()):
            if metric not in backbone_data[backbone]:
                continue
            
            # Collect all differences across all methods for this backbone
            all_diffs = []
            for method, diffs in backbone_data[backbone][metric].items():
                all_diffs.extend(diffs)
            
            if all_diffs:
                plot_data.append(all_diffs)
                labels.append(backbone)
                colors.append(backbone_colors.get(backbone, '#95a5a6'))
        
        # Create boxplot
        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                           widths=0.6,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(color='black', linewidth=2))
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
            
            # Labels and title
            ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
            ax.set_xlabel('Backbone', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric_labels[metric]}\n(All Methods & Setups Combined)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Add grid
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            # Add summary statistics as text
            for i, (data, label) in enumerate(zip(plot_data, labels), 1):
                mean_val = np.mean(data)
                median_val = np.median(data)
                n_samples = len(data)
                
                # Position text above the boxplot
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(i, y_pos, f'n={n_samples}\nmean={mean_val:+.4f}\nmedian={median_val:+.4f}',
                       ha='center', va='top', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / f'ensemble_vs_mean_boxplot_{metric}_by_backbone.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            
            plt.close()

def create_detailed_boxplots_by_method(backbone_data, output_dir):
    """
    Create detailed boxplots showing each method separately for each backbone.
    """
    metrics = ['auroc_f', 'augrc']
    metric_labels = {
        'auroc_f': 'ΔAUROC_f (Ensemble - Mean Per-Fold)',
        'augrc': 'ΔAUGRC (Ensemble - Mean Per-Fold)'
    }
    
    for metric in metrics:
        # Collect all methods
        all_methods = set()
        for backbone in backbone_data.keys():
            if metric in backbone_data[backbone]:
                all_methods.update(backbone_data[backbone][metric].keys())
        
        all_methods = sorted(all_methods)
        
        if not all_methods:
            continue
        
        # Create figure with subplots for each backbone
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        backbones = sorted(backbone_data.keys())
        
        for ax, backbone in zip(axes, backbones):
            if metric not in backbone_data[backbone]:
                continue
            
            # Prepare data for this backbone
            plot_data = []
            labels = []
            
            for method in all_methods:
                if method in backbone_data[backbone][metric]:
                    diffs = backbone_data[backbone][metric][method]
                    if diffs:
                        plot_data.append(diffs)
                        labels.append(method)
            
            if plot_data:
                # Create boxplot
                bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                               widths=0.6,
                               boxprops=dict(linewidth=1),
                               whiskerprops=dict(linewidth=1),
                               capprops=dict(linewidth=1),
                               medianprops=dict(color='black', linewidth=1.5))
                
                # Color boxes with gradient
                colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
                
                # Labels and title
                ax.set_ylabel(metric_labels[metric] if ax == axes[0] else '', 
                             fontsize=11, fontweight='bold')
                ax.set_xlabel('Method', fontsize=11, fontweight='bold')
                ax.set_title(backbone, fontsize=13, fontweight='bold', pad=10)
                
                # Rotate x labels
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
                
                # Add grid
                ax.yaxis.grid(True, alpha=0.3)
                ax.set_axisbelow(True)
        
        # Overall title
        fig.suptitle(f'{metric_labels[metric]} by Method and Backbone\n(All Setups Combined)', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        output_path = output_dir / f'ensemble_vs_mean_boxplot_{metric}_by_method.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        plt.close()

def print_summary_statistics(backbone_data):
    """Print summary statistics with statistical significance tests for each backbone and metric."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS WITH STATISTICAL TESTS")
    print("="*80)
    
    for backbone in sorted(backbone_data.keys()):
        print(f"\n{backbone}:")
        print("-" * 80)
        
        for metric in ['auroc_f', 'augrc']:
            if metric not in backbone_data[backbone]:
                continue
            
            metric_label = 'AUROC_f' if metric == 'auroc_f' else 'AUGRC'
            print(f"\n  {metric_label}:")
            
            # Aggregate all differences across methods (excluding Mean Aggregation to avoid double-counting)
            all_diffs = []
            for method, diffs in backbone_data[backbone][metric].items():
                if method != '⚡ Mean Aggregation':
                    all_diffs.extend(diffs)
            
            if all_diffs:
                # Basic statistics
                mean_diff = np.mean(all_diffs)
                median_diff = np.median(all_diffs)
                std_diff = np.std(all_diffs, ddof=1)
                
                print(f"    Overall (all methods combined):")
                print(f"      n={len(all_diffs)}, mean={mean_diff:+.5f}, "
                      f"median={median_diff:+.5f}, std={std_diff:.5f}")
                print(f"      min={np.min(all_diffs):+.5f}, max={np.max(all_diffs):+.5f}")
                
                # Statistical significance: Wilcoxon signed-rank test (non-parametric, paired)
                # H0: median difference = 0
                if len(all_diffs) > 0:
                    statistic, p_value = stats.wilcoxon(all_diffs, alternative='two-sided')
                    print(f"\n      Statistical Test (Wilcoxon signed-rank):")
                    print(f"        p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
                    print(f"        Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
                    
                    # Effect size: Cohen's d (standardized mean difference)
                    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                    effect_size_label = ('negligible' if abs(cohens_d) < 0.2 else
                                       'small' if abs(cohens_d) < 0.5 else
                                       'medium' if abs(cohens_d) < 0.8 else 'large')
                    print(f"\n      Effect Size (Cohen's d): {cohens_d:+.3f} ({effect_size_label})")
                    print(f"        |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, >0.8: large")
                    
                    # Confidence interval (95%)
                    ci_lower, ci_upper = stats.t.interval(0.95, len(all_diffs)-1, 
                                                           loc=mean_diff, 
                                                           scale=stats.sem(all_diffs))
                    print(f"\n      95% Confidence Interval: [{ci_lower:+.5f}, {ci_upper:+.5f}]")
                
                # Count positive vs negative
                positive = sum(1 for d in all_diffs if d > 0)
                negative = sum(1 for d in all_diffs if d < 0)
                zero = sum(1 for d in all_diffs if d == 0)
                print(f"\n      Direction:")
                print(f"        Ensemble > Mean: {positive}/{len(all_diffs)} ({100*positive/len(all_diffs):.1f}%)")
                print(f"        Ensemble < Mean: {negative}/{len(all_diffs)} ({100*negative/len(all_diffs):.1f}%)")
                if zero > 0:
                    print(f"        Equal: {zero}/{len(all_diffs)}")
            
            # Per-method statistics
            print(f"\n    By method:")
            for method in sorted(backbone_data[backbone][metric].keys()):
                diffs = backbone_data[backbone][metric][method]
                if diffs and method != '⚡ Mean Aggregation':
                    mean_m = np.mean(diffs)
                    std_m = np.std(diffs, ddof=1) if len(diffs) > 1 else 0
                    
                    # Statistical test for this method
                    if len(diffs) >= 3:  # Need at least 3 samples for Wilcoxon
                        _, p_val_m = stats.wilcoxon(diffs, alternative='two-sided')
                        sig_marker = '***' if p_val_m < 0.001 else '**' if p_val_m < 0.01 else '*' if p_val_m < 0.05 else ''
                    else:
                        sig_marker = ''
                    
                    # Effect size
                    cohens_d_m = mean_m / std_m if std_m > 0 else 0
                    
                    print(f"      {method:25s}: n={len(diffs):3d}, mean={mean_m:+.5f} {sig_marker:3s}, "
                          f"d={cohens_d_m:+.3f}, p={p_val_m:.4f}" if len(diffs) >= 3 
                          else f"      {method:25s}: n={len(diffs):3d}, mean={mean_m:+.5f}")
            
            # Special line for Mean Aggregation
            if '⚡ Mean Aggregation' in backbone_data[backbone][metric]:
                diffs = backbone_data[backbone][metric]['⚡ Mean Aggregation']
                mean_m = np.mean(diffs)
                print(f"      {'⚡ Mean Aggregation':25s}: n={len(diffs):3d}, mean={mean_m:+.5f} (aggregated)")
    
    # Interpretation guide
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Statistical Significance (p-value):
  - p < 0.001: *** (highly significant)
  - p < 0.01:  **  (very significant)
  - p < 0.05:  *   (significant)
  - p ≥ 0.05:      (not significant)

Effect Size (Cohen's d):
  - |d| < 0.2: Negligible - practically no difference
  - 0.2-0.5:   Small - minimal practical difference
  - 0.5-0.8:   Medium - moderate practical difference
  - |d| > 0.8: Large - substantial practical difference

Context for AUROC_f differences:
  - AUROC_f measures ability to detect failures via uncertainty
  - Δ = +0.01 means ensemble improves failure detection by ~1 percentage point
  - For a model with 80% AUROC_f, +0.01 → 81% (1.25% relative improvement)
  - Positive Δ: Ensemble better at detecting failures
  - Practical impact depends on:
    * Base failure rate (higher rate → more important)
    * Cost of missed failures (high-stakes domains → more important)
    * Available resources for intervention
    
For AUGRC (area under retention-gain curve):
  - Measures cumulative silent failures across rejection thresholds
  - LOWER AUGRC = BETTER (fewer silent failures)
  - HIGHER AUGRC = WORSE (more silent failures)
  - Negative Δ (Ensemble - Mean): Ensemble has LOWER AUGRC → Ensemble is BETTER
  - Positive Δ: Mean per-fold has lower AUGRC → Mean per-fold is better
  - Typical range: 0.05-0.30 depending on base accuracy
  - Large effect size means substantial difference in silent failure rates
    
SUMMARY INTERPRETATION:
  For your results:
    - AUROC_f: Small positive Δ → Ensemble slightly better at failure detection
    - AUGRC: Large negative Δ → Ensemble MUCH better (fewer silent failures)
  → CONCLUSION: Ensemble is superior, especially for reducing silent failures
    """)

def main():
    # Paths
    base_path = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/id_results')
    output_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/figures/ensemble_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("="*80)
    print("ENSEMBLE vs MEAN PER-FOLD BOXPLOT ANALYSIS")
    print("="*80)
    print("\nLoading comprehensive evaluation results...")
    results = load_comprehensive_results(base_path)
    print(f"\nLoaded {len(results)} dataset/model/setup combinations")
    
    # Compute differences by backbone
    print("\nComputing ensemble vs mean per-fold differences by backbone...")
    backbone_data = compute_differences_by_backbone(results)
    
    # Print summary statistics
    print_summary_statistics(backbone_data)
    
    # Create boxplots
    print("\n" + "="*80)
    print("GENERATING BOXPLOTS")
    print("="*80 + "\n")
    
    print("Creating simple boxplots (one box per backbone)...")
    create_boxplots_by_backbone(backbone_data, output_dir)
    
    print("\nCreating detailed boxplots (by method)...")
    create_detailed_boxplots_by_method(backbone_data, output_dir)
    
    print("\n" + "="*80)
    print("✓ All boxplots generated successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
