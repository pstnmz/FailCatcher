"""
Script to parse UQ benchmark JSON files and plot test_accuracy vs augrc_mean
from Mean_Aggregation_Ensemble method.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_json_files(base_dir):
    """
    Parse all JSON files from the three shift subfolders.
    
    Returns:
        list of dicts with keys: config_name, shift_type, test_accuracy, augrc_mean
    """
    results = []
    
    # Define shift types and their corresponding folders
    shift_folders = {
        'in_distribution': 'ID',
        'corruption_shifts': 'Corruption',
        'population_shifts': 'Population'
    }
    
    for folder, shift_type in shift_folders.items():
        folder_path = Path(base_dir) / folder
        
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            continue
        
        # Get all JSON files in the folder
        json_files = list(folder_path.glob('*.json'))
        print(f"Found {len(json_files)} JSON files in {folder}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract required values
                test_accuracy = data.get('test_accuracy')
                augrc_mean = data.get('methods', {}).get('Mean_Aggregation_Ensemble', {}).get('augrc_mean')
                
                # Skip if either value is missing
                if test_accuracy is None or augrc_mean is None:
                    print(f"Skipping {json_file.name}: missing required fields")
                    continue
                
                # Parse configuration from filename
                # Expected format: uq_benchmark_{dataset}_{backbone}_{setup}_{shift_info}_{timestamp}.json
                filename = json_file.stem  # Remove .json extension
                parts = filename.replace('uq_benchmark_', '').split('_')
                
                # Extract dataset and backbone
                dataset = parts[0] if len(parts) > 0 else 'unknown'
                
                # Handle ViT backbone (vit_b_16) vs ResNet (resnet18)
                if len(parts) > 1 and parts[1] == 'vit' and len(parts) > 2 and parts[2] == 'b':
                    # ViT model: vit_b_16
                    backbone = 'vit_b_16'
                    setup_start_idx = 3  # Setup starts after 'vit', 'b', '16'
                    if len(parts) > 3 and parts[3] == '16':
                        setup_start_idx = 4
                    else:
                        setup_start_idx = 3
                else:
                    # ResNet or other model
                    backbone = parts[1] if len(parts) > 1 else 'unknown'
                    setup_start_idx = 2
                
                # Determine setup (DADO, DA, DO, or empty/baseline)
                setup = ''
                if len(parts) > setup_start_idx:
                    # Check if the part at setup position is a setup keyword
                    if parts[setup_start_idx] in ['DADO', 'DA', 'DO']:
                        setup = parts[setup_start_idx]
                
                if not setup:
                    setup = 'Baseline'
                
                # Create configuration name
                config_name = f"{shift_type}_{dataset}_{backbone}_{setup}"
                
                results.append({
                    'config_name': config_name,
                    'shift_type': shift_type,
                    'dataset': dataset,
                    'backbone': backbone,
                    'setup': setup,
                    'test_accuracy': test_accuracy,
                    'augrc_mean': augrc_mean,
                    'filename': json_file.name
                })
                
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
    
    return results

def create_scatter_plot(results, output_path='ensemble_vs_accuracy_plot.png'):
    """
    Create a scatter plot with hierarchical configuration labels on x-axis and
    test_accuracy and augrc_mean on y-axis.
    """
    if not results:
        print("No results to plot!")
        return
    
    # Define custom sort order for shift types: ID, Corruption, Population
    shift_order = {'ID': 0, 'Corruption': 1, 'Population': 2}
    
    # Sort results by shift type (custom order), then dataset, then backbone, then setup
    results_sorted = sorted(results, key=lambda x: (shift_order.get(x['shift_type'], 99), x['dataset'], x['backbone'], x['setup']))
    
    # Replace 'Baseline' with 'S' (Standard/Single)
    for r in results_sorted:
        if r['setup'] == 'Baseline':
            r['setup'] = 'S'
    
    # Extract data
    test_accuracies = [r['test_accuracy'] for r in results_sorted]
    augrc_means = [r['augrc_mean'] for r in results_sorted]
    
    # Create figure with larger size to accommodate many configurations
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # X positions for each configuration
    x_pos = np.arange(len(results_sorted))
    
    # Convert to error rates (1 - accuracy)
    test_error_rates = [1 - acc for acc in test_accuracies]
    
    # Use nice colors from tab20 colormap
    cmap = plt.cm.tab20
    blue_color = cmap(0)      # Nice blue
    yellow_color = cmap(17)    # Orange/yellow
    orange_color = cmap(7)    # Orange
    
    # Plot both metrics
    ax.scatter(x_pos, test_error_rates, alpha=0.7, s=100, label='Test Error Rate (1-Acc)', color=blue_color, marker='o')
    ax.scatter(x_pos, augrc_means, alpha=0.8, s=200, label='Mean Agg+Ens AUGRC', color=yellow_color, marker='$\u26A1$', edgecolors='black', linewidths=0.5)
    
    # Configure plot
    ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
    ax.set_title('Test Error Rate (1-Acc) vs AUGRC Mean Aggregation + Ensemble Across Configurations', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Create hierarchical tick labels
    # Level 1 (top): Setup (DA, DO, DADO, S)
    setup_labels = [r['setup'] for r in results_sorted]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(setup_labels, rotation=45, ha='right', fontsize=8)
    
    # Level 2: Backbone - add text labels with separators
    current_shift = None
    current_dataset = None
    current_backbone = None
    
    # Draw vertical separators and add grouped labels
    for i, r in enumerate(results_sorted):
        # Add vertical lines to separate shift types only
        if current_shift != r['shift_type']:
            if i > 0:
                ax.axvline(x=i-0.5, color='black', linestyle='-', linewidth=2)
            current_shift = r['shift_type']
        
        # Track dataset and backbone changes without drawing lines
        if current_dataset != r['dataset']:
            current_dataset = r['dataset']
        
        if current_backbone != r['backbone']:
            current_backbone = r['backbone']
    
    # Add group labels for backbone
    backbone_groups = {}
    for i, r in enumerate(results_sorted):
        key = (r['shift_type'], r['dataset'], r['backbone'])
        if key not in backbone_groups:
            backbone_groups[key] = []
        backbone_groups[key].append(i)
    
    for key, indices in backbone_groups.items():
        mid_pos = (indices[0] + indices[-1]) / 2
        ax.text(mid_pos, -0.08, key[2], ha='center', va='top', fontsize=9, 
                transform=ax.get_xaxis_transform(), fontweight='bold', rotation=15)
    
    # Add group labels for dataset
    dataset_groups = {}
    for i, r in enumerate(results_sorted):
        key = (r['shift_type'], r['dataset'])
        if key not in dataset_groups:
            dataset_groups[key] = []
        dataset_groups[key].append(i)
    
    for key, indices in dataset_groups.items():
        mid_pos = (indices[0] + indices[-1]) / 2
        # Shorten long dataset names for display
        dataset_name = key[1].replace('dermamnist-e-external', 'dermamnist-e-ext')
        ax.text(mid_pos, -0.12, dataset_name, ha='center', va='top', fontsize=10, 
                transform=ax.get_xaxis_transform(), fontweight='bold', color='black', rotation=15)
    
    # Add group labels for shift type
    shift_groups = {}
    for i, r in enumerate(results_sorted):
        key = r['shift_type']
        if key not in shift_groups:
            shift_groups[key] = []
        shift_groups[key].append(i)
    
    for key, indices in shift_groups.items():
        mid_pos = (indices[0] + indices[-1]) / 2
        ax.text(mid_pos, -0.20, key, ha='center', va='top', fontsize=11, 
                transform=ax.get_xaxis_transform(), fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    # Adjust layout to make room for the hierarchical labels
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total configurations: {len(results_sorted)}")
    print(f"\nTest Error Rate (1-Accuracy):")
    print(f"  Mean: {np.mean(test_error_rates):.4f}")
    print(f"  Std:  {np.std(test_error_rates):.4f}")
    print(f"  Min:  {np.min(test_error_rates):.4f}")
    print(f"  Max:  {np.max(test_error_rates):.4f}")
    print(f"\nEnsemble AUGRC Mean:")
    print(f"  Mean: {np.mean(augrc_means):.4f}")
    print(f"  Std:  {np.std(augrc_means):.4f}")
    print(f"  Min:  {np.min(augrc_means):.4f}")
    print(f"  Max:  {np.max(augrc_means):.4f}")

def main():
    # Base directory containing the three subfolders
    base_dir = '/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/jsons_results'
    
    print("Parsing JSON files...")
    results = parse_json_files(base_dir)
    
    print(f"\nSuccessfully parsed {len(results)} configurations")
    
    # Create the scatter plot
    output_path = '/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/ensemble_vs_accuracy_plot.png'
    create_scatter_plot(results, output_path)
    
    # Optionally save results to CSV for further analysis
    import csv
    csv_path = '/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/ensemble_vs_accuracy_data.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['config_name', 'shift_type', 'dataset', 'backbone', 'setup', 'test_accuracy', 'augrc_mean', 'filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda x: (x['shift_type'], x['dataset'], x['backbone'], x['setup'])):
            writer.writerow(result)
    print(f"Data saved to: {csv_path}")

if __name__ == '__main__':
    main()
