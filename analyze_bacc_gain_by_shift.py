#!/usr/bin/env python3
"""
Analyze balanced accuracy gain between ResNet18 and ViT-B16 across different shift types.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Base directory
base_dir = Path("/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/utils/comprehensive_evaluation_results")

# Define shift categories
shift_categories = {
    "corruption_shifts": base_dir / "corruption_shifts",
    "in_distribution": base_dir / "in_distribution", 
    "population_shift": base_dir / "population_shift"
}

def extract_balanced_accuracy(json_file):
    """Extract mean balanced accuracy across folds from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    # corruption_shifts format uses 'per_fold'
    per_fold = data.get('per_fold', [])
    # in_distribution/population_shift format uses 'per_fold_metrics'
    if not per_fold:
        per_fold = data.get('per_fold_metrics', [])
    
    if not per_fold:
        return None
    
    bacc_values = [fold.get('balanced_accuracy') for fold in per_fold if fold.get('balanced_accuracy') is not None]
    if not bacc_values:
        return None
    
    return np.mean(bacc_values)

def parse_filename(filename):
    """Parse filename to extract dataset, model, and setup info."""
    # Remove .json extension
    name = filename.replace('.json', '')
    
    # Handle corruption_shifts format: {dataset}_{model}_{setup}_severity3.json
    if 'severity3' in name:
        parts = name.replace('_severity3', '').split('_')
        if 'resnet18' in name:
            model_idx = parts.index('resnet18')
            dataset = '_'.join(parts[:model_idx])
            model = 'resnet18'
            setup = '_'.join(parts[model_idx+1:]) if model_idx+1 < len(parts) else 'standard'
        elif 'vit_b_16' in name:
            # Find where vit_b_16 starts
            vit_parts = []
            i = 0
            while i < len(parts):
                if parts[i] == 'vit' and i+2 < len(parts) and parts[i+1] == 'b' and parts[i+2] == '16':
                    break
                vit_parts.append(parts[i])
                i += 1
            dataset = '_'.join(vit_parts)
            model = 'vit_b_16'
            setup = '_'.join(parts[i+3:]) if i+3 < len(parts) else 'standard'
        else:
            return None, None, None
    # Handle in_distribution/population_shift format: comprehensive_metrics_{dataset}_{model}_{setup}.json
    elif 'comprehensive_metrics' in name:
        parts = name.replace('comprehensive_metrics_', '').split('_')
        if 'resnet18' in name:
            model_idx = parts.index('resnet18')
            dataset = '_'.join(parts[:model_idx])
            model = 'resnet18'
            setup = '_'.join(parts[model_idx+1:]) if model_idx+1 < len(parts) else 'standard'
        elif 'vit' in name:
            # Find where vit_b_16 starts
            vit_parts = []
            i = 0
            while i < len(parts):
                if parts[i] == 'vit' and i+2 < len(parts) and parts[i+1] == 'b' and parts[i+2] == '16':
                    break
                vit_parts.append(parts[i])
                i += 1
            dataset = '_'.join(vit_parts)
            model = 'vit_b_16'
            setup = '_'.join(parts[i+3:]) if i+3 < len(parts) else 'standard'
        else:
            return None, None, None
    else:
        return None, None, None
    
    return dataset, model, setup

def main():
    # Storage for results
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Process each shift category
    for shift_name, shift_dir in shift_categories.items():
        print(f"\n{'='*80}")
        print(f"Processing: {shift_name}")
        print(f"{'='*80}")
        
        # Get all JSON files
        json_files = list(shift_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        # Process each file
        for json_file in json_files:
            dataset, model, setup = parse_filename(json_file.name)
            if dataset is None or model is None:
                continue
                
            b_acc = extract_balanced_accuracy(json_file)
            if b_acc is not None:
                results[shift_name][dataset][model].append({
                    'setup': setup,
                    'b_acc': b_acc
                })
    
    # Calculate statistics and gains
    print("\n" + "="*100)
    print("BALANCED ACCURACY GAIN ANALYSIS: ViT-B16 vs ResNet18")
    print("="*100)
    
    shift_summary = {}
    
    for shift_name in shift_categories.keys():
        print(f"\n{shift_name.upper().replace('_', ' ')}")
        print("-" * 100)
        
        resnet_values = []
        vit_values = []
        gains = []
        dataset_gains = []
        
        for dataset in sorted(results[shift_name].keys()):
            dataset_data = results[shift_name][dataset]
            
            if 'resnet18' in dataset_data and 'vit_b_16' in dataset_data:
                # Get all balanced accuracies for this dataset
                resnet_baccs = [item['b_acc'] for item in dataset_data['resnet18']]
                vit_baccs = [item['b_acc'] for item in dataset_data['vit_b_16']]
                
                # Calculate means
                resnet_mean = np.mean(resnet_baccs)
                vit_mean = np.mean(vit_baccs)
                gain = vit_mean - resnet_mean
                gain_pct = (gain / resnet_mean) * 100
                
                resnet_values.extend(resnet_baccs)
                vit_values.extend(vit_baccs)
                gains.append(gain)
                dataset_gains.append({
                    'dataset': dataset,
                    'resnet_mean': resnet_mean,
                    'vit_mean': vit_mean,
                    'gain': gain,
                    'gain_pct': gain_pct,
                    'n_resnet': len(resnet_baccs),
                    'n_vit': len(vit_baccs)
                })
                
                print(f"  {dataset:25s} | ResNet18: {resnet_mean:.4f} | ViT-B16: {vit_mean:.4f} | "
                      f"Gain: {gain:+.4f} ({gain_pct:+.2f}%) | N: {len(resnet_baccs)}")
        
        # Overall statistics for this shift type
        if resnet_values and vit_values:
            overall_resnet = np.mean(resnet_values)
            overall_vit = np.mean(vit_values)
            overall_gain = overall_vit - overall_resnet
            overall_gain_pct = (overall_gain / overall_resnet) * 100
            mean_gain = np.mean(gains)
            
            print(f"\n  {'OVERALL (mean across all)':25s} | ResNet18: {overall_resnet:.4f} | ViT-B16: {overall_vit:.4f} | "
                  f"Gain: {overall_gain:+.4f} ({overall_gain_pct:+.2f}%)")
            print(f"  {'Mean gain per dataset:':25s} {mean_gain:+.4f}")
            print(f"  {'Std of gains:':25s} {np.std(gains):.4f}")
            
            shift_summary[shift_name] = {
                'overall_resnet': overall_resnet,
                'overall_vit': overall_vit,
                'overall_gain': overall_gain,
                'overall_gain_pct': overall_gain_pct,
                'mean_gain_per_dataset': mean_gain,
                'std_gain': np.std(gains),
                'n_datasets': len(dataset_gains),
                'n_samples_resnet': len(resnet_values),
                'n_samples_vit': len(vit_values)
            }
    
    # Summary across all shifts
    print("\n" + "="*100)
    print("SUMMARY ACROSS ALL SHIFT TYPES")
    print("="*100)
    
    for shift_name, summary in shift_summary.items():
        print(f"\n{shift_name.upper().replace('_', ' ')}:")
        print(f"  ResNet18 mean b_acc:     {summary['overall_resnet']:.4f}")
        print(f"  ViT-B16 mean b_acc:      {summary['overall_vit']:.4f}")
        print(f"  Gain:                    {summary['overall_gain']:+.4f} ({summary['overall_gain_pct']:+.2f}%)")
        print(f"  Datasets analyzed:       {summary['n_datasets']}")
        print(f"  Total samples (ResNet):  {summary['n_samples_resnet']}")
        print(f"  Total samples (ViT):     {summary['n_samples_vit']}")
    
    # Overall comparison
    print("\n" + "="*100)
    print("CROSS-SHIFT COMPARISON")
    print("="*100)
    
    all_gains = [(name, summary['overall_gain']) for name, summary in shift_summary.items()]
    all_gains.sort(key=lambda x: x[1], reverse=True)
    
    print("\nShift types ranked by ViT-B16 gain over ResNet18:")
    for i, (shift_name, gain) in enumerate(all_gains, 1):
        summary = shift_summary[shift_name]
        print(f"  {i}. {shift_name:25s}: {gain:+.4f} ({summary['overall_gain_pct']:+.2f}%)")

if __name__ == "__main__":
    main()
