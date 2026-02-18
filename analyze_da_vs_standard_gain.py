#!/usr/bin/env python3
"""
Analyze balanced accuracy gain between DA and standard setups across different shift types.
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

def extract_per_fold_balanced_accuracy(json_file):
    """Extract per-fold balanced accuracy from a JSON file and return mean."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if "per_fold" in data:
        # corruption_shifts format
        b_accs = [fold["balanced_accuracy"] for fold in data["per_fold"]]
        return np.mean(b_accs)
    elif "per_fold_metrics" in data:
        # in_distribution/population_shift format
        b_accs = [fold["balanced_accuracy"] for fold in data["per_fold_metrics"]]
        return np.mean(b_accs)
    else:
        return None

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
    # Storage for results: shift_type -> dataset -> model -> setup -> b_acc
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
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
                
            b_acc = extract_per_fold_balanced_accuracy(json_file)
            if b_acc is not None:
                results[shift_name][dataset][model][setup] = b_acc
    
    # Calculate DA vs standard gains
    print("\n" + "="*100)
    print("BALANCED ACCURACY GAIN ANALYSIS: DA vs Standard Setup")
    print("="*100)
    
    shift_summary = {}
    
    for shift_name in shift_categories.keys():
        print(f"\n{shift_name.upper().replace('_', ' ')}")
        print("-" * 100)
        
        da_values = []
        standard_values = []
        gains_per_dataset = []
        
        # Organize by dataset, combining both models
        dataset_results = {}
        
        for dataset in sorted(results[shift_name].keys()):
            dataset_data = results[shift_name][dataset]
            
            # Collect all DA and standard values across both models
            da_baccs = []
            standard_baccs = []
            
            for model in ['resnet18', 'vit_b_16']:
                if model in dataset_data:
                    if 'DA' in dataset_data[model]:
                        da_baccs.append(dataset_data[model]['DA'])
                    if 'standard' in dataset_data[model]:
                        standard_baccs.append(dataset_data[model]['standard'])
            
            if da_baccs and standard_baccs:
                da_mean = np.mean(da_baccs)
                standard_mean = np.mean(standard_baccs)
                gain = da_mean - standard_mean
                gain_pct = (gain / standard_mean) * 100
                
                da_values.extend(da_baccs)
                standard_values.extend(standard_baccs)
                gains_per_dataset.append(gain)
                
                print(f"  {dataset:25s} | Standard: {standard_mean:.4f} | DA: {da_mean:.4f} | "
                      f"Gain: {gain:+.4f} ({gain_pct:+.2f}%) | N_std: {len(standard_baccs)}, N_DA: {len(da_baccs)}")
        
        # Overall statistics for this shift type
        if standard_values and da_values:
            overall_standard = np.mean(standard_values)
            overall_da = np.mean(da_values)
            overall_gain = overall_da - overall_standard
            overall_gain_pct = (overall_gain / overall_standard) * 100
            mean_gain = np.mean(gains_per_dataset)
            
            print(f"\n  {'OVERALL (mean across all)':25s} | Standard: {overall_standard:.4f} | DA: {overall_da:.4f} | "
                  f"Gain: {overall_gain:+.4f} ({overall_gain_pct:+.2f}%)")
            print(f"  {'Mean gain per dataset:':25s} {mean_gain:+.4f}")
            print(f"  {'Std of gains:':25s} {np.std(gains_per_dataset):.4f}")
            
            shift_summary[shift_name] = {
                'overall_standard': overall_standard,
                'overall_da': overall_da,
                'overall_gain': overall_gain,
                'overall_gain_pct': overall_gain_pct,
                'mean_gain_per_dataset': mean_gain,
                'std_gain': np.std(gains_per_dataset),
                'n_datasets': len(gains_per_dataset),
                'n_samples_standard': len(standard_values),
                'n_samples_da': len(da_values)
            }
    
    # Summary across all shifts
    print("\n" + "="*100)
    print("SUMMARY ACROSS ALL SHIFT TYPES")
    print("="*100)
    
    for shift_name, summary in shift_summary.items():
        print(f"\n{shift_name.upper().replace('_', ' ')}:")
        print(f"  Standard mean b_acc:     {summary['overall_standard']:.4f}")
        print(f"  DA mean b_acc:           {summary['overall_da']:.4f}")
        print(f"  Gain:                    {summary['overall_gain']:+.4f} ({summary['overall_gain_pct']:+.2f}%)")
        print(f"  Mean gain per dataset:   {summary['mean_gain_per_dataset']:+.4f}")
        print(f"  Std of gains:            {summary['std_gain']:.4f}")
        print(f"  Datasets analyzed:       {summary['n_datasets']}")
        print(f"  Total samples (Std):     {summary['n_samples_standard']}")
        print(f"  Total samples (DA):      {summary['n_samples_da']}")
    
    # Overall comparison
    print("\n" + "="*100)
    print("CROSS-SHIFT COMPARISON")
    print("="*100)
    
    all_gains = [(name, summary['overall_gain']) for name, summary in shift_summary.items()]
    all_gains.sort(key=lambda x: x[1], reverse=True)
    
    print("\nShift types ranked by DA gain over standard:")
    for i, (shift_name, gain) in enumerate(all_gains, 1):
        summary = shift_summary[shift_name]
        print(f"  {i}. {shift_name:25s}: {gain:+.4f} ({summary['overall_gain_pct']:+.2f}%)")
    
    # Breakdown by model type
    print("\n" + "="*100)
    print("BREAKDOWN BY MODEL TYPE")
    print("="*100)
    
    for shift_name in shift_categories.keys():
        print(f"\n{shift_name.upper().replace('_', ' ')}")
        print("-" * 100)
        
        for model in ['resnet18', 'vit_b_16']:
            model_da_values = []
            model_standard_values = []
            model_gains = []
            
            for dataset in sorted(results[shift_name].keys()):
                dataset_data = results[shift_name][dataset]
                
                if model in dataset_data:
                    if 'DA' in dataset_data[model] and 'standard' in dataset_data[model]:
                        da_val = dataset_data[model]['DA']
                        std_val = dataset_data[model]['standard']
                        gain = da_val - std_val
                        
                        model_da_values.append(da_val)
                        model_standard_values.append(std_val)
                        model_gains.append(gain)
            
            if model_standard_values and model_da_values:
                model_std_mean = np.mean(model_standard_values)
                model_da_mean = np.mean(model_da_values)
                model_gain = model_da_mean - model_std_mean
                model_gain_pct = (model_gain / model_std_mean) * 100
                
                print(f"  {model:15s} | Standard: {model_std_mean:.4f} | DA: {model_da_mean:.4f} | "
                      f"Gain: {model_gain:+.4f} ({model_gain_pct:+.2f}%) | N: {len(model_standard_values)}")

if __name__ == "__main__":
    main()
