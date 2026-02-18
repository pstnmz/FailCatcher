#!/usr/bin/env python3
"""
Analyze balanced accuracy gain between ensemble and per-fold mean across different shift types.
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

def extract_balanced_accuracies(json_file):
    """Extract both per-fold mean and ensemble balanced accuracy from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract per-fold mean
    per_fold = data.get('per_fold', [])
    if not per_fold:
        per_fold = data.get('per_fold_metrics', [])
    
    if not per_fold:
        return None, None
    
    bacc_values = [fold.get('balanced_accuracy') for fold in per_fold if fold.get('balanced_accuracy') is not None]
    if not bacc_values:
        return None, None
    
    perfold_mean = np.mean(bacc_values)
    
    # Extract ensemble
    if "ensemble" in data:
        # corruption_shifts format
        ensemble_bacc = data["ensemble"]["balanced_accuracy"]
    elif "ensemble_metrics" in data:
        # in_distribution/population_shift format
        ensemble_bacc = data["ensemble_metrics"]["balanced_accuracy"]
    else:
        ensemble_bacc = None
    
    return perfold_mean, ensemble_bacc

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
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
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
                
            perfold_mean, ensemble_bacc = extract_balanced_accuracies(json_file)
            if perfold_mean is not None and ensemble_bacc is not None:
                results[shift_name][dataset][model][setup] = {
                    'perfold_mean': perfold_mean,
                    'ensemble': ensemble_bacc
                }
    
    # Calculate statistics and gains
    print("\n" + "="*100)
    print("BALANCED ACCURACY COMPARISON: Ensemble vs Per-Fold Mean")
    print("="*100)
    
    shift_summary = {}
    
    for shift_name in shift_categories.keys():
        print(f"\n{shift_name.upper().replace('_', ' ')}")
        print("-" * 100)
        
        perfold_values = []
        ensemble_values = []
        gains = []
        
        for dataset in sorted(results[shift_name].keys()):
            for model in sorted(results[shift_name][dataset].keys()):
                for setup in sorted(results[shift_name][dataset][model].keys()):
                    data = results[shift_name][dataset][model][setup]
                    
                    perfold = data['perfold_mean']
                    ensemble = data['ensemble']
                    gain = ensemble - perfold
                    gain_pct = (gain / perfold) * 100
                    
                    perfold_values.append(perfold)
                    ensemble_values.append(ensemble)
                    gains.append(gain)
                    
                    print(f"  {dataset:20s} | {model:10s} | {setup:8s} | Per-fold: {perfold:.4f} | "
                          f"Ensemble: {ensemble:.4f} | Gain: {gain:+.4f} ({gain_pct:+.2f}%)")
        
        # Overall statistics for this shift type
        if perfold_values and ensemble_values:
            overall_perfold = np.mean(perfold_values)
            overall_ensemble = np.mean(ensemble_values)
            overall_gain = overall_ensemble - overall_perfold
            overall_gain_pct = (overall_gain / overall_perfold) * 100
            mean_gain = np.mean(gains)
            
            print(f"\n  {'OVERALL (mean across all)':45s} | Per-fold: {overall_perfold:.4f} | "
                  f"Ensemble: {overall_ensemble:.4f} | Gain: {overall_gain:+.4f} ({overall_gain_pct:+.2f}%)")
            print(f"  {'Mean gain:':45s} {mean_gain:+.4f}")
            print(f"  {'Std of gains:':45s} {np.std(gains):.4f}")
            print(f"  {'Min gain:':45s} {np.min(gains):+.4f}")
            print(f"  {'Max gain:':45s} {np.max(gains):+.4f}")
            
            shift_summary[shift_name] = {
                'overall_perfold': overall_perfold,
                'overall_ensemble': overall_ensemble,
                'overall_gain': overall_gain,
                'overall_gain_pct': overall_gain_pct,
                'mean_gain': mean_gain,
                'std_gain': np.std(gains),
                'min_gain': np.min(gains),
                'max_gain': np.max(gains),
                'n_samples': len(perfold_values)
            }
    
    # Summary across all shifts
    print("\n" + "="*100)
    print("SUMMARY ACROSS ALL SHIFT TYPES")
    print("="*100)
    
    for shift_name, summary in shift_summary.items():
        print(f"\n{shift_name.upper().replace('_', ' ')}:")
        print(f"  Per-fold mean b_acc:     {summary['overall_perfold']:.4f}")
        print(f"  Ensemble b_acc:          {summary['overall_ensemble']:.4f}")
        print(f"  Gain:                    {summary['overall_gain']:+.4f} ({summary['overall_gain_pct']:+.2f}%)")
        print(f"  Mean gain:               {summary['mean_gain']:+.4f}")
        print(f"  Std gain:                {summary['std_gain']:.4f}")
        print(f"  Range:                   [{summary['min_gain']:+.4f}, {summary['max_gain']:+.4f}]")
        print(f"  Total samples:           {summary['n_samples']}")
    
    # Overall comparison
    print("\n" + "="*100)
    print("CROSS-SHIFT COMPARISON")
    print("="*100)
    
    all_gains = [(name, summary['overall_gain']) for name, summary in shift_summary.items()]
    all_gains.sort(key=lambda x: x[1], reverse=True)
    
    print("\nShift types ranked by ensemble gain over per-fold mean:")
    for i, (shift_name, gain) in enumerate(all_gains, 1):
        summary = shift_summary[shift_name]
        print(f"  {i}. {shift_name:25s}: {gain:+.4f} ({summary['overall_gain_pct']:+.2f}%)")
    
    # Grand total across all shifts
    print("\n" + "="*100)
    print("GRAND TOTAL ACROSS ALL SHIFTS")
    print("="*100)
    
    all_perfold = []
    all_ensemble = []
    all_gains_flat = []
    
    for shift_name in shift_categories.keys():
        for dataset in results[shift_name].keys():
            for model in results[shift_name][dataset].keys():
                for setup in results[shift_name][dataset][model].keys():
                    data = results[shift_name][dataset][model][setup]
                    all_perfold.append(data['perfold_mean'])
                    all_ensemble.append(data['ensemble'])
                    all_gains_flat.append(data['ensemble'] - data['perfold_mean'])
    
    if all_perfold and all_ensemble:
        grand_perfold = np.mean(all_perfold)
        grand_ensemble = np.mean(all_ensemble)
        grand_gain = grand_ensemble - grand_perfold
        grand_gain_pct = (grand_gain / grand_perfold) * 100
        
        print(f"  Per-fold mean b_acc:     {grand_perfold:.4f}")
        print(f"  Ensemble b_acc:          {grand_ensemble:.4f}")
        print(f"  Overall gain:            {grand_gain:+.4f} ({grand_gain_pct:+.2f}%)")
        print(f"  Mean gain:               {np.mean(all_gains_flat):+.4f}")
        print(f"  Std gain:                {np.std(all_gains_flat):.4f}")
        print(f"  Range:                   [{np.min(all_gains_flat):+.4f}, {np.max(all_gains_flat):+.4f}]")
        print(f"  Total samples:           {len(all_perfold)}")

if __name__ == "__main__":
    main()
