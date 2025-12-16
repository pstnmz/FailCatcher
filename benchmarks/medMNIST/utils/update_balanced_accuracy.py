"""
Update UQ benchmark JSON files with balanced accuracy from training runs.
Searches for metrics_ensemble.json in benchmarks/medMNIST/runs/*/models/ folders.
"""

import json
import os
from pathlib import Path
import re


def find_metrics_ensemble(dataset_name, model_backbone='resnet18', setup=''):
    """
    Find metrics_ensemble.json for a given dataset, model, and setup.
    
    Args:
        dataset_name: e.g., 'breastmnist', 'octmnist'
        model_backbone: e.g., 'resnet18', 'vit_b_16'
        setup: Training setup ('', 'DA', 'DO', 'DADO')
               '' or unspecified = standard (randaug0)
               'DA' = data augmentation (randaug1)
               'DO' = dropout enabled
               'DADO' = both DA and DO
    
    Returns:
        Path to metrics_ensemble.json or None
    """
    # Base path to runs
    runs_dir = Path(__file__).parent.parent / 'benchmarks' / 'medMNIST' / 'runs' / dataset_name
    
    if not runs_dir.exists():
        return None
    
    # Map setup to randaug value
    # DA means randaug1, standard/empty means randaug0
    randaug = 1 if setup in ['DA', 'DADO'] else 0
    
    # Look for folders matching the model backbone and randaug setting
    pattern = f"{model_backbone}_224_*_randaug{randaug}_*"
    matching_dirs = list(runs_dir.glob(pattern))
    
    if not matching_dirs:
        return None
    
    # Filter based on dropout requirement
    if setup == 'DA':
        # DA only: exclude folders with dropout
        matching_dirs = [d for d in matching_dirs if 'dropout' not in d.name]
    elif setup == 'DO':
        # DO only: must have dropout AND randaug0
        matching_dirs = [d for d in matching_dirs if 'dropout' in d.name]
    elif setup == 'DADO':
        # DADO: must have dropout AND randaug1
        matching_dirs = [d for d in matching_dirs if 'dropout' in d.name]
    else:
        # Standard setup: exclude folders with dropout
        matching_dirs = [d for d in matching_dirs if 'dropout' not in d.name]
    
    if not matching_dirs:
        return None
    
    # Use the most recent one (by folder name timestamp)
    latest_dir = max(matching_dirs, key=lambda p: p.name)
    
    # Debug output
    print(f"    [DEBUG] Found {len(matching_dirs)} folders for {dataset_name}/{model_backbone} (setup={setup}, randaug={randaug})")
    print(f"    [DEBUG] Using: {latest_dir.name}")
    
    metrics_file = latest_dir / 'metrics_ensemble.json'
    
    if metrics_file.exists():
        return metrics_file
    
    return None


def extract_balanced_accuracy(metrics_file):
    """
    Extract balanced accuracy from metrics_ensemble.json.
    
    Returns:
        float: balanced accuracy or None
    """
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # Look for balanced_accuracy in the JSON
        # It might be under different keys depending on structure
        if 'balanced_accuracy' in data:
            return float(data['balanced_accuracy'])
        elif 'test' in data and 'balanced_accuracy' in data['test']:
            return float(data['test']['balanced_accuracy'])
        elif 'balanced_acc' in data:
            return float(data['balanced_acc'])
        elif 'test' in data and 'balanced_acc' in data['test']:
            return float(data['test']['balanced_acc'])
        
        # If not found, try to find it in any nested structure
        def search_dict(d):
            if isinstance(d, dict):
                for key, value in d.items():
                    if 'balanced' in key.lower() and 'acc' in key.lower():
                        if isinstance(value, (int, float)):
                            return float(value)
                    result = search_dict(value)
                    if result is not None:
                        return result
            return None
        
        return search_dict(data)
    
    except Exception as e:
        print(f"  Error reading {metrics_file}: {e}")
        return None


def update_uq_benchmark_files(results_dir=None):
    """
    Update all uq_benchmark_*.json files with balanced accuracy.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent
    else:
        results_dir = Path(results_dir)
    
    # Find all uq_benchmark JSON files
    json_files = list(results_dir.glob('uq_benchmark_*.json'))
    
    print(f"Found {len(json_files)} UQ benchmark files")
    print("=" * 80)
    
    updated_count = 0
    skipped_count = 0
    
    for json_file in sorted(json_files):
        try:
            # Load the UQ benchmark file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            flag = data.get('flag', '')
            model_backbone = data.get('model_backbone', 'resnet18')
            setup = data.get('setup', '')  # Get setup (DA, DO, DADO, or '')
            
            # Skip if already has balanced_accuracy
            if 'ensemble_balanced_accuracy' in data:
                print(f"  ✓ {json_file.name}: already has balanced_accuracy")
                skipped_count += 1
                continue
            
            # Find corresponding metrics_ensemble.json with matching setup
            metrics_file = find_metrics_ensemble(flag, model_backbone, setup)
            
            if metrics_file is None:
                setup_str = f" (setup: {setup})" if setup else ""
                print(f"  ⚠️  {json_file.name}: metrics_ensemble.json not found for {flag}/{model_backbone}{setup_str}")
                skipped_count += 1
                continue
            
            # Extract balanced accuracy
            bal_acc = extract_balanced_accuracy(metrics_file)
            
            if bal_acc is None:
                print(f"  ⚠️  {json_file.name}: could not extract balanced_accuracy from {metrics_file}")
                skipped_count += 1
                continue
            
            # Add to UQ benchmark file
            data['ensemble_balanced_accuracy'] = bal_acc
            
            # Write back
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  ✓ {json_file.name}: added balanced_accuracy = {bal_acc:.4f}")
            updated_count += 1
        
        except Exception as e:
            print(f"  ❌ {json_file.name}: error - {e}")
            skipped_count += 1
    
    print("=" * 80)
    print(f"Updated: {updated_count}, Skipped: {skipped_count}")


if __name__ == '__main__':
    update_uq_benchmark_files()
