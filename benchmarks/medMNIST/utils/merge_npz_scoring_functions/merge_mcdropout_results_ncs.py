#!/usr/bin/env python3
"""
Script to merge new MCDropout results (k=30) from to_merge/ subdirectory
into main new_class_shifts result files, replacing old MCDropout data.
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple
import re

def extract_file_key(filename: str) -> Tuple[str, str, str]:
    """Extract (dataset, model, setup) from filename for matching."""
    # Pattern: uq_benchmark_{dataset}_{model}_{setup}_{timestamp}.json
    # or: all_metrics_{dataset}_{model}_{setup}_{timestamp}.npz
    
    # Remove prefix and suffix
    if filename.startswith('uq_benchmark_'):
        name = filename.replace('uq_benchmark_', '')
    elif filename.startswith('all_metrics_'):
        name = filename.replace('all_metrics_', '')
    else:
        return None
    
    # Remove timestamp suffix: _{timestamp}.json or _{timestamp}.npz
    name = re.sub(r'_\d+_\d+\.(json|npz)$', '', name)
    
    # Parse: dataset_model_setup
    parts = name.split('_')
    
    # Handle cases like vit_b_16
    # Setup is last part (DADO, DO, DA, or no setup for vanilla)
    if parts[-1] in ['DADO', 'DO', 'DA']:
        setup = parts[-1]
        rest = parts[:-1]
    else:
        setup = 'vanilla'
        rest = parts
    
    # Model: resnet18 or vit_b_16
    if 'vit' in rest:
        idx = rest.index('vit')
        model = '_'.join(rest[idx:])
        dataset = '_'.join(rest[:idx])
    elif 'resnet18' in rest:
        model = 'resnet18'
        dataset = '_'.join(rest[:-1])
    else:
        return None
    
    return (dataset, model, setup)

def merge_json_files(parent_dir: Path, merge_dir: Path) -> None:
    """Merge MCDropout results from merge_dir JSONs into parent_dir JSONs."""
    
    # Build index of merge files
    merge_files = {}
    for merge_file in merge_dir.glob('uq_benchmark_*.json'):
        key = extract_file_key(merge_file.name)
        if key:
            merge_files[key] = merge_file
    
    print(f"Found {len(merge_files)} MCDropout result files to merge")
    
    # Process parent files
    updated_count = 0
    not_found_count = 0
    
    for parent_file in parent_dir.glob('uq_benchmark_*.json'):
        key = extract_file_key(parent_file.name)
        
        if not key:
            print(f"Warning: Could not parse {parent_file.name}")
            continue
        
        if key not in merge_files:
            not_found_count += 1
            continue
        
        # Load both files
        with open(parent_file, 'r') as f:
            parent_data = json.load(f)
        
        with open(merge_files[key], 'r') as f:
            merge_data = json.load(f)
        
        # Replace MCDropout section
        if 'MCDropout' not in merge_data.get('methods', {}):
            print(f"Warning: No MCDropout in merge file for {key}")
            continue
        
        if 'methods' not in parent_data:
            print(f"Warning: No methods in parent file {parent_file.name}")
            continue
        
        # Replace MCDropout
        parent_data['methods']['MCDropout'] = merge_data['methods']['MCDropout']
        
        # Update Mean_Aggregation metrics if they exist
        # These need to be recalculated with new MCDropout data
        if 'Mean_Aggregation' in parent_data['methods']:
            # Calculate new mean across all methods (excluding aggregations)
            all_auroc_f = []
            all_augrc = []
            
            for method_name, method_data in parent_data['methods'].items():
                if method_name not in ['Mean_Aggregation', 'Mean_Aggregation_Ensemble']:
                    if 'auroc_f' in method_data:
                        all_auroc_f.append(method_data['auroc_f'])
                    if 'augrc' in method_data:
                        all_augrc.append(method_data['augrc'])
            
            if all_auroc_f:
                parent_data['methods']['Mean_Aggregation']['auroc_f_mean'] = np.mean(all_auroc_f)
            if all_augrc:
                parent_data['methods']['Mean_Aggregation']['augrc_mean'] = np.mean(all_augrc)
        
        # Save updated parent file
        with open(parent_file, 'w') as f:
            json.dump(parent_data, f, indent=2)
        
        updated_count += 1
        print(f"✓ Updated {parent_file.name}")
    
    print(f"\nJSON Summary: Updated {updated_count} files, {not_found_count} had no matching merge file")

def merge_npz_files(parent_dir: Path, merge_dir: Path) -> None:
    """Merge MCDropout results from merge_dir NPZ files into parent_dir NPZ files."""
    
    # Build index of merge files
    merge_files = {}
    for merge_file in merge_dir.glob('all_metrics_*.npz'):
        key = extract_file_key(merge_file.name)
        if key:
            merge_files[key] = merge_file
    
    print(f"\nFound {len(merge_files)} MCDropout NPZ files to merge")
    
    # Process parent files
    updated_count = 0
    not_found_count = 0
    
    for parent_file in parent_dir.glob('all_metrics_*.npz'):
        key = extract_file_key(parent_file.name)
        
        if not key:
            print(f"Warning: Could not parse {parent_file.name}")
            continue
        
        if key not in merge_files:
            not_found_count += 1
            continue
        
        # Load both files
        parent_data = dict(np.load(parent_file, allow_pickle=True))
        merge_data = dict(np.load(merge_files[key], allow_pickle=True))
        
        # Find MCDropout keys in merge data
        mcd_keys = [k for k in merge_data.keys() if k.startswith('MCDropout')]
        
        if not mcd_keys:
            print(f"Warning: No MCDropout keys in merge file for {key}")
            continue
        
        # Replace MCDropout keys
        for mcd_key in mcd_keys:
            parent_data[mcd_key] = merge_data[mcd_key]
        
        # Save updated parent file
        np.savez(parent_file, **parent_data)
        
        updated_count += 1
        print(f"✓ Updated {parent_file.name}")
    
    print(f"\nNPZ Summary: Updated {updated_count} files, {not_found_count} had no matching merge file")

def main():
    ncs_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/new_class_shifts')
    parent_dir = ncs_dir
    merge_dir = ncs_dir / 'to_merge'
    
    print("=" * 80)
    print("Merging MCDropout Results (k=30) for NEW CLASS SHIFTS")
    print("=" * 80)
    
    # Test file key extraction
    test_files = [
        'uq_benchmark_amos2022_resnet18_20260114_161249.json',
        'uq_benchmark_amos2022_vit_b_16_DADO_20260118_140333.json',
        'all_metrics_midog_resnet18_DO_20260203_203524.npz',
        'uq_benchmark_midog_vit_b_16_DA_20260203_213450.json',
    ]
    
    print("\nTesting file key extraction:")
    for tf in test_files:
        key = extract_file_key(tf)
        print(f"  {tf[:60]:<60} -> {key}")
    
    print("\n" + "=" * 80)
    print("Processing JSON files...")
    print("=" * 80)
    merge_json_files(parent_dir, merge_dir)
    
    print("\n" + "=" * 80)
    print("Processing NPZ files...")
    print("=" * 80)
    merge_npz_files(parent_dir, merge_dir)
    
    print("\n" + "=" * 80)
    print("✓ Merge complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
