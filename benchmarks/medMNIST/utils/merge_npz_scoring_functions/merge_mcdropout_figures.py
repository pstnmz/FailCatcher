#!/usr/bin/env python3
"""
Script to replace MCDropout figures from to_merge/figures subdirectory
into main corruption_shifts/figures, matching by dataset/backbone/setup.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

def extract_backbone_setup_from_figures(figure_dir: Path) -> Optional[Tuple[str, str]]:
    """Extract (backbone, setup) from figure filenames in a directory."""
    # Look for any MCDropout figure file
    for filename in os.listdir(figure_dir):
        if 'MCDropout' in filename and filename.endswith('.png'):
            # Pattern: {backbone}_{setup}_corrupt_severity3_test_MCDropout_{type}.png
            # E.g.: vit_b_16_DADO_corrupt_severity3_test_MCDropout_distributions.png
            parts = filename.split('_corrupt_severity3_test_')
            if len(parts) >= 2:
                prefix = parts[0]  # e.g., "vit_b_16_DADO"
                
                # Split by underscore and identify backbone vs setup
                prefix_parts = prefix.split('_')
                
                # Setup is the last part (DADO, DO, DA) or empty for vanilla
                if prefix_parts[-1] in ['DADO', 'DO', 'DA']:
                    setup = prefix_parts[-1]
                    backbone = '_'.join(prefix_parts[:-1])
                else:
                    setup = 'vanilla'
                    backbone = prefix
                
                return (backbone, setup)
    
    return None

def find_matching_folder(parent_dataset_dir: Path, target_backbone: str, target_setup: str) -> Optional[Path]:
    """Find folder in parent dataset directory that matches backbone and setup."""
    for timestamp_dir in parent_dataset_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue
        
        result = extract_backbone_setup_from_figures(timestamp_dir)
        if result:
            backbone, setup = result
            if backbone == target_backbone and setup == target_setup:
                return timestamp_dir
    
    return None

def merge_mcdropout_figures(parent_figures_dir: Path, merge_figures_dir: Path) -> None:
    """Replace MCDropout figures from merge_figures_dir into parent_figures_dir."""
    
    updated_count = 0
    skipped_count = 0
    
    # Iterate through datasets in merge directory
    for dataset_dir in merge_figures_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        parent_dataset_dir = parent_figures_dir / dataset_name
        
        if not parent_dataset_dir.exists():
            print(f"⚠ Dataset not found in parent: {dataset_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Iterate through timestamp folders in new figures
        for new_timestamp_dir in dataset_dir.iterdir():
            if not new_timestamp_dir.is_dir():
                continue
            
            # Extract backbone and setup from new figures
            result = extract_backbone_setup_from_figures(new_timestamp_dir)
            if not result:
                print(f"⚠ Could not extract backbone/setup from {new_timestamp_dir.name}")
                skipped_count += 1
                continue
            
            backbone, setup = result
            
            # Find matching folder in parent
            old_folder = find_matching_folder(parent_dataset_dir, backbone, setup)
            
            if not old_folder:
                print(f"⚠ No matching folder for {backbone}_{setup}")
                skipped_count += 1
                continue
            
            # Get MCDropout figure files
            mcd_figures = [f for f in os.listdir(new_timestamp_dir) if 'MCDropout' in f and f.endswith('.png')]
            
            if not mcd_figures:
                print(f"⚠ No MCDropout figures found in {new_timestamp_dir.name}")
                skipped_count += 1
                continue
            
            # Delete old MCDropout figures
            old_mcd_figures = [f for f in os.listdir(old_folder) if 'MCDropout' in f and f.endswith('.png')]
            for old_fig in old_mcd_figures:
                old_fig_path = old_folder / old_fig
                old_fig_path.unlink()
                print(f"  🗑️  Deleted: {old_fig}")
            
            # Copy new MCDropout figures
            for new_fig in mcd_figures:
                src = new_timestamp_dir / new_fig
                dst = old_folder / new_fig
                shutil.copy2(src, dst)
                print(f"  ✓ Copied: {new_fig}")
            
            print(f"  → Updated {backbone}_{setup} in {old_folder.name}")
            updated_count += 1
    
    print(f"\n{'='*80}")
    print(f"✓ Merge complete!")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"{'='*80}")

def main():
    corruption_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/corruption_shifts')
    parent_figures_dir = corruption_dir / 'figures'
    merge_figures_dir = corruption_dir / 'to_merge' / 'figures'
    
    print("=" * 80)
    print("Merging MCDropout Figures")
    print("=" * 80)
    print(f"Source: {merge_figures_dir}")
    print(f"Target: {parent_figures_dir}")
    
    merge_mcdropout_figures(parent_figures_dir, merge_figures_dir)

if __name__ == '__main__':
    main()
