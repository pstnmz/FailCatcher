#!/usr/bin/env python3
"""
Compare ViT-B-16 vs ResNet18 performance across all shift types:
- ID (in-distribution)
- CS (corruption shifts)
- PS (population shifts)
- NCS (new class shifts)

For each shift, compute mean difference in AUROC-F and AUGRC between ViT and ResNet18.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Directories
ID_DIR = Path("uq_benchmark_results/in_distribution")
CS_DIR = Path("uq_benchmark_results/corruption_shifts")
PS_DIR = Path("uq_benchmark_results/population_shifts")
NCS_DIR = Path("uq_benchmark_results/new_class_shifts")

# Dataset mappings (ID name -> base name)
DATASETS_ID = {
    "dermamnist-e-id": "dermamnist",
    "breastmnist": "breastmnist",
    "tissuemnist": "tissuemnist",
    "pneumoniamnist": "pneumoniamnist",
    "octmnist": "octmnist",
    "organamnist": "organamnist",
    "bloodmnist": "bloodmnist"
}

# Population shift datasets (external test sets)
DATASETS_PS = {
    "dermamnist-e-external": "dermamnist",
    "amos2022": "organamnist"
}

# New class shift datasets
DATASETS_NCS = {
    "amos2022": "organamnist"
}

# Model architectures
MODELS = ["resnet18", "vit_b_16"]

# Setups (setup suffix in filename)
SETUPS = ["", "DA", "DO", "DADO"]  # "" corresponds to standard (S)


def find_json_files_by_pattern(directory, pattern_parts, exclude_patterns=None):
    """Find JSON files matching pattern parts and excluding certain patterns."""
    all_files = list(directory.glob("uq_benchmark_*.json"))
    
    matching_files = []
    for f in all_files:
        # Check if all pattern parts are in the filename
        if all(part in f.stem for part in pattern_parts):
            # Check exclusions
            if exclude_patterns:
                if any(excl in f.stem for excl in exclude_patterns):
                    continue
            matching_files.append(f)
    
    if len(matching_files) == 0:
        return None
    elif len(matching_files) == 1:
        return matching_files[0]
    else:
        # Take the latest by timestamp
        return max(matching_files, key=lambda p: p.stem.split('_')[-1])


def find_id_file(dataset_id, model, setup):
    """Find in-distribution file."""
    pattern_parts = [dataset_id, model]
    
    # For standard setup, exclude DA/DO/DADO
    if setup == "":
        exclude_patterns = ['_DA_', '_DO_', '_DADO_']
    else:
        # Need to be specific to avoid DA matching DADO, DO matching DADO
        if setup == "DA":
            pattern_parts.append("_DA_")
            exclude_patterns = ['_DADO_']
        elif setup == "DO":
            pattern_parts.append("_DO_")
            exclude_patterns = ['_DADO_', '_DA_']
        elif setup == "DADO":
            pattern_parts.append("_DADO_")
            exclude_patterns = None
        else:
            pattern_parts.append(setup)
            exclude_patterns = None
    
    return find_json_files_by_pattern(ID_DIR, pattern_parts, exclude_patterns)


def find_cs_file(dataset_id, model, setup):
    """Find corruption shift file."""
    pattern_parts = [dataset_id, model, 'corrupt_severity3_test']
    
    # For standard setup, exclude DA/DO/DADO
    if setup == "":
        exclude_patterns = ['_DA_corrupt', '_DO_corrupt', '_DADO_corrupt']
    else:
        # Need to check the order: should be {model}_{setup}_corrupt
        exclude_patterns = []
        if setup == "DA":
            exclude_patterns = ['_DO_corrupt', '_DADO_corrupt']
        elif setup == "DO":
            exclude_patterns = ['_DA_corrupt', '_DADO_corrupt']
        elif setup == "DADO":
            exclude_patterns = ['_DA_corrupt', '_DO_corrupt']
        pattern_parts.append(setup)
    
    return find_json_files_by_pattern(CS_DIR, pattern_parts, exclude_patterns)


def find_ps_file(dataset_id, model, setup):
    """Find population shift file (external test set)."""
    pattern_parts = [dataset_id, model]
    
    # Exclude new_class_shift and corrupt files
    exclude_patterns = ['corrupt', 'new_class_shift']
    
    # For standard setup, exclude DA/DO/DADO
    if setup == "":
        exclude_patterns.extend(['_DA_', '_DO_', '_DADO_'])
    else:
        # Need to be specific to avoid DA matching DADO, DO matching DADO
        if setup == "DA":
            pattern_parts.append("_DA_")
            exclude_patterns.append('_DADO_')
        elif setup == "DO":
            pattern_parts.append("_DO_")
            exclude_patterns.extend(['_DADO_', '_DA_'])
        elif setup == "DADO":
            pattern_parts.append("_DADO_")
        else:
            pattern_parts.append(setup)
    
    return find_json_files_by_pattern(PS_DIR, pattern_parts, exclude_patterns)


def find_ncs_file(dataset_id, model, setup):
    """Find new class shift file.
    
    Note: NCS files don't have 'new_class_shift' in the filename,
    they're just in the new_class_shifts folder with regular naming.
    """
    pattern_parts = [dataset_id, model]
    
    # For standard setup, exclude DA/DO/DADO
    if setup == "":
        exclude_patterns = ['_DA_', '_DO_', '_DADO_']
    else:
        # Need to be specific to avoid DA matching DADO, DO matching DADO
        if setup == "DA":
            pattern_parts.append("_DA_")
            exclude_patterns = ['_DADO_']
        elif setup == "DO":
            pattern_parts.append("_DO_")
            exclude_patterns = ['_DADO_', '_DA_']
        elif setup == "DADO":
            pattern_parts.append("_DADO_")
            exclude_patterns = None
        else:
            pattern_parts.append(setup)
            exclude_patterns = None
    
    return find_json_files_by_pattern(NCS_DIR, pattern_parts, exclude_patterns)


def extract_best_metrics(json_path):
    """Extract best AUROC-F and AUGRC across all UQ methods from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    best_auroc_f = -np.inf
    best_augrc = -np.inf
    best_auroc_method = None
    best_augrc_method = None
    
    methods = data.get("methods", {})
    
    for method_name, method_data in methods.items():
        if isinstance(method_data, dict):
            # ALL methods use 'auroc_f_mean' and 'augrc_mean' fields
            auroc_f = method_data.get("auroc_f_mean", None)
            augrc = method_data.get("augrc_mean", None)
            
            if auroc_f is not None and auroc_f > best_auroc_f:
                best_auroc_f = auroc_f
                best_auroc_method = method_name
            
            if augrc is not None and augrc > best_augrc:
                best_augrc = augrc
                best_augrc_method = method_name
    
    return {
        "auroc_f": best_auroc_f if best_auroc_f != -np.inf else None,
        "augrc": best_augrc if best_augrc != -np.inf else None,
        "auroc_f_method": best_auroc_method,
        "augrc_method": best_augrc_method
    }


def main():
    """Main comparison function."""
    
    # Store results per shift type
    results = {
        "ID": {"auroc_f_diffs": [], "augrc_diffs": [], "details": []},
        "CS": {"auroc_f_diffs": [], "augrc_diffs": [], "details": []},
        "PS": {"auroc_f_diffs": [], "augrc_diffs": [], "details": []},
        "NCS": {"auroc_f_diffs": [], "augrc_diffs": [], "details": []}
    }
    
    print("=" * 80)
    print("Comparing ViT-B-16 vs ResNet18 Across All Shifts")
    print("=" * 80)
    print()
    
    # Process ID shift
    print("Processing ID (In-Distribution)...")
    for dataset_id, dataset_base in DATASETS_ID.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            
            r18_file = find_id_file(dataset_id, "resnet18", setup)
            vit_file = find_id_file(dataset_id, "vit_b_16", setup)
            
            if r18_file is None or vit_file is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING FILES")
                continue
            
            r18_metrics = extract_best_metrics(r18_file)
            vit_metrics = extract_best_metrics(vit_file)
            
            if r18_metrics["auroc_f"] is None or vit_metrics["auroc_f"] is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING METRICS")
                continue
            
            # ViT - ResNet18
            auroc_diff = vit_metrics["auroc_f"] - r18_metrics["auroc_f"]
            augrc_diff = vit_metrics["augrc"] - r18_metrics["augrc"]
            
            results["ID"]["auroc_f_diffs"].append(auroc_diff)
            results["ID"]["augrc_diffs"].append(augrc_diff)
            results["ID"]["details"].append({
                "dataset": dataset_base,
                "setup": setup_label,
                "shift": "ID",
                "r18_auroc_f": r18_metrics["auroc_f"],
                "vit_auroc_f": vit_metrics["auroc_f"],
                "auroc_f_diff": auroc_diff,
                "r18_augrc": r18_metrics["augrc"],
                "vit_augrc": vit_metrics["augrc"],
                "augrc_diff": augrc_diff
            })
            
            print(f"  {dataset_base:15s} {setup_label:4s}: R18={r18_metrics['auroc_f']:.4f} "
                  f"ViT={vit_metrics['auroc_f']:.4f} Diff={auroc_diff:+.4f}")
    
    # Process CS shift
    print("\nProcessing CS (Corruption Shifts)...")
    for dataset_id, dataset_base in DATASETS_ID.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            
            r18_file = find_cs_file(dataset_id, "resnet18", setup)
            vit_file = find_cs_file(dataset_id, "vit_b_16", setup)
            
            if r18_file is None or vit_file is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING FILES")
                continue
            
            r18_metrics = extract_best_metrics(r18_file)
            vit_metrics = extract_best_metrics(vit_file)
            
            if r18_metrics["auroc_f"] is None or vit_metrics["auroc_f"] is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING METRICS")
                continue
            
            auroc_diff = vit_metrics["auroc_f"] - r18_metrics["auroc_f"]
            augrc_diff = vit_metrics["augrc"] - r18_metrics["augrc"]
            
            results["CS"]["auroc_f_diffs"].append(auroc_diff)
            results["CS"]["augrc_diffs"].append(augrc_diff)
            results["CS"]["details"].append({
                "dataset": dataset_base,
                "setup": setup_label,
                "shift": "CS",
                "r18_auroc_f": r18_metrics["auroc_f"],
                "vit_auroc_f": vit_metrics["auroc_f"],
                "auroc_f_diff": auroc_diff,
                "r18_augrc": r18_metrics["augrc"],
                "vit_augrc": vit_metrics["augrc"],
                "augrc_diff": augrc_diff
            })
            
            print(f"  {dataset_base:15s} {setup_label:4s}: R18={r18_metrics['auroc_f']:.4f} "
                  f"ViT={vit_metrics['auroc_f']:.4f} Diff={auroc_diff:+.4f}")
    
    # Process PS shift
    print("\nProcessing PS (Population Shifts)...")
    for dataset_id, dataset_base in DATASETS_PS.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            
            r18_file = find_ps_file(dataset_id, "resnet18", setup)
            vit_file = find_ps_file(dataset_id, "vit_b_16", setup)
            
            if r18_file is None or vit_file is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING FILES")
                continue
            
            r18_metrics = extract_best_metrics(r18_file)
            vit_metrics = extract_best_metrics(vit_file)
            
            if r18_metrics["auroc_f"] is None or vit_metrics["auroc_f"] is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING METRICS")
                continue
            
            auroc_diff = vit_metrics["auroc_f"] - r18_metrics["auroc_f"]
            augrc_diff = vit_metrics["augrc"] - r18_metrics["augrc"]
            
            results["PS"]["auroc_f_diffs"].append(auroc_diff)
            results["PS"]["augrc_diffs"].append(augrc_diff)
            results["PS"]["details"].append({
                "dataset": dataset_base,
                "setup": setup_label,
                "shift": "PS",
                "r18_auroc_f": r18_metrics["auroc_f"],
                "vit_auroc_f": vit_metrics["auroc_f"],
                "auroc_f_diff": auroc_diff,
                "r18_augrc": r18_metrics["augrc"],
                "vit_augrc": vit_metrics["augrc"],
                "augrc_diff": augrc_diff
            })
            
            print(f"  {dataset_base:15s} {setup_label:4s}: R18={r18_metrics['auroc_f']:.4f} "
                  f"ViT={vit_metrics['auroc_f']:.4f} Diff={auroc_diff:+.4f}")
    
    # Process NCS shift
    print("\nProcessing NCS (New Class Shifts)...")
    for dataset_id, dataset_base in DATASETS_NCS.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            
            r18_file = find_ncs_file(dataset_id, "resnet18", setup)
            vit_file = find_ncs_file(dataset_id, "vit_b_16", setup)
            
            if r18_file is None or vit_file is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING FILES")
                continue
            
            r18_metrics = extract_best_metrics(r18_file)
            vit_metrics = extract_best_metrics(vit_file)
            
            if r18_metrics["auroc_f"] is None or vit_metrics["auroc_f"] is None:
                print(f"  {dataset_base:15s} {setup_label:4s}: MISSING METRICS")
                continue
            
            auroc_diff = vit_metrics["auroc_f"] - r18_metrics["auroc_f"]
            augrc_diff = vit_metrics["augrc"] - r18_metrics["augrc"]
            
            results["NCS"]["auroc_f_diffs"].append(auroc_diff)
            results["NCS"]["augrc_diffs"].append(augrc_diff)
            results["NCS"]["details"].append({
                "dataset": dataset_base,
                "setup": setup_label,
                "shift": "NCS",
                "r18_auroc_f": r18_metrics["auroc_f"],
                "vit_auroc_f": vit_metrics["auroc_f"],
                "auroc_f_diff": auroc_diff,
                "r18_augrc": r18_metrics["augrc"],
                "vit_augrc": vit_metrics["augrc"],
                "augrc_diff": augrc_diff
            })
            
            print(f"  {dataset_base:15s} {setup_label:4s}: R18={r18_metrics['auroc_f']:.4f} "
                  f"ViT={vit_metrics['auroc_f']:.4f} Diff={auroc_diff:+.4f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: ViT vs ResNet18 Performance Differences (ViT - ResNet18)")
    print("=" * 80)
    
    for shift_type in ["ID", "CS", "PS", "NCS"]:
        auroc_diffs = results[shift_type]["auroc_f_diffs"]
        augrc_diffs = results[shift_type]["augrc_diffs"]
        
        print(f"\n{shift_type} Shift:")
        if len(auroc_diffs) > 0:
            mean_auroc_diff = np.mean(auroc_diffs)
            std_auroc_diff = np.std(auroc_diffs)
            mean_augrc_diff = np.mean(augrc_diffs)
            std_augrc_diff = np.std(augrc_diffs)
            
            print(f"  AUROC-F Diff (ViT - R18): {mean_auroc_diff:+.4f} ± {std_auroc_diff:.4f} (n={len(auroc_diffs)})")
            print(f"  AUGRC Diff (ViT - R18):   {mean_augrc_diff:+.4f} ± {std_augrc_diff:.4f} (n={len(augrc_diffs)})")
            
            # Positive AUROC-F diff means ViT is better
            # Positive AUGRC diff means ViT has higher AUGRC (better)
            print(f"  Range AUROC-F: [{min(auroc_diffs):+.4f}, {max(auroc_diffs):+.4f}]")
            print(f"  Range AUGRC:   [{min(augrc_diffs):+.4f}, {max(augrc_diffs):+.4f}]")
        else:
            print(f"  No data available")
    
    # Save detailed results
    output_file = "vit_vs_resnet_all_shifts_results.json"
    with open(output_file, 'w') as f:
        results_serializable = {}
        for shift_type in ["ID", "CS", "PS", "NCS"]:
            diffs = results[shift_type]["auroc_f_diffs"]
            results_serializable[shift_type] = {
                "mean_auroc_f_diff": float(np.mean(diffs)) if diffs else None,
                "std_auroc_f_diff": float(np.std(diffs)) if diffs else None,
                "mean_augrc_diff": float(np.mean(results[shift_type]["augrc_diffs"])) if results[shift_type]["augrc_diffs"] else None,
                "std_augrc_diff": float(np.std(results[shift_type]["augrc_diffs"])) if results[shift_type]["augrc_diffs"] else None,
                "n_samples": len(diffs),
                "details": results[shift_type]["details"]
            }
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
