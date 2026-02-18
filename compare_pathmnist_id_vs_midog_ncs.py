#!/usr/bin/env python3
"""
Compare PathMNIST In-Distribution vs MIDOG New Class Shift for both ResNet18 and ViT-B-16.
Computes the performance difference (ID - NCS) for AUROC-F and AUGRC.
"""

import json
import os
from pathlib import Path
import numpy as np

# Directories
ID_DIR = Path("uq_benchmark_results/full_results/in_distribution")
NCS_DIR = Path("uq_benchmark_results/full_results/new_class_shifts")

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


def find_dataset_file(directory, dataset_name, model, setup):
    """Find dataset file for a given model and setup."""
    pattern_parts = [dataset_name, model]
    
    # For standard setup, exclude DA/DO/DADO
    if setup == "":
        exclude_patterns = ['_DA_', '_DO_', '_DADO_']
        # Also exclude corrupt and new_class_shift in filenames
        if directory == ID_DIR:
            exclude_patterns.extend(['corrupt', 'new_class_shift'])
    else:
        # Need to be specific about which setup to match
        # DA should exclude DADO (since DADO contains DA)
        # DO should exclude DADO (since DADO contains DO)
        if setup == "DA":
            # Match _DA_ but not _DADO_
            pattern_parts.append("_DA_")
            exclude_patterns = ['_DADO_']
        elif setup == "DO":
            # Match _DO_ but not _DADO_
            pattern_parts.append("_DO_")
            exclude_patterns = ['_DADO_', '_DA_']
        elif setup == "DADO":
            pattern_parts.append("_DADO_")
            exclude_patterns = None
        else:
            pattern_parts.append(setup)
            exclude_patterns = None
        
        if directory == ID_DIR and exclude_patterns:
            exclude_patterns.extend(['corrupt', 'new_class_shift'])
        elif directory == ID_DIR:
            exclude_patterns = ['corrupt', 'new_class_shift']
    
    return find_json_files_by_pattern(directory, pattern_parts, exclude_patterns)


def extract_best_metrics(json_path):
    """Extract best AUROC-F and AUGRC across all UQ methods from a JSON file.
    
    Note: Best AUROC-F is highest (maximum), best AUGRC is lowest (minimum).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    best_auroc_f = -np.inf
    best_augrc = np.inf  # Lower AUGRC is better!
    best_auroc_method = None
    best_augrc_method = None
    
    methods = data.get("methods", {})
    
    for method_name, method_data in methods.items():
        if isinstance(method_data, dict):
            # Mean_Aggregation_Ensemble only has 'auroc_f'/'augrc' (no '_mean' suffix)
            # All other methods have 'auroc_f_mean'/'augrc_mean'
            if method_name == "Mean_Aggregation_Ensemble":
                auroc_f = method_data.get("auroc_f", None)
                augrc = method_data.get("augrc", None)
            else:
                auroc_f = method_data.get("auroc_f_mean", None)
                augrc = method_data.get("augrc_mean", None)
            
            if auroc_f is not None and auroc_f > best_auroc_f:
                best_auroc_f = auroc_f
                best_auroc_method = method_name
            
            if augrc is not None and augrc < best_augrc:  # Lower is better!
                best_augrc = augrc
                best_augrc_method = method_name
    
    return {
        "auroc_f": best_auroc_f if best_auroc_f != -np.inf else None,
        "augrc": best_augrc if best_augrc != np.inf else None,
        "auroc_f_method": best_auroc_method,
        "augrc_method": best_augrc_method
    }


def main():
    """Main comparison function."""
    
    # Store results per model
    results = {
        "resnet18": {
            "auroc_f_diffs": [],  # ID - NCS
            "augrc_diffs": [],
            "details": []
        },
        "vit_b_16": {
            "auroc_f_diffs": [],
            "augrc_diffs": [],
            "details": []
        }
    }
    
    print("=" * 80)
    print("Comparing PathMNIST In-Distribution vs MIDOG New Class Shift")
    print("=" * 80)
    print()
    
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"Model: {model.upper()}")
        print(f"{'='*80}")
        
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            
            # Find ID (PathMNIST) and NCS (MIDOG) files
            id_file = find_dataset_file(ID_DIR, "pathmnist", model, setup)
            ncs_file = find_dataset_file(NCS_DIR, "midog", model, setup)
            
            if id_file is None or ncs_file is None:
                print(f"  Setup {setup_label:4s}: MISSING FILES (ID: {id_file is not None}, NCS: {ncs_file is not None})")
                continue
            
            # Extract metrics
            id_metrics = extract_best_metrics(id_file)
            ncs_metrics = extract_best_metrics(ncs_file)
            
            if id_metrics["auroc_f"] is None or ncs_metrics["auroc_f"] is None:
                print(f"  Setup {setup_label:4s}: MISSING METRICS")
                continue
            
            # Compute differences (ID - NCS)
            # Positive means ID performs better, negative means NCS performs better
            auroc_f_diff = id_metrics["auroc_f"] - ncs_metrics["auroc_f"]
            augrc_diff = id_metrics["augrc"] - ncs_metrics["augrc"]
            
            # Store
            results[model]["auroc_f_diffs"].append(auroc_f_diff)
            results[model]["augrc_diffs"].append(augrc_diff)
            results[model]["details"].append({
                "model": model,
                "setup": setup_label,
                "id_auroc_f": id_metrics["auroc_f"],
                "ncs_auroc_f": ncs_metrics["auroc_f"],
                "auroc_f_diff": auroc_f_diff,
                "id_augrc": id_metrics["augrc"],
                "ncs_augrc": ncs_metrics["augrc"],
                "augrc_diff": augrc_diff,
                "id_auroc_method": id_metrics["auroc_f_method"],
                "ncs_auroc_method": ncs_metrics["auroc_f_method"],
                "id_augrc_method": id_metrics["augrc_method"],
                "ncs_augrc_method": ncs_metrics["augrc_method"]
            })
            
            print(f"  Setup {setup_label:4s}:")
            print(f"    AUROC-F: ID={id_metrics['auroc_f']:.4f} NCS={ncs_metrics['auroc_f']:.4f} Diff(ID-NCS)={auroc_f_diff:+.4f}")
            print(f"    AUGRC:   ID={id_metrics['augrc']:.4f} NCS={ncs_metrics['augrc']:.4f} Diff(ID-NCS)={augrc_diff:+.4f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: ID vs NCS Performance Differences (ID - NCS)")
    print("=" * 80)
    print("Positive = ID performs better, Negative = NCS performs better")
    print()
    
    for model in MODELS:
        print(f"\n{model.upper()}:")
        
        auroc_diffs = results[model]["auroc_f_diffs"]
        augrc_diffs = results[model]["augrc_diffs"]
        
        if len(auroc_diffs) > 0:
            mean_auroc_diff = np.mean(auroc_diffs)
            std_auroc_diff = np.std(auroc_diffs)
            mean_augrc_diff = np.mean(augrc_diffs)
            std_augrc_diff = np.std(augrc_diffs)
            
            print(f"  AUROC-F Diff (ID - NCS): {mean_auroc_diff:+.4f} ± {std_auroc_diff:.4f} (n={len(auroc_diffs)})")
            print(f"  AUGRC Diff (ID - NCS):   {mean_augrc_diff:+.4f} ± {std_augrc_diff:.4f} (n={len(augrc_diffs)})")
            
            # Show min/max differences
            print(f"  AUROC-F range: [{min(auroc_diffs):+.4f}, {max(auroc_diffs):+.4f}]")
            print(f"  AUGRC range:   [{min(augrc_diffs):+.4f}, {max(augrc_diffs):+.4f}]")
        else:
            print(f"  No data available")
    
    # Breakdown by backbone (averaging over setups)
    print("\n" + "=" * 80)
    print("BREAKDOWN BY BACKBONE (Averaged Over All Setups)")
    print("=" * 80)
    print()
    
    # Collect all metrics by model for computing means
    for model in MODELS:
        id_aurocs = []
        ncs_aurocs = []
        id_augrcs = []
        ncs_augrcs = []
        
        for detail in results[model]["details"]:
            id_aurocs.append(detail["id_auroc_f"])
            ncs_aurocs.append(detail["ncs_auroc_f"])
            id_augrcs.append(detail["id_augrc"])
            ncs_augrcs.append(detail["ncs_augrc"])
        
        if id_aurocs and ncs_aurocs:
            mean_id_auroc = np.mean(id_aurocs)
            mean_ncs_auroc = np.mean(ncs_aurocs)
            mean_id_augrc = np.mean(id_augrcs)
            mean_ncs_augrc = np.mean(ncs_augrcs)
            
            mean_auroc_diff = mean_id_auroc - mean_ncs_auroc
            mean_augrc_diff = mean_id_augrc - mean_ncs_augrc
            
            print(f"{model.upper()}:")
            print(f"  ID Performance:  AUROC-F={mean_id_auroc:.4f}, AUGRC={mean_id_augrc:.4f}")
            print(f"  NCS Performance: AUROC-F={mean_ncs_auroc:.4f}, AUGRC={mean_ncs_augrc:.4f}")
            print(f"  Difference:      AUROC-F={mean_auroc_diff:+.4f}, AUGRC={mean_augrc_diff:+.4f}")
            print(f"  (n={len(id_aurocs)} setups)")
            print()
    
    # Print detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    
    for model in MODELS:
        print(f"\n{model.upper()}:")
        print(f"{'Setup':<6} {'ID AUROC-F':<12} {'NCS AUROC-F':<12} {'Diff(ID-NCS)':<14} {'ID Method':<25} {'NCS Method':<25}")
        print("-" * 130)
        
        for detail in results[model]["details"]:
            print(f"{detail['setup']:<6} "
                  f"{detail['id_auroc_f']:<12.4f} {detail['ncs_auroc_f']:<12.4f} "
                  f"{detail['auroc_f_diff']:+14.4f} "
                  f"{detail['id_auroc_method']:<25} {detail['ncs_auroc_method']:<25}")
        
        print()
        print(f"{'Setup':<6} {'ID AUGRC':<12} {'NCS AUGRC':<12} {'Diff(ID-NCS)':<14} {'ID Method':<25} {'NCS Method':<25}")
        print("-" * 130)
        
        for detail in results[model]["details"]:
            print(f"{detail['setup']:<6} "
                  f"{detail['id_augrc']:<12.4f} {detail['ncs_augrc']:<12.4f} "
                  f"{detail['augrc_diff']:+14.4f} "
                  f"{detail['id_augrc_method']:<25} {detail['ncs_augrc_method']:<25}")
    
    # Save detailed results
    output_file = "pathmnist_id_vs_midog_ncs_comparison.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model in MODELS:
            # Compute backbone-averaged metrics
            if results[model]["details"]:
                id_aurocs = [d["id_auroc_f"] for d in results[model]["details"]]
                ncs_aurocs = [d["ncs_auroc_f"] for d in results[model]["details"]]
                id_augrcs = [d["id_augrc"] for d in results[model]["details"]]
                ncs_augrcs = [d["ncs_augrc"] for d in results[model]["details"]]
                
                backbone_avg_id_auroc = float(np.mean(id_aurocs))
                backbone_avg_ncs_auroc = float(np.mean(ncs_aurocs))
                backbone_avg_id_augrc = float(np.mean(id_augrcs))
                backbone_avg_ncs_augrc = float(np.mean(ncs_augrcs))
            else:
                backbone_avg_id_auroc = None
                backbone_avg_ncs_auroc = None
                backbone_avg_id_augrc = None
                backbone_avg_ncs_augrc = None
            
            results_serializable[model] = {
                "mean_auroc_f_diff": float(np.mean(results[model]["auroc_f_diffs"])) if results[model]["auroc_f_diffs"] else None,
                "std_auroc_f_diff": float(np.std(results[model]["auroc_f_diffs"])) if results[model]["auroc_f_diffs"] else None,
                "mean_augrc_diff": float(np.mean(results[model]["augrc_diffs"])) if results[model]["augrc_diffs"] else None,
                "std_augrc_diff": float(np.std(results[model]["augrc_diffs"])) if results[model]["augrc_diffs"] else None,
                "n_samples": len(results[model]["auroc_f_diffs"]),
                "backbone_averaged": {
                    "id_auroc_f": backbone_avg_id_auroc,
                    "ncs_auroc_f": backbone_avg_ncs_auroc,
                    "id_augrc": backbone_avg_id_augrc,
                    "ncs_augrc": backbone_avg_ncs_augrc,
                    "auroc_f_diff": backbone_avg_id_auroc - backbone_avg_ncs_auroc if backbone_avg_id_auroc is not None else None,
                    "augrc_diff": backbone_avg_id_augrc - backbone_avg_ncs_augrc if backbone_avg_id_augrc is not None else None
                },
                "details": results[model]["details"]
            }
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
