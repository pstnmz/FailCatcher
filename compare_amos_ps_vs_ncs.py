#!/usr/bin/env python3
"""
Compare AMOS2022 Population Shift vs New Class Shift for both ResNet18 and ViT-B-16.
Computes the performance difference (PS - NCS) for AUROC-F and AUGRC.
"""

import json
import os
from pathlib import Path
import numpy as np

# Directories
PS_DIR = Path("uq_benchmark_results/population_shifts")
NCS_DIR = Path("uq_benchmark_results/new_class_shifts")

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


def find_amos_file(directory, model, setup):
    """Find AMOS2022 file for a given model and setup."""
    pattern_parts = ["amos2022", model]
    
    # For standard setup, exclude DA/DO/DADO
    if setup == "":
        exclude_patterns = ['_DA_', '_DO_', '_DADO_']
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
    
    return find_json_files_by_pattern(directory, pattern_parts, exclude_patterns)


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
    
    # Store results per model
    results = {
        "resnet18": {
            "auroc_f_diffs": [],  # PS - NCS
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
    print("Comparing AMOS2022 Population Shift vs New Class Shift")
    print("=" * 80)
    print()
    
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"Model: {model.upper()}")
        print(f"{'='*80}")
        
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            
            # Find PS and NCS files
            ps_file = find_amos_file(PS_DIR, model, setup)
            ncs_file = find_amos_file(NCS_DIR, model, setup)
            
            if ps_file is None or ncs_file is None:
                print(f"  Setup {setup_label:4s}: MISSING FILES (PS: {ps_file is not None}, NCS: {ncs_file is not None})")
                continue
            
            # Extract metrics
            ps_metrics = extract_best_metrics(ps_file)
            ncs_metrics = extract_best_metrics(ncs_file)
            
            if ps_metrics["auroc_f"] is None or ncs_metrics["auroc_f"] is None:
                print(f"  Setup {setup_label:4s}: MISSING METRICS")
                continue
            
            # Compute differences (PS - NCS)
            # Positive means PS performs better, negative means NCS performs better
            auroc_f_diff = ps_metrics["auroc_f"] - ncs_metrics["auroc_f"]
            augrc_diff = ps_metrics["augrc"] - ncs_metrics["augrc"]
            
            # Store
            results[model]["auroc_f_diffs"].append(auroc_f_diff)
            results[model]["augrc_diffs"].append(augrc_diff)
            results[model]["details"].append({
                "model": model,
                "setup": setup_label,
                "ps_auroc_f": ps_metrics["auroc_f"],
                "ncs_auroc_f": ncs_metrics["auroc_f"],
                "auroc_f_diff": auroc_f_diff,
                "ps_augrc": ps_metrics["augrc"],
                "ncs_augrc": ncs_metrics["augrc"],
                "augrc_diff": augrc_diff,
                "ps_auroc_method": ps_metrics["auroc_f_method"],
                "ncs_auroc_method": ncs_metrics["auroc_f_method"],
                "ps_augrc_method": ps_metrics["augrc_method"],
                "ncs_augrc_method": ncs_metrics["augrc_method"]
            })
            
            print(f"  Setup {setup_label:4s}:")
            print(f"    AUROC-F: PS={ps_metrics['auroc_f']:.4f} NCS={ncs_metrics['auroc_f']:.4f} Diff(PS-NCS)={auroc_f_diff:+.4f}")
            print(f"    AUGRC:   PS={ps_metrics['augrc']:.4f} NCS={ncs_metrics['augrc']:.4f} Diff(PS-NCS)={augrc_diff:+.4f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: PS vs NCS Performance Differences (PS - NCS)")
    print("=" * 80)
    print("Positive = PS performs better, Negative = NCS performs better")
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
            
            print(f"  AUROC-F Diff (PS - NCS): {mean_auroc_diff:+.4f} ± {std_auroc_diff:.4f} (n={len(auroc_diffs)})")
            print(f"  AUGRC Diff (PS - NCS):   {mean_augrc_diff:+.4f} ± {std_augrc_diff:.4f} (n={len(augrc_diffs)})")
            
            # Show min/max differences
            print(f"  AUROC-F range: [{min(auroc_diffs):+.4f}, {max(auroc_diffs):+.4f}]")
            print(f"  AUGRC range:   [{min(augrc_diffs):+.4f}, {max(augrc_diffs):+.4f}]")
        else:
            print(f"  No data available")
    
    # Print detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    
    for model in MODELS:
        print(f"\n{model.upper()}:")
        print(f"{'Setup':<6} {'PS AUROC-F':<12} {'NCS AUROC-F':<12} {'Diff(PS-NCS)':<14} {'PS Method':<20} {'NCS Method':<20}")
        print("-" * 120)
        
        for detail in results[model]["details"]:
            print(f"{detail['setup']:<6} "
                  f"{detail['ps_auroc_f']:<12.4f} {detail['ncs_auroc_f']:<12.4f} "
                  f"{detail['auroc_f_diff']:+14.4f} "
                  f"{detail['ps_auroc_method']:<20} {detail['ncs_auroc_method']:<20}")
        
        print()
        print(f"{'Setup':<6} {'PS AUGRC':<12} {'NCS AUGRC':<12} {'Diff(PS-NCS)':<14} {'PS Method':<20} {'NCS Method':<20}")
        print("-" * 120)
        
        for detail in results[model]["details"]:
            print(f"{detail['setup']:<6} "
                  f"{detail['ps_augrc']:<12.4f} {detail['ncs_augrc']:<12.4f} "
                  f"{detail['augrc_diff']:+14.4f} "
                  f"{detail['ps_augrc_method']:<20} {detail['ncs_augrc_method']:<20}")
    
    # Save detailed results
    output_file = "amos_ps_vs_ncs_comparison.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model in MODELS:
            results_serializable[model] = {
                "mean_auroc_f_diff": float(np.mean(results[model]["auroc_f_diffs"])) if results[model]["auroc_f_diffs"] else None,
                "std_auroc_f_diff": float(np.std(results[model]["auroc_f_diffs"])) if results[model]["auroc_f_diffs"] else None,
                "mean_augrc_diff": float(np.mean(results[model]["augrc_diffs"])) if results[model]["augrc_diffs"] else None,
                "std_augrc_diff": float(np.std(results[model]["augrc_diffs"])) if results[model]["augrc_diffs"] else None,
                "n_samples": len(results[model]["auroc_f_diffs"]),
                "details": results[model]["details"]
            }
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
