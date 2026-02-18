#!/usr/bin/env python3
"""
Analyze the performance drop between in-distribution (ID) and population shift (PS) datasets.
Compares:
- Dermamnist ID vs Dermamnist-e-external (PS)
- Organamnist ID vs AMOS2022 (PS)

Computes mean AUROC-F and AUGRC drops across all setups for ResNet18 and ViT-B-16.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Directories
ID_DIR = Path("uq_benchmark_results/full_results/in_distribution")
PS_DIR = Path("uq_benchmark_results/full_results/population_shifts")

# Dataset mappings: (ID dataset name, PS dataset name, display name)
DATASET_PAIRS = [
    ("dermamnist-e-id", "dermamnist-e-external", "dermamnist"),
    ("organamnist", "amos2022", "organamnist->amos2022")
]

# Model architectures
MODELS = ["resnet18", "vit_b_16"]

# Setups (setup suffix in filename)
SETUPS = ["", "DA", "DO", "DADO"]  # "" corresponds to standard (S)


def find_json_file(directory, dataset_id, model, setup):
    """Find the JSON benchmark file for a given configuration."""
    # Build pattern
    if setup == "":
        pattern = f"uq_benchmark_{dataset_id}_{model}_*.json"
    else:
        pattern = f"uq_benchmark_{dataset_id}_{model}_{setup}_*.json"
    
    # Find matching files
    matching_files = list(directory.glob(pattern))
    
    # For standard setup (S, empty string), exclude files with DA, DO, DADO in their names
    # For DA/DO setups, exclude DADO to prevent matching
    if setup == "":
        filtered = []
        for f in matching_files:
            stem = f.stem
            # Should end with: {model}_{timestamp} NOT {model}_{setup}_{timestamp}
            # Check that there's no DA, DO, or DADO between model and timestamp
            # Also exclude corrupt and new_class_shift files
            if ('_DA_' not in stem and '_DO_' not in stem and '_DADO_' not in stem 
                and 'corrupt' not in stem and 'new_class_shift' not in stem):
                filtered.append(f)
        matching_files = filtered
    elif setup in ["DA", "DO"]:
        # For DA and DO, exclude DADO files
        filtered = []
        for f in matching_files:
            stem = f.stem
            if '_DADO' not in stem and 'corrupt' not in stem and 'new_class_shift' not in stem:
                filtered.append(f)
        matching_files = filtered
    else:
        # For DADO, just exclude corrupt and new_class_shift
        filtered = []
        for f in matching_files:
            stem = f.stem
            if 'corrupt' not in stem and 'new_class_shift' not in stem:
                filtered.append(f)
        matching_files = filtered
    
    if len(matching_files) == 0:
        return None
    elif len(matching_files) == 1:
        return matching_files[0]
    else:
        # Multiple files, take the latest by timestamp
        return max(matching_files, key=lambda p: p.stem.split('_')[-1])


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
        # Check if this is a valid UQ method result
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
    """Main analysis function."""
    
    # Store results
    results = {
        "resnet18": {
            "auroc_f_drops": [],
            "augrc_drops": [],
            "details": []
        },
        "vit_b_16": {
            "auroc_f_drops": [],
            "augrc_drops": [],
            "details": []
        }
    }
    
    print("=" * 80)
    print("Analyzing ID vs PS Performance Drops")
    print("=" * 80)
    print()
    
    # Iterate over all configurations
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"Model: {model.upper()}")
        print(f"{'='*80}")
        
        for id_dataset, ps_dataset, display_name in DATASET_PAIRS:
            print(f"\n  Dataset Pair: {display_name} (ID: {id_dataset}, PS: {ps_dataset})")
            
            for setup in SETUPS:
                setup_label = setup if setup != "" else "S"
                
                # Find ID file
                id_file = find_json_file(ID_DIR, id_dataset, model, setup)
                # Find PS file
                ps_file = find_json_file(PS_DIR, ps_dataset, model, setup)
                
                if id_file is None or ps_file is None:
                    print(f"    Setup {setup_label:4s}: MISSING FILES (ID: {id_file is not None}, PS: {ps_file is not None})")
                    continue
                
                # Extract metrics
                id_metrics = extract_best_metrics(id_file)
                ps_metrics = extract_best_metrics(ps_file)
                
                if id_metrics["auroc_f"] is None or ps_metrics["auroc_f"] is None:
                    print(f"    Setup {setup_label:4s}: MISSING METRICS")
                    continue
                
                # Compute drops (ID - PS)
                auroc_f_drop = id_metrics["auroc_f"] - ps_metrics["auroc_f"]
                augrc_drop = id_metrics["augrc"] - ps_metrics["augrc"]
                
                # Store
                results[model]["auroc_f_drops"].append(auroc_f_drop)
                results[model]["augrc_drops"].append(augrc_drop)
                results[model]["details"].append({
                    "dataset_pair": display_name,
                    "id_dataset": id_dataset,
                    "ps_dataset": ps_dataset,
                    "setup": setup_label,
                    "id_auroc_f": id_metrics["auroc_f"],
                    "ps_auroc_f": ps_metrics["auroc_f"],
                    "auroc_f_drop": auroc_f_drop,
                    "id_augrc": id_metrics["augrc"],
                    "ps_augrc": ps_metrics["augrc"],
                    "augrc_drop": augrc_drop,
                    "id_auroc_method": id_metrics["auroc_f_method"],
                    "ps_auroc_method": ps_metrics["auroc_f_method"],
                    "id_augrc_method": id_metrics["augrc_method"],
                    "ps_augrc_method": ps_metrics["augrc_method"]
                })
                
                print(f"    Setup {setup_label:4s}: AUROC-F: {id_metrics['auroc_f']:.4f} -> {ps_metrics['auroc_f']:.4f} (Δ={auroc_f_drop:+.4f}) | "
                      f"AUGRC: {id_metrics['augrc']:.4f} -> {ps_metrics['augrc']:.4f} (Δ={augrc_drop:+.4f})")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Drops Across All Setups and Dataset Pairs")
    print("=" * 80)
    
    for model in MODELS:
        print(f"\n{model.upper()}:")
        
        auroc_drops = results[model]["auroc_f_drops"]
        augrc_drops = results[model]["augrc_drops"]
        
        if len(auroc_drops) > 0:
            mean_auroc_drop = np.mean(auroc_drops)
            std_auroc_drop = np.std(auroc_drops)
            mean_augrc_drop = np.mean(augrc_drops)
            std_augrc_drop = np.std(augrc_drops)
            
            print(f"  AUROC-F Drop: {mean_auroc_drop:+.4f} ± {std_auroc_drop:.4f} (n={len(auroc_drops)})")
            print(f"  AUGRC Drop:   {mean_augrc_drop:+.4f} ± {std_augrc_drop:.4f} (n={len(augrc_drops)})")
            
            # Show min/max drops
            print(f"  AUROC-F range: [{min(auroc_drops):+.4f}, {max(auroc_drops):+.4f}]")
            print(f"  AUGRC range:   [{min(augrc_drops):+.4f}, {max(augrc_drops):+.4f}]")
        else:
            print(f"  No data available")
    
    # Print per-dataset-pair breakdown
    print("\n" + "=" * 80)
    print("PER-DATASET-PAIR BREAKDOWN")
    print("=" * 80)
    
    for model in MODELS:
        print(f"\n{model.upper()}:")
        
        for id_dataset, ps_dataset, display_name in DATASET_PAIRS:
            dataset_auroc_drops = []
            dataset_augrc_drops = []
            
            for detail in results[model]["details"]:
                if detail["dataset_pair"] == display_name:
                    dataset_auroc_drops.append(detail["auroc_f_drop"])
                    dataset_augrc_drops.append(detail["augrc_drop"])
            
            if dataset_auroc_drops:
                mean_auroc = np.mean(dataset_auroc_drops)
                mean_augrc = np.mean(dataset_augrc_drops)
                print(f"  {display_name:30s}: AUROC-F Drop: {mean_auroc:+.4f} | AUGRC Drop: {mean_augrc:+.4f} (n={len(dataset_auroc_drops)})")
    
    # Print detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    
    for model in MODELS:
        print(f"\n{model.upper()}:")
        print(f"{'Dataset Pair':<30} {'Setup':<6} {'ID AUROC-F':<12} {'PS AUROC-F':<12} {'Drop':<10} {'ID Method':<25} {'PS Method':<25}")
        print("-" * 150)
        
        for detail in results[model]["details"]:
            print(f"{detail['dataset_pair']:<30} {detail['setup']:<6} "
                  f"{detail['id_auroc_f']:<12.4f} {detail['ps_auroc_f']:<12.4f} "
                  f"{detail['auroc_f_drop']:+10.4f} "
                  f"{detail['id_auroc_method']:<25} {detail['ps_auroc_method']:<25}")
    
    # Save detailed results
    output_file = "id_ps_analysis_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model in MODELS:
            results_serializable[model] = {
                "mean_auroc_f_drop": float(np.mean(results[model]["auroc_f_drops"])) if results[model]["auroc_f_drops"] else None,
                "std_auroc_f_drop": float(np.std(results[model]["auroc_f_drops"])) if results[model]["auroc_f_drops"] else None,
                "mean_augrc_drop": float(np.mean(results[model]["augrc_drops"])) if results[model]["augrc_drops"] else None,
                "std_augrc_drop": float(np.std(results[model]["augrc_drops"])) if results[model]["augrc_drops"] else None,
                "n_samples": len(results[model]["auroc_f_drops"]),
                "details": results[model]["details"]
            }
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
