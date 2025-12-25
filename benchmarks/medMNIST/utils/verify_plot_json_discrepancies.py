"""
Verify discrepancies between plotted AUROC_f values and JSON-saved values.

The issue: 
- JSON saves AUROC_f from AVERAGED per-fold uncertainties
- Plots show AUROC_f from TRUE ensemble uncertainties
These are different values!
"""

import json
import numpy as np
from pathlib import Path
import re

def verify_discrepancies(uq_results_dir, figures_dir):
    """
    Compare JSON values with what should be in plots.
    
    Since OCR is unreliable, we'll:
    1. Load the JSON values
    2. Load the .npz files with raw uncertainties
    3. Recompute what the plots SHOULD show
    4. Report discrepancies
    """
    from sklearn.metrics import roc_auc_score
    
    uq_results_path = Path(uq_results_dir)
    figures_path = Path(figures_dir)
    
    discrepancies = []
    
    # Find all uq_benchmark JSON files
    json_files = sorted(uq_results_path.glob('uq_benchmark_*.json'))
    
    print(f"Found {len(json_files)} JSON files to verify\n")
    print("="*100)
    
    for json_file in json_files:
        # Parse filename to get dataset, model, setup, timestamp
        filename = json_file.stem
        parts = filename.replace('uq_benchmark_', '').rsplit('_', 1)
        
        if len(parts) != 2:
            continue
            
        prefix, timestamp = parts
        
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        flag = data.get('flag', '')
        model_backbone = data.get('model_backbone', '')
        setup = data.get('setup', '')
        
        # Load corresponding .npz file
        npz_filename = f"all_metrics_{prefix}_{timestamp}.npz"
        
        npz_path = uq_results_path / npz_filename
        
        if not npz_path.exists():
            print(f"⚠️  NPZ file not found: {npz_filename}")
            continue
        
        # Load uncertainties
        npz_data = np.load(npz_path)
        
        # Get ensemble predictions from JSON (stored during evaluation)
        test_accuracy = data.get('test_accuracy')
        
        print(f"\n{'─'*100}")
        print(f"📁 {flag} | {model_backbone} | {setup} | {timestamp}")
        print(f"{'─'*100}")
        
        # Check each method
        methods = data.get('methods', {})
        
        for method_name, method_metrics in methods.items():
            json_auroc = method_metrics.get('auroc_f')
            json_auroc_mean = method_metrics.get('auroc_f_mean')
            
            # Check if per-fold data exists
            per_fold_key = f'{method_name}_per_fold'
            ensemble_key = f'{method_name}_ensemble'
            
            has_per_fold = per_fold_key in npz_data
            has_ensemble = ensemble_key in npz_data
            
            if has_per_fold and has_ensemble:
                # This method has separate ensemble uncertainties
                # JSON should store AUROC_f from averaged per-fold
                # Plot should show AUROC_f from ensemble uncertainties
                
                uncertainties_averaged = npz_data[method_name]  # What JSON used
                uncertainties_ensemble = npz_data[ensemble_key]  # What plot should show
                
                # Check if they're different
                diff = np.abs(uncertainties_averaged - uncertainties_ensemble).mean()
                
                if diff > 0.001:  # Significant difference
                    print(f"\n  🔴 {method_name}:")
                    print(f"     JSON auroc_f: {json_auroc:.4f} (from averaged per-fold uncertainties)")
                    print(f"     JSON auroc_f_mean: {json_auroc_mean:.4f} (mean of per-fold AUROC_f values)")
                    print(f"     Mean uncertainty diff: {diff:.6f}")
                    print(f"     ⚠️  Plot likely shows ensemble AUROC_f (from ensemble uncertainties)")
                    print(f"     ⚠️  This does NOT match JSON's top-level auroc_f!")
                    
                    discrepancies.append({
                        'dataset': flag,
                        'model': model_backbone,
                        'setup': setup,
                        'method': method_name,
                        'json_auroc': json_auroc,
                        'json_auroc_mean': json_auroc_mean,
                        'uncertainty_diff': diff
                    })
            elif not has_per_fold:
                # Ensemble-only method (like Ensembling)
                print(f"  ✓ {method_name}: Ensemble-only (no per-fold data)")
            else:
                print(f"  ℹ️  {method_name}: Has per-fold but no separate ensemble key")
    
    print("\n" + "="*100)
    print(f"\n📊 SUMMARY: Found {len(discrepancies)} methods with potential plot/JSON discrepancies")
    
    if discrepancies:
        print("\n🔧 RECOMMENDED FIX:")
        print("   The plotting code should use 'uncertainties' (averaged per-fold) for ensemble curve,")
        print("   NOT 'ensemble_uncertainties' (true ensemble), to match the JSON values.")
        print("\n   OR: JSON should save BOTH values:")
        print("   - 'auroc_f_ensemble': from true ensemble uncertainties")
        print("   - 'auroc_f_averaged': from averaged per-fold uncertainties")
    
    return discrepancies

if __name__ == '__main__':
    uq_results_dir = '/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/id_results'
    figures_dir = '/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/id_results/figures'
    
    discrepancies = verify_discrepancies(uq_results_dir, figures_dir)
