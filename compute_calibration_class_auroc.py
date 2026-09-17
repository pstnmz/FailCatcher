"""Compute mean per-class RAW (uncentered) AUROC-F on the CALIBRATION set itself (not the
test set), one row per (flag, backbone, setup, fold, mass_threshold), for use as the x-axis
in plot_mean_class_auroc_vs_centering_gap.py.

Lighter than the full compute_calibration_centered_auroc.py sweep: only needs the per-fold
model (for the SHAP classifier head) and the cached train/calib activations - no test-loader
construction or test-set feature extraction, since the calibration set is fixed across shifts.

Usage:
    python compute_calibration_class_auroc.py --mass-thresholds 0.8
    python compute_calibration_class_auroc.py --flags breastmnist octmnist --gpu 1
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from compare_shap_normalization_auroc import (  # noqa: E402
    discover_shap_cache_files, load_cached_activations, compute_shap_values_for_fold,
)
from compute_calibration_centered_auroc import (  # noqa: E402
    build_namespace, check_calib_problems, fit_per_class_knn, knn_distance, metric_key,
)

DEFAULT_SHAP_CACHE_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'shap_cache'
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'cache'
DEFAULT_OUTPUT_CSV = WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results_calib_centered_full_sweep' / 'calibration_class_auroc.csv'
DEFAULT_MASS_THRESHOLDS = [0.8]


def mean_per_class_calib_auroc(fitted, calib_df, predicted_calib, true_calib):
    per_class_aurocs = []
    for c, fitted_c in fitted.items():
        mask = predicted_calib == c
        if mask.sum() == 0:
            continue
        dist = knn_distance(fitted_c, calib_df, mask)
        true_here = true_calib[mask]
        labels = (true_here != c).astype(int)
        if len(np.unique(labels)) < 2:
            continue
        per_class_aurocs.append(roc_auc_score(labels, dist))
    return float(np.mean(per_class_aurocs)) if per_class_aurocs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--flags', nargs='+', default=None)
    parser.add_argument('--backbones', nargs='+', choices=['resnet18', 'vit_b_16'], default=None)
    parser.add_argument('--setups', nargs='+', default=None)
    parser.add_argument('--mass-thresholds', nargs='+', default=DEFAULT_MASS_THRESHOLDS)
    parser.add_argument('--shap-cache-dir', default=str(DEFAULT_SHAP_CACHE_DIR))
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument('--output-csv', default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    mass_thresholds = [float(t) if t != 'all' else 'all' for t in args.mass_thresholds]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch  # deferred: must be imported after CUDA_VISIBLE_DEVICES is set
    from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (physical GPU {args.gpu})")

    cache_dir = Path(args.cache_dir)
    output_csv_path = Path(args.output_csv)
    rows = []

    groups = discover_shap_cache_files(args.shap_cache_dir)
    if args.flags:
        groups = {k: v for k, v in groups.items() if k[0] in args.flags}
    if args.backbones:
        groups = {k: v for k, v in groups.items() if k[1] in args.backbones}
    if args.setups:
        requested_setups = {'' if setup == 'standard' else setup for setup in args.setups}
        groups = {k: v for k, v in groups.items() if k[2] in requested_setups}
    print(f"Processing {len(groups)} (dataset, backbone, setup) group(s)\n")

    for (flag, backbone, setup), fold_entries in groups.items():
        namespace = build_namespace(flag, backbone, setup)
        calib_path = cache_dir / f'{namespace}_calib_results.npz'
        if not calib_path.exists():
            print(f"[SKIP GROUP] {namespace}: no calibration cache at {calib_path}")
            continue
        calib_cache = np.load(calib_path, allow_pickle=True)
        true_calib = calib_cache['y_true']
        problems, n_classes = check_calib_problems(true_calib, calib_cache['per_fold_predictions'])
        if problems:
            print(f"[SKIP GROUP] {namespace}: calibration set has unusable predicted classes, skipping")
            continue

        print(f"=== {namespace} ({len(fold_entries)} fold file(s)) ===")
        try:
            models = tr.load_models(flag, device=device, size=224, model_backbone=backbone, setup=setup)
        except Exception as e:
            print(f"[SKIP GROUP] {namespace}: failed to load models ({e})")
            continue

        for fold_idx, bg_size, shap_cache_path in sorted(fold_entries):
            if fold_idx >= len(models):
                print(f" [SKIP] fold {fold_idx}: only {len(models)} models loaded")
                continue
            cached = load_cached_activations(shap_cache_path)
            if cached is None:
                print(f" [SKIP] fold {fold_idx}: shap cache missing train/calib activations")
                continue
            features_train, labels_train, features_calib, labels_calib = cached
            if not np.array_equal(labels_calib, true_calib):
                print(f" [SKIP] fold {fold_idx}: calib label order mismatch between shap_cache and calib_results.npz")
                continue
            predicted_calib = calib_cache['per_fold_predictions'][fold_idx]

            model_fold = models[fold_idx]
            model_fold.eval()
            print(f" fold {fold_idx}: computing SHAP values on train activations...")
            shap_values = compute_shap_values_for_fold(model_fold, 'avgpool', features_train, features_calib)
            calib_df = pd.DataFrame(features_calib, columns=[f"Feature_{i}" for i in range(features_calib.shape[1])])

            for mass_threshold in mass_thresholds:
                fitted = fit_per_class_knn(features_train, labels_train, shap_values, mass_threshold)
                mean_auroc = mean_per_class_calib_auroc(fitted, calib_df, predicted_calib, true_calib)
                if mean_auroc is None:
                    print(f"   mass={mass_threshold}: no class had both successes and failures, skipping")
                    continue
                print(f"   mass={mass_threshold}: mean per-class calibration AUROC-F = {mean_auroc:.4f}")
                rows.append({'flag': flag, 'backbone': backbone, 'setup': setup or 'standard',
                              'fold': fold_idx, 'mass_threshold': mass_threshold,
                              'mean_calib_class_auroc': mean_auroc})

            if rows:
                output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(output_csv_path, index=False)

    print(f"\nSaved calibration per-class AUROC-F results to {output_csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
