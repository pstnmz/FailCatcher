"""For each (flag, backbone, setup, fold), plot a scatter point with:
  x = mean per-class CALIBRATION-set AUROC-F for that fold (unweighted average across
      predicted classes of the per-class success-vs-failure AUROC on the calibration set,
      using the calibration KNN distances in the same fitted subspace used for centering)
  y = AUROC-F(midpoint-centered) - AUROC-F(auroc-offset-centered) on the test set, at a
      fixed mass threshold (one point per shift, ID/CS both plotted against the same
      calibration x-value since the calibration set doesn't change with shift)

Purpose: check whether the two centering methods diverge more when the calibration set's
own per-class separability is already poor/uneven, versus already well separated.

Calibration AUROC-F is recomputed here (not cached anywhere) the same way
compute_calibration_centered_auroc.py does for its midpoints/offsets: cached penultimate-layer
train+calibration activations (shap_cache) -> per-class SHAP feature selection -> per-class
StandardScaler+PCA(0.9)+KNN fit on train -> calibration distances -> per-class AUROC-F. No
test-set feature extraction is needed, so this is much lighter than the full sweep.

Usage:
    python plot_mean_calib_class_auroc_vs_centering_gap.py
    python plot_mean_calib_class_auroc_vs_centering_gap.py --mass-threshold 0.8 --gpu 1
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from compare_shap_normalization_auroc import (  # noqa: E402
    discover_shap_cache_files, load_cached_activations, compute_shap_values_for_fold,
)
from compute_calibration_centered_auroc import (  # noqa: E402
    build_namespace, fit_per_class_knn, knn_distance, check_calib_problems,
    DEFAULT_SHAP_CACHE_DIR, DEFAULT_CACHE_DIR,
)

DEFAULT_SWEEP_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results_calib_centered_full_sweep'
DEFAULT_OUTPUT = WORKSPACE_ROOT / 'uq_benchmark_results' / 'mean_calib_class_auroc_vs_centering_gap_heatmap.png'
DEFAULT_CALIB_CACHE = WORKSPACE_ROOT / 'uq_benchmark_results' / 'mean_calib_class_auroc_cache.csv'
SHIFT_MARKERS = {'ID': 'o', 'CS': '^'}
SHIFT_SUFFIX = {'ID': '', 'CS': '_corrupt3_test'}


def load_sweep(sweep_dir, mass_threshold):
    midpoint = pd.read_csv(sweep_dir / 'midpoint_auroc.csv')
    aurocoffset = pd.read_csv(sweep_dir / 'aurocoffset_auroc.csv')
    key_cols = ['flag', 'backbone', 'setup', 'fold', 'shift', 'mass_threshold']
    midpoint = midpoint[midpoint['mass_threshold'].astype(str) == str(mass_threshold)]
    aurocoffset = aurocoffset[aurocoffset['mass_threshold'].astype(str) == str(mass_threshold)]
    merged = midpoint.merge(aurocoffset, on=key_cols, suffixes=('_midpoint', '_aurocoffset'))
    merged['auroc_gap'] = merged['auroc_after_midpoint'] - merged['auroc_after_aurocoffset']
    return merged[key_cols + ['auroc_gap']]


def mean_calib_class_auroc(fitted, calib_df, predicted_calib, true_calib):
    per_class_aurocs = []
    for c, fitted_c in fitted.items():
        mask = predicted_calib == c
        if mask.sum() == 0:
            continue
        dist = knn_distance(fitted_c, calib_df, mask)
        labels_here = (true_calib[mask] != c).astype(int)
        if len(np.unique(labels_here)) < 2:
            continue
        per_class_aurocs.append(roc_auc_score(labels_here, dist))
    return float(np.mean(per_class_aurocs)) if per_class_aurocs else None


def max_class_dissimilarity(calib_predicted, test_predicted):
    n_classes = int(max(calib_predicted.max(), test_predicted.max()) + 1)
    calib_counts = np.bincount(calib_predicted, minlength=n_classes).astype(float)
    test_counts = np.bincount(test_predicted, minlength=n_classes).astype(float)
    calib_distribution = calib_counts / calib_counts.sum()
    test_distribution = test_counts / test_counts.sum()
    return float(np.max(np.abs(calib_distribution - test_distribution)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep-dir', default=str(DEFAULT_SWEEP_DIR))
    parser.add_argument('--shap-cache-dir', default=str(DEFAULT_SHAP_CACHE_DIR))
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument('--mass-threshold', default='0.8')
    parser.add_argument('--setups', nargs='+', default=None)
    parser.add_argument('--flags', nargs='+', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--calib-cache', default=str(DEFAULT_CALIB_CACHE),
                        help='CSV cache for computed calibration mean per-class AUROC-F values.')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    mass_threshold = float(args.mass_threshold) if args.mass_threshold != 'all' else 'all'

    merged = load_sweep(Path(args.sweep_dir), args.mass_threshold)
    if args.setups:
        merged = merged[merged['setup'].isin(args.setups)]
    if args.flags:
        merged = merged[merged['flag'].isin(args.flags)]
    needed_groups = set(merged[['flag', 'backbone', 'setup']].apply(tuple, axis=1))
    needed_groups = {(flag, backbone, '' if setup == 'standard' else setup)
                     for flag, backbone, setup in needed_groups}

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch  # deferred: must be imported after CUDA_VISIBLE_DEVICES is set
    from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (physical GPU {args.gpu})")

    cache_dir = Path(args.cache_dir)
    groups = discover_shap_cache_files(args.shap_cache_dir)
    groups = {k: v for k, v in groups.items() if k in needed_groups}
    print(f"Computing calibration per-class AUROC-F for {len(groups)} group(s)")

    calib_cache_path = Path(args.calib_cache)
    cache_columns = ['flag', 'backbone', 'setup', 'fold', 'mass_threshold', 'mean_calib_class_auroc']
    calib_auroc_by_key = {}
    calib_cache_by_key = {}
    if calib_cache_path.exists():
        cached_rows = pd.read_csv(calib_cache_path)
        for _, cached_row in cached_rows.iterrows():
            cache_key = (cached_row['flag'], cached_row['backbone'], cached_row['setup'],
                         int(cached_row['fold']), str(cached_row['mass_threshold']))
            calib_cache_by_key[cache_key] = float(cached_row['mean_calib_class_auroc'])
        selected_key = (lambda key: key[-1] == str(args.mass_threshold))
        calib_auroc_by_key = {key[:4]: value for key, value in calib_cache_by_key.items()
                              if selected_key(key)}
        print(f"Loaded {len(calib_auroc_by_key)} calibration AUROC-F values from {calib_cache_path}")

    def save_calib_cache():
        rows = []
        for (cached_flag, cached_backbone, cached_setup, cached_fold, cached_mass), value in calib_cache_by_key.items():
            rows.append({'flag': cached_flag, 'backbone': cached_backbone, 'setup': cached_setup,
                         'fold': cached_fold, 'mass_threshold': cached_mass,
                         'mean_calib_class_auroc': value})
        calib_cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=cache_columns).sort_values(
            ['flag', 'backbone', 'setup', 'fold']).to_csv(calib_cache_path, index=False)

    for (flag, backbone, setup), fold_entries in groups.items():
        namespace = build_namespace(flag, backbone, setup)
        calib_path = cache_dir / f'{namespace}_calib_results.npz'
        if not calib_path.exists():
            print(f"[SKIP GROUP] {namespace}: no calibration cache at {calib_path}")
            continue
        calib_cache = np.load(calib_path, allow_pickle=True)
        true_calib = calib_cache['y_true']
        problems, _ = check_calib_problems(true_calib, calib_cache['per_fold_predictions'])
        if problems:
            print(f"[SKIP GROUP] {namespace}: calibration set has unusable predicted classes")
            continue
        try:
            models = tr.load_models(flag, device=device, size=224, model_backbone=backbone, setup=setup)
        except Exception as e:
            print(f"[SKIP GROUP] {namespace}: failed to load models ({e})")
            continue

        for fold_idx, bg_size, shap_cache_path in sorted(fold_entries):
            setup_key = setup or 'standard'
            cache_key = (flag, backbone, setup_key, fold_idx)
            if cache_key in calib_auroc_by_key:
                print(f" {namespace} fold {fold_idx}: loaded cached calibration AUROC-F = "
                      f"{calib_auroc_by_key[cache_key]:.4f}")
                continue
            if fold_idx >= len(models):
                continue
            cached = load_cached_activations(shap_cache_path)
            if cached is None:
                continue
            features_train, labels_train, features_calib, labels_calib = cached
            if not np.array_equal(labels_calib, true_calib):
                continue
            predicted_calib = calib_cache['per_fold_predictions'][fold_idx]
            model_fold = models[fold_idx]
            model_fold.eval()
            shap_values = compute_shap_values_for_fold(model_fold, 'avgpool', features_train, features_calib)
            fitted = fit_per_class_knn(features_train, labels_train, shap_values, mass_threshold)
            calib_df = pd.DataFrame(features_calib, columns=[f"Feature_{i}" for i in range(features_calib.shape[1])])
            mean_auroc = mean_calib_class_auroc(fitted, calib_df, predicted_calib, true_calib)
            if mean_auroc is not None:
                calib_auroc_by_key[cache_key] = mean_auroc
                calib_cache_by_key[(*cache_key, str(args.mass_threshold))] = mean_auroc
                save_calib_cache()
                print(f" {namespace} fold {fold_idx}: mean calib per-class AUROC-F = {mean_auroc:.4f}")

    xs, ys, zs, flags, shifts = [], [], [], [], []
    for _, row in merged.iterrows():
        key = (row['flag'], row['backbone'], row['setup'], row['fold'])
        if key not in calib_auroc_by_key:
            continue
        namespace = build_namespace(row['flag'], row['backbone'], '' if row['setup'] == 'standard' else row['setup'])
        calib_path = cache_dir / f'{namespace}_calib_results.npz'
        test_path = cache_dir / f'{namespace}{SHIFT_SUFFIX[row["shift"]]}_test_results.npz'
        if not calib_path.exists() or not test_path.exists():
            continue
        calib_cache = np.load(calib_path, allow_pickle=True)
        test_cache = np.load(test_path, allow_pickle=True)
        calib_predicted = calib_cache['per_fold_predictions'][row['fold']]
        test_predicted = test_cache['per_fold_predictions'][row['fold']]
        xs.append(calib_auroc_by_key[key])
        ys.append(row['auroc_gap'])
        zs.append(max_class_dissimilarity(calib_predicted, test_predicted))
        flags.append(row['flag'])
        shifts.append(row['shift'])
    print(f"Plotted {len(xs)} / {len(merged)} rows (missing calibration data skipped)")

    unique_flags = sorted(set(flags))
    flag_cmap = plt.get_cmap('tab10')
    flag_colors = {flag: flag_cmap(i % 10) for i, flag in enumerate(unique_flags)}
    heatmap = plt.get_cmap('viridis')
    n_rows = int(np.ceil(len(unique_flags) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.2 * n_rows), sharex=True, sharey=True,
                             squeeze=False)
    axes_flat = axes.ravel()
    z_min, z_max = min(zs), max(zs)
    norm = plt.Normalize(z_min, z_max if z_max > z_min else z_min + 1e-12)
    for ax, flag in zip(axes_flat, unique_flags):
        for shift in ('ID', 'CS'):
            points = [(x, y, z) for x, y, z, point_flag, point_shift
                      in zip(xs, ys, zs, flags, shifts)
                      if point_flag == flag and point_shift == shift]
            if not points:
                continue
            x_values, y_values, z_values = zip(*points)
            ax.scatter(x_values, y_values, c=z_values, cmap=heatmap, norm=norm,
                       marker=SHIFT_MARKERS[shift], alpha=0.85, edgecolors='black',
                       linewidths=0.35, s=52, label=shift)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(flag)
        ax.grid(alpha=0.2)

    for ax in axes_flat[len(unique_flags):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel('Mean per-class CALIBRATION-set AUROC-F')
    for ax in axes[:, 0]:
        ax.set_ylabel('AUROC-F(midpoint) - AUROC-F(auroc-offset)')
    fig.suptitle(f'Calibration separability vs. centering gap (mass={args.mass_threshold})', y=0.995)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=heatmap)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=axes_flat[:len(unique_flags)].tolist(),
                 label='Calibration/test class dissimilarity\nmax_c |p_calib[c] - p_test[c]|',
                 shrink=0.9)
    shift_handles = [plt.Line2D([0], [0], marker=SHIFT_MARKERS[s], color='gray', linestyle='None',
                                markersize=8, label=f'{s} shift') for s in ('ID', 'CS')]
    fig.legend(handles=shift_handles, loc='upper right', bbox_to_anchor=(0.92, 0.99), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
