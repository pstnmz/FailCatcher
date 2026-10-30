"""For each (flag, backbone, setup, fold, shift), plot a scatter point with:
  x = calibration-to-test dissimilarity in predicted-class proportions split by success/failure
      (two selectable metrics, see --metric)
    y = relative AUROC-F gain of KNN-SHAP over KNN_Raw, in percent

Purpose: check whether KNN-SHAP's relative performance over KNN_Raw varies with the
calibration set's predicted-class distribution dissimilarity from the test set.

Uses per_fold_predictions and y_true from the cache to compute predicted-class frequencies
and class-specific success/failure proportions for each fold.

Metrics (--metric):
  max-diff (default): max_c |p_calib[c] - p_test[c]| (Chebyshev/L-inf distance), 0 = identical,
      1 = the single worst class is fully present in one set and absent in the other. Sensitive
      to the single most-shifted class rather than aggregate overlap.
  intersection: 1 - sum_c min(p_calib[c], p_test[c]) (histogram intersection), 0 = identical,
      1 = disjoint supports. Aggregate overlap, insensitive to how mass is spread across classes.

Usage:
    python plot_class_distribution_shift_vs_centering_gap.py
    python plot_class_distribution_shift_vs_centering_gap.py --setups DADO
    python plot_class_distribution_shift_vs_centering_gap.py --metric intersection
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = WORKSPACE_ROOT / 'Benchmarks' / 'medMNIST' / 'results'
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'cache'
DEFAULT_OUTPUT = WORKSPACE_ROOT / 'uq_benchmark_results' / 'class_distribution_shift_vs_knn_shap_gain.png'
SHIFT_SUFFIX = {'ID': '', 'CS': '_corrupt3_test', 'PS': ''}
SHIFT_MARKERS = {'ID': 'o', 'CS': '^', 'PS': 's'}
SHIFT_DIRS = {
    'ID': 'in_distribution',
    'CS': 'corruption_shifts',
    'PS': 'population_shifts',
}
FLAG_PALETTE = (
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79',
)


def build_namespace(flag, backbone, setup):
    setup_suffix = f'_{setup}' if setup and setup != 'standard' else ''
    return f'{flag}_{backbone}{setup_suffix}'


def class_outcome_histogram(predicted, labels, n_classes):
    """Joint proportions for each predicted class split into success and failure."""
    outcome = (predicted != labels).astype(int)
    bins = predicted * 2 + outcome
    counts = np.bincount(bins, minlength=2 * n_classes).astype(float)
    return counts / counts.sum()


def max_class_dissimilarity(p, q):
    return float(np.max(np.abs(p - q)))


def intersection_dissimilarity(p, q):
    return 1.0 - float(np.minimum(p, q).sum())


DISSIMILARITY_METRICS = {
    'max-diff': max_class_dissimilarity,
    'intersection': intersection_dissimilarity,
}


def load_method_fold_scores(json_dir, method_name, shift):
    """Load per-fold AUROC-F, keeping the latest JSON for each dataset configuration."""
    latest = {}
    for json_path in sorted(Path(json_dir).glob('*.json')):
        try:
            with json_path.open() as handle:
                data = json.load(handle)
            flag = data.get('flag')
            backbone = data.get('model_backbone')
            if not flag or not backbone:
                continue
            setup = data.get('setup') or 'standard'
            method = data.get('methods', {}).get(method_name, {})
            fold_metrics = method.get('per_fold_metrics', [])
            timestamp = data.get('timestamp', json_path.name)
            key = (flag, backbone, setup)
            if fold_metrics and (key not in latest or timestamp > latest[key][0]):
                latest[key] = (timestamp, [fold.get('auroc_f') for fold in fold_metrics])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

    scores = {}
    for (flag, backbone, setup), (_, fold_values) in latest.items():
        for fold, value in enumerate(fold_values):
            if value is not None:
                scores[(flag, backbone, setup, fold, shift)] = float(value)
    return scores


def load_comparison(results_dir):
    """Pair KNN-SHAP and KNN_Raw fold metrics, then compute relative gain percent."""
    rows = []
    for shift, directory in SHIFT_DIRS.items():
        raw_dir = Path(results_dir) / 'jsons_results' / directory
        shap_dir = Path(results_dir) / 'knn_shap_results' / 'jsons_results' / directory
        raw_scores = load_method_fold_scores(raw_dir, 'KNN_Raw', shift)
        shap_scores = load_method_fold_scores(shap_dir, 'KNN_SHAP', shift)
        for key in raw_scores.keys() & shap_scores.keys():
            raw = raw_scores[key]
            if raw == 0:
                continue
            rows.append((*key, 100.0 * (shap_scores[key] - raw) / raw))
    return pd.DataFrame(rows, columns=[
        'flag', 'backbone', 'setup', 'fold', 'shift', 'relative_gain_pct',
    ])


def compute_outcome_dissimilarity(row, cache_dir, cache_arrays, metric_fn):
    namespace = build_namespace(row['flag'], row['backbone'], row['setup'])
    calib_key = f'{namespace}_calib_results.npz'
    test_key = f'{namespace}{SHIFT_SUFFIX[row["shift"]]}_test_results.npz'
    for key in (calib_key, test_key):
        if key not in cache_arrays:
            path = cache_dir / key
            if not path.exists():
                return None
            cache_arrays[key] = np.load(path, allow_pickle=True)
    calib_pred = cache_arrays[calib_key]['per_fold_predictions'][row['fold']]
    test_pred = cache_arrays[test_key]['per_fold_predictions'][row['fold']]
    calib_labels = cache_arrays[calib_key]['y_true']
    test_labels = cache_arrays[test_key]['y_true']
    n_classes = int(max(calib_pred.max(), test_pred.max(), calib_labels.max(), test_labels.max()) + 1)
    p_calib_outcomes = class_outcome_histogram(calib_pred, calib_labels, n_classes)
    p_test_outcomes = class_outcome_histogram(test_pred, test_labels, n_classes)
    return metric_fn(p_calib_outcomes, p_test_outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument('--setups', nargs='+', default=None)
    parser.add_argument('--flags', nargs='+', default=None)
    parser.add_argument('--metric', choices=list(DISSIMILARITY_METRICS), default='max-diff')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    merged = load_comparison(Path(args.results_dir))
    if args.setups:
        merged = merged[merged['setup'].isin(args.setups)]
    if args.flags:
        merged = merged[merged['flag'].isin(args.flags)]

    metric_fn = DISSIMILARITY_METRICS[args.metric]
    cache_dir = Path(args.cache_dir)
    cache_arrays = {}
    x_outcomes, ys, flags, shifts = [], [], [], []
    for _, row in merged.iterrows():
        outcome_dissimilarity = compute_outcome_dissimilarity(row, cache_dir, cache_arrays, metric_fn)
        if outcome_dissimilarity is None:
            continue
        x_outcomes.append(outcome_dissimilarity)
        ys.append(row['relative_gain_pct'])
        flags.append(row['flag'])
        shifts.append(row['shift'])
    print(f"Plotted {len(x_outcomes)} / {len(merged)} rows (missing cache files skipped)")

    unique_flags = sorted(set(flags))
    flag_colors = {flag: FLAG_PALETTE[i] for i, flag in enumerate(unique_flags)}

    fig, ax = plt.subplots(figsize=(8, 6))
    metric_label = ('max_c |p_calib[c] - p_test[c]|' if args.metric == 'max-diff'
                    else '1 - histogram intersection')
    for shift in ('ID', 'CS', 'PS'):
        mask = [value == shift for value in shifts]
        if not any(mask):
            continue
        x_shift = [x for x, include in zip(x_outcomes, mask) if include]
        y_shift = [y for y, include in zip(ys, mask) if include]
        c_shift = [flag_colors[flag] for flag, include in zip(flags, mask) if include]
        ax.scatter(x_shift, y_shift, c=c_shift, marker=SHIFT_MARKERS[shift], alpha=0.7,
                   edgecolors='black', linewidths=0.3, label=f'{shift} shift', s=50)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(f'Calibration vs Test success/failure proportions dissimilarity\n({metric_label}, 0 = identical)')
    ax.set_ylabel('KNN-SHAP vs KNN AUROC-F change (%)')
    ax.set_title('Predicted-class success/failure distribution shift vs. KNN-SHAP gain')

    flag_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=flag_colors[f],
                               markersize=8, label=f) for f in unique_flags]
    shift_handles = [plt.Line2D([0], [0], marker=SHIFT_MARKERS[s], color='gray', linestyle='None',
                                markersize=8, label=f'{s} shift') for s in ('ID', 'CS', 'PS')]
    ax.legend(handles=flag_handles + shift_handles, bbox_to_anchor=(1.02, 1),
              loc='upper left', fontsize=8)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
