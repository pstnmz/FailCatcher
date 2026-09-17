"""For each (flag, backbone, setup, fold, shift), plot a scatter point with:
  x = class-distribution dissimilarity between the calibration set and the test set
      (predicted-class histograms; two selectable metrics, see --metric)
  y = AUROC-F(midpoint-centered) - AUROC-F(auroc-offset-centered), at a fixed mass threshold

Purpose: check whether the two centering methods diverge more when the calibration set's
predicted-class distribution differs from the test set's (i.e. whether auroc-offset earns its
extra flexibility mainly under class-distribution shift, or regardless of it).

Uses per_fold_predictions from the cache (not true labels), since centering is applied
per predicted class and per_fold_predictions is what varies fold-to-fold/ID-vs-CS.

Metrics (--metric):
  max-diff (default): max_c |p_calib[c] - p_test[c]| (Chebyshev/L-inf distance), 0 = identical,
      1 = the single worst class is fully present in one set and absent in the other. Sensitive
      to the single most-shifted class rather than aggregate overlap.
  intersection: 1 - sum_c min(p_calib[c], p_test[c]) (histogram intersection), 0 = identical,
      1 = disjoint supports. Aggregate overlap, insensitive to how mass is spread across classes.

Usage:
    python plot_class_distribution_shift_vs_centering_gap.py
    python plot_class_distribution_shift_vs_centering_gap.py --mass-threshold 0.8 --setups DADO
    python plot_class_distribution_shift_vs_centering_gap.py --metric intersection
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results_calib_centered_full_sweep'
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'cache'
DEFAULT_OUTPUT = WORKSPACE_ROOT / 'uq_benchmark_results' / 'class_distribution_shift_vs_centering_gap.png'
SHIFT_SUFFIX = {'ID': '', 'CS': '_corrupt3_test'}
SHIFT_MARKERS = {'ID': 'o', 'CS': '^'}


def build_namespace(flag, backbone, setup):
    setup_suffix = f'_{setup}' if setup and setup != 'standard' else ''
    return f'{flag}_{backbone}{setup_suffix}'


def class_histogram(predicted, n_classes):
    counts = np.bincount(predicted, minlength=n_classes).astype(float)
    return counts / counts.sum()


def max_class_dissimilarity(p, q):
    return float(np.max(np.abs(p - q)))


def intersection_dissimilarity(p, q):
    return 1.0 - float(np.minimum(p, q).sum())


DISSIMILARITY_METRICS = {
    'max-diff': max_class_dissimilarity,
    'intersection': intersection_dissimilarity,
}


def load_sweep(sweep_dir, mass_threshold):
    midpoint = pd.read_csv(sweep_dir / 'midpoint_auroc.csv')
    aurocoffset = pd.read_csv(sweep_dir / 'aurocoffset_auroc.csv')
    key_cols = ['flag', 'backbone', 'setup', 'fold', 'shift', 'mass_threshold']
    midpoint = midpoint[midpoint['mass_threshold'].astype(str) == str(mass_threshold)]
    aurocoffset = aurocoffset[aurocoffset['mass_threshold'].astype(str) == str(mass_threshold)]
    merged = midpoint.merge(aurocoffset, on=key_cols, suffixes=('_midpoint', '_aurocoffset'))
    merged['auroc_gap'] = merged['auroc_after_midpoint'] - merged['auroc_after_aurocoffset']
    return merged[key_cols + ['auroc_gap']]


def compute_dissimilarity(row, cache_dir, cache_arrays, metric_fn):
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
    n_classes = int(max(calib_pred.max(), test_pred.max()) + 1)
    p_calib = class_histogram(calib_pred, n_classes)
    p_test = class_histogram(test_pred, n_classes)
    return metric_fn(p_calib, p_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep-dir', default=str(DEFAULT_SWEEP_DIR))
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument('--mass-threshold', default='0.8')
    parser.add_argument('--setups', nargs='+', default=None)
    parser.add_argument('--flags', nargs='+', default=None)
    parser.add_argument('--metric', choices=list(DISSIMILARITY_METRICS), default='max-diff')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    merged = load_sweep(Path(args.sweep_dir), args.mass_threshold)
    if args.setups:
        merged = merged[merged['setup'].isin(args.setups)]
    if args.flags:
        merged = merged[merged['flag'].isin(args.flags)]

    metric_fn = DISSIMILARITY_METRICS[args.metric]
    cache_dir = Path(args.cache_dir)
    cache_arrays = {}
    xs, ys, flags, shifts = [], [], [], []
    for _, row in merged.iterrows():
        dissimilarity = compute_dissimilarity(row, cache_dir, cache_arrays, metric_fn)
        if dissimilarity is None:
            continue
        xs.append(dissimilarity)
        ys.append(row['auroc_gap'])
        flags.append(row['flag'])
        shifts.append(row['shift'])
    print(f"Plotted {len(xs)} / {len(merged)} rows (missing cache files skipped)")

    unique_flags = sorted(set(flags))
    cmap = plt.get_cmap('tab10')
    flag_colors = {flag: cmap(i % 10) for i, flag in enumerate(unique_flags)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for shift in ('ID', 'CS'):
        mask = [s == shift for s in shifts]
        if not any(mask):
            continue
        x_shift = [x for x, m in zip(xs, mask) if m]
        y_shift = [y for y, m in zip(ys, mask) if m]
        c_shift = [flag_colors[f] for f, m in zip(flags, mask) if m]
        ax.scatter(x_shift, y_shift, c=c_shift, marker=SHIFT_MARKERS[shift], alpha=0.7,
                   edgecolors='black', linewidths=0.3, label=f'{shift} shift', s=50)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    metric_label = ('max_c |p_calib[c] - p_test[c]|' if args.metric == 'max-diff'
                    else '1 - histogram intersection')
    ax.set_xlabel(f'Calibration vs test predicted-class distribution dissimilarity\n({metric_label}, 0 = identical)')
    ax.set_ylabel('AUROC-F(midpoint) - AUROC-F(auroc-offset)')
    ax.set_title(f'Class-distribution shift vs. centering-method gap (mass={args.mass_threshold}, metric={args.metric})')

    flag_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=flag_colors[f],
                               markersize=8, label=f) for f in unique_flags]
    shift_handles = [plt.Line2D([0], [0], marker=SHIFT_MARKERS[s], color='gray', linestyle='None',
                                markersize=8, label=f'{s} shift') for s in ('ID', 'CS')]
    ax.legend(handles=flag_handles + shift_handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
