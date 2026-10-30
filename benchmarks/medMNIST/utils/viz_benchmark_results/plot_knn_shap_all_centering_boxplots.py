"""Generate raw, calibration, midpoint-centered, and AUROC-offset boxplots together.

The four figures are saved in the same figure directory for one dataset/backbone/setup/shift
and mass threshold. Midpoint-centered files use the normal namespace in raw_results_calib_centered;
AUROC-offset files use the ``_aurocoffset`` namespace in raw_results_calib_centered_full_sweep.

Example:
    python plot_knn_shap_all_centering_boxplots.py \
        --flag tissuemnist --backbone resnet18 --shift ID --mass-threshold 80
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results'
DEFAULT_MIDPOINT_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results_calib_centered'
DEFAULT_AUROCOFFSET_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results_calib_centered_full_sweep'
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / 'results' / 'cache'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'results' / 'figures' / 'knn_shap_class_boxplots'
MASS_THRESHOLD_CHOICES = ['20', '50', '70', '80', '90', 'all']
SUCCESS_COLOR, FAILURE_COLOR = 'tab:blue', 'tab:red'

from plot_knn_shap_class_boxplot import (  # noqa: E402
    build_namespace,
    load_fold_data,
    reconstruct_per_class,
    plot_fold_row,
    _boxplot_with_pct,
)
from plot_calibration_set_boxplot import (  # noqa: E402
    rebuild_calib_per_class,
    DEFAULT_SHAP_CACHE_DIR,
)


def _boxplot_shadow(ax, values, position, width, color):
    """Transparent 'shadow' box for the raw (uncentered) distribution, drawn at the same
    x position as the centered box so their vertical offset visualizes the applied shift."""
    if len(values) == 0:
        return
    ax.boxplot(values, positions=[position], widths=width, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.15, edgecolor=color),
               medianprops=dict(color=color, alpha=0.5),
               whiskerprops=dict(color=color, alpha=0.3), capprops=dict(color=color, alpha=0.3))


def plot_fold_row_with_raw_overlay(ax_classes, ax_pooled, centered_per_class, raw_per_class,
                                    centered_pooled_scores, centered_pooled_labels, auroc, fold_idx):
    classes = sorted(centered_per_class.keys())
    width = 0.35
    total = len(centered_pooled_scores)
    for c in classes:
        success_c, failure_c = centered_per_class[c]
        success_r, failure_r = raw_per_class.get(c, (np.array([]), np.array([])))
        pos_success = c - width / 2 - 0.02
        pos_failure = c + width / 2 + 0.02
        _boxplot_shadow(ax_classes, success_r, pos_success, width, SUCCESS_COLOR)
        _boxplot_shadow(ax_classes, failure_r, pos_failure, width, FAILURE_COLOR)
        _boxplot_with_pct(ax_classes, success_c, pos_success, width, SUCCESS_COLOR, total)
        _boxplot_with_pct(ax_classes, failure_c, pos_failure, width, FAILURE_COLOR, total)
    ax_classes.set_xticks(classes)
    ax_classes.set_xticklabels([str(c) for c in classes])
    ax_classes.set_ylabel(f'Fold {fold_idx}\nKNN distance')
    ax_classes.grid(axis='y', alpha=0.3)
    ax_classes.margins(y=0.15)

    success_c = centered_pooled_scores[centered_pooled_labels == 0]
    failure_c = centered_pooled_scores[centered_pooled_labels == 1]
    raw_success_blocks = [v[0] for v in raw_per_class.values() if len(v[0])]
    raw_failure_blocks = [v[1] for v in raw_per_class.values() if len(v[1])]
    raw_success_all = np.concatenate(raw_success_blocks) if raw_success_blocks else np.array([])
    raw_failure_all = np.concatenate(raw_failure_blocks) if raw_failure_blocks else np.array([])
    _boxplot_shadow(ax_pooled, raw_success_all, 0, 0.5, SUCCESS_COLOR)
    _boxplot_shadow(ax_pooled, raw_failure_all, 1, 0.5, FAILURE_COLOR)
    _boxplot_with_pct(ax_pooled, success_c, 0, 0.5, SUCCESS_COLOR, total)
    _boxplot_with_pct(ax_pooled, failure_c, 1, 0.5, FAILURE_COLOR, total)
    ax_pooled.set_xticks([0, 1])
    ax_pooled.set_xticklabels(['Success', 'Failure'])
    ax_pooled.set_title(f'AUROC-F = {auroc:.4f}')
    ax_pooled.grid(axis='y', alpha=0.3)
    ax_pooled.margins(y=0.15)


def save_test_figure(fold_data, output_path, title, raw_fold_data=None):
    folds = sorted(fold_data)
    n_classes_max = max(len(item[0]) for item in fold_data.values())
    fig, axes = plt.subplots(len(folds), 2,
                             figsize=(max(8, 0.6 * n_classes_max) + 3, 3 * len(folds)),
                             sharex=False, squeeze=False,
                             gridspec_kw={'width_ratios': [n_classes_max, 2]})
    for (ax_classes, ax_pooled), fold_idx in zip(axes, folds):
        if raw_fold_data is not None and fold_idx in raw_fold_data:
            centered_per_class, pooled_scores, pooled_labels, auroc = fold_data[fold_idx]
            raw_per_class = raw_fold_data[fold_idx][0]
            plot_fold_row_with_raw_overlay(ax_classes, ax_pooled, centered_per_class, raw_per_class,
                                           pooled_scores, pooled_labels, auroc, fold_idx)
        else:
            plot_fold_row(ax_classes, ax_pooled, *fold_data[fold_idx], fold_idx)
    axes[-1, 0].set_xlabel('Class')
    axes[0, 0].set_title(title)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SUCCESS_COLOR, alpha=0.6, label='Success'),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAILURE_COLOR, alpha=0.6, label='Failure'),
    ]
    if raw_fold_data is not None:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.15, edgecolor='gray', label='Raw (uncentered, shadow)'))
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.0, 1.0),
               bbox_transform=fig.transFigure, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


def save_calibration_figure(fold_data, output_path, namespace, mass_threshold):
    folds = sorted(fold_data)
    n_classes_max = max(len(item[0]) for item in fold_data.values())
    fig, axes = plt.subplots(len(folds), 2,
                             figsize=(max(8, 0.6 * n_classes_max) + 3, 3 * len(folds)),
                             sharex=False, squeeze=False,
                             gridspec_kw={'width_ratios': [n_classes_max, 2]})
    for (ax_classes, ax_pooled), fold_idx in zip(axes, folds):
        per_class, pooled_scores, pooled_labels, auroc = fold_data[fold_idx]
        classes = sorted(per_class)
        width = 0.35
        total = len(pooled_scores)
        for c in classes:
            success, failure = per_class[c]
            midpoint = (np.median(success) + np.median(failure)) / 2
            for values, position, color in ((success, c - width / 2 - 0.02, SUCCESS_COLOR),
                                             (failure, c + width / 2 + 0.02, FAILURE_COLOR)):
                if len(values) == 0:
                    continue
                bp = ax_classes.boxplot(values, positions=[position], widths=width,
                                        showfliers=False, patch_artist=True,
                                        boxprops=dict(facecolor=color, alpha=0.6),
                                        medianprops=dict(color='black'))
                ax_classes.hlines(np.median(values), position - width / 2, position + width / 2,
                                 color='black', linewidth=1.0, alpha=0.7)
                ax_classes.hlines(midpoint, position - width / 2 - 0.05,
                                  position + width / 2 + 0.05, color='darkslategray',
                                  linewidth=1.0, linestyle='--', alpha=0.9)
                top = bp['caps'][1].get_ydata()[0]
                ax_classes.text(position, top, f'{100 * len(values) / total:.0f}%',
                                ha='center', va='bottom', fontsize=7)
        ax_classes.set_xticks(classes)
        ax_classes.set_xticklabels([str(c) for c in classes])
        ax_classes.set_ylabel(f'Fold {fold_idx}\nKNN distance')
        ax_classes.grid(axis='y', alpha=0.3)
        ax_classes.margins(y=0.15)
        success = pooled_scores[pooled_labels == 0]
        failure = pooled_scores[pooled_labels == 1]
        for values, position, color in ((success, 0, SUCCESS_COLOR), (failure, 1, FAILURE_COLOR)):
            if len(values):
                bp = ax_pooled.boxplot(values, positions=[position], widths=0.5,
                                       showfliers=False, patch_artist=True,
                                       boxprops=dict(facecolor=color, alpha=0.6),
                                       medianprops=dict(color='black'))
                ax_pooled.text(position, bp['caps'][1].get_ydata()[0],
                               f'{100 * len(values) / total:.0f}%', ha='center', va='bottom', fontsize=7)
        ax_pooled.set_xticks([0, 1])
        ax_pooled.set_xticklabels(['Success', 'Failure'])
        ax_pooled.set_title(f'AUROC-F = {auroc:.4f}')
        ax_pooled.grid(axis='y', alpha=0.3)
        ax_pooled.margins(y=0.15)
    axes[-1, 0].set_xlabel('Class')
    axes[0, 0].set_title('Per-class calibration distances')
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=SUCCESS_COLOR, alpha=0.6, label='Success'),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAILURE_COLOR, alpha=0.6, label='Failure'),
    ], loc='upper right', bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f'{namespace} calibration-set KNN-SHAP distances (mass-{mass_threshold}%)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--flag', required=True)
    parser.add_argument('--backbone', required=True, choices=['resnet18', 'vit_b_16'])
    parser.add_argument('--setup', default='')
    parser.add_argument('--shift', required=True, choices=['ID', 'CS'])
    parser.add_argument('--mass-threshold', required=True, choices=MASS_THRESHOLD_CHOICES)
    parser.add_argument('--bg-size', type=int, default=1000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--raw-results-dir', default=str(DEFAULT_RAW_DIR))
    parser.add_argument('--midpoint-results-dir', default=str(DEFAULT_MIDPOINT_DIR))
    parser.add_argument('--aurocoffset-results-dir', default=str(DEFAULT_AUROCOFFSET_DIR))
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument('--shap-cache-dir', default=str(DEFAULT_SHAP_CACHE_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    namespace = build_namespace(args.flag, args.backbone, args.setup)
    suffix = '_corrupt3_test' if args.shift == 'CS' else ''
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = f'{namespace}_{args.shift}_mass{args.mass_threshold}'

    raw = {}
    midpoint = {}
    aurocoffset = {}
    for fold in range(args.n_folds):
        test_path = cache_dir / f'{namespace}{suffix}_test_results.npz'
        if not test_path.exists():
            continue
        test_cache = np.load(test_path, allow_pickle=True)
        true_labels = test_cache['y_true']
        predicted_classes = test_cache['per_fold_predictions'][fold]
        for target, directory, extra in ((raw, Path(args.raw_results_dir), ''),
                                          (midpoint, Path(args.midpoint_results_dir), ''),
                                          (aurocoffset, Path(args.aurocoffset_results_dir), '_aurocoffset')):
            raw_path = directory / f'{namespace}{suffix}_fold{fold}_bg{args.bg_size}{extra}.npz'
            if not raw_path.exists():
                continue
            raw_cache = np.load(raw_path, allow_pickle=True)
            key = 'knn_all' if args.mass_threshold == 'all' else f'knn_mass{args.mass_threshold}'
            if f'{key}_scores' not in raw_cache.files:
                continue
            scores = raw_cache[f'{key}_scores']
            labels = raw_cache[f'{key}_labels']
            per_class = reconstruct_per_class(scores, labels, true_labels, predicted_classes)
            target[fold] = per_class, scores, labels, roc_auc_score(labels, scores)

    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    calibration = {}
    for fold in range(args.n_folds):
        try:
            calibration[fold] = rebuild_calib_per_class(
                args.flag, args.backbone, args.setup, fold, args.mass_threshold,
                Path(args.shap_cache_dir), cache_dir, device)
        except Exception as exc:
            print(f' [SKIP] calibration fold {fold}: {exc}')

    save_test_figure(raw, output_dir / f'{output_base}_raw.png', 'Per-class raw KNN-SHAP distances')
    save_calibration_figure(calibration, output_dir / f'{output_base}_calibration.png', namespace, args.mass_threshold)
    save_test_figure(midpoint, output_dir / f'{output_base}_midpoint_centered.png',
                     'Per-class midpoint-centered KNN-SHAP distances (raw shown as shadow)', raw_fold_data=raw)
    save_test_figure(aurocoffset, output_dir / f'{output_base}_aurocoffset_centered.png',
                     'Per-class AUROC-offset-centered KNN-SHAP distances (raw shown as shadow)', raw_fold_data=raw)


if __name__ == '__main__':
    main()
