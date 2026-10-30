"""Plot octmnist ResNet18 ID boxplots from the stratified AUROC-offset run.

Creates two figures:
  1. test-set AUROC-offset-centered per-class boxes, with raw test boxes as shadows;
  2. calibration-set per-class boxes using the same fold-specific calibration subsamples
     whose predicted-class proportions match the ID test fold.

The calibration distances are recomputed from cached activations using the same fitted
per-class scaler/PCA/KNN spaces as the centering run. Sampling is deterministic (seed 42).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_shap_normalization_auroc import load_cached_activations, compute_shap_values_for_fold
from compute_calibration_centered_auroc import (
    build_namespace,
    check_calib_problems,
    fit_per_class_knn,
    knn_distance,
    compute_auroc_optimal_offsets,
    stratified_calibration_indices,
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'cache'
SHAP_DIR = ROOT / 'uq_benchmark_results' / 'shap_cache'
RAW_DIR = ROOT / 'uq_benchmark_results' / 'raw_results'
STRATIFIED_DIR = ROOT / 'uq_benchmark_results' / 'raw_results_calib_centered_oct_stratified'
OUTPUT_DIR = ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'figures' / 'knn_shap_class_boxplots'
SUCCESS_COLOR, FAILURE_COLOR = 'tab:blue', 'tab:red'


def reconstruct_per_class(scores, labels, true_labels, predicted_classes):
    n_classes = int(max(true_labels.max(), predicted_classes.max()) + 1)
    result = {}
    offset = 0
    for c in range(n_classes):
        mask = predicted_classes == c
        if not mask.any():
            continue
        true_here = true_labels[mask]
        n_success = int((true_here == c).sum())
        n_failure = int((true_here != c).sum())
        block_labels = labels[offset:offset + n_success + n_failure]
        if not (np.all(block_labels[:n_success] == 0) and np.all(block_labels[n_success:] == 1)):
            raise ValueError(f'pooled block mismatch at class {c}')
        result[c] = (scores[offset:offset + n_success],
                     scores[offset + n_success:offset + n_success + n_failure])
        offset += n_success + n_failure
    if offset != len(scores):
        raise ValueError(f'reconstructed {offset} samples, expected {len(scores)}')
    return result


def draw_box(ax, values, position, width, color, alpha=0.6, annotate_total=None):
    if len(values) == 0:
        return
    box = ax.boxplot(values, positions=[position], widths=width, showfliers=False,
                     patch_artist=True,
                     boxprops={'facecolor': color, 'alpha': alpha, 'edgecolor': color},
                     medianprops={'color': 'black'},
                     whiskerprops={'color': color}, capprops={'color': color})
    if annotate_total:
        top = box['caps'][1].get_ydata()[0]
        ax.text(position, top, f'{100 * len(values) / annotate_total:.0f}%',
                ha='center', va='bottom', fontsize=7)


def draw_shadow(ax, values, position, width, color):
    draw_box(ax, values, position, width, color, alpha=0.15)


def draw_class_row(ax, per_class, ylabel, title=None, shadow=None, offsets=None, pooled=False):
    classes = sorted(per_class)
    width = 0.35
    total = sum(len(values) for pair in per_class.values() for values in pair)
    for c in classes:
        success, failure = per_class[c]
        positions = (c - width / 2 - 0.02, c + width / 2 + 0.02)
        if shadow is not None:
            draw_shadow(ax, shadow.get(c, (np.array([]), np.array([])))[0], positions[0], width, SUCCESS_COLOR)
            draw_shadow(ax, shadow.get(c, (np.array([]), np.array([])))[1], positions[1], width, FAILURE_COLOR)
        draw_box(ax, success, positions[0], width, SUCCESS_COLOR, annotate_total=total)
        draw_box(ax, failure, positions[1], width, FAILURE_COLOR, annotate_total=total)
    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    ax.margins(y=0.15)


def load_test_per_class(fold, namespace, cache_dir, result_dir, mass_key, suffix='', extra=''):
    test_cache = np.load(cache_dir / f'{namespace}{suffix}_test_results.npz', allow_pickle=True)
    raw_cache = np.load(result_dir / f'{namespace}{suffix}_fold{fold}_bg1000{extra}.npz', allow_pickle=True)
    scores = raw_cache[f'{mass_key}_scores']
    labels = raw_cache[f'{mass_key}_labels']
    return reconstruct_per_class(scores, labels, test_cache['y_true'],
                                 test_cache['per_fold_predictions'][fold]), scores, labels


def calibration_per_class(flag, backbone, setup, fold, mass_threshold, seed, device, models):
    namespace = build_namespace(flag, backbone, setup)
    calib_cache = np.load(CACHE_DIR / f'{namespace}_calib_results.npz', allow_pickle=True)
    true_calib = calib_cache['y_true']
    predicted_calib = calib_cache['per_fold_predictions'][fold]
    test_cache = np.load(CACHE_DIR / f'{namespace}_test_results.npz', allow_pickle=True)
    predicted_test = test_cache['per_fold_predictions'][fold]
    indices = stratified_calibration_indices(
        predicted_calib, true_calib, predicted_test, test_cache['y_true'],
        random_state=seed + fold)

    shap_files = sorted(SHAP_DIR.glob(f'shap_cache_{namespace}_fold{fold}_bg*.npz'))
    if not shap_files:
        raise FileNotFoundError(f'no SHAP cache for {namespace} fold {fold}')
    cached = load_cached_activations(shap_files[0])
    features_train, labels_train, features_calib, labels_calib = cached
    model = models[fold]
    shap_values = compute_shap_values_for_fold(model, 'avgpool', features_train, features_calib)
    fitted = fit_per_class_knn(features_train, labels_train, shap_values, mass_threshold)
    calib_df = pd.DataFrame(features_calib, columns=[f'Feature_{i}' for i in range(features_calib.shape[1])])
    sampled_df = calib_df.iloc[indices]
    sampled_pred = predicted_calib[indices]
    sampled_true = true_calib[indices]
    per_class = {}
    scores, labels = [], []
    for c, fitted_c in fitted.items():
        mask = sampled_pred == c
        if not mask.any():
            continue
        distances = knn_distance(fitted_c, sampled_df, mask)
        success = distances[sampled_true[mask] == c]
        failure = distances[sampled_true[mask] != c]
        per_class[c] = (success, failure)
        scores.extend([*success, *failure])
        labels.extend([0] * len(success) + [1] * len(failure))
    auroc = roc_auc_score(labels, scores)
    offsets, optimized_auroc = compute_auroc_optimal_offsets(
        fitted, sampled_df, sampled_pred, sampled_true)
    return (per_class, auroc, len(indices),
            np.bincount(sampled_pred, minlength=int(max(predicted_calib.max(), predicted_test.max()) + 1)),
            offsets, optimized_auroc)


def load_models(flag, backbone, setup, device):
    from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
    return tr.load_models(flag, device=device, size=224, model_backbone=backbone, setup=setup)


def save_test_figure(data, output, title, n_classes):
    fig, axes = plt.subplots(len(data), 1, figsize=(max(8, 0.8 * n_classes), 3 * len(data)), squeeze=False)
    for row, fold in enumerate(sorted(data)):
        ax = axes[row, 0]
        centered, raw = data[fold]
        draw_class_row(ax, centered, f'Fold {fold}\nKNN distance', shadow=raw)
    axes[-1, 0].set_xlabel('Predicted class')
    axes[0, 0].set_title(title)
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=SUCCESS_COLOR, alpha=0.6, label='Success'),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAILURE_COLOR, alpha=0.6, label='Failure'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.15, label='Raw shadow'),
    ], loc='upper right')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output}')


def save_calibration_figure(data, output, namespace, mass_threshold):
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 3 * len(data)), squeeze=False)
    for row, fold in enumerate(sorted(data)):
        ax = axes[row, 0]
        per_class, auroc, sample_size, counts, offsets, optimized_auroc = data[fold]
        draw_class_row(ax, per_class, f'Fold {fold}\nKNN distance',
                   title=f'AUROC-F = {auroc:.4f}; optimized = {optimized_auroc:.4f}',
                   offsets=offsets)
    axes[-1, 0].set_xlabel('Predicted class')
    axes[0, 0].set_title(f'{namespace} stratified calibration distances (mass-{mass_threshold}%)')
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=SUCCESS_COLOR, alpha=0.6, label='Success'),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAILURE_COLOR, alpha=0.6, label='Failure'),
    ], loc='upper right')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = parser.parse_args()
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    flag, backbone, setup, shift = 'octmnist', 'resnet18', '', 'ID'
    namespace = build_namespace(flag, backbone, setup)
    mass_key = 'knn_mass80'
    raw_data, calibration_data = {}, {}
    models = load_models(flag, backbone, setup, device)
    for fold in range(5):
        raw_per_class, _, _ = load_test_per_class(fold, namespace, CACHE_DIR, RAW_DIR, mass_key)
        stratified_per_class, _, _ = load_test_per_class(
            fold, namespace, CACHE_DIR, STRATIFIED_DIR, mass_key, suffix='', extra='_aurocoffset')
        raw_data[fold] = (stratified_per_class, raw_per_class)
        calibration_data[fold] = calibration_per_class(
            flag, backbone, setup, fold, 0.8, args.seed, device, models)
        print(f'fold {fold}: calibration sample size = {calibration_data[fold][2]}')
    output_dir = Path(args.output_dir)
    save_test_figure(raw_data, output_dir / 'octmnist_resnet18_ID_mass80_aurocoffset_stratified.png',
                     'OctMNIST ResNet18 ID AUROC-offset boxes (stratified calibration)', 4)
    save_calibration_figure(calibration_data, output_dir / 'octmnist_resnet18_ID_mass80_calibration_stratified.png',
                            namespace, 80)


if __name__ == '__main__':
    main()
