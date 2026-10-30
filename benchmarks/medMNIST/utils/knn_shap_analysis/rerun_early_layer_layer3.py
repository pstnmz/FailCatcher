"""
Recompute *only* the early-layer radiomics KNN block (default: layer3 instead of layer1)
for resnet18 folds, reusing the penultimate-layer scores already cached in
uq_benchmark_results/raw_results/*.npz (key 'penultimate_scores'/'penultimate_labels') instead
of recomputing SHAP + penultimate KNN from scratch.

This relies on the pooled score/label arrays in both runs being built by iterating over the
same classes (np.unique(predicted_classes), deterministic from the cached test-result files)
in the same correct/incorrect order - so the freshly computed early-layer pooled array lines
up index-for-index with the cached penultimate pooled array and can be averaged directly.

Usage:
    python rerun_early_layer_layer3.py
    python rerun_early_layer_layer3.py --early-layer-name layer3 --flags organamnist pathmnist
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from compare_shap_normalization_auroc import (
    WORKSPACE_ROOT, IMAGE_SIZE, COLOR_FLAGS, MODEL_SOURCE_FLAG, POPULATION_SHIFT_SOURCE,
    SPECIAL_TEST_LOADERS, DEFAULT_SHAP_CACHE_DIR, DEFAULT_RESULTS_CACHE_DIR,
    DEFAULT_RADIOMICS_CACHE_DIR, discover_shap_cache_files, pooled_auroc,
    extract_early_layer_activations, compute_radiomics_features, get_or_compute_radiomics,
)
from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation import dataset_utils
from ToolBox.methods.latent import get_layer_from_model

DEFAULT_RAW_RESULTS_DIR = str(WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results')
# Corrupted test_results caches live under the medMNIST results tree, not uq_benchmark_results/cache.
DEFAULT_CORRUPT_RESULTS_CACHE_DIR = str(WORKSPACE_ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'cache')


def compute_early_layer_only(early_features_train, early_labels_train, early_features_test,
                              true_labels, predicted_classes):
    """Same early-layer KNN block as compute_early_layer_and_combined_auroc(), minus the
    penultimate/combined parts (those are loaded from the existing raw-results cache instead)."""
    early_df_train = pd.DataFrame(early_features_train)
    early_df_test = pd.DataFrame(early_features_test)
    correct, incorrect = {}, {}
    classes_present = []
    for c in np.unique(predicted_classes):
        train_mask = (early_labels_train == c)
        pred_mask = (predicted_classes == c)
        if train_mask.sum() == 0 or pred_mask.sum() == 0:
            continue
        true_here = true_labels[pred_mask]
        scaler = StandardScaler()
        train_std = scaler.fit_transform(early_df_train[train_mask].values)
        pca = PCA(n_components=0.8).fit(train_std)
        train_low = pca.transform(train_std)
        knn = NearestNeighbors(n_neighbors=min(5, len(train_low))).fit(train_low)
        test_low = pca.transform(scaler.transform(early_df_test[pred_mask].values))
        dist = knn.kneighbors(test_low)[0].mean(axis=1)
        classes_present.append(c)
        correct[c], incorrect[c] = dist[true_here == c], dist[true_here != c]
    if not classes_present:
        return None
    auroc, scores, labels = pooled_auroc(correct, incorrect, classes_present, return_raw=True)
    return {'n_classes': len(classes_present), 'auroc_early': auroc, 'scores': scores, 'labels': labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shap-cache-dir', default=DEFAULT_SHAP_CACHE_DIR,
                         help='Only used to discover (flag, backbone, setup, fold, bg) groups - '
                              'no SHAP is recomputed.')
    parser.add_argument('--results-cache-dir', default=DEFAULT_RESULTS_CACHE_DIR,
                         help='Where clean *_test_results.npz caches live.')
    parser.add_argument('--corrupt-results-cache-dir', default=DEFAULT_CORRUPT_RESULTS_CACHE_DIR,
                         help='Where corrupted *_test_results.npz caches live (used when '
                              '--corruption-severity is set).')
    parser.add_argument('--raw-results-dir', default=DEFAULT_RAW_RESULTS_DIR,
                         help='Directory holding existing *_fold{N}_bg{N}.npz files with cached '
                              'penultimate_scores/penultimate_labels to reuse.')
    parser.add_argument('--radiomics-cache-dir', default=DEFAULT_RADIOMICS_CACHE_DIR)
    parser.add_argument('--early-layer-name', default='layer3')
    parser.add_argument('--output-suffix', default=None,
                         help='Suffix for new npz/csv outputs (default: "_" + early-layer-name).')
    parser.add_argument('--radiomics-n-jobs', type=int, default=None)
    parser.add_argument('--corruption-severity', type=int, choices=range(1, 6), default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--flags', nargs='+', default=None)
    parser.add_argument('--output', default=None, help='CSV path for the new early/penultimate/combined summary.')
    args = parser.parse_args()
    suffix = args.output_suffix or f'_{args.early_layer_name}'

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (physical GPU {args.gpu})")

    corruption_suffix = f'_corrupt{args.corruption_severity}_test' if args.corruption_severity else ''
    results_cache_dir = Path(args.corrupt_results_cache_dir) if args.corruption_severity else Path(args.results_cache_dir)
    raw_results_dir = Path(args.raw_results_dir)
    if args.output is None:
        args.output = str(WORKSPACE_ROOT / 'uq_benchmark_results' / f'knn_early_layer_combined{suffix}{corruption_suffix}.csv')

    groups = discover_shap_cache_files(args.shap_cache_dir)
    for target_flag, source_flag in POPULATION_SHIFT_SOURCE.items():
        for (src_flag, backbone, setup), fold_entries in list(groups.items()):
            if src_flag == source_flag:
                groups.setdefault((target_flag, backbone, setup), fold_entries)

    groups = {k: v for k, v in groups.items() if k[1] == 'resnet18'}
    if args.flags:
        groups = {k: v for k, v in groups.items() if k[0] in args.flags}
    print(f"Processing {len(groups)} resnet18 (dataset, backbone, setup) group(s)\n")

    dataset_cache = {}
    early_train_loader_cache = {}
    rows = []

    for (flag, backbone, setup), fold_entries in groups.items():
        setup_suffix = f'_{setup}' if setup else ''
        namespace = f'{flag}_{backbone}{setup_suffix}'
        test_cache_path = results_cache_dir / f'{namespace}{corruption_suffix}_test_results.npz'
        if not test_cache_path.exists():
            print(f" [SKIP] {namespace}: missing test result cache at {test_cache_path}")
            continue

        base_flag, test_subset = flag, 'all'
        if flag == 'dermamnist-e-id':
            base_flag, test_subset = 'dermamnist-e', 'id'
        elif flag == 'dermamnist-e-external':
            base_flag, test_subset = 'dermamnist-e', 'external'
        elif flag in MODEL_SOURCE_FLAG:
            base_flag = MODEL_SOURCE_FLAG[flag]

        print(f"=== {namespace} ({len(fold_entries)} fold(s)) ===")
        try:
            models = tr.load_models(base_flag, device=device, size=IMAGE_SIZE, model_backbone=backbone, setup=setup)
            color = base_flag in COLOR_FLAGS
            special_marker = flag if flag in SPECIAL_TEST_LOADERS else None
            dataset_key = (base_flag, test_subset, args.corruption_severity, special_marker)
            if dataset_key not in dataset_cache:
                transform, transform_tta = dataset_utils.get_transforms(color, IMAGE_SIZE)
                if flag in SPECIAL_TEST_LOADERS:
                    [study_dataset, _, _], [_, _, _], _ = tr.load_datasets(
                        base_flag, color, IMAGE_SIZE, transform, batch_size=256
                    )
                    if flag == 'amos2022':
                        test_dataset, test_loader, _, _, _ = dataset_utils.load_amos_dataset(
                            transform, transform_tta, batch_size=256, workspace_root=WORKSPACE_ROOT
                        )
                    else:
                        test_dataset, test_loader, _ = dataset_utils.load_hmu_crc_dataset(
                            transform, transform_tta, batch_size=256, workspace_root=WORKSPACE_ROOT, num_workers=0
                        )
                else:
                    [study_dataset, _, test_dataset], [_, _, test_loader], _ = tr.load_datasets(
                        base_flag, color, IMAGE_SIZE, transform, batch_size=256, test_subset=test_subset
                    )
                    if args.corruption_severity:
                        corruption_flag = 'dermamnist' if 'dermamnist' in base_flag else base_flag
                        test_dataset = dataset_utils.apply_random_corruptions(
                            test_dataset, corruption_flag, args.corruption_severity, cache=True, seed=42
                        )
                        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
                dataset_cache[dataset_key] = (study_dataset, test_dataset, test_loader)
            study_dataset, test_dataset, test_loader = dataset_cache[dataset_key]
            if dataset_key not in early_train_loader_cache:
                early_train_loader_cache[dataset_key] = DataLoader(study_dataset, batch_size=64, shuffle=False)
        except Exception as e:
            print(f" [SKIP] {namespace}: failed to load models/dataset ({e})")
            continue

        test_cache = np.load(test_cache_path, allow_pickle=True)
        train_loader_full = early_train_loader_cache[dataset_key]

        for fold_idx, bg_size, _shap_path in sorted(fold_entries):
            if fold_idx >= len(models):
                print(f" [SKIP] fold {fold_idx}: only {len(models)} models loaded")
                continue
            raw_path = raw_results_dir / f'{namespace}{corruption_suffix}_fold{fold_idx}_bg{bg_size}.npz'
            if not raw_path.exists():
                print(f" [SKIP] fold {fold_idx}: no cached raw results at {raw_path}")
                continue
            cached_raw = np.load(raw_path)
            if 'penultimate_scores' not in cached_raw.files:
                print(f" [SKIP] fold {fold_idx}: cached raw results have no penultimate_scores (early layer wasn't enabled for this run)")
                continue
            pen_scores = cached_raw['penultimate_scores']
            pen_labels = cached_raw['penultimate_labels']

            try:
                model_fold = models[fold_idx]
                model_fold.eval()
                cache_key_train = f'{namespace}_fold{fold_idx}_{args.early_layer_name}_train'
                cache_key_test = f'{namespace}{corruption_suffix}_fold{fold_idx}_{args.early_layer_name}_test'
                early_features_train, early_labels_train = get_or_compute_radiomics(
                    args.radiomics_cache_dir, cache_key_train, model_fold, train_loader_full,
                    device, args.early_layer_name, args.radiomics_n_jobs
                )
                early_features_test, _ = get_or_compute_radiomics(
                    args.radiomics_cache_dir, cache_key_test, model_fold, test_loader,
                    device, args.early_layer_name, args.radiomics_n_jobs
                )
                predicted_classes = test_cache['per_fold_predictions'][fold_idx]
                true_labels = test_cache['y_true']
                early_metrics = compute_early_layer_only(
                    early_features_train, early_labels_train, early_features_test,
                    true_labels, predicted_classes
                )
            except Exception as e:
                print(f" [FAIL] fold {fold_idx} (bg{bg_size}): {e}")
                continue

            if early_metrics is None:
                print(f" [SKIP] fold {fold_idx}: no classes with both correct and incorrect predictions")
                continue
            scores_early, labels_early = early_metrics['scores'], early_metrics['labels']
            if not np.array_equal(labels_early, pen_labels):
                print(f" [FAIL] fold {fold_idx}: pooled labels don't line up with the cached "
                      f"penultimate labels (fresh={len(labels_early)}, cached={len(pen_labels)}) "
                      f"- class sets diverged, skipping combine")
                continue

            combined_scores = (scores_early + pen_scores) / 2
            auroc_early = early_metrics['auroc_early']
            auroc_penultimate = roc_auc_score(pen_labels, pen_scores)
            auroc_combined = roc_auc_score(labels_early, combined_scores)

            print(f"  fold {fold_idx} (bg{bg_size}, {early_metrics['n_classes']} classes): "
                  f"early={auroc_early:.4f}  penultimate={auroc_penultimate:.4f}  combined={auroc_combined:.4f}")

            out_path = raw_results_dir / f'{namespace}{corruption_suffix}_fold{fold_idx}_bg{bg_size}{suffix}.npz'
            np.savez(out_path,
                     early_scores=scores_early, early_labels=labels_early,
                     penultimate_scores=pen_scores, penultimate_labels=pen_labels,
                     combined_scores=combined_scores, combined_labels=labels_early)

            rows.append({'flag': flag, 'backbone': backbone, 'setup': setup or 'standard', 'fold': fold_idx,
                         'n_classes': early_metrics['n_classes'], 'auroc_early': auroc_early,
                         'auroc_penultimate': auroc_penultimate, 'auroc_combined': auroc_combined})
            pd.DataFrame(rows).to_csv(args.output, index=False)
        print()

    if not rows:
        print("No results computed.")
        return

    df = pd.DataFrame(rows)
    print("=" * 100)
    print(f"EARLY-LAYER ({args.early_layer_name}) + COMBINED SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMean across all folds:")
    print(df[['auroc_early', 'auroc_penultimate', 'auroc_combined']].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
