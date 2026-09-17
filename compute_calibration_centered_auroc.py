"""
Calibration-centered KNN-SHAP distances: for each predicted class, shift its RAW KNN
distances (test set) by the calibration-set midpoint between that class's success and
failure distance means, then re-pool into a single AUROC-F.

midpoint[c] = (mean(calib success distances) + mean(calib failure distances)) / 2, and both
the calibration and test distances behind it are projected through the exact same per-class
StandardScaler+PCA(0.9)+KNN subspace (fit once per fold on train data, mass-threshold
SHAP-selected features - SHAP computed on train activations with calibration activations
as the background, never on the calibration set itself). Test distances are therefore
recomputed fresh here (a forward pass over the test set to extract penultimate features,
then projected through that same fitted subspace) rather than reused from the independently
computed uq_benchmark_results/raw_results cache, so calibration and test are guaranteed to
live in the same subspace instead of two separately-fit ones that may have selected
different SHAP features.

Calibration data does not change between ID and CS (only the test set is corrupted for
CS), so one calibration fit per (dataset, backbone, setup, fold, mass-threshold) is reused
for both shifts. Population-shift-only flags (amos2022, hmu-crc, dermamnist-e-external)
have no calibration set of their own and are out of scope here - see
compare_shap_normalization_auroc.py for those.

If a predicted class in the calibration set has zero samples, or has samples but no
successes/no failures, no midpoint can be computed for it - the entire (dataset, backbone,
setup) group (all 5 folds, both shifts, all mass-thresholds) is skipped with a message,
rather than silently producing partial/inconsistent results.

Usage:
    python compute_calibration_centered_auroc.py
    python compute_calibration_centered_auroc.py --flags octmnist organamnist --gpu 1
    python compute_calibration_centered_auroc.py --mass-thresholds 0.7 0.8 0.9
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from compare_shap_normalization_auroc import (  # noqa: E402
    discover_shap_cache_files, load_cached_activations, compute_shap_values_for_fold,
    select_features_by_mass, COLOR_FLAGS, IMAGE_SIZE,
)
from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation import dataset_utils  # noqa: E402
from ToolBox.methods.latent import extract_latent_space_and_compute_shap_importance, get_layer_from_model  # noqa: E402

DEFAULT_SHAP_CACHE_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'shap_cache'
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / 'Benchmarks' / 'medMNIST' / 'results' / 'cache'
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / 'uq_benchmark_results' / 'raw_results_calib_centered'

DEFAULT_MASS_THRESHOLDS = [0.2, 0.5, 0.7, 0.8, 0.9, 'all']
CORRUPTION_SEVERITY = 3
SHIFTS = [('ID', ''), ('CS', '_corrupt3_test')]


def metric_key(mass_threshold):
    return 'knn_all' if mass_threshold == 'all' else f'knn_mass{int(mass_threshold * 100)}'


def build_namespace(flag, backbone, setup):
    setup_suffix = f'_{setup}' if setup else ''
    return f'{flag}_{backbone}{setup_suffix}'


def check_calib_problems(calib_true, per_fold_predictions):
    """Predicted classes in the calibration set with no samples, or with samples but no
    successes/failures, make a success/failure midpoint impossible to compute."""
    n_classes = int(max(calib_true.max(), per_fold_predictions.max()) + 1)
    problems = []
    for fold in range(per_fold_predictions.shape[0]):
        pred = per_fold_predictions[fold]
        for c in range(n_classes):
            mask = pred == c
            if mask.sum() == 0:
                problems.append((fold, c, 'no predicted samples'))
                continue
            true_here = calib_true[mask]
            if (true_here == c).sum() == 0:
                problems.append((fold, c, 'no successes (all failures)'))
            elif (true_here != c).sum() == 0:
                problems.append((fold, c, 'no failures (all successes)'))
    return problems, n_classes


def fit_per_class_knn(features_train, labels_train, shap_values, mass_threshold, knn_k=5):
    """Same per-class StandardScaler+PCA(0.9)+KNN fit as compute_metrics_for_fold() in
    compare_shap_normalization_auroc.py, returning the fitted objects instead of distances."""
    train_df = pd.DataFrame(features_train, columns=[f"Feature_{i}" for i in range(features_train.shape[1])])
    fitted = {}
    for c in np.unique(labels_train):
        train_mask = (labels_train == c)
        if mass_threshold == 'all':
            feats = list(train_df.columns)
        else:
            mass_idx, _ = select_features_by_mass(shap_values, labels_train, c, mass_threshold)
            feats = [f"Feature_{i}" for i in mass_idx]
        scaler = StandardScaler()
        train_std = scaler.fit_transform(train_df[train_mask][feats].values)
        pca = PCA(n_components=0.9)
        train_pca = pca.fit_transform(train_std)
        knn = NearestNeighbors(n_neighbors=min(knn_k, len(train_pca)))
        knn.fit(train_pca)
        fitted[c] = (scaler, pca, knn, feats)
    return fitted


def knn_distance(fitted_c, features_df, mask):
    scaler, pca, knn, feats = fitted_c
    x = scaler.transform(features_df[mask][feats].values)
    x = pca.transform(x)
    return knn.kneighbors(x)[0].mean(axis=1)


def compute_midpoints(fitted, calib_df, predicted_calib, true_calib):
    midpoints = {}
    for c, fitted_c in fitted.items():
        calib_mask = predicted_calib == c
        dist = knn_distance(fitted_c, calib_df, calib_mask)
        true_here = true_calib[calib_mask]
        succ, fail = dist[true_here == c], dist[true_here != c]
        midpoints[c] = (succ.mean() + fail.mean()) / 2
    return midpoints


def compute_test_distances(fitted, test_df, true_test, predicted_test, n_classes):
    """Project the test set through the exact same per-class fitted subspace used for the
    calibration midpoints (never a separately-fit one), pooling in the same ascending
    predicted-class / correct-then-incorrect block order shift_pooled_scores() expects."""
    scores, labels = [], []
    for c in range(n_classes):
        mask = predicted_test == c
        if mask.sum() == 0:
            continue
        if c not in fitted:
            raise ValueError(f"no fitted subspace for predicted class {c} (present in test set)")
        dist = knn_distance(fitted[c], test_df, mask)
        true_here = true_test[mask]
        correct, incorrect = dist[true_here == c], dist[true_here != c]
        scores.append(correct)
        labels.append(np.zeros(len(correct), dtype=int))
        scores.append(incorrect)
        labels.append(np.ones(len(incorrect), dtype=int))
    return np.concatenate(scores), np.concatenate(labels)


def build_test_loaders(flag, batch_size=256):
    """ID and CS (corruption severity 3) test DataLoaders for a dataset flag, mirroring the
    loading in compare_shap_normalization_auroc.py's main loop (standard flags only - see
    module docstring for what's out of scope)."""
    color = flag in COLOR_FLAGS
    transform, _ = dataset_utils.get_transforms(color, IMAGE_SIZE)
    [_, _, test_dataset], [_, _, test_loader], _ = tr.load_datasets(
        flag, color, IMAGE_SIZE, transform, batch_size=batch_size
    )
    corruption_flag = 'dermamnist' if 'dermamnist' in flag else flag
    cs_test_dataset = dataset_utils.apply_random_corruptions(
        test_dataset, corruption_flag, CORRUPTION_SEVERITY, cache=True, seed=42
    )
    cs_test_loader = DataLoader(cs_test_dataset, batch_size=batch_size, shuffle=False)
    return {'': test_loader, '_corrupt3_test': cs_test_loader}


def extract_test_features(model, layer_name, test_loader, device):
    """Fresh penultimate-layer features for the test set (no SHAP/importance needed here -
    only used to be projected through the already-fitted per-class subspace)."""
    layer = get_layer_from_model(model, layer_name)
    features, _, _, _ = extract_latent_space_and_compute_shap_importance(
        model, test_loader, device, layer, importance=False
    )
    return features.numpy()


def shift_pooled_scores(test_scores, true_test, predicted_test, midpoints, n_classes):
    """Subtract midpoints[predicted class] from each per-class contiguous block of the
    pooled test array, using the exact block order/sizes compute_metrics_for_fold() used
    (ascending predicted class, correct-block then incorrect-block) - see
    plot_knn_shap_class_boxplot.reconstruct_per_class() for the same derivation."""
    shifted = np.empty_like(test_scores)
    offset = 0
    for c in range(n_classes):
        mask = predicted_test == c
        n = int(mask.sum())
        if n == 0:
            continue
        if c not in midpoints:
            raise ValueError(f"no calibration midpoint for predicted class {c} (present in test set)")
        shifted[offset:offset + n] = test_scores[offset:offset + n] - midpoints[c]
        offset += n
    if offset != len(test_scores):
        raise ValueError(f"reconstructed {offset} samples but pooled test array has {len(test_scores)}")
    return shifted


def compute_auroc_optimal_offsets(fitted, calib_df, predicted_calib, true_calib, n_passes=5,
                                   coarse_grid=41, fine_grid=41, fine_halfwidth_frac=0.1):
    """Alternative to compute_midpoints(): learn one additive offset b_c per predicted class
    c directly maximizing the pooled calibration AUROC-F, s(x) = d(x) + b_{y_hat(x)} (failures
    = positive class), instead of centering on the success/failure mean midpoint.

    Shifting every sample of a class by the same b_c cannot change the within-class ranking
    (only where each class's block sits relative to the others), so this is block coordinate
    ascent: cycle through classes, and for each one grid-search (coarse then a finer pass
    around the best coarse point) the single b_c that maximizes the pooled AUROC-F with all
    other offsets held fixed, repeating until no class improves. One class's offset is fixed
    at 0 throughout to remove the arbitrary global translation (only relative offsets matter
    for AUROC).
    """
    classes = sorted(fitted.keys())
    ref_class = classes[0]

    distances = np.zeros(len(predicted_calib), dtype=float)
    labels = np.zeros(len(predicted_calib), dtype=int)
    for c in classes:
        mask = predicted_calib == c
        if mask.sum() == 0:
            continue
        distances[mask] = knn_distance(fitted[c], calib_df, mask)
        labels[mask] = (true_calib[mask] != c).astype(int)

    offsets = {c: 0.0 for c in classes}
    per_sample_offset = np.zeros(len(predicted_calib), dtype=float)
    scale = float(np.std(distances)) or 1.0
    lo, hi = -4 * scale, 4 * scale

    def pooled_auroc():
        return roc_auc_score(labels, distances + per_sample_offset)

    best_auroc = pooled_auroc()
    for _ in range(n_passes):
        improved = False
        for c in classes:
            if c == ref_class:
                continue
            mask = predicted_calib == c
            best_b, best_score = offsets[c], best_auroc
            for width_frac, center in ((1.0, None), (fine_halfwidth_frac, best_b)):
                if center is None:
                    grid = np.linspace(lo, hi, coarse_grid)
                else:
                    half_width = (hi - lo) * width_frac
                    grid = np.linspace(center - half_width, center + half_width, fine_grid)
                for b in grid:
                    per_sample_offset[mask] = b
                    score = pooled_auroc()
                    if score > best_score:
                        best_score, best_b = score, b
            per_sample_offset[mask] = best_b
            offsets[c] = best_b
            if best_score > best_auroc + 1e-12:
                best_auroc = best_score
                improved = True
        if not improved:
            break
    return offsets, best_auroc


def apply_offsets(test_scores, true_test, predicted_test, offsets, n_classes):
    """Add offsets[predicted class] to each per-class block of the pooled test array (the
    sign convention of compute_auroc_optimal_offsets()'s s(x) = d(x) + b_c, as opposed to
    shift_pooled_scores()'s subtraction of the midpoint) - same block order as
    shift_pooled_scores()."""
    shifted = np.empty_like(test_scores)
    offset = 0
    for c in range(n_classes):
        mask = predicted_test == c
        n = int(mask.sum())
        if n == 0:
            continue
        if c not in offsets:
            raise ValueError(f"no calibration offset for predicted class {c} (present in test set)")
        shifted[offset:offset + n] = test_scores[offset:offset + n] + offsets[c]
        offset += n
    if offset != len(test_scores):
        raise ValueError(f"reconstructed {offset} samples but pooled test array has {len(test_scores)}")
    return shifted


def stratified_calibration_indices(predicted_calib, true_calib, predicted_test, true_test,
                                   random_state=42):
    """Sample calibration rows toward test proportions for (predicted class, success/failure)."""
    rng = np.random.default_rng(random_state)
    classes = np.arange(int(max(predicted_calib.max(), predicted_test.max()) + 1))
    calib_success = (true_calib == predicted_calib).astype(int)
    test_success = (true_test == predicted_test).astype(int)
    test_cells = np.zeros((len(classes), 2), dtype=float)
    calib_cells = np.zeros((len(classes), 2), dtype=int)
    for c in classes:
        test_cells[c] = np.bincount(test_success[predicted_test == c], minlength=2)
        calib_cells[c] = np.bincount(calib_success[predicted_calib == c], minlength=2)
    test_proportions = test_cells / test_cells.sum()
    feasible_sizes = [calib_cells[c, outcome] / test_proportions[c, outcome]
                      for c in classes for outcome in (0, 1)
                      if test_proportions[c, outcome] > 0]
    n_samples = min(len(predicted_calib), int(np.floor(min(feasible_sizes))))
    if n_samples < 1:
        raise ValueError('no feasible calibration subsample matches test class proportions')
    desired = test_proportions * n_samples
    counts = np.floor(desired).astype(int)
    counts = np.minimum(counts, calib_cells)

    remaining = n_samples - int(counts.sum())
    order = np.dstack(np.unravel_index(np.argsort(-(desired - np.floor(desired)), axis=None), desired.shape))[0]
    while remaining:
        added = False
        for c, outcome in order:
            if counts[c, outcome] < calib_cells[c, outcome]:
                counts[c, outcome] += 1
                remaining -= 1
                added = True
                if not remaining:
                    break
        if not added:
            break

    indices = []
    for c in classes:
        for outcome in (0, 1):
            count = counts[c, outcome]
            class_indices = np.flatnonzero((predicted_calib == c) & (calib_success == outcome))
            if count:
                indices.append(rng.choice(class_indices, size=count, replace=False))
    if not indices:
        raise ValueError('could not sample any calibration rows')
    return np.sort(np.concatenate(indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--flags', nargs='+', default=None,
                         help='Restrict to these dataset flags (default: all discovered ID/CS groups).')
    parser.add_argument('--backbones', nargs='+', choices=['resnet18', 'vit_b_16'], default=None,
                         help='Restrict to these backbones.')
    parser.add_argument('--setups', nargs='+', default=None,
                         help='Restrict to these training setups; use standard for the empty setup.')
    parser.add_argument('--shifts', nargs='+', choices=['ID', 'CS'], default=None,
                         help='Restrict to these test shifts (default: both ID and CS).')
    parser.add_argument('--mass-thresholds', nargs='+', default=DEFAULT_MASS_THRESHOLDS,
                         help="Mass thresholds to compute, floats or 'all' (default: 0.2 0.5 0.7 0.8 0.9 all).")
    parser.add_argument('--centering-method', choices=['midpoint', 'auroc-offset'], default='midpoint',
                         help="'midpoint' (default): shift by the calib success/failure mean midpoint. "
                              "'auroc-offset': per-class additive offset directly maximizing calibration AUROC-F "
                              "(compute_auroc_optimal_offsets). Output files get an '_aurocoffset' suffix so both "
                              "methods can be run into the same --output-dir without overwriting each other.")
    parser.add_argument('--stratify-calibration-to-test', action='store_true',
                        help='For AUROC-offset, subsample calibration rows per shift/fold toward '
                            'the test proportions of (predicted class, success/failure).')
    parser.add_argument('--calibration-sampling-seed', type=int, default=42)
    parser.add_argument('--shap-cache-dir', default=str(DEFAULT_SHAP_CACHE_DIR))
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--results-csv', default=str(DEFAULT_OUTPUT_DIR / 'calibration_centered_auroc.csv'))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    mass_thresholds = [float(t) if t != 'all' else 'all' for t in args.mass_thresholds]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch  # deferred: must be imported after CUDA_VISIBLE_DEVICES is set
    global tr
    from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (physical GPU {args.gpu})")

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    results_csv_path = Path(args.results_csv)
    rows = []

    groups = discover_shap_cache_files(args.shap_cache_dir)
    if args.flags:
        groups = {k: v for k, v in groups.items() if k[0] in args.flags}
    if args.backbones:
        groups = {k: v for k, v in groups.items() if k[1] in args.backbones}
    if args.setups:
        requested_setups = {'' if setup == 'standard' else setup for setup in args.setups}
        groups = {k: v for k, v in groups.items() if k[2] in requested_setups}
    selected_shifts = [(shift, suffix) for shift, suffix in SHIFTS
                       if args.shifts is None or shift in args.shifts]
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
            print(f"[SKIP GROUP] {namespace}: calibration set has predicted classes with no "
                  f"usable success/failure split, skipping all 5 folds:")
            for fold, c, reason in problems:
                print(f"    fold {fold}, predicted class {c}: {reason}")
            continue

        print(f"=== {namespace} ({len(fold_entries)} fold file(s)) ===")
        try:
            models = tr.load_models(flag, device=device, size=224, model_backbone=backbone, setup=setup)
        except Exception as e:
            print(f"[SKIP GROUP] {namespace}: failed to load models ({e})")
            continue
        try:
            test_loaders = build_test_loaders(flag)
        except Exception as e:
            print(f"[SKIP GROUP] {namespace}: failed to build test loaders ({e})")
            continue
        test_caches = {}
        for shift, corruption_suffix in selected_shifts:
            test_path = cache_dir / f'{namespace}{corruption_suffix}_test_results.npz'
            if test_path.exists():
                test_caches[shift] = np.load(test_path, allow_pickle=True)

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

            print(f" fold {fold_idx}: extracting fresh test-set features (ID + CS)...")
            test_dfs = {}
            for shift, corruption_suffix in selected_shifts:
                if shift not in test_caches:
                    continue
                features_test = extract_test_features(model_fold, 'avgpool', test_loaders[corruption_suffix], device)
                test_dfs[shift] = pd.DataFrame(features_test, columns=[f"Feature_{i}" for i in range(features_test.shape[1])])

            out_payload = {shift: {} for shift, _ in selected_shifts}
            for mass_threshold in mass_thresholds:
                key = metric_key(mass_threshold)
                fitted = fit_per_class_knn(features_train, labels_train, shap_values, mass_threshold)
                if args.centering_method == 'midpoint':
                    midpoints = compute_midpoints(fitted, calib_df, predicted_calib, true_calib)

                for shift, corruption_suffix in selected_shifts:
                    if shift not in test_dfs:
                        print(f"   [SKIP] {shift} mass={mass_threshold}: missing test cache")
                        continue
                    test_cache = test_caches[shift]
                    true_test = test_cache['y_true']
                    predicted_test = test_cache['per_fold_predictions'][fold_idx]
                    try:
                        if args.centering_method == 'auroc-offset':
                            calib_indices = np.arange(len(predicted_calib))
                            if args.stratify_calibration_to_test:
                                calib_indices = stratified_calibration_indices(
                                    predicted_calib, true_calib, predicted_test, true_test,
                                    random_state=args.calibration_sampling_seed + fold_idx)
                                print(f"   {shift} mass={mass_threshold}: sampled "
                                      f"{len(calib_indices)}/{len(predicted_calib)} calibration rows "
                                      "to match test class proportions")
                            offsets, calib_auroc = compute_auroc_optimal_offsets(
                                fitted, calib_df.iloc[calib_indices], predicted_calib[calib_indices],
                                true_calib[calib_indices])
                            print(f"   {shift} mass={mass_threshold}: calibration AUROC-F after offset search = "
                                  f"{calib_auroc:.4f}")
                        test_scores, test_labels = compute_test_distances(
                            fitted, test_dfs[shift], true_test, predicted_test, n_classes)
                        if args.centering_method == 'midpoint':
                            shifted = shift_pooled_scores(test_scores, true_test, predicted_test, midpoints, n_classes)
                        else:
                            shifted = apply_offsets(test_scores, true_test, predicted_test, offsets, n_classes)
                    except ValueError as e:
                        print(f"   [SKIP] {shift} mass={mass_threshold}: {e}")
                        continue

                    auroc_before = roc_auc_score(test_labels, test_scores)
                    auroc_after = roc_auc_score(test_labels, shifted)
                    print(f"   {shift} mass={mass_threshold}: AUROC-F {auroc_before:.4f} -> {auroc_after:.4f}")
                    out_payload[shift][f'{key}_scores'] = shifted
                    out_payload[shift][f'{key}_labels'] = test_labels
                    rows.append({'flag': flag, 'backbone': backbone, 'setup': setup or 'standard',
                                  'fold': fold_idx, 'shift': shift, 'mass_threshold': mass_threshold,
                                  'centering_method': args.centering_method,
                                  'auroc_before': auroc_before, 'auroc_after': auroc_after})

            file_suffix = '' if args.centering_method == 'midpoint' else '_aurocoffset'
            for shift, corruption_suffix in selected_shifts:
                if not out_payload[shift]:
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f'{namespace}{corruption_suffix}_fold{fold_idx}_bg{bg_size}{file_suffix}.npz'
                np.savez(out_path, **out_payload[shift])
                print(f"   saved {out_path}")

            if rows:
                results_csv_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(results_csv_path, index=False)

    print(f"\nSaved AUROC-F results to {results_csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
