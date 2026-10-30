"""Compute ID SHAP-selected feature counts per class, fold, setup, and backbone."""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import torch

from compare_shap_normalization_auroc import (
    IMAGE_SIZE,
    SHAP_CACHE_RE,
    compute_shap_values_for_fold,
    get_layer_from_model,
    load_cached_activations,
    parse_namespace,
    select_features_by_mass,
    tr,
)

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
DEFAULT_CACHE_DIR = RESULTS_DIR / 'knn_shap_results' / 'shap_cache'
DEFAULT_OUTPUT = RESULTS_DIR / 'knn_shap_results' / 'shap_selected_feature_counts_by_class_id.csv'
ID_FLAGS = {
    'tissuemnist', 'breastmnist', 'pneumoniamnist', 'octmnist',
    'pathmnist', 'organamnist', 'bloodmnist', 'dermamnist-e-id',
}
MASS_THRESHOLDS = (20, 50, 70, 80, 90)


def discover_id_caches(cache_dir):
    groups = defaultdict(list)
    for path in sorted(Path(cache_dir).glob('shap_cache_*.npz')):
        match = SHAP_CACHE_RE.match(path.name)
        if match is None or '_corrupt' in path.name:
            continue
        parsed = parse_namespace(match.group('namespace'))
        if parsed is None or parsed[0] not in ID_FLAGS:
            continue
        flag, backbone, setup = parsed
        groups[(flag, backbone, setup or 'standard')].append(
            (int(match.group('fold')), int(match.group('bg')), path)
        )
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--flags', nargs='+', choices=sorted(ID_FLAGS), default=None)
    parser.add_argument('--backbones', nargs='+', choices=['resnet18', 'vit_b_16'], default=None)
    parser.add_argument('--setups', nargs='+', choices=['standard', 'DA', 'DO', 'DADO'], default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda was requested, but CUDA is unavailable')
    device = torch.device(
        'cuda:0' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu'
    )
    groups = discover_id_caches(args.cache_dir)
    if args.flags:
        groups = {key: value for key, value in groups.items() if key[0] in args.flags}
    if args.backbones:
        groups = {key: value for key, value in groups.items() if key[1] in args.backbones}
    if args.setups:
        groups = {key: value for key, value in groups.items() if key[2] in args.setups}

    print(f'Found {sum(map(len, groups.values()))} ID fold caches in {len(groups)} groups; device={device}')
    rows = pd.read_csv(args.output).to_dict('records') if args.output.exists() else []
    print(f'Resuming with {len(rows)} saved rows')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for (flag, backbone, setup), fold_entries in sorted(groups.items()):
        model_flag = 'dermamnist-e' if flag == 'dermamnist-e-id' else flag
        try:
            models = tr.load_models(
                model_flag, device=device, size=IMAGE_SIZE,
                model_backbone=backbone, setup='' if setup == 'standard' else setup,
            )
        except Exception as exc:
            print(f'[SKIP] {flag}/{backbone}/{setup}: could not load models: {exc}')
            continue

        for fold, bg_size, cache_path in sorted(fold_entries):
            if fold >= len(models):
                print(f'[SKIP] {cache_path.name}: fold index exceeds loaded model count')
                continue
            cached = load_cached_activations(cache_path)
            if cached is None:
                print(f'[SKIP] {cache_path.name}: train/calibration activations missing')
                continue
            features_train, labels_train, features_calib, _ = cached
            fold_key = (flag, backbone, setup, fold, bg_size)
            saved = [row for row in rows if (
                row['flag'], row['backbone'], row['setup'], row['fold'], row['background_size']
            ) == fold_key]
            expected = {(int(class_idx), mass) for class_idx in np.unique(labels_train)
                        for mass in MASS_THRESHOLDS}
            if len(saved) == len(expected) and {
                (row['class_idx'], row['mass_threshold_pct']) for row in saved
            } == expected:
                print(f'[SKIP] {cache_path.name}: already complete')
                continue
            model = models[fold]
            model.eval()
            layer = get_layer_from_model(model, 'avgpool')
            try:
                shap_values = compute_shap_values_for_fold(
                    model, 'avgpool', features_train, features_calib,
                )
                fold_rows = []
                for class_idx in np.unique(labels_train):
                    class_count = int(np.count_nonzero(labels_train == class_idx))
                    for mass in MASS_THRESHOLDS:
                        _, selected_count = select_features_by_mass(
                            shap_values, labels_train, int(class_idx), mass / 100,
                        )
                        fold_rows.append({
                            'flag': flag,
                            'backbone': backbone,
                            'setup': setup,
                            'fold': fold,
                            'background_size': bg_size,
                            'class_idx': int(class_idx),
                            'n_train_samples_class': class_count,
                            'mass_threshold_pct': mass,
                            'n_selected_features': int(selected_count),
                        })
            except Exception as exc:
                print(f'[FAIL] {cache_path.name}: {exc}')
                continue

            if saved:
                rows = [row for row in rows if (
                    row['flag'], row['backbone'], row['setup'], row['fold'], row['background_size']
                ) != fold_key]
            rows.extend(fold_rows)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f'{flag} {backbone} {setup} fold={fold}: wrote {len(fold_rows)} class/threshold rows')

    print(f'Saved {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()