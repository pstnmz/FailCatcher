"""Shared vs class-unique SHAP-selected features (mass threshold) between the 2 classes of one fold.

Usage:
    .venv/bin/python analyze_shared_unique_shap_features.py --flag breastmnist --fold 0 --mass 0.8
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from compare_shap_normalization_auroc import (
    IMAGE_SIZE, compute_shap_values_for_fold, load_cached_activations, select_features_by_mass, tr,
)
from plot_pca_feature_norms_by_class import CACHE_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--flag', default='breastmnist')
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--setup', default='')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--mass', type=float, default=0.8)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    ns = f"{args.flag}_{args.backbone}{'_' + args.setup if args.setup else ''}"
    features_train, labels_train, features_calib, _ = load_cached_activations(
        CACHE_DIR / f'shap_cache_{ns}_fold{args.fold}_bg1000.npz')
    model = tr.load_models(args.flag, device=torch.device('cpu'), size=IMAGE_SIZE,
                           model_backbone=args.backbone, setup=args.setup)[args.fold].eval()
    shap_values = compute_shap_values_for_fold(model, 'avgpool', features_train, features_calib)
    n_feat = features_train.shape[1]

    # per-class mean |SHAP| over that class's train samples, normalized to a mass fraction per feature
    mass = {}
    for c in (0, 1):
        m = np.abs(shap_values[labels_train == c]).mean(axis=0)
        mass[c] = m / m.sum()
    sel = {c: set(select_features_by_mass(shap_values, labels_train, c, args.mass)[0].tolist()) for c in (0, 1)}
    shared, only0, only1 = sel[0] & sel[1], sel[0] - sel[1], sel[1] - sel[0]
    neither = set(range(n_feat)) - sel[0] - sel[1]
    print(f'{ns} fold{args.fold}, mass={args.mass}')
    print(f'class0 selected: {len(sel[0])}, class1 selected: {len(sel[1])}')
    print(f'shared: {len(shared)}, only class0: {len(only0)}, only class1: {len(only1)}, neither: {len(neither)}')
    print(f'Jaccard = {len(shared) / len(sel[0] | sel[1]):.3f}; '
          f'shared = {len(shared) / len(sel[0]):.1%} of class0 set, {len(shared) / len(sel[1]):.1%} of class1 set')

    groups = {'shared': shared, 'only class0': only0, 'only class1': only1}
    print('\nSHAP mass fraction covered (class0 / class1) by each group:')
    for name, g in groups.items():
        idx = sorted(g)
        print(f'  {name:12s}: {mass[0][idx].sum():.3f} / {mass[1][idx].sum():.3f}')

    # Per-class PCA on its own selected features (as in the KNN-SHAP fit)
    print('\nMean PCA loading norm over retained PCs (class0 PCA / class1 PCA):')
    pca_norm = {}
    for c in (0, 1):
        idx = np.array(sorted(sel[c]))
        x = features_train[labels_train == c][:, idx]
        pca = PCA(n_components=0.9).fit(StandardScaler().fit_transform(x))
        pca_norm[c] = dict(zip(idx.tolist(), np.linalg.norm(pca.components_, axis=0)))
        print(f'  class {c}: {len(idx)} features, {pca.n_components_} retained PCs')
    for name, g in groups.items():
        v0 = [pca_norm[0][i] for i in g if i in pca_norm[0]]
        v1 = [pca_norm[1][i] for i in g if i in pca_norm[1]]
        f = lambda v: f'{np.mean(v):.4f}' if v else '  n/a '
        print(f'  {name:12s}: {f(v0)} / {f(v1)}')

    # Rank correlation of SHAP importance across classes (binary head -> expected ~1 by symmetry)
    ranks = [np.argsort(np.argsort(mass[c])) for c in (0, 1)]
    print(f'\nSpearman rho of per-class mean|SHAP| across all {n_feat} features: '
          f'{np.corrcoef(ranks[0], ranks[1])[0, 1]:.3f}')

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {'shared': 'tab:green', 'only class0': 'tab:blue', 'only class1': 'tab:red'}
    for name, g in groups.items():
        idx = sorted(g)
        ax.scatter(mass[0][idx], mass[1][idx], s=12, alpha=0.7, c=colors[name], label=f'{name} (n={len(idx)})')
    if neither:
        idx = sorted(neither)
        ax.scatter(mass[0][idx], mass[1][idx], s=12, alpha=0.5, c='gray', label=f'neither (n={len(idx)})')
    ax.set(xlabel='class 0: mean|SHAP| mass fraction', ylabel='class 1: mean|SHAP| mass fraction',
           title=f'{ns} fold{args.fold}, mass={args.mass}')
    ax.legend()
    fig.tight_layout()
    out = args.output or Path(f'uq_benchmark_results/shared_unique_shap_features_{ns}_fold{args.fold}_mass{args.mass}.png')
    fig.savefig(out, dpi=150)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
