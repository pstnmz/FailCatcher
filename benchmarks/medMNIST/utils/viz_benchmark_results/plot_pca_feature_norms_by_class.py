"""KNN-SHAP up to (not including) the KNN fit: for one fold, select SHAP features per class by
mass threshold, standardize + PCA (90% variance) per class on train data, then take the Euclidean
norm of each feature's loadings across the retained PCs and scatter class0 (x) vs class1 (y).

Usage:
    .venv/bin/python plot_pca_feature_norms_by_class.py --flag breastmnist --fold 0 --mass 1.0
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

CACHE_DIR = Path(__file__).resolve().parent / 'Benchmarks/medMNIST/results/knn_shap_results/shap_cache'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--flag', default='breastmnist')
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--setup', default='', help="'' = standard, or DA/DO/DADO")
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--mass', type=float, default=1.0)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    ns = f"{args.flag}_{args.backbone}{'_' + args.setup if args.setup else ''}"
    cache = CACHE_DIR / f'shap_cache_{ns}_fold{args.fold}_bg1000.npz'
    features_train, labels_train, features_calib, _ = load_cached_activations(cache)

    model = tr.load_models(args.flag, device=torch.device('cpu'), size=IMAGE_SIZE,
                           model_backbone=args.backbone, setup=args.setup)[args.fold].eval()
    shap_values = compute_shap_values_for_fold(model, 'avgpool', features_train, features_calib)

    n_feat = features_train.shape[1]
    norms = {}
    for c in (0, 1):
        sel, k = select_features_by_mass(shap_values, labels_train, c, args.mass)
        x = features_train[labels_train == c][:, sel]
        pca = PCA(n_components=0.9).fit(StandardScaler().fit_transform(x))
        # components_: [n_PCs, n_selected] -> norm over PCs for each selected feature
        full = np.full(n_feat, np.nan)
        full[sel] = np.linalg.norm(pca.components_, axis=0)
        norms[c] = full
        print(f'class {c}: n_train={len(x)}, selected features={k}/{n_feat}, retained PCs={pca.n_components_}')

    both = ~np.isnan(norms[0]) & ~np.isnan(norms[1])
    print(f'features selected in both classes: {both.sum()}')
    for c in (0, 1):
        v = norms[c][~np.isnan(norms[c])]
        print(f'class {c} norms: min={v.min():.4f} median={np.median(v):.4f} max={v.max():.4f}')
    r = np.corrcoef(norms[0][both], norms[1][both])[0, 1]
    print(f'Pearson r (class0 vs class1 norms) = {r:.4f}')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(norms[0][both], norms[1][both], s=12, alpha=0.7)
    lim = max(np.nanmax(norms[0]), np.nanmax(norms[1])) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8)
    ax.set(xlabel='class 0: ||loadings across retained PCs||₂', ylabel='class 1: ||loadings across retained PCs||₂',
           title=f'{ns} fold{args.fold}, mass={args.mass} (r={r:.2f})')
    ax.set_aspect('equal')
    fig.tight_layout()
    out = args.output or Path(f'uq_benchmark_results/pca_feature_norms_{ns}_fold{args.fold}_mass{args.mass}.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
