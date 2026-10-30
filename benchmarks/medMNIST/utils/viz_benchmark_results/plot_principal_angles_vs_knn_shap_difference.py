"""Plot dimension-aware S_PCA against fold AUROC-F(KNN) - AUROC-F(KNN-SHAP).

For similarity only, uses one global training scaler before separate class
PCAs retaining 90% variance. KNN-SHAP still uses its original class-wise scalers.
S_PCA = sum(cos(principal_angles)**2) / max(k0,k1); higher is more similar.
Complete subspaces are retained, and unequal dimensions are penalized.
Only controlled recomputed UQ results are used, never historical JSON metrics.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

RESULTS = Path(__file__).resolve().parent / "results"
DATASETS = ("breastmnist", "pneumoniamnist")
BACKBONES = ("resnet18", "vit_b_16")
SETUPS = ("standard", "DA", "DO", "DADO")
COLORS = {"breastmnist": "tab:purple", "pneumoniamnist": "tab:orange"}
MARKERS = {"standard": "o", "DA": "^", "DO": "s", "DADO": "D"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recomputed-csv", type=Path,
                        default=RESULTS / "controlled_knn_k5_pca90_mass1" / "per_fold_results.csv",
                        help="Controlled recomputed scores with dimension-aware S_PCA similarity")
    args = parser.parse_args()
    output = args.recomputed_csv.parent / "figures"
    keys = ["dataset", "backbone", "setup", "shift", "fold"]
    joined = pd.read_csv(args.recomputed_csv)
    if (len(joined) != 160 or joined.duplicated(keys).any()
            or not joined.k.eq(5).all() or not joined.mass_threshold.eq(1).all()
            or not joined.pca_variance.eq(0.9).all()
            or joined.calibrate_with_auroc.any()):
        raise ValueError("Expected 160 controlled k=5, mass=1, PCA90, uncalibrated fold results")
    if not joined.similarity_normalization.eq("global_training_standard_scaler").all():
        raise ValueError("Principal-angle subanalysis requires one shared training scaler")
    for row in joined.itertuples():
        angles = np.array(json.loads(row.principal_angles_deg))
        if (angles.ndim != 1 or len(angles) != min(
                row.similarity_n_pcs_class0, row.similarity_n_pcs_class1)
                or not np.isfinite(angles).all() or (angles < 0).any() or (angles > 90).any()):
            raise ValueError("Invalid individual principal angles")
        np.testing.assert_allclose(angles.mean(), row.mean_principal_angle_deg)
        expected_similarity = np.sum(np.cos(np.radians(angles)) ** 2) / max(
            row.similarity_n_pcs_class0, row.similarity_n_pcs_class1)
        np.testing.assert_allclose(row.similarity_s_pca, expected_similarity)
        if not 0 <= row.similarity_s_pca <= 1:
            raise ValueError("S_PCA must lie in [0, 1]")
    joined["auroc_f_knn_minus_knn_shap"] = joined.KNN_Raw_auroc_f - joined.KNN_SHAP_auroc_f
    if not np.isfinite(joined[["similarity_s_pca", "auroc_f_knn_minus_knn_shap"]]).all().all():
        raise ValueError("Nonfinite plotting values")
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "matched_per_fold_results.csv", index=False)
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 16})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True,
                             layout="constrained")
    correlations = []
    for row, backbone in enumerate(BACKBONES):
        for col, shift in enumerate(("ID", "CS")):
            ax = axes[row, col]
            subset = joined[(joined.backbone == backbone) & (joined["shift"] == shift)]
            for dataset in DATASETS:
                for setup in SETUPS:
                    group = subset[(subset.dataset == dataset) & (subset.setup == setup)]
                    if len(group) != 5:
                        raise ValueError(f"Incomplete plot group: {dataset}/{backbone}/{setup}/{shift}")
                    ax.scatter(group.similarity_s_pca, group.auroc_f_knn_minus_knn_shap,
                               c=COLORS[dataset], marker=MARKERS[setup], s=75, alpha=0.8,
                               edgecolors="white", linewidths=0.5)
            result = pearsonr(subset.similarity_s_pca, subset.auroc_f_knn_minus_knn_shap)
            correlations.append(dict(backbone=backbone, shift=shift, n=len(subset),
                                     pearson_r=float(result.statistic)))
            ax.axhline(0, color="gray", ls="--", lw=1.2)
            ax.set_title(f"{backbone} / {shift}\nAssociation r = {result.statistic:.3f} (n={len(subset)})")
            ax.grid(alpha=0.15)
            ax.set_xlim(0, 1)
            if row == 1:
                ax.set_xlabel(r"$S_{\mathrm{PCA}}$ (higher = greater similarity)")
            if col == 0:
                ax.set_ylabel("AUROC-F: KNN - KNN-SHAP")
    handles = [Line2D([], [], color=COLORS[d], marker="o", ls="", label=d) for d in DATASETS]
    handles += [Line2D([], [], color="black", marker=MARKERS[s], ls="", label=s) for s in SETUPS]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=12)
    heading = "Controlled k=5, mass=1, no AUROC-F calibration; PCA90"
    fig.suptitle(heading + "\n"
                 "S_PCA: shared training scaler | Positive difference favors KNN", fontsize=17)
    fig.savefig(output / "s_pca_vs_knn_minus_knn_shap.png", dpi=180)
    fig.savefig(output / "s_pca_vs_knn_minus_knn_shap.pdf")
    plt.close(fig)
    for (backbone, shift, dataset), group in joined.groupby(["backbone", "shift", "dataset"]):
        correlations.append(dict(backbone=backbone, shift=shift, dataset=dataset, n=len(group),
                                 pearson_r=float(pearsonr(
                                     group.similarity_s_pca, group.auroc_f_knn_minus_knn_shap).statistic)))
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame["similarity_metric"] = "S_PCA"
    correlation_frame["auroc_difference"] = "KNN - KNN-SHAP"
    correlation_frame.to_csv(output / "association_correlations.csv", index=False)
    for stem in ("pc_norm_vs_knn_minus_knn_shap", "principal_angles_vs_knn_minus_knn_shap"):
        for extension in ("png", "pdf"):
            obsolete = output / f"{stem}.{extension}"
            if obsolete.exists():
                obsolete.unlink()
    print(correlation_frame.to_string(index=False))
    print(f"Validated all 160 paired folds. Saved plot and data to {output}")


if __name__ == "__main__":
    main()
