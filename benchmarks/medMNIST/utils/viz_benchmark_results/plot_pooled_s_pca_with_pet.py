"""Pool controlled MedMNIST results and overlay five matched PET/CT folds.

PET uncertainty scores use the same k=5/PCA90/mass1/no-calibration settings.
Similarity alone uses a shared training scaler before separate class PCAs.
MedMNIST AUROCs and predictions are read unchanged from controlled results.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from recompute_controlled_knn_comparison import OUTPUT, compare

PET_STEM = "external_DualStream_archDenseNet_do0.1_wd0.0001_bs8_bm1_les1"
PET_CACHE = Path("/workspace/train_PET_CT/UQ/knn_shap_cache")


def pet_results():
    prediction_path = Path("/workspace/train_PET_CT/Inference") / f"{PET_STEM}_predictions.csv"
    predictions = pd.read_csv(prediction_path, dtype={"Patient_ID": str})
    if predictions.Patient_ID.duplicated().any():
        raise ValueError("Duplicate PET prediction patient IDs")
    predictions = predictions.set_index("Patient_ID")
    rows = []
    for fold in range(5):
        train_path = PET_CACHE / f"shap_cache_{PET_STEM}_fold{fold}_fold0_bg1000.npz"
        external_path = PET_CACHE / f"{PET_STEM}_external_features_fold{fold}.npz"
        with np.load(train_path, allow_pickle=False) as cache:
            train, labels = cache["features_train"], cache["labels_train"]
        with np.load(external_path, allow_pickle=True) as cache:
            test, ids = cache["features"], cache["patient_ids"].astype(str)
        if len(np.unique(ids)) != len(ids) or set(ids) != set(predictions.index):
            raise ValueError(f"PET feature/prediction patient mismatch in fold {fold}")
        aligned = predictions.loc[ids]
        truth = aligned.label.to_numpy()
        probabilities = aligned[f"prob_fold_{fold}"].to_numpy()
        predicted = (probabilities >= 0.5).astype(int)
        metrics, raw, shap = compare(train, labels, test, truth, predicted)
        score_path = OUTPUT / f"PET_fold{fold}_per_patient.csv"
        pd.DataFrame({
            "Patient_ID": ids, "label": truth, "probability": probabilities,
            "prediction": predicted, "error": predicted != truth,
            "knn_distance": raw, "knn_shap_distance": shap,
        }).to_csv(score_path, index=False)
        rows.append(dict(dataset="PET/CT", backbone="DenseNet121", setup="bm1_les1",
                         shift="external", fold=fold, **metrics,
                         score_csv=str(score_path), train_cache=str(train_path)))
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "pet_per_fold_results.csv", index=False)
    return result


def main():
    med = pd.read_csv(OUTPUT / "per_fold_results.csv")
    if len(med) != 160 or not med.similarity_normalization.eq("global_training_standard_scaler").all():
        raise ValueError("Expected 160 shared-scaler controlled MedMNIST results")
    pet = pet_results()
    combined = pd.concat([med, pet], ignore_index=True)
    combined.to_csv(OUTPUT / "pooled_with_pet_per_fold_results.csv", index=False)
    x, y = "similarity_s_pca", "auroc_f_knn_minus_knn_shap"
    associations = []
    for name, frame in (("MedMNIST only", med), ("MedMNIST + PET/CT", combined)):
        associations.append(dict(scope=name, n=len(frame),
                                 pearson_r=float(pearsonr(frame[x], frame[y]).statistic)))
    pd.DataFrame(associations).to_csv(OUTPUT / "figures" / "pooled_associations.csv", index=False)
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17})
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    colors = {"breastmnist": "tab:purple", "pneumoniamnist": "tab:orange"}
    markers = {"resnet18": "o", "vit_b_16": "^"}
    for (dataset, backbone, shift), group in med.groupby(["dataset", "backbone", "shift"]):
        ax.scatter(group[x], group[y], marker=markers[backbone], s=75,
                   facecolors=colors[dataset] if shift == "ID" else "none",
                   edgecolors=colors[dataset], linewidths=1.3, alpha=0.7)
    ax.scatter(pet[x], pet[y], marker="X", s=180, c="black", edgecolors="white",
               linewidths=0.6, zorder=5)
    for row in pet.itertuples():
        ax.annotate(f"F{row.fold}", (row.similarity_s_pca, row.auroc_f_knn_minus_knn_shap),
                    xytext=(7, 5), textcoords="offset points", fontsize=11)
    ax.axhline(0, color="gray", ls="--", lw=1.2)
    ax.set(xlim=(0, 1), xlabel=r"$S_{\mathrm{PCA}}$ (higher = greater similarity)",
           ylabel="AUROC-F: KNN - KNN-SHAP",
           title="KNN vs. KNN-SHAP AUROC-F Diff vs. PCA Subspace Similarity in Binary Classification\n"
                 f"MedMNIST r={associations[0]['pearson_r']:.3f} (n=160) ; "
                 f"including PET + seg r={associations[1]['pearson_r']:.3f} (n=165)")
    ax.title.set_fontsize(13)
    ax.grid(alpha=0.15)
    handles = [Line2D([], [], marker="o", color=color, ls="", label=dataset)
               for dataset, color in colors.items()]
    handles += [Line2D([], [], marker=marker, color="gray", ls="", label=backbone)
                for backbone, marker in markers.items()]
    handles += [Line2D([], [], marker="o", color="gray", ls="", label="ID (filled)"),
                Line2D([], [], marker="o", color="gray", markerfacecolor="none", ls="", label="CS (open)"),
                Line2D([], [], marker="X", color="black", ls="", markersize=10, label="PET + seg BMI prediction")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / "figures" / f"pooled_s_pca_with_pet.{extension}", dpi=180)
    plt.close(fig)
    print(pd.DataFrame(associations).to_string(index=False))
    print(pet[["fold", x, "KNN_Raw_auroc_f", "KNN_SHAP_auroc_f", y]].to_string(index=False))


if __name__ == "__main__":
    main()
