"""Recompute matched fold KNN comparisons without reading historical UQ results.

KNN: all training samples, one StandardScaler/PCA(0.9), k=5.
KNN-SHAP mass=1: all features, true-class training reference sets, separate
StandardScaler/PCA(0.9), k=5, predicted-class routing, no offsets or calibration.
SHAP attribution calculation is unnecessary when retaining every feature.
Distances are unwhitened mean Euclidean k-neighbor distances, without class
normalization. Larger distance means greater uncertainty.

Fresh ID and severity-3 CS inputs are shared by both methods, every fold and
every setup. Existing cached training embeddings are reused; test embeddings
and predictions are extracted together from the corresponding model.

For similarity only, one scaler is fitted on all training samples before
separate class PCAs; angles compare globally standardized loading subspaces.
The class-specific KNN-SHAP scalers and distances are unchanged. Unequal retained
dimensions yield min(k0, k1) angles. The dimension-aware summary S_PCA is
sum(cos(angle)**2) / max(k0, k1), ranging from 0 to 1 (higher = more similar).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation import dataset_utils as du
from ToolBox.methods.latent import (
    ClassifierHeadWrapper, extract_latent_space_and_compute_shap_importance, get_layer_from_model,
)

RESULTS = Path(__file__).resolve().parent / "results"
CACHE = RESULTS / "knn_shap_results" / "shap_cache"
OUTPUT = RESULTS / "controlled_knn_k5_pca90_mass1"
K = 5
VARIANCE = 0.9


def principal_angles(components0, components1):
    """Angles in ascending degrees from SVD(U1.T @ U0), with PCs as columns."""
    bases = [np.asarray(components, dtype=np.float64).T
             for components in (components0, components1)]
    if any(b.ndim != 2 or b.shape[1] == 0 or not np.isfinite(b).all() for b in bases):
        raise ValueError("PCA bases must be finite, nonempty matrices")
    if bases[0].shape[0] != bases[1].shape[0]:
        raise ValueError("PCA bases must use the same feature coordinates")
    for basis in bases:
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-5, rtol=1e-5):
            raise ValueError("PCA basis columns must be orthonormal")
    singular_values = np.linalg.svd(bases[1].T @ bases[0], compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular_values, 0.0, 1.0)))


def subspace_metrics(pca0, pca1):
    angles = principal_angles(pca0.components_, pca1.components_)
    similarity = np.sum(np.cos(np.radians(angles)) ** 2) / max(
        len(pca0.components_), len(pca1.components_))
    return {
        "similarity_s_pca": float(similarity),
        "principal_angles_deg": json.dumps(angles.tolist()),
        "mean_principal_angle_deg": float(angles.mean()),
        "n_principal_angles": len(angles),
    }


def globally_scaled_subspace_metrics(train, labels):
    """Fit one train-only scaler, then separate centered class PCAs at 90%."""
    if (train.ndim != 2 or labels.shape != (len(train),)
            or not np.isfinite(train).all() or not np.array_equal(np.unique(labels), [0, 1])):
        raise ValueError("Expected finite training features and aligned binary labels")
    standardized = StandardScaler().fit_transform(train)
    pcas = [PCA(n_components=VARIANCE, svd_solver="full").fit(
        standardized[labels == c]) for c in (0, 1)]
    for pca in pcas:
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        if cumulative[-1] < VARIANCE or (len(cumulative) > 1 and cumulative[-2] > VARIANCE):
            raise ValueError("Similarity PCA must retain the minimal PCs reaching 90% variance")
    return {
        **subspace_metrics(*pcas),
        "similarity_normalization": "global_training_standard_scaler",
        **{f"similarity_n_pcs_class{c}": pcas[c].n_components_ for c in (0, 1)},
        **{f"similarity_variance_class{c}": float(pcas[c].explained_variance_ratio_.sum())
           for c in (0, 1)},
    }


def fit_reference(train):
    if len(train) < K or not np.isfinite(train).all():
        raise ValueError("Reference set must contain at least five finite training samples")
    scaler = StandardScaler().fit(train)
    standardized = scaler.transform(train)
    pca = PCA(n_components=VARIANCE, svd_solver="full").fit(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    if cumulative[-1] < VARIANCE or (len(cumulative) > 1 and cumulative[-2] > VARIANCE):
        raise ValueError("PCA did not retain the minimal number of PCs reaching 90% variance")
    neighbors = NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=2).fit(
        pca.transform(standardized))
    return scaler, pca, neighbors


def distances(reference, test):
    scaler, pca, neighbors = reference
    return neighbors.kneighbors(pca.transform(scaler.transform(test)))[0].mean(axis=1)


def compare(train, labels, test, truth, predictions):
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Expected binary training labels")
    if len(test) != len(truth) or predictions.shape != truth.shape:
        raise ValueError("Test embeddings, labels and predictions must align")
    error = predictions != truth
    if len(np.unique(error)) != 2:
        raise ValueError("AUROC-F requires both correct and incorrect predictions")
    global_ref = fit_reference(train)
    class_refs = {c: fit_reference(train[labels == c]) for c in (0, 1)}
    raw = distances(global_ref, test)
    shap = np.full(len(test), np.nan)
    for c in (0, 1):
        mask = predictions == c
        if mask.any():
            shap[mask] = distances(class_refs[c], test[mask])
    if not np.isfinite(shap).all() or not np.isfinite(raw).all():
        raise ValueError("Invalid distances or unassigned predicted class")
    knn_auroc = float(roc_auc_score(error, raw))
    shap_auroc = float(roc_auc_score(error, shap))
    metrics = {
        **globally_scaled_subspace_metrics(train, labels),
        "KNN_Raw_auroc_f": knn_auroc, "KNN_SHAP_auroc_f": shap_auroc,
        "auroc_f_knn_minus_knn_shap": knn_auroc - shap_auroc,
        "n_pcs_knn": global_ref[1].n_components_,
        "n_pcs_class0": class_refs[0][1].n_components_,
        "n_pcs_class1": class_refs[1][1].n_components_,
        "variance_knn": float(global_ref[1].explained_variance_ratio_.sum()),
        "variance_class0": float(class_refs[0][1].explained_variance_ratio_.sum()),
        "variance_class1": float(class_refs[1][1].explained_variance_ratio_.sum()),
        "n_features": train.shape[1], "n_train": len(train),
        "n_test": len(test), "n_errors": int(error.sum()),
        "k": K, "pca_variance": VARIANCE, "mass_threshold": 1.0,
        "calibrate_with_auroc": False,
    }
    return metrics, raw, shap


def save_angle_outputs(results):
    keys = ["dataset", "backbone", "setup", "shift", "fold"]
    individual = []
    for row in results.to_dict("records"):
        angles = json.loads(row["principal_angles_deg"])
        if len(angles) != min(row["similarity_n_pcs_class0"], row["similarity_n_pcs_class1"]):
            raise ValueError("Principal angle count does not match subspace dimensions")
        individual.extend({
            **{key: row[key] for key in keys}, "angle_index": index,
            "principal_angle_deg": angle,
            "similarity_n_pcs_class0": row["similarity_n_pcs_class0"],
            "similarity_n_pcs_class1": row["similarity_n_pcs_class1"],
            "similarity_normalization": row["similarity_normalization"],
        } for index, angle in enumerate(angles, 1))
    pd.DataFrame(individual).to_csv(OUTPUT / "individual_principal_angles.csv", index=False)
    summary = results.groupby(keys[:-1]).agg(
        n_folds=("fold", "nunique"),
        s_pca_mean=("similarity_s_pca", "mean"),
        s_pca_sd=("similarity_s_pca", "std"),
        mean_angle_deg=("mean_principal_angle_deg", "mean"),
        sd_mean_angle_deg=("mean_principal_angle_deg", "std"),
        similarity_n_pcs_class0_min=("similarity_n_pcs_class0", "min"),
        similarity_n_pcs_class0_max=("similarity_n_pcs_class0", "max"),
        similarity_n_pcs_class1_min=("similarity_n_pcs_class1", "min"),
        similarity_n_pcs_class1_max=("similarity_n_pcs_class1", "max"),
    ).reset_index()
    summary.to_csv(OUTPUT / "subspace_angle_summary.csv", index=False)


def refresh_similarity():
    """Replace only similarity fields; preserve every controlled UQ result."""
    path = OUTPUT / "per_fold_results.csv"
    results = pd.read_csv(path)
    keys = ["dataset", "backbone", "setup", "shift", "fold"]
    if len(results) != 160 or results.duplicated(keys).any():
        raise ValueError("Expected 160 unique controlled fold results")
    similarity_columns = [
        "pearson_r", "spearman_rho", "principal_angles_deg",
        "mean_principal_angle_deg", "n_principal_angles",
    ] + [column for column in results if column.startswith("similarity_")]
    preserved = results.drop(columns=similarity_columns, errors="ignore")
    replacements = {}
    for row in results.itertuples():
        cache_path = row.train_cache
        if cache_path not in replacements:
            with np.load(cache_path, allow_pickle=False) as cache:
                train, labels = cache["features_train"], cache["labels_train"]
            replacements[cache_path] = globally_scaled_subspace_metrics(train, labels)
    updated = pd.concat([
        preserved.reset_index(drop=True),
        pd.DataFrame([replacements[row.train_cache] for row in results.itertuples()]),
    ], axis=1)
    pd.testing.assert_frame_equal(updated[preserved.columns], preserved)
    updated.to_csv(path, index=False)
    save_angle_outputs(updated)
    config_path = OUTPUT / "experiment_config.json"
    config = json.loads(config_path.read_text())
    config["pca_similarity"] = {
        "method": "dimension-aware S_PCA", "range": [0, 1],
        "basis": "complete retained globally standardized class PCA loading subspaces",
        "normalization": "one StandardScaler fitted on all fold training samples; similarity only",
        "unequal_dimensions": "complete subspaces; denominator max(k0,k1)",
        "summary": "sum(cos(principal_angles)**2) / max(k0,k1); higher means more similar",
        "detailed_outputs": "individual principal angles in degrees and PCA dimensions",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Updated {len(updated)} folds; all UQ fields unchanged.", flush=True)


def materialize(dataset):
    samples = [dataset[i] for i in range(len(dataset))]
    images = torch.stack([x for x, _ in samples])
    labels = torch.tensor([int(np.asarray(y).item()) for _, y in samples])
    if not torch.isfinite(images).all():
        raise ValueError("Nonfinite input image")
    return TensorDataset(images, labels)


def test_sets(dataset, output):
    transform, _ = du.get_transforms(False, 224)
    data_class = getattr(tr.medmnist, tr.INFO[dataset]["python_class"])
    clean = data_class(split="test", transform=transform, size=224, download=False)
    if not du.MEDMNISTC_AVAILABLE or not du.get_available_corruptions(dataset):
        raise RuntimeError("CS requires working medmnistc and ImageMagick; clean fallback is forbidden")
    # Reset corruption-object generators, not just numpy's legacy global RNG.
    for corruption in du.get_available_corruptions(dataset).values():
        corruption.rng = np.random.default_rng(42)
    corrupted = du.apply_random_corruptions(clean, dataset, severity=3, cache=True, seed=42)
    if corrupted is clean:
        raise RuntimeError("Corruption wrapper returned clean images")
    sets = {"ID": materialize(clean), "CS": materialize(corrupted)}
    np.testing.assert_array_equal(sets["ID"].tensors[1], sets["CS"].tensors[1])
    if torch.equal(sets["ID"].tensors[0], sets["CS"].tensors[0]):
        raise ValueError("CS images are identical to ID images")
    # Persist the exact tensors so the controlled experiment can be reproduced.
    for shift, tensors in sets.items():
        torch.save(tensors.tensors, output / f"{dataset}_{shift}_input_tensors.pt")
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-similarity-only", action="store_true",
                        help="Update S_PCA and principal angles without changing UQ results")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=["breastmnist", "pneumoniamnist"],
                        choices=["breastmnist", "pneumoniamnist"])
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "vit_b_16"],
                        choices=["resnet18", "vit_b_16"])
    parser.add_argument("--setups", nargs="+", default=["standard", "DA", "DO", "DADO"],
                        choices=["standard", "DA", "DO", "DADO"])
    args = parser.parse_args()
    if args.refresh_similarity_only:
        refresh_similarity()
        return
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model_root = Path(__file__).resolve().parent / "models"
    os.environ["MEDMNIST_MODELS_DIR"] = str(model_root)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    embeddings_dir = OUTPUT / "test_embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    rows = []
    for dataset in args.datasets:
        sets = test_sets(dataset, OUTPUT)
        fingerprints = {
            shift: hashlib.sha256(data.tensors[0].numpy().tobytes()).hexdigest()
            for shift, data in sets.items()
        }
        for backbone in args.backbones:
            for setup in args.setups:
                models = tr.load_models(dataset, torch.device("cpu"), size=224,
                                        model_backbone=backbone, setup="" if setup == "standard" else setup)
                namespace = f"{dataset}_{backbone}" + (f"_{setup}" if setup != "standard" else "")
                for fold, model in enumerate(models):
                    randaug = int(setup in ("DA", "DADO"))
                    dropout = ("_dropout03" if backbone == "resnet18" else "_dropout01") if setup in ("DO", "DADO") else ""
                    checkpoint = model_root / "224*224" / (
                        f"{dataset}_{backbone}_224_randaug{randaug}{dropout}_fold_{fold}.pt")
                    with checkpoint.open("rb") as stream:
                        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                    train_path = CACHE / f"shap_cache_{namespace}_fold{fold}_bg1000.npz"
                    with np.load(train_path, allow_pickle=False) as cache:
                        train, labels = cache["features_train"], cache["labels_train"]
                    if not np.isfinite(train).all() or labels.shape != (len(train),):
                        raise ValueError(f"Invalid training cache: {train_path}")
                    model.to(device).eval()
                    for shift, data in sets.items():
                        key = f"{namespace}_{shift}_fold{fold}"
                        cache_path = embeddings_dir / f"{key}.npz"
                        if cache_path.exists():
                            with np.load(cache_path, allow_pickle=False) as cached:
                                if str(cached["input_sha256"]) != fingerprints[shift]:
                                    raise ValueError(f"Cached test inputs changed: {cache_path}")
                                if str(cached["checkpoint_sha256"]) != checkpoint_hash:
                                    raise ValueError(f"Cached checkpoint changed: {cache_path}")
                                test, truth, probabilities = (
                                    cached["features"], cached["labels"], cached["probabilities"])
                            np.testing.assert_array_equal(truth, data.tensors[1].numpy())
                        else:
                            loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True)
                            features, truth, success, probabilities = extract_latent_space_and_compute_shap_importance(
                                model, loader, device, get_layer_from_model(model, "avgpool"), importance=False)
                            test = features.numpy()
                            probabilities = np.asarray(probabilities)
                            head = ClassifierHeadWrapper(model, "avgpool")
                            with torch.no_grad():
                                reconstructed = torch.sigmoid(head(features.to(device))).flatten().cpu().numpy()
                            np.testing.assert_allclose(reconstructed, probabilities, atol=2e-6, rtol=1e-5)
                            np.testing.assert_array_equal((probabilities > 0.5) == truth, success.astype(bool))
                            np.savez_compressed(cache_path, features=test, labels=truth,
                                                probabilities=probabilities, input_sha256=fingerprints[shift],
                                                checkpoint_sha256=checkpoint_hash)
                        predictions = (probabilities > 0.5).astype(int)
                        metrics, raw, shap = compare(train, labels, test, truth, predictions)
                        score_path = OUTPUT / f"{key}_per_patient.csv"
                        pd.DataFrame({
                            "sample_index": np.arange(len(truth)), "label": truth,
                            "probability": probabilities, "prediction": predictions,
                            "error": predictions != truth, "knn_distance": raw,
                            "knn_shap_distance": shap,
                        }).to_csv(score_path, index=False)
                        rows.append(dict(dataset=dataset, backbone=backbone, setup=setup, shift=shift,
                                         fold=fold, **metrics, train_cache=str(train_path),
                                         score_csv=str(score_path), input_sha256=fingerprints[shift],
                                         checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash))
                        pd.DataFrame(rows).to_csv(OUTPUT / "per_fold_results.csv", index=False)
                        print(f"{key}: KNN={metrics['KNN_Raw_auroc_f']:.3f}, "
                              f"KNN-SHAP={metrics['KNN_SHAP_auroc_f']:.3f}, "
                              f"difference={metrics['auroc_f_knn_minus_knn_shap']:.3f}", flush=True)
                    model.to("cpu")
                del models
                torch.cuda.empty_cache()
    expected = len(args.datasets) * len(args.backbones) * len(args.setups) * 10
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} fold results, got {len(rows)}")
    save_angle_outputs(pd.DataFrame(rows))
    metadata = {
        "k": K, "pca_variance": VARIANCE, "mass_threshold": 1,
        "calibrate_with_auroc": False, "seed": 42, "corruption_severity": 3,
        "training_cache_shift": "ID for both ID and CS",
        "features": "all, no SHAP filtering or weighting", "pca_whiten": False,
        "score": "raw mean Euclidean distance to 5 training neighbors",
        "historical_uq_metrics_used": False, "arguments": vars(args),
        "pca_similarity": {
            "method": "dimension-aware S_PCA", "range": [0, 1],
            "basis": "complete retained globally standardized class PCA loading subspaces",
            "normalization": "one StandardScaler fitted on all fold training samples; similarity only",
            "unequal_dimensions": "complete subspaces; denominator max(k0,k1)",
            "summary": "sum(cos(principal_angles)**2) / max(k0,k1); higher means more similar",
            "detailed_outputs": "individual principal angles in degrees and PCA dimensions",
        },
    }
    (OUTPUT / "experiment_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {len(rows)} fold comparisons: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
