"""Checks for the controlled, all-feature KNN comparison."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from scipy.linalg import subspace_angles

from .recompute_controlled_knn_comparison import (
    compare, fit_reference, distances, principal_angles, globally_scaled_subspace_metrics,
    subspace_metrics,
)


def test_distances_are_five_neighbor_means():
    rng = np.random.default_rng(12)
    train, test = rng.normal(size=(30, 8)), rng.normal(size=(9, 8))
    reference = fit_reference(train)
    scaler, pca, _ = reference
    all_distances = cdist(pca.transform(scaler.transform(test)),
                          pca.transform(scaler.transform(train)))
    expected = np.sort(all_distances, axis=1)[:, :5].mean(axis=1)
    np.testing.assert_allclose(distances(reference, test), expected)
    assert pca.explained_variance_ratio_.sum() >= 0.9
    assert pca.explained_variance_ratio_[:-1].sum() <= 0.9


def test_class_distances_route_by_prediction_without_offsets():
    rng = np.random.default_rng(15)
    labels = np.repeat([0, 1], 25)
    train = rng.normal(size=(50, 7)) + labels[:, None] * 8
    truth = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 0, 1])
    test = train[[0, 1, 25, 26]] + 0.1
    metrics, raw, shap = compare(train, labels, test, truth, predictions)
    expected = np.empty(4)
    for c in (0, 1):
        mask = predictions == c
        expected[mask] = distances(fit_reference(train[labels == c]), test[mask])
    np.testing.assert_allclose(shap, expected)
    assert shap[1] > shap[0] and shap[2] > shap[3]
    assert metrics["KNN_SHAP_auroc_f"] == roc_auc_score(predictions != truth, expected)
    assert metrics["KNN_Raw_auroc_f"] == roc_auc_score(predictions != truth, raw)
    assert metrics["mass_threshold"] == 1 and not metrics["calibrate_with_auroc"]
    assert metrics["n_features"] == 7
    assert "pearson_r" not in metrics and "spearman_rho" not in metrics
    assert 0 <= metrics["mean_principal_angle_deg"] <= 90
    assert 0 <= metrics["similarity_s_pca"] <= 1
    assert metrics["n_principal_angles"] == min(
        metrics["similarity_n_pcs_class0"], metrics["similarity_n_pcs_class1"])


def test_too_few_neighbors_is_explicit_error():
    with pytest.raises(ValueError, match="at least five"):
        fit_reference(np.zeros((4, 8)))


def test_principal_angles_alignment_orthogonality_and_rotation():
    basis = np.eye(6)[:2]
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    np.testing.assert_allclose(principal_angles(basis, rotation @ basis), 0, atol=2e-6)
    np.testing.assert_allclose(principal_angles(basis, np.eye(6)[2:4]), 90, atol=1e-10)
    # Containment yields zero angles even when one subspace has extra dimensions.
    np.testing.assert_allclose(principal_angles(basis, np.eye(6)[:3]), [0, 0], atol=1e-10)


def test_principal_angles_known_values_and_unequal_dimensions():
    radians = np.radians([20, 65])
    basis0 = np.eye(6)[:2]
    basis1 = np.zeros((3, 6))
    basis1[:2, :2] = np.diag(np.cos(radians))
    basis1[:2, 2:4] = np.diag(np.sin(radians))
    basis1[2, 4] = 1
    np.testing.assert_allclose(principal_angles(basis0, basis1), [20, 65], atol=1e-10)
    np.testing.assert_allclose(principal_angles(basis1, basis0), [20, 65], atol=1e-10)


def test_principal_angles_agree_with_scipy():
    rng = np.random.default_rng(42)
    q0 = np.linalg.qr(rng.normal(size=(20, 4)))[0]
    q1 = np.linalg.qr(rng.normal(size=(20, 7)))[0]
    np.testing.assert_allclose(principal_angles(q0.T, q1.T),
                               np.sort(np.degrees(subspace_angles(q0, q1))), atol=1e-10)
    np.testing.assert_allclose(principal_angles(-q0.T, q1.T),
                               principal_angles(q0.T, q1.T), atol=1e-10)


@pytest.mark.parametrize("other", [np.ones((2, 6)), np.eye(5)[:2], np.empty((0, 6)),
                                  np.full((2, 6), np.nan)])
def test_principal_angles_reject_invalid_bases(other):
    with pytest.raises(ValueError):
        principal_angles(np.eye(6)[:2], other)


def test_similarity_uses_global_scaling_before_class_pca():
    import json
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(72)
    labels = np.repeat([0, 1], 40)
    train = rng.normal(size=(80, 9))
    train[labels == 1] *= np.arange(1, 10)
    train[labels == 1] += 3
    standardized = StandardScaler().fit_transform(train)
    pcas = [PCA(n_components=.9, svd_solver="full").fit(
        standardized[labels == c]) for c in (0, 1)]
    expected = np.sort(np.degrees(subspace_angles(
        pcas[0].components_.T.astype(float), pcas[1].components_.T.astype(float))))
    result = globally_scaled_subspace_metrics(train, labels)
    np.testing.assert_allclose(json.loads(result["principal_angles_deg"]), expected, atol=3e-6)
    assert result["similarity_normalization"] == "global_training_standard_scaler"
    assert result["similarity_n_pcs_class0"] == pcas[0].n_components_
    assert result["similarity_n_pcs_class1"] == pcas[1].n_components_
    separate = [PCA(n_components=.9, svd_solver="full").fit(
        StandardScaler().fit_transform(train[labels == c])) for c in (0, 1)]
    assert abs(result["mean_principal_angle_deg"] -
               principal_angles(separate[0].components_, separate[1].components_).mean()) > 1
    expected_similarity = np.linalg.norm(
        pcas[0].components_.astype(float) @ pcas[1].components_.astype(float).T, "fro") ** 2 / max(
            pcas[0].n_components_, pcas[1].n_components_)
    np.testing.assert_allclose(result["similarity_s_pca"], expected_similarity, atol=1e-12)


@pytest.mark.parametrize("indices, expected", [
    ([0, 1], 1.0), ([2, 3], 0.0), ([0, 1, 2, 3], 0.5),
])
def test_s_pca_alignment_orthogonality_and_dimension_penalty(indices, expected):
    from types import SimpleNamespace

    pca0 = SimpleNamespace(components_=np.eye(6)[:2])
    pca1 = SimpleNamespace(components_=np.eye(6)[indices])
    result = subspace_metrics(pca0, pca1)
    np.testing.assert_allclose(result["similarity_s_pca"], expected, atol=1e-12)
    np.testing.assert_allclose(subspace_metrics(pca1, pca0)["similarity_s_pca"], expected, atol=1e-12)


def test_s_pca_equals_frobenius_overlap_and_is_rotation_invariant():
    from types import SimpleNamespace

    rng = np.random.default_rng(114)
    u0 = np.linalg.qr(rng.normal(size=(20, 4)))[0]
    u1 = np.linalg.qr(rng.normal(size=(20, 7)))[0]
    rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    expected = np.linalg.norm(u0.T @ u1, "fro") ** 2 / 7
    for basis in (u0, u0 @ rotation, -u0):
        result = subspace_metrics(SimpleNamespace(components_=basis.T),
                                  SimpleNamespace(components_=u1.T))
        np.testing.assert_allclose(result["similarity_s_pca"], expected, atol=1e-12)
        assert 0 <= result["similarity_s_pca"] <= 4 / 7 + 1e-12
