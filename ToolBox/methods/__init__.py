"""
UQ methods: TTA, GPS, ensemble, distance-based, latent-space, calibration.
"""
# Class-based API
from .tta import TTAMethod, GPSMethod

# Functional API
from .tta import (
    TTA,
    apply_augmentations,
    apply_randaugment_and_store_results,
    extract_gps_augmentations_info,
)
# Ensemble
from .ensemble import (
    EnsembleSTDMethod,
    ensembling_predictions,
    ensembling_stds_computation,
    ensembling_variance_computation,
)
# Distance-based
from .distance import (
    DistanceToHardLabelsMethod,
    CalibrationMethod,
    TemperatureScaler,
    distance_to_hard_labels_computation,
    posthoc_calibration,
    fit_temperature_scaling,
    compute_class_weights,
)
# Latent space
from .latent import (
    ClassifierHeadWrapper,
    KNNLatentSHAPMethod,
    KNNLatentMethod,
    HyperplaneDistanceMethod,
    get_layer_from_model,
    extract_latent_space_and_compute_shap_importance,
    compute_mean_shap_values,
    display_shap_values,
    feature_engineering_pipeline,
    analyze_hyperplane_distance,
    compute_knn_distances_to_train_data,
)

__all__ = [
    # TTA Classes
    "TTAMethod",
    "GPSMethod",
    # TTA functions
    "TTA",
    "apply_augmentations",
    "apply_randaugment_and_store_results",
    "extract_gps_augmentations_info",
    # Ensemble Classes
    "EnsembleSTDMethod",
    # Ensemble Functions
    "ensembling_predictions",
    "ensembling_stds_computation",
    "ensembling_variance_computation",
    # Distance Classes
    "DistanceToHardLabelsMethod",
    "CalibrationMethod",
    "TemperatureScaler",
    # Distance Functions
    "distance_to_hard_labels_computation",
    "posthoc_calibration",
    "fit_temperature_scaling",
    "compute_class_weights",
    # Latent Classes
    "ClassifierHeadWrapper",
    "KNNLatentSHAPMethod",
    "KNNLatentMethod",
    "HyperplaneDistanceMethod",
    # Latent Functions
    "get_layer_from_model",
    "extract_latent_space_and_compute_shap_importance",
    "compute_mean_shap_values",
    "display_shap_values",
    "feature_engineering_pipeline",
    "analyze_hyperplane_distance",
    "compute_knn_distances_to_train_data"
]