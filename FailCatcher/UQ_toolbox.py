"""
UQ_Toolbox: Uncertainty Quantification for Deep Learning Models

Backward-compatible API. Import everything from submodules and expose at top level.
Users can still do: import UQ_toolbox as uq; uq.TTA(...)
"""

# Core utilities
from .core.utils import (
    build_monai_cache_dataset,
    evaluate_models_on_loader,
    get_prediction,
    get_batch_predictions,
    average_predictions,
    compute_stds,
    AddBatchDimension,
    EnsurePIL,
)

# TTA methods
from .methods.tta import (
    TTAMethod,
    GPSMethod,
    TTA,
    apply_augmentations,
    apply_randaugment_and_store_results,
    extract_gps_augmentations_info,
)

# Ensemble methods
from .methods.ensemble import (
    EnsembleSTDMethod,
    ensembling_predictions,
    ensembling_stds_computation,
    ensembling_variance_computation,
)

# Distance-based methods
from .methods.distance import (
    DistanceToHardLabelsMethod,
    CalibrationMethod,
    TemperatureScaler,
    distance_to_hard_labels_computation,
    posthoc_calibration,
    fit_temperature_scaling,
    compute_class_weights,
)

# Latent space methods
from .methods.latent import (
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

# Greedy search
from .search.greedy import (
    perform_greedy_policy_search,
    select_greedily_on_ens,
    greedy_search,
    load_npz_files_for_greedy_search,
)

# Visualization
from .visualization.plots import (
    plot_calibration_curve,
    model_calibration_plot,
    UQ_method_plot,
    roc_curve_UQ_method_computation,
    roc_curve_UQ_methods_plot,
    plot_auc_curves,
    compare_uq_methods,
)
from .visualization.shap_viz import (
    visualize_input_shap_overlayed_multimodel,
    plot_shap_importance,
    plot_clustered_feature_heatmap,
    visualize_umap_with_labels,
)

# Evaluation metrics
from .evaluation.evaluation import (
    compute_auroc,
    compute_roc_curve,
    compute_aurc,
    compute_augrc,
    compute_all_metrics,
    compute_all_metrics_per_fold,
    plot_risk_coverage_curve,
    plot_roc_curve_failure_prediction,
    plot_uncertainty_distributions,
    save_all_evaluation_plots,
)

# Base classes
from .core.base import UQMethod, UQResult

__version__ = "2.0.0"

__all__ = [
    # Core
    "build_monai_cache_dataset",
    "evaluate_models_on_loader",
    "get_prediction",
    "get_batch_predictions",
    "average_predictions",
    "compute_stds",
    "AddBatchDimension",
    "EnsurePIL",
    # TTA
    "TTAMethod",
    "GPSMethod",
    "TTA",
    "apply_augmentations",
    "apply_randaugment_and_store_results",
    "extract_gps_augmentations_info",
    # Ensemble
    "EnsembleSTDMethod",
    "ensembling_predictions",
    "ensembling_stds_computation",
    "ensembling_variance_computation",
    # Distance
    "DistanceToHardLabelsMethod",
    "CalibrationMethod",
    "TemperatureScaler",
    "distance_to_hard_labels_computation",
    "posthoc_calibration",
    "fit_temperature_scaling",
    "compute_class_weights",
    # Latent
    "ClassifierHeadWrapper",
    "KNNLatentSHAPMethod",
    "KNNLatentMethod",
    "HyperplaneDistanceMethod",
    "get_layer_from_model",
    "extract_latent_space_and_compute_shap_importance",
    "compute_mean_shap_values",
    "display_shap_values",
    "feature_engineering_pipeline",
    "analyze_hyperplane_distance",
    "compute_knn_distances_to_train_data",
    # Search
    "perform_greedy_policy_search",
    "select_greedily_on_ens",
    "greedy_search",
    "load_npz_files_for_greedy_search",
    # Visualization
    "plot_calibration_curve",
    "model_calibration_plot",
    "UQ_method_plot",
    "roc_curve_UQ_method_computation",
    "roc_curve_UQ_methods_plot",
    "plot_auc_curves",
    "compare_uq_methods",
    "visualize_input_shap_overlayed_multimodel",
    "plot_shap_importance",
    "plot_clustered_feature_heatmap",
    "visualize_umap_with_labels",
    # Evaluation
    "compute_auroc",
    "compute_roc_curve",
    "compute_aurc",
    "compute_augrc",
    "compute_all_metrics",
    "compute_all_metrics_per_fold",
    "plot_risk_coverage_curve",
    "plot_roc_curve_failure_prediction",
    "plot_uncertainty_distributions",
    "save_all_evaluation_plots",
    # Base classes
    "UQMethod",
    "UQResult",
]