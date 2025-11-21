"""
Visualization utilities for uncertainty quantification.
"""
from .plots import (
    plot_calibration_curve,
    model_calibration_plot,
    UQ_method_plot,
    roc_curve_UQ_method_computation,
    roc_curve_UQ_methods_plot,
    plot_auc_curves,
    compare_uq_methods,
)
from .shap_viz import (
    visualize_input_shap_overlayed_multimodel,
    plot_shap_importance,
    plot_clustered_feature_heatmap,
    visualize_umap_with_labels,
)

__all__ = [
    # Calibration plots
    "plot_calibration_curve",
    "model_calibration_plot",
    # UQ comparison
    "UQ_method_plot",
    "roc_curve_UQ_method_computation",
    "roc_curve_UQ_methods_plot",
    "plot_auc_curves",
    "compare_uq_methods",
    # SHAP visualization
    "visualize_input_shap_overlayed_multimodel",
    "plot_shap_importance",
    "plot_clustered_feature_heatmap",
    "visualize_umap_with_labels",
]