"""
Ensemble-based uncertainty quantification methods.
Compute uncertainty from prediction variance across multiple models.
"""
import numpy as np
import torch

from ..core.base import UQMethod
from ..core.utils import evaluate_models_on_loader, get_prediction


# ============================================================================
# CLASS-BASED API (new, recommended)
# ============================================================================

class EnsembleSTDMethod(UQMethod):
    """
    Uncertainty quantification via standard deviation across ensemble models.
    
    Higher std indicates higher disagreement between models → higher uncertainty.
    """
    def __init__(self):
        super().__init__("Ensemble-STD")
    
    def compute(self, models, data_loader, device):
        """
        Compute per-sample uncertainty as std across ensemble predictions.
        
        Args:
            models: List of PyTorch models
            data_loader: DataLoader for evaluation
            device: torch.device
        
        Returns:
            np.ndarray: Per-sample uncertainty scores (N,)
        """
        _, _, _, correct_idx, incorrect_idx, indiv_scores = evaluate_models_on_loader(
            models, data_loader, device
        )
        stds = ensembling_stds_computation(indiv_scores)
        return stds


# ============================================================================
# FUNCTIONAL API (existing, for backward compatibility)
# ============================================================================

def ensembling_predictions(models, image, device=None):
    """
    Get predictions from all models in an ensemble for a single image.
    
    Args:
        models: List of PyTorch models
        image: Input image tensor or PIL image
        device: torch.device (optional, inferred from image if not provided)
    
    Returns:
        list: Predictions from each model
    
    Example:
        >>> preds = ensembling_predictions(models, image, device)
        >>> # preds: [tensor([0.8]), tensor([0.75]), tensor([0.82])]
    """
    if device is None and isinstance(image, torch.Tensor):
        device = image.device
    
    ensemble_preds = [get_prediction(model, image, device) for model in models]
    return ensemble_preds


def ensembling_stds_computation(models_predictions):
    """
    Compute standard deviation of model predictions for ensembling.
    
    Args:
        models_predictions: Array of predictions with shape:
            - (num_models, num_samples) for binary classification
            - (num_models, num_samples, num_classes) for multiclass
    
    Returns:
        np.ndarray: Per-sample uncertainty scores
            - Binary: std across models (N,)
            - Multiclass: mean std across classes (N,)
    
    Raises:
        ValueError: If array is not 2D or 3D
    
    Example:
        >>> # Binary case: 5 models, 100 samples
        >>> preds = np.random.rand(5, 100)
        >>> stds = ensembling_stds_computation(preds)
        >>> stds.shape  # (100,)
        
        >>> # Multiclass case: 5 models, 100 samples, 10 classes
        >>> preds = np.random.rand(5, 100, 10)
        >>> stds = ensembling_stds_computation(preds)
        >>> stds.shape  # (100,)
    """
    models_predictions = np.asarray(models_predictions)
    
    if models_predictions.ndim == 2:
        # Binary classification: (num_models, num_samples)
        stds = np.std(models_predictions, axis=0)
    
    elif models_predictions.ndim == 3:
        # Multiclass: (num_models, num_samples, num_classes)
        # Compute std per class, then average across classes
        class_wise_stds = np.std(models_predictions, axis=0)  # (num_samples, num_classes)
        stds = np.mean(class_wise_stds, axis=1)  # (num_samples,)
    
    else:
        raise ValueError(
            f"Unexpected shape {models_predictions.shape}. "
            "Expected 2D (binary) or 3D (multiclass) array."
        )
    
    return stds


def ensembling_variance_computation(models_predictions):
    """
    Compute variance of model predictions (alternative to std).
    
    Args:
        models_predictions: Same format as ensembling_stds_computation
    
    Returns:
        np.ndarray: Per-sample variance scores
    
    Note:
        Variance = std², more sensitive to outliers.
    """
    models_predictions = np.asarray(models_predictions)
    
    if models_predictions.ndim == 2:
        variances = np.var(models_predictions, axis=0)
    
    elif models_predictions.ndim == 3:
        class_wise_vars = np.var(models_predictions, axis=0)
        variances = np.mean(class_wise_vars, axis=1)
    
    else:
        raise ValueError(
            f"Unexpected shape {models_predictions.shape}. "
            "Expected 2D (binary) or 3D (multiclass) array."
        )
    
    return variances