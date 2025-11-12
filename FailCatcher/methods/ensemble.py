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

def ensembling_stds_computation(individual_scores):
    """
    Compute standard deviation across ensemble predictions.
    
    Args:
        individual_scores: numpy array of shape [N, K, C] where:
            N = number of samples
            K = number of models
            C = number of classes
    
    Returns:
        list: Per-sample standard deviations (length N)
    
    Example:
        >>> # 100 samples, 5 models, 2 classes
        >>> scores = np.random.rand(100, 5, 2)
        >>> stds = ensembling_stds_computation(scores)
        >>> len(stds)  # 100
    """
    if not isinstance(individual_scores, np.ndarray):
        individual_scores = np.array(individual_scores)
    
    # Ensure shape is [N, K, C]
    if individual_scores.ndim != 3:
        raise ValueError(
            f"individual_scores must be 3D [N, K, C], got shape {individual_scores.shape}"
        )
    
    N, K, C = individual_scores.shape
    
    # Compute std across models (axis=1), then average across classes (axis=1 after reduction)
    stds_per_class = np.std(individual_scores, axis=1)  # [N, C]
    stds = np.mean(stds_per_class, axis=1)  # [N]
    
    return stds.tolist()


def ensembling_predictions(individual_scores):
    """
    Average predictions across ensemble models.
    
    Args:
        individual_scores: numpy array of shape [N, K, C]
    
    Returns:
        numpy.ndarray: Averaged predictions of shape [N, C]
    """
    if not isinstance(individual_scores, np.ndarray):
        individual_scores = np.array(individual_scores)
    
    # Average across models (axis=1)
    return np.mean(individual_scores, axis=1)  # [N, C]


def ensembling_variance_computation(individual_scores):
    """
    Compute variance across ensemble predictions.
    
    Args:
        individual_scores: numpy array of shape [N, K, C]
    
    Returns:
        list: Per-sample variances (length N)
    """
    if not isinstance(individual_scores, np.ndarray):
        individual_scores = np.array(individual_scores)
    
    # Compute variance across models (axis=1), then average across classes
    vars_per_class = np.var(individual_scores, axis=1)  # [N, C]
    vars = np.mean(vars_per_class, axis=1)  # [N]
    
    return vars.tolist()