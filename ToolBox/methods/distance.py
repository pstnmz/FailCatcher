"""
Distance-based uncertainty quantification methods.
Includes distance to hard labels, calibration methods (Platt, Isotonic, Temperature Scaling).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from ..core.base import UQMethod


# ============================================================================
# CLASS-BASED API (new, recommended)
# ============================================================================

class DistanceToHardLabelsMethod(UQMethod):
    """
    Uncertainty quantification via distance to hard decision boundaries.
    
    For binary: distance = 0.5 - |pred - 0.5| (max uncertainty at 0.5)
    For multiclass: distance = 1 - max(probs) (max uncertainty when uniform)
    """
    def __init__(self):
        super().__init__("DistanceToHardLabels")
    
    def compute(self, predictions):
        """
        Compute per-sample uncertainty as distance to hard labels.
        
        Args:
            predictions: np.ndarray of shape (N,) for binary or (N, C) for multiclass
        
        Returns:
            np.ndarray: Per-sample uncertainty scores (N,)
        """
        return distance_to_hard_labels_computation(predictions)


class CalibrationMethod(UQMethod):
    """
    Post-hoc calibration wrapper (Platt, Isotonic, Temperature Scaling).
    """
    def __init__(self, method='platt'):
        """
        Args:
            method: 'platt', 'isotonic', or 'temperature'
        """
        super().__init__(f"Calibration-{method}")
        self.method = method
        self.model = None
    
    def fit(self, y_scores, y_true):
        """
        Fit calibration model on validation set.
        
        Args:
            y_scores: Predicted probabilities or logits
            y_true: True labels
        
        Returns:
            self (for chaining)
        """
        _, self.model = posthoc_calibration(y_scores, y_true, self.method)
        return self
    
    def compute(self, y_scores):
        """
        Apply calibration to scores.
        
        Args:
            y_scores: Predicted probabilities or logits
        
        Returns:
            np.ndarray: Calibrated probabilities
        """
        if self.model is None:
            raise RuntimeError("Call fit() before compute()")
        
        if self.method == 'temperature':
            logits_tensor = torch.from_numpy(y_scores).float()
            if logits_tensor.ndim == 1:
                logits_tensor = logits_tensor.unsqueeze(1)
            calibrated_logits = self.model(logits_tensor).detach()
            
            if calibrated_logits.ndim == 1 or calibrated_logits.shape[1] == 1:
                return torch.sigmoid(calibrated_logits).numpy().squeeze()
            else:
                return torch.softmax(calibrated_logits, dim=1).numpy()
        
        elif self.method == 'platt':
            return self.model.predict_proba(y_scores.reshape(-1, 1))[:, 1]
        
        elif self.method == 'isotonic':
            return self.model.predict(y_scores)


# ============================================================================
# TEMPERATURE SCALING COMPONENTS
# ============================================================================

class TemperatureScaler(nn.Module):
    """
    Temperature scaling module for calibration.
    
    Divides logits by learned temperature T:
        calibrated_logits = logits / T
    
    Temperature is parameterized in log-space for numerical stability and 
    constrained to reasonable range [0.1, 10.0] to prevent extreme values.
    """
    def __init__(self, init_temp=1.0):
        super().__init__()
        # Initialize at T=1.0 (no scaling) - standard practice
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([init_temp])))

    def forward(self, logits):
        """Apply temperature scaling to logits."""
        temperature = torch.exp(self.log_temperature)
        # Wider bounds allow for poorly calibrated models
        temperature = torch.clamp(temperature, min=0.1, max=10.0)
        return logits / temperature


def compute_class_weights(labels_np):
    """
    Compute inverse frequency class weights for imbalanced datasets.
    
    Args:
        labels_np: np.ndarray of integer labels
    
    Returns:
        torch.Tensor: Class weights (one per class)
    
    Example:
        >>> labels = np.array([0, 0, 0, 1, 1, 2])
        >>> weights = compute_class_weights(labels)
        >>> # weights ≈ [0.67, 1.0, 2.0] (inverse frequency)
    """
    classes, counts = np.unique(labels_np, return_counts=True)
    total = sum(counts)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float)


def fit_temperature_scaling(logits, labels, max_iter=1000):
    """
    Fit temperature scaling parameter using LBFGS optimization.
    
    Args:
        logits: np.ndarray of raw model outputs (N,) or (N, C)
        labels: np.ndarray of true labels
        max_iter: Maximum optimization iterations
    
    Returns:
        TemperatureScaler: Trained temperature scaling model
    
    Example:
        >>> logits = np.random.randn(100, 10)  # 100 samples, 10 classes
        >>> labels = np.random.randint(0, 10, 100)
        >>> model = fit_temperature_scaling(logits, labels)
        >>> calibrated_logits = model(torch.from_numpy(logits).float())
    """
    logits = torch.from_numpy(logits).float()
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)

    labels_np = labels.copy()
    labels = torch.from_numpy(labels).float()
    model = TemperatureScaler()

    if logits.shape[1] == 1:
        # Binary classification - use unweighted loss for standard calibration
        criterion = nn.BCEWithLogitsLoss()
    else:
        # Multiclass classification - use unweighted loss for standard calibration
        # Temperature scaling aims to match predicted probabilities to empirical frequencies,
        # not to handle class imbalance (that's what class_weight during training is for)
        labels = labels.long()
        criterion = nn.CrossEntropyLoss()

    # Optimize log-temperature using LBFGS with higher learning rate
    optimizer = optim.LBFGS([model.log_temperature], lr=0.01, max_iter=max_iter)
    
    print(f"Initial log-temperature: {model.log_temperature.item():.4f}")
    print(f"Initial temperature: {torch.exp(model.log_temperature).item():.4f}")
    
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(logits).squeeze(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    
    print(f"Optimized log-temperature: {model.log_temperature.item():.4f}")
    print(f"Optimized temperature: {torch.exp(model.log_temperature).item():.4f}")
    
    return model


# ============================================================================
# FUNCTIONAL API (existing, for backward compatibility)
# ============================================================================

def _should_use_balanced_platt(y_true, min_samples=200, min_imbalance_ratio=2.0):
    """
    Determine whether to use class_weight='balanced' for Platt scaling.
    
    Rule: Use balanced weighting when:
    1. Calibration set is large enough (>= min_samples)
    2. Classes are moderately to highly imbalanced (ratio >= min_imbalance_ratio)
    
    Args:
        y_true: True labels from calibration set
        min_samples: Minimum calibration set size to consider balanced (default: 200)
        min_imbalance_ratio: Minimum class imbalance ratio to use balanced (default: 2.0)
    
    Returns:
        'balanced' or None
    """
    n_samples = len(y_true)
    class_counts = np.bincount(y_true)
    
    # Compute imbalance ratio (majority / minority)
    majority_count = np.max(class_counts)
    minority_count = np.min(class_counts)
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else 1.0
    
    # Use balanced if dataset is large enough AND imbalanced
    use_balanced = (n_samples >= min_samples) and (imbalance_ratio >= min_imbalance_ratio)
    
    return 'balanced' if use_balanced else None


def posthoc_calibration(y_scores, y_true, method_calibration='platt', 
                       auto_tune_platt=False, platt_val_ratio=0.3, verbose=False):
    """
    Perform post-hoc calibration using Platt scaling, isotonic regression, or temperature scaling.

    Args:
        y_scores: Predicted probabilities (or logits if method='temperature')
            - Binary: (N,) or (N, 2) where [:,1] is positive class
            - Multiclass: (N, C)
        y_true: True labels (np.ndarray)
        method_calibration: 'platt', 'isotonic', or 'temperature'
        auto_tune_platt: If True and method_calibration='platt', automatically determine
            class_weight based on calibration set size and class imbalance
        platt_val_ratio: (Deprecated, unused)
        verbose: Print configuration if True

    Returns:
        tuple: (calibrated_probs, calibration_model)
            - calibrated_probs: Calibrated probabilities
                - Binary: (N,) array of positive class probabilities
                - Multiclass: (N, C) array of class probabilities
            - calibration_model: Fitted calibration model
    
    Note:
        Auto-tuning uses simple heuristic: class_weight='balanced' if calibration set
        has >= 200 samples AND imbalance ratio >= 2.0, otherwise None.
    """
    # Ensure y_scores is numpy array
    y_scores = np.asarray(y_scores)
    
    # Determine if binary or multiclass
    num_classes = len(np.unique(y_true))
    is_binary = num_classes == 2
    
    # For binary classification with shape (N, 2), extract positive class probability
    if y_scores.ndim == 2 and y_scores.shape[1] == 2:
        y_scores_input = y_scores[:, 1]  # Positive class probability
    else:
        y_scores_input = y_scores.ravel() if y_scores.ndim == 1 else y_scores
    
    if method_calibration == 'temperature':
        model = fit_temperature_scaling(y_scores, y_true)
        logits_tensor = torch.from_numpy(y_scores).float()
        if logits_tensor.ndim == 1:
            logits_tensor = logits_tensor.unsqueeze(1)
        calibrated_logits = model(logits_tensor).detach()

        # Binary or multiclass?
        if calibrated_logits.ndim == 1 or calibrated_logits.shape[1] == 1:
            # Binary: return (N,) probabilities
            calibrated_probs = torch.sigmoid(calibrated_logits).numpy().squeeze()
        else:
            # Multiclass: return (N, C) probabilities
            calibrated_probs = torch.softmax(calibrated_logits, dim=1).numpy()

    elif method_calibration == 'platt':
        # Automatic selection based on dataset characteristics
        # Use balanced weighting for large, imbalanced datasets
        # Use unweighted (None) for small or balanced datasets
        if auto_tune_platt:
            class_weight = _should_use_balanced_platt(y_true)
            if verbose:
                n_samples = len(y_true)
                class_counts = np.bincount(y_true)
                ratio = np.max(class_counts) / np.min(class_counts)
                weight_str = class_weight if class_weight else 'None'
                print(f"Platt config: N={n_samples}, imbalance={ratio:.2f} → class_weight={weight_str}")
        else:
            # Default: no class weighting (works better for small calibration sets)
            class_weight = None
        
        # Use moderate regularization (C=1.0) as default
        model = LogisticRegression(C=1.0, class_weight=class_weight, max_iter=1000)
        model.fit(y_scores_input.reshape(-1, 1), y_true)
        calibrated_probs = model.predict_proba(y_scores_input.reshape(-1, 1))[:, 1]

    elif method_calibration == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(y_scores_input, y_true)
        calibrated_probs = model.predict(y_scores_input)

    else:
        raise ValueError("Invalid method. Choose 'platt', 'isotonic', or 'temperature'.")

    # Compute Brier score (only for binary classification)
    if is_binary:
        # For binary, calibrated_probs is (N,) - probability of positive class
        y_prob_true = calibrated_probs
        brier = brier_score_loss(y_true, y_prob_true)
        print(f"Brier Score Loss ({method_calibration}): {brier:.4f}")
    else:
        # For multiclass, calibrated_probs is (N, C) - no Brier score
        print(f"Calibration complete ({method_calibration}) - Brier score not computed for multiclass")

    return calibrated_probs, model


def distance_to_hard_labels_computation(predictions):
    """
    Compute distance to hard decision boundaries as uncertainty metric.

    Args:
        predictions: Predicted probabilities
            - Binary: (N,) or (N, 1)
            - Multiclass: (N, C)

    Returns:
        np.ndarray: Per-sample uncertainty scores (N,)
            - Binary: 0.5 - |pred - 0.5| (max=0.5 at pred=0.5, min=0 at pred=0/1)
            - Multiclass: 1 - max(probs) (max=1-1/C for uniform, min=0 for confident)
    
    Example:
        >>> # Binary: confident predictions have low distance
        >>> preds = np.array([0.1, 0.5, 0.9])
        >>> dist = distance_to_hard_labels_computation(preds)
        >>> # [0.4, 0.5, 0.4] → middle prediction is most uncertain
        
        >>> # Multiclass
        >>> preds = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])
        >>> dist = distance_to_hard_labels_computation(preds)
        >>> # [0.3, 0.6] → second prediction is more uncertain
    """
    predictions = np.asarray(predictions)
    
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        # Binary classification
        predictions = predictions.ravel()
        distances = 0.5 - np.abs(predictions - 0.5)
    else:
        # Multiclass classification
        distances = 1.0 - np.max(predictions, axis=1)

    return distances


def maximum_logit_score_computation(logits):
    """
    Compute Maximum Logit Score (MLS) as uncertainty metric.
    
    MLS uses the negative of the maximum unnormalized logit as a confidence scoring function (CSF).
    Higher logit values indicate more confident predictions, so negating gives uncertainty.
    
    Args:
        logits: Raw logits (unnormalized scores) from model
            - Binary: (N,) or (N, 1) single logit value
            - Multiclass: (N, C) logits for each class
    
    Returns:
        np.ndarray: Per-sample uncertainty scores (N,)
            - Negative of maximum logit: -max(logits)
            - Higher values = more uncertain (lower max logit)
            - Lower values = more confident (higher max logit)
    
    Example:
        >>> # Binary with single logit
        >>> logits = np.array([3.0, 0.5, -2.0])  # Confident pos, uncertain, confident neg
        >>> unc = maximum_logit_score_computation(logits)
        >>> # [-3.0, -0.5, 2.0] → negative logit is uncertain
        
        >>> # Multiclass
        >>> logits = np.array([[5.0, 1.0, 0.5], [2.0, 1.8, 1.5]])  # Confident, uncertain
        >>> unc = maximum_logit_score_computation(logits)
        >>> # [-5.0, -2.0] → first is more confident (lower uncertainty)
    """
    logits = np.asarray(logits)
    
    if logits.ndim == 1:
        # Binary with single logit: use absolute value of logit for confidence
        uncertainties = -np.abs(logits)
    elif logits.shape[1] == 1:
        # Binary with shape (N, 1)
        logits = logits.ravel()
        uncertainties = -np.abs(logits)
    else:
        # Multiclass: negative of maximum logit
        uncertainties = -np.max(logits, axis=1)
    
    return uncertainties