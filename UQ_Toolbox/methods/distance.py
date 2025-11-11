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
    
    Temperature is parameterized in log-space for stability and constrained to [0.5, 5.0].
    """
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([init_temp])))

    def forward(self, logits):
        """Apply temperature scaling to logits."""
        temperature = torch.exp(self.log_temperature)
        temperature = torch.clamp(temperature, min=0.5, max=5.0)
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
        # Binary classification
        class_weights = compute_class_weights(labels_np)
        # Map class weights to sample weights based on labels
        sample_weights = torch.where(labels == 1, class_weights[1], class_weights[0])
        criterion = nn.BCEWithLogitsLoss(weight=sample_weights)
    else:
        # Multiclass classification
        labels = labels.long()
        class_weights = compute_class_weights(labels_np)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimize log-temperature using LBFGS
    optimizer = optim.LBFGS([model.log_temperature], lr=0.001, max_iter=max_iter)
    
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

def posthoc_calibration(y_scores, y_true, method_calibration='platt'):
    """
    Perform post-hoc calibration using Platt scaling, isotonic regression, or temperature scaling.

    Args:
        y_scores: Predicted probabilities (or logits if method='temperature')
        y_true: True labels (np.ndarray)
        method_calibration: 'platt', 'isotonic', or 'temperature'

    Returns:
        tuple: (calibrated_probs, calibration_model)
            - calibrated_probs: Calibrated probabilities (np.ndarray)
            - calibration_model: Fitted calibration model
    
    Example:
        >>> # Platt scaling
        >>> probs, model = posthoc_calibration(y_scores, y_true, 'platt')
        
        >>> # Temperature scaling (requires logits)
        >>> logits = model.predict(X)  # raw outputs before softmax
        >>> cal_probs, temp_model = posthoc_calibration(logits, y_true, 'temperature')
    """
    if method_calibration == 'temperature':
        model = fit_temperature_scaling(y_scores, y_true)
        logits_tensor = torch.from_numpy(y_scores).float()
        if logits_tensor.ndim == 1:
            logits_tensor = logits_tensor.unsqueeze(1)
        calibrated_logits = model(logits_tensor).detach()

        # Binary or multiclass?
        if calibrated_logits.ndim == 1 or calibrated_logits.shape[1] == 1:
            calibrated_probs = torch.sigmoid(calibrated_logits).numpy()
            y_prob_true = calibrated_probs.squeeze()
        else:
            calibrated_probs = torch.softmax(calibrated_logits, dim=1).numpy()
            y_prob_true = calibrated_probs[np.arange(len(y_true)), y_true]

    elif method_calibration == 'platt':
        model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000)
        model.fit(y_scores.reshape(-1, 1), y_true)
        calibrated_probs = model.predict_proba(y_scores.reshape(-1, 1))[:, 1]
        y_prob_true = calibrated_probs

    elif method_calibration == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(y_scores, y_true)
        calibrated_probs = model.predict(y_scores)
        y_prob_true = calibrated_probs

    else:
        raise ValueError("Invalid method. Choose 'platt', 'isotonic', or 'temperature'.")

    # Compute Brier score
    if y_scores.ndim > 1 and y_scores.shape[1] > 1:
        # Multiclass: compare true class probability
        brier = brier_score_loss(y_true, y_prob_true)
    else:
        # Binary
        brier = brier_score_loss(y_true, y_prob_true)
    
    print(f"Brier Score Loss ({method_calibration}): {brier:.4f}")

    return y_prob_true, model


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