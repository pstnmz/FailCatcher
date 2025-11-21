from abc import ABC, abstractmethod
import numpy as np

class UQMethod(ABC):
    """Base class for all UQ methods."""
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute(self, **kwargs) -> np.ndarray:
        """Return per-sample uncertainty scores (higher = more uncertain)."""
        pass

class UQResult:
    """Container for UQ evaluation results."""
    def __init__(self, scores, correct_idx, incorrect_idx):
        self.scores = scores
        self.correct_idx = correct_idx
        self.incorrect_idx = incorrect_idx
        self.auc = None  # computed lazily
    
    def compute_auc(self):
        from ..methods.distance import roc_curve_UQ_method_computation
        _, _, auc = roc_curve_UQ_method_computation(
            [self.scores[i] for i in self.correct_idx],
            [self.scores[i] for i in self.incorrect_idx]
        )
        self.auc = auc
        return auc