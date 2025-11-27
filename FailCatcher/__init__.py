"""
UQ_Toolbox: Uncertainty Quantification for Deep Learning

A modular toolkit for quantifying prediction uncertainty using:
- Test-Time Augmentation (TTA)
- Greedy Policy Search (GPS)
- Ensemble methods
- Distance-based metrics
- Latent space analysis (SHAP, KNN, SVM)
- Calibration methods (Platt, Isotonic, Temperature Scaling)
- Evaluation metrics (AUROC, AURC, AUGRC)
"""

# Import from relative paths (we're already inside UQ_Toolbox/)
from .UQ_toolbox import *
from .core.base import UQMethod, UQResult
from .evaluation import evaluation

__version__ = "2.0.0"