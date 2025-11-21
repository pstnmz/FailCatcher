"""
Visualization utilities for uncertainty quantification.
Includes ROC curves, calibration plots, and UQ method comparisons.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================================
# CALIBRATION PLOTS
# ============================================================================

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Compute calibration curve data (fraction of positives vs predicted probability).
    
    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        tuple: (prob_true, prob_pred)
            - prob_true: Fraction of positives in each bin
            - prob_pred: Mean predicted probability in each bin
    
    Example:
        >>> prob_true, prob_pred = plot_calibration_curve(y_true, y_scores)
        >>> plt.plot(prob_pred, prob_true, 'o-')
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    return prob_true, prob_pred


def model_calibration_plot(true_labels, predictions, n_bins=20):
    """
    Plot calibration curve (reliability diagram) for model predictions.
    
    - Binary classification: standard calibration curve
    - Multiclass: top-1 reliability diagram (confidence vs accuracy)
    
    Args:
        true_labels: True class labels (np.ndarray)
        predictions: Model predictions
            - Binary: (N,) probabilities
            - Multiclass: (N, C) probability matrix
        n_bins: Number of bins for calibration
    
    Example:
        >>> # Binary
        >>> model_calibration_plot(y_true, y_probs)
        
        >>> # Multiclass
        >>> model_calibration_plot(y_true, softmax_probs)  # (N, 10)
    """
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        # Binary classification
        predictions_flat = predictions.ravel()
        prob_true, prob_pred = plot_calibration_curve(
            true_labels, predictions_flat, n_bins=n_bins
        )
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, 
                 label='Model Calibration')
        plt.xlabel('Predicted Probability', fontsize=14)
        plt.ylabel('Fraction of Positives', fontsize=14)
        plt.title('Calibration Curve (Binary Classification)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    else:
        # Multiclass: top-1 reliability diagram
        top1_preds = np.argmax(predictions, axis=1)
        top1_confs = np.max(predictions, axis=1)
        top1_correct = (top1_preds == true_labels).astype(int)
        
        prob_true, prob_pred = calibration_curve(
            top1_correct, top1_confs, n_bins=n_bins, strategy='uniform'
        )
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
                 label='Top-1 Reliability')
        plt.xlabel('Confidence (Max Probability)', fontsize=14)
        plt.ylabel('Accuracy (Fraction Correct)', fontsize=14)
        plt.title('Top-1 Reliability Diagram (Multiclass)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# UQ METHOD COMPARISON PLOTS
# ============================================================================

def UQ_method_plot(correct_predictions, incorrect_predictions, y_title, title, 
                   flag=None, swarmplot=False, save_path=None):
    """
    Plot boxplot comparing UQ metric distributions for correct vs incorrect predictions.
    
    Args:
        correct_predictions: UQ values for correctly classified samples
        incorrect_predictions: UQ values for incorrectly classified samples
        y_title: Y-axis label (e.g., "Std Deviation", "Distance")
        title: Plot title
        flag: Optional dataset identifier (for filename)
        swarmplot: Whether to overlay swarmplot (can be slow for large datasets)
        save_path: Optional file path to save plot
    
    Example:
        >>> UQ_method_plot(
        ...     stds[correct_idx], stds[incorrect_idx],
        ...     y_title="Ensemble STD", title="Ensemble Uncertainty",
        ...     flag="breastmnist"
        ... )
    """
    # Build DataFrame
    df = pd.DataFrame({
        y_title: list(correct_predictions) + list(incorrect_predictions),
        'Category': ['Success'] * len(correct_predictions) + ['Failure'] * len(incorrect_predictions)
    })
    
    plt.figure(figsize=(10, 6))
    
    # Boxplot
    sns.boxplot(x='Category', y=y_title, data=df, palette='Set2')
    
    # Optional swarmplot (warning: slow for large N)
    if swarmplot:
        if len(df) > 1000:
            print(f"Warning: swarmplot with {len(df)} points may be slow")
        sns.swarmplot(x='Category', y=y_title, data=df, color='k', alpha=0.3, size=3)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Outcome', fontsize=14)
    plt.ylabel(y_title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# ROC CURVE UTILITIES
# ============================================================================

def roc_curve_UQ_method_computation(correct_predictions, incorrect_predictions):
    """
    Compute ROC curve and AUC for uncertainty quantification method.
    
    Convention: Higher uncertainty should correspond to incorrect predictions.
    If the opposite is true, AUC is negated.
    
    Args:
        correct_predictions: UQ metric values for correct predictions
        incorrect_predictions: UQ metric values for incorrect predictions
    
    Returns:
        tuple: (fpr, tpr, auc_score)
            - fpr: False positive rates
            - tpr: True positive rates
            - auc_score: Area under ROC curve (signed; positive = good separation)
    
    Example:
        >>> fpr, tpr, auc = roc_curve_UQ_method_computation(
        ...     stds[correct_idx], stds[incorrect_idx]
        ... )
        >>> print(f"AUC: {auc:.3f}")
    """
    correct_predictions = np.asarray(correct_predictions).ravel()
    incorrect_predictions = np.asarray(incorrect_predictions).ravel()
    
    # Labels: 1 = failure (should have high uncertainty), 0 = success
    labels = np.concatenate([
        np.ones(len(incorrect_predictions)),
        np.zeros(len(correct_predictions))
    ])
    
    predictions = np.concatenate([incorrect_predictions, correct_predictions])
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_score = roc_auc_score(labels, predictions)
    
    # Sign convention: incorrect should have higher values
    # If incorrect has lower mean, flip AUC sign
    if np.mean(incorrect_predictions) < np.mean(correct_predictions):
        auc_score = -auc_score
    
    return fpr, tpr, auc_score


def roc_curve_UQ_methods_plot(method_names, fprs, tprs, auc_scores, title=None, save_path=None):
    """
    Plot ROC curves for multiple UQ methods on the same figure.
    
    Args:
        method_names: List of method names (e.g., ["Ensemble-STD", "TTA", "GPS"])
        fprs: List of false positive rate arrays
        tprs: List of true positive rate arrays
        auc_scores: List of AUC scores
        title: Optional custom title
        save_path: Optional file path to save plot
    
    Example:
        >>> roc_curve_UQ_methods_plot(
        ...     ["Ensemble", "TTA", "GPS"],
        ...     [fpr1, fpr2, fpr3],
        ...     [tpr1, tpr2, tpr3],
        ...     [auc1, auc2, auc3]
        ... )
    """
    plt.figure(figsize=(10, 8))
    
    # Plot each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))
    for fpr, tpr, auc_score, method_name, color in zip(fprs, tprs, auc_scores, method_names, colors):
        plt.plot(fpr, tpr, lw=2.5, label=f'{method_name} (AUC={auc_score:.3f})', color=color)
    
    # Reference diagonal
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title or 'ROC Curves for UQ Methods', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_auc_curves(results, title=None, save_path=None):
    """
    Plot AUC progression over iterations for greedy policy searches.
    
    Args:
        results: List of (best_metric, policy_indices, auc_history) tuples
        title: Optional custom title
        save_path: Optional file path to save plot
    
    Example:
        >>> results = [(0.85, [1,5,7], [0.6, 0.7, 0.85]), ...]
        >>> plot_auc_curves(results)
    """
    if not results:
        print("No results to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for idx, (best_metric, _, auc_history) in enumerate(results):
        if not auc_history:
            continue
        plt.plot(
            range(1, len(auc_history) + 1), 
            auc_history, 
            marker='o', 
            linewidth=2,
            markersize=5,
            label=f"Search {idx + 1} (best={best_metric:.3f})",
            color=colors[idx]
        )
    
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("ROC AUC", fontsize=14)
    plt.title(title or "ROC AUC Progress Over Iterations (Greedy Search)", 
              fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved AUC curves to {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# UTILITY: Compare multiple UQ methods
# ============================================================================

def compare_uq_methods(
    method_results, 
    correct_idx, 
    incorrect_idx,
    dataset_name=None,
    save_dir=None
):
    """
    Convenience function to generate all comparison plots for multiple UQ methods.
    
    Args:
        method_results: Dict mapping method names to UQ scores
            Example: {"Ensemble": stds, "TTA": tta_stds, "GPS": gps_stds}
        correct_idx: Indices of correct predictions
        incorrect_idx: Indices of incorrect predictions
        dataset_name: Optional dataset identifier for titles/filenames
        save_dir: Optional directory to save plots
    
    Example:
        >>> compare_uq_methods(
        ...     {"Ensemble": ens_stds, "TTA": tta_stds, "GPS": gps_stds},
        ...     correct_idx, incorrect_idx,
        ...     dataset_name="breastmnist"
        ... )
    """
    method_names = list(method_results.keys())
    fprs, tprs, aucs = [], [], []
    
    print(f"\n{'='*60}")
    print(f"UQ Method Comparison{f' ({dataset_name})' if dataset_name else ''}")
    print(f"{'='*60}")
    
    for method_name, scores in method_results.items():
        fpr, tpr, auc = roc_curve_UQ_method_computation(
            [scores[i] for i in correct_idx],
            [scores[i] for i in incorrect_idx]
        )
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc)
        print(f"{method_name:20s} AUC = {auc:.4f}")
    
    print(f"{'='*60}\n")
    
    # ROC curves
    roc_title = f"ROC Curves{f' - {dataset_name}' if dataset_name else ''}"
    roc_path = f"{save_dir}/roc_comparison.png" if save_dir else None
    roc_curve_UQ_methods_plot(method_names, fprs, tprs, aucs, title=roc_title, save_path=roc_path)
    
    # Individual boxplots
    for method_name, scores in method_results.items():
        box_title = f"{method_name} Uncertainty Distribution"
        box_path = f"{save_dir}/{method_name.lower()}_boxplot.png" if save_dir else None
        UQ_method_plot(
            [scores[i] for i in correct_idx],
            [scores[i] for i in incorrect_idx],
            y_title=f"{method_name} Score",
            title=box_title,
            flag=dataset_name,
            save_path=box_path
        )