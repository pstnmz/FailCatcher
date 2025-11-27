"""
Evaluation metrics for uncertainty quantification methods.

Includes:
- AUROC_f: Area Under ROC Curve (for failure prediction)
- AURC: Area Under Risk-Coverage Curve
- AUGRC: Area Under Generalized Risk-Coverage Curve (Traub et al., NeurIPS 2024)

AUGRC measures the average rate of silent failures (undetected errors) across all working points.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path


def compute_auroc(uncertainties, correct_idx, incorrect_idx):
    """
    Compute AUROC for failure prediction (classification: correct vs incorrect).
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        correct_idx: Indices of correctly classified samples
        incorrect_idx: Indices of incorrectly classified samples
    
    Returns:
        float: AUROC score (higher = better uncertainty quantification)
    """
    # Create binary labels: 1 = incorrect (failure), 0 = correct
    n_samples = len(uncertainties)
    labels = np.zeros(n_samples)
    labels[list(incorrect_idx)] = 1
    
    # AUROC: higher uncertainty should predict failures
    auroc = roc_auc_score(labels, uncertainties)
    
    return auroc


def compute_roc_curve(uncertainties, correct_idx, incorrect_idx):
    """
    Compute ROC curve for failure prediction.
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        correct_idx: Indices of correctly classified samples
        incorrect_idx: Indices of incorrectly classified samples
    
    Returns:
        tuple: (fpr, tpr, thresholds, auroc)
    """
    n_samples = len(uncertainties)
    labels = np.zeros(n_samples)
    labels[list(incorrect_idx)] = 1
    
    fpr, tpr, thresholds = roc_curve(labels, uncertainties)
    auroc = roc_auc_score(labels, uncertainties)
    
    return fpr, tpr, thresholds, auroc


def compute_aurc(uncertainties, predictions, labels, num_bins=1000):
    """
    Compute Area Under Risk-Coverage Curve (AURC).
    
    Risk-Coverage curve plots classification error vs coverage, where samples
    are rejected in order of decreasing uncertainty. AURC measures the quality
    of uncertainty estimates for selective prediction.
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        num_bins: Number of coverage bins (default: 1000)
    
    Returns:
        float: AURC score (lower = better, range [0, 1])
        dict: Additional metrics including:
            - coverages: Coverage values
            - risks: Risk (error rate) at each coverage
            - optimal_risk: Risk when covering only correct predictions
    """
    n_samples = len(uncertainties)
    
    # Sort by uncertainty (descending: most uncertain first)
    sorted_indices = np.argsort(uncertainties)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Compute cumulative errors and coverage
    errors = (sorted_predictions != sorted_labels).astype(float)
    
    # Coverage percentages
    coverages = []
    risks = []
    
    for n_rejected in range(n_samples + 1):
        if n_rejected == n_samples:
            # All samples rejected
            coverage = 0.0
            risk = 0.0
        else:
            # Keep samples from n_rejected onwards (reject most uncertain)
            kept_errors = errors[n_rejected:]
            coverage = (n_samples - n_rejected) / n_samples
            risk = kept_errors.mean() if len(kept_errors) > 0 else 0.0
        
        coverages.append(coverage)
        risks.append(risk)
    
    coverages = np.array(coverages)
    risks = np.array(risks)
    
    # Compute AURC using trapezoidal rule
    # Sort by coverage (ascending)
    sorted_idx = np.argsort(coverages)
    coverages = coverages[sorted_idx]
    risks = risks[sorted_idx]
    
    aurc = np.trapz(risks, coverages)
    
    # Optimal risk: if we could perfectly identify errors
    total_errors = errors.sum()
    optimal_risk = total_errors / n_samples
    
    return aurc, {
        'coverages': coverages,
        'risks': risks,
        'optimal_risk': optimal_risk,
        'n_samples': n_samples,
        'n_errors': int(total_errors)
    }


def compute_augrc(uncertainties, predictions, labels, num_bins=1000):
    """
    Compute Area Under Generalized Risk-Coverage Curve (AUGRC).
    
    From Traub et al. (NeurIPS 2024): "Overcoming Common Flaws in the Evaluation 
    of Selective Classification Systems"
    
    AUGRC evaluates selective classification in terms of the rate of silent failures
    (undetected errors) averaged over all working points. Unlike AURC, it uses the
    Generalized Risk P(fail, accept) instead of Selective Risk P(fail|accept).
    
    The formula is:
        AUGRC = (1 - AUROC_f) * acc * (1 - acc) + 0.5 * (1 - acc)^2
    
    Where:
    - AUROC_f: AUROC for failure prediction (how well uncertainties rank errors)
    - acc: Overall classification accuracy
    
    This metric:
    - Is bounded to [0, 0.5], where lower is better
    - Is directly interpretable as average risk of silent failures
    - Maintains monotonicity with respect to both AUROC_f and accuracy
    - Resolves the excessive weighting of high-confidence failures in AURC
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        num_bins: Number of coverage bins (default: 1000, not used but kept for compatibility)
    
    Returns:
        float: AUGRC score (0-0.5, lower is better)
        dict: Additional metrics including:
            - auroc_f: AUROC for failure prediction
            - acc: Overall accuracy
            - error_rate: Classification error rate (1 - acc)
            - aurc: Raw AURC value (for comparison)
            - coverages: Coverage values from risk-coverage curve
            - risks: Risk at each coverage
    """
    n_samples = len(uncertainties)
    errors = (predictions != labels).astype(float)
    n_errors = errors.sum()
    
    # Compute accuracy
    acc = 1.0 - (n_errors / n_samples)
    error_rate = n_errors / n_samples
    
    # Compute AUROC_f (AUROC for failure prediction)
    correct_idx = np.where(~errors.astype(bool))[0]
    incorrect_idx = np.where(errors.astype(bool))[0]
    
    if len(incorrect_idx) > 0 and len(correct_idx) > 0:
        auroc_f = compute_auroc(uncertainties, correct_idx, incorrect_idx)
    else:
        # Edge case: perfect model or completely wrong model
        auroc_f = 0.5  # Random baseline
    
    # Compute AUGRC using the formula from Traub et al. (Equation 7)
    # AUGRC = (1 - AUROC_f) * acc * (1 - acc) + 0.5 * (1 - acc)^2
    term1 = (1.0 - auroc_f) * acc * (1.0 - acc)
    term2 = 0.5 * (1.0 - acc) ** 2
    augrc = term1 + term2
    
    # Also compute AURC for comparison (legacy metric)
    aurc, aurc_metrics = compute_aurc(uncertainties, predictions, labels, num_bins)
    
    return augrc, {
        'auroc_f': float(auroc_f),
        'acc': float(acc),
        'error_rate': float(error_rate),
        'n_errors': int(n_errors),
        'aurc': float(aurc),  # For comparison with old metric
        'coverages': aurc_metrics['coverages'],
        'risks': aurc_metrics['risks']
    }


def compute_all_metrics(uncertainties, predictions, labels, correct_idx=None, incorrect_idx=None):
    """
    Compute all UQ evaluation metrics.
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        correct_idx: Optional indices of correct predictions (computed if None)
        incorrect_idx: Optional indices of incorrect predictions (computed if None)
    
    Returns:
        dict: All metrics including AUROC_f, AURC, AUGRC, and auxiliary information
    """
    # Compute correct/incorrect indices if not provided
    if correct_idx is None or incorrect_idx is None:
        errors = (predictions != labels)
        correct_idx = np.where(~errors)[0]
        incorrect_idx = np.where(errors)[0]
    
    # AUROC_f (for failure prediction)
    if len(incorrect_idx) > 0 and len(correct_idx) > 0:
        auroc_f = compute_auroc(uncertainties, correct_idx, incorrect_idx)
    else:
        auroc_f = np.nan
    
    # AURC and AUGRC
    aurc, aurc_metrics = compute_aurc(uncertainties, predictions, labels)
    augrc, augrc_metrics = compute_augrc(uncertainties, predictions, labels)
    
    return {
        'auroc_f': float(auroc_f),
        'aurc': float(aurc),
        'augrc': float(augrc),
        'accuracy': float(augrc_metrics['acc']),
        'error_rate': float(augrc_metrics['error_rate']),
        'n_correct': len(correct_idx),
        'n_incorrect': len(incorrect_idx),
        'n_total': len(uncertainties),
    }


def plot_risk_coverage_curve(uncertainties, predictions, labels, ax=None, 
                              show_optimal=True, save_path=None):
    """
    Plot Risk-Coverage curve (Selective Risk) and Generalized Risk.
    
    Baselines:
    - Oracle: Perfect UQ that ranks ALL errors as most uncertain (AUROC_f=1.0)
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        ax: Matplotlib axis (creates new figure if None)
        show_optimal: Whether to show oracle baseline (perfect ranking)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib figure and axis
    """
    
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig = ax.get_figure()
        ax1 = ax
        ax2 = None
    
    # Compute curves
    aurc, metrics = compute_aurc(uncertainties, predictions, labels)
    augrc, augrc_metrics = compute_augrc(uncertainties, predictions, labels)
    
    coverages = metrics['coverages']
    risks = metrics['risks']
    error_rate = augrc_metrics['error_rate']
    auroc_f = augrc_metrics['auroc_f']
    acc = augrc_metrics['acc']
    n_samples = len(uncertainties)
    n_errors = int(augrc_metrics['n_errors'])
    
    # Compute generalized risk: P(fail, accept) = P(fail | accept) * P(accept)
    # P(fail | accept) = risks, P(accept) = coverages
    generalized_risks = risks * coverages
    
    # ===== Plot 1: Selective Risk (standard risk-coverage curve) =====
    ax1.plot(coverages, risks, 'b-', linewidth=2.5, 
            label=f'UQ Method (AUROC_f={auroc_f:.3f})')
    
    # Plot oracle baseline (perfect ranking of failures)
    if show_optimal and n_errors > 0 and n_errors < n_samples:
        # Oracle: accept correct predictions first (low risk), then errors (risk increases)
        # At coverage (1 - error_rate): only correct predictions, risk = 0
        # At coverage 1.0: all predictions accepted, risk = error_rate
        cov_correct = (n_samples - n_errors) / n_samples
        oracle_cov = [0.001, cov_correct, 1.0]  # Small epsilon to avoid division issues
        oracle_risk = [0, 0, error_rate]
        ax1.plot(oracle_cov, oracle_risk, 'g--', linewidth=2, label='Oracle (perfect ranking)')
    
    ax1.set_xlabel('Coverage (fraction of predictions accepted)', fontsize=13)
    ax1.set_ylabel('Selective Risk: P(fail | accept)', fontsize=13)
    ax1.set_title('Risk-Coverage Curve\n(Error rate among accepted predictions)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, max(max(risks) * 1.1, error_rate * 1.1) if len(risks) > 0 and max(risks) > 0 else 1])
    
    # ===== Plot 2: Generalized Risk (rate of silent failures) =====
    if ax2 is not None:
        ax2.plot(coverages, generalized_risks, 'b-', linewidth=2.5, 
                label=f'UQ Method')
        
        # Oracle baseline for generalized risk
        if show_optimal and n_errors > 0 and n_errors < n_samples:
            # Oracle rejects all errors first → no silent failures until we accept them
            # Generalized risk = P(fail, accept) = risk * coverage
            # For oracle: risk=0 until coverage exceeds fraction of correct predictions
            cov_correct = (n_samples - n_errors) / n_samples
            oracle_gen_cov = np.array([0, cov_correct, 1.0])
            oracle_gen_risk = np.array([0, 0, error_rate * 1.0])  # Only errors accepted at end
            oracle_augrc = 0.5 * (1.0 - acc) ** 2
            ax2.plot(oracle_gen_cov, oracle_gen_risk, 'g--', linewidth=2, 
                    label=f'Oracle (AUGRC={oracle_augrc:.6f})')
        
        # Shade the area under the curve (this is AUGRC)
        ax2.fill_between(coverages, 0, generalized_risks, alpha=0.3, color='blue',
                        label=f'AUGRC = {augrc:.6f}')
        
        ax2.set_xlabel('Coverage (fraction of predictions accepted)', fontsize=13)
        ax2.set_ylabel('Generalized Risk: P(fail, accept)', fontsize=13)
        ax2.set_title('Generalized Risk-Coverage Curve\n(Absolute rate of silent failures across dataset)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        max_gen_risk = max(generalized_risks) if len(generalized_risks) > 0 else error_rate
        ax2.set_ylim([0, max(max_gen_risk * 1.1, error_rate * 0.5)])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved risk-coverage curves to {save_path}")
    
    return fig, (ax1, ax2)


def plot_roc_curve_failure_prediction(uncertainties, predictions, labels,
                                       method_name='UQ Method', save_path=None):
    """
    Plot ROC curve for failure prediction (AUROC_f).
    
    This shows how well the uncertainty scores separate correct from incorrect predictions.
    
    Baselines:
    - Oracle (perfect): AUROC_f = 1.0, separates all failures perfectly
    - Random: AUROC_f = 0.5, diagonal line (no discrimination)
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        method_name: Name of the UQ method for plot title
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib figure
    """
    errors = (predictions != labels)
    correct_idx = np.where(~errors)[0]
    incorrect_idx = np.where(errors)[0]
    
    # Compute ROC curve
    fpr, tpr, thresholds, auroc_f = compute_roc_curve(uncertainties, correct_idx, incorrect_idx)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUROC_f = {auroc_f:.4f}')
    
    # Plot diagonal (random baseline)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7)
    
    # Styling
    ax.set_xlabel('False Positive Rate\n(Fraction of correct predictions flagged as uncertain)', fontsize=12)
    ax.set_ylabel('True Positive Rate\n(Fraction of incorrect predictions flagged as uncertain)', fontsize=12)
    ax.set_title(f'ROC Curve for Failure Prediction\n{method_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    # Add interpretive text
    info_text = 'Baselines:\n'
    info_text += '• Oracle: AUROC_f = 1.0\n'
    info_text += '  (perfect separation)\n'
    info_text += '• Random: AUROC_f = 0.5\n'
    info_text += '  (diagonal, no skill)'
    
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve to {save_path}")
    
    return fig


def plot_uncertainty_distributions(uncertainties, predictions, labels, 
                                   method_name='UQ Method', save_path=None):
    """
    Plot uncertainty score distributions for correct vs incorrect predictions.
    
    Creates:
    1. Boxplot comparing distributions
    2. Histogram overlays
    3. Summary statistics
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        method_name: Name of the UQ method for plot title
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib figure
    """
    errors = (predictions != labels)
    correct_idx = ~errors
    incorrect_idx = errors
    
    unc_correct = uncertainties[correct_idx]
    unc_incorrect = uncertainties[incorrect_idx]
    
    # Compute statistics
    auroc_f = compute_auroc(uncertainties, 
                           np.where(correct_idx)[0], 
                           np.where(incorrect_idx)[0]) if len(unc_incorrect) > 0 and len(unc_correct) > 0 else np.nan
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Plot 1: Boxplot =====
    ax1 = axes[0]
    
    box_data = [unc_correct, unc_incorrect]
    box_labels = [f'Correct\n(n={len(unc_correct)})', 
                  f'Incorrect\n(n={len(unc_incorrect)})']
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                     notch=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                     medianprops=dict(linewidth=2, color='black'))
    
    # Color the boxes
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Uncertainty Score', fontsize=13)
    ax1.set_title(f'{method_name}\nUncertainty Distribution by Prediction Outcome', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f'AUROC_f = {auroc_f:.4f}\n\n'
    stats_text += f'Correct:\n  Mean: {unc_correct.mean():.3f}\n  Median: {np.median(unc_correct):.3f}\n'
    stats_text += f'  Std: {unc_correct.std():.3f}\n\n'
    stats_text += f'Incorrect:\n  Mean: {unc_incorrect.mean():.3f}\n  Median: {np.median(unc_incorrect):.3f}\n'
    stats_text += f'  Std: {unc_incorrect.std():.3f}'
    
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== Plot 2: Histogram overlay =====
    ax2 = axes[1]
    
    # Determine bins
    bins = np.linspace(min(uncertainties.min(), 0), 
                      uncertainties.max(), 50)
    
    ax2.hist(unc_correct, bins=bins, alpha=0.6, color='green', 
            label=f'Correct (n={len(unc_correct)})', density=True, edgecolor='black')
    ax2.hist(unc_incorrect, bins=bins, alpha=0.6, color='red', 
            label=f'Incorrect (n={len(unc_incorrect)})', density=True, edgecolor='black')
    
    # Add vertical lines for means
    ax2.axvline(unc_correct.mean(), color='darkgreen', linestyle='--', 
               linewidth=2, label=f'Correct mean: {unc_correct.mean():.3f}')
    ax2.axvline(unc_incorrect.mean(), color='darkred', linestyle='--', 
               linewidth=2, label=f'Incorrect mean: {unc_incorrect.mean():.3f}')
    
    ax2.set_xlabel('Uncertainty Score', fontsize=13)
    ax2.set_ylabel('Density', fontsize=13)
    ax2.set_title(f'{method_name}\nUncertainty Distribution Overlap', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved uncertainty distributions to {save_path}")
    
    return fig


def save_all_evaluation_plots(uncertainties, predictions, labels, 
                               method_name='UQ Method', output_dir='./figures'):
    """
    Generate and save all evaluation plots for a UQ method.
    
    Creates:
    1. ROC curve for failure prediction (AUROC_f)
    2. Risk-coverage curves (selective + generalized risk)
    3. Uncertainty distribution plots (boxplot + histogram)
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        method_name: Name of the UQ method
        output_dir: Directory to save figures
    
    Returns:
        dict: Paths to saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize method name for filename
    safe_name = method_name.replace(' ', '_').replace('/', '_')
    
    # Save ROC curve for failure prediction
    roc_path = output_dir / f'{safe_name}_roc_curve.png'
    fig_roc = plot_roc_curve_failure_prediction(uncertainties, predictions, labels,
                                                method_name=method_name,
                                                save_path=roc_path)
    plt.close(fig_roc)
    
    # Save risk-coverage curves
    rc_path = output_dir / f'{safe_name}_risk_coverage.png'
    fig_rc, _ = plot_risk_coverage_curve(uncertainties, predictions, labels, 
                                         save_path=rc_path)
    plt.close(fig_rc)
    
    # Save uncertainty distributions
    dist_path = output_dir / f'{safe_name}_distributions.png'
    fig_dist = plot_uncertainty_distributions(uncertainties, predictions, labels,
                                              method_name=method_name,
                                              save_path=dist_path)
    plt.close(fig_dist)
    
    return {
        'roc_curve': str(roc_path),
        'risk_coverage': str(rc_path),
        'distributions': str(dist_path)
    }
