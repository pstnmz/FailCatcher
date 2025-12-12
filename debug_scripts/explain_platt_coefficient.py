"""
Deep dive into why Platt scaling gives such a small coefficient.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

project_root = Path(__file__).parent.parent

# Load calibration set
cache_dir = project_root / 'uq_benchmark_results' / 'cache'
calib_cache = np.load(cache_dir / 'breastmnist_resnet18_calib_results.npz')

y_true_calib = calib_cache['y_true']
y_scores_calib = calib_cache['y_scores']  # [N_calib, 2]

print("="*70)
print("UNDERSTANDING PLATT SCALING")
print("="*70)

# Extract positive class probability
calib_pos_prob = y_scores_calib[:, 1]

print(f"\nCalibration set: {len(y_true_calib)} samples")
print(f"  Label distribution: Class 0={np.sum(y_true_calib==0)}, Class 1={np.sum(y_true_calib==1)}")
print(f"  Imbalance ratio: {np.sum(y_true_calib==1) / len(y_true_calib):.2%}")

# Analyze model predictions by true label
class0_preds = calib_pos_prob[y_true_calib == 0]
class1_preds = calib_pos_prob[y_true_calib == 1]

print(f"\nModel predictions for TRUE class 0 (negative):")
print(f"  Mean: {class0_preds.mean():.4f}, Std: {class0_preds.std():.4f}")
print(f"  Min: {class0_preds.min():.4f}, Max: {class0_preds.max():.4f}")
print(f"  Model correctly predicts <0.5: {np.sum(class0_preds < 0.5)}/{len(class0_preds)} ({np.sum(class0_preds < 0.5)/len(class0_preds)*100:.1f}%)")

print(f"\nModel predictions for TRUE class 1 (positive):")
print(f"  Mean: {class1_preds.mean():.4f}, Std: {class1_preds.std():.4f}")
print(f"  Min: {class1_preds.min():.4f}, Max: {class1_preds.max():.4f}")
print(f"  Model correctly predicts >0.5: {np.sum(class1_preds > 0.5)}/{len(class1_preds)} ({np.sum(class1_preds > 0.5)/len(class1_preds)*100:.1f}%)")

# Check calibration: do predictions match empirical frequencies?
print("\n" + "="*70)
print("CALIBRATION ANALYSIS: Do predicted probabilities match true frequencies?")
print("="*70)

bins_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
for i in range(len(bins_edges)-1):
    low, high = bins_edges[i], bins_edges[i+1]
    mask = (calib_pos_prob >= low) & (calib_pos_prob < high)
    n_samples = np.sum(mask)
    if n_samples > 0:
        mean_pred = calib_pos_prob[mask].mean()
        true_freq = np.sum(y_true_calib[mask] == 1) / n_samples
        print(f"Predicted prob {low:.1f}-{high:.1f}: n={n_samples:3d}, mean_pred={mean_pred:.3f}, true_freq={true_freq:.3f}, gap={true_freq-mean_pred:+.3f}")
    else:
        print(f"Predicted prob {low:.1f}-{high:.1f}: n=  0 (no samples)")

# Now fit Platt scaling
print("\n" + "="*70)
print("FITTING PLATT SCALING")
print("="*70)

# Standard Platt (what the code does)
platt_model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000)
platt_model.fit(calib_pos_prob.reshape(-1, 1), y_true_calib)

print(f"\nPlatt model (C=0.01, class_weight='balanced'):")
print(f"  Coefficient (a): {platt_model.coef_[0][0]:.4f}")
print(f"  Intercept (b): {platt_model.intercept_[0]:.4f}")
print(f"  Formula: P(y=1) = sigmoid({platt_model.coef_[0][0]:.4f} * pred + {platt_model.intercept_[0]:.4f})")

# What happens with different regularization?
print("\nEffect of regularization (C parameter):")
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    model = LogisticRegression(C=C, class_weight='balanced', max_iter=1000)
    model.fit(calib_pos_prob.reshape(-1, 1), y_true_calib)
    print(f"  C={C:6.3f}: coef={model.coef_[0][0]:7.4f}, intercept={model.intercept_[0]:7.4f}")

# Without class weighting?
print("\nWithout class_weight='balanced':")
for C in [0.01, 1.0, 100.0]:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(calib_pos_prob.reshape(-1, 1), y_true_calib)
    print(f"  C={C:6.3f}: coef={model.coef_[0][0]:7.4f}, intercept={model.intercept_[0]:7.4f}")

# Apply calibration with current model
calibrated_probs = platt_model.predict_proba(calib_pos_prob.reshape(-1, 1))[:, 1]

print(f"\nAfter Platt calibration:")
print(f"  Min: {calibrated_probs.min():.4f}, Max: {calibrated_probs.max():.4f}")
print(f"  Mean: {calibrated_probs.mean():.4f} (vs true frequency {np.mean(y_true_calib):.4f})")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original predictions vs true labels
ax = axes[0, 0]
ax.scatter(calib_pos_prob[y_true_calib==0], np.zeros(np.sum(y_true_calib==0)), 
           alpha=0.5, label='True class 0', s=30)
ax.scatter(calib_pos_prob[y_true_calib==1], np.ones(np.sum(y_true_calib==1)), 
           alpha=0.5, label='True class 1', s=30)
ax.axvline(0.5, color='red', linestyle='--', alpha=0.3, label='Decision boundary')
ax.set_xlabel('Model prediction (P(y=1))')
ax.set_ylabel('True label')
ax.set_title('Original Model Predictions')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Calibration curve
ax = axes[0, 1]
bin_centers = []
bin_true_freqs = []
bin_pred_means = []
for i in range(len(bins_edges)-1):
    low, high = bins_edges[i], bins_edges[i+1]
    mask = (calib_pos_prob >= low) & (calib_pos_prob < high)
    if np.sum(mask) > 0:
        bin_centers.append((low + high) / 2)
        bin_pred_means.append(calib_pos_prob[mask].mean())
        bin_true_freqs.append(np.mean(y_true_calib[mask]))

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
ax.plot(bin_pred_means, bin_true_freqs, 'o-', markersize=10, linewidth=2, label='Model calibration')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True frequency')
ax.set_title('Calibration Curve (Reliability Diagram)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Plot 3: Platt transformation
ax = axes[1, 0]
x = np.linspace(0, 1, 100)
y = platt_model.predict_proba(x.reshape(-1, 1))[:, 1]
ax.plot(x, y, 'b-', linewidth=2, label=f'Platt: sigmoid({platt_model.coef_[0][0]:.2f}*x + {platt_model.intercept_[0]:.2f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change (y=x)')
ax.scatter(calib_pos_prob, calibrated_probs, alpha=0.3, s=20, label='Calibration data')
ax.set_xlabel('Original prediction')
ax.set_ylabel('Calibrated prediction')
ax.set_title('Platt Transformation')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Before/After histograms
ax = axes[1, 1]
ax.hist(calib_pos_prob, bins=20, alpha=0.5, label='Before calibration', color='blue')
ax.hist(calibrated_probs, bins=20, alpha=0.5, label='After calibration', color='red')
ax.axvline(0.5, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Count')
ax.set_title('Distribution Before/After Calibration')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = project_root / 'debug_scripts' / 'platt_coefficient_explanation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n📊 Visualization saved to: {output_path}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. Small coefficient (0.21) means Platt is HEAVILY compressing the predictions
   - Original predictions 0.0-1.0 → Calibrated predictions ~0.47-0.52
   
2. Why so small?
   - Strong L2 regularization (C=0.01 is low, more regularization)
   - Class imbalance (92 class-1 vs 33 class-0) with balanced weighting
   - Model might be poorly calibrated or overconfident
   
3. This is BAD for failure detection:
   - Removes discriminative power
   - Can't tell confident correct from uncertain incorrect anymore
   
4. Platt scaling is designed for CALIBRATION (matching predicted probs to true freqs)
   NOT for failure detection (separating correct from incorrect predictions)
""")
