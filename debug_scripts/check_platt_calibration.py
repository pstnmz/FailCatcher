"""
Debug script to understand Platt calibration behavior on breastmnist.
"""
import numpy as np
import sys
import torch
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'benchmarks' / 'medMNIST'))
sys.path.insert(0, str(project_root))

from utils.train_models_load_datasets import load_datasets, load_models
from torch.utils.data import DataLoader
from torchvision import transforms
from FailCatcher.methods.distance import posthoc_calibration
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Load data
flag = 'breastmnist'
color = False  # breastmnist is grayscale
# Need to repeat grayscale to 3 channels for ResNet
class RepeatGrayToRGB:
    def __call__(self, img):
        return img.repeat(3, 1, 1) if img.shape[0] == 1 else img

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    RepeatGrayToRGB()
])
datasets_list, loaders_list, info = load_datasets(flag, color, 224, transform, batch_size=256)
train_dataset, calib_dataset, test_dataset = datasets_list

# Get labels
calib_labels = np.array([int(calib_dataset[i][1]) for i in range(len(calib_dataset))])
test_labels = np.array([int(test_dataset[i][1]) for i in range(len(test_dataset))])

print(f"Dataset info:")
print(f"  Calib: {len(calib_labels)} samples, class dist: {np.bincount(calib_labels)}")
print(f"  Test: {len(test_labels)} samples, class dist: {np.bincount(test_labels)}")

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = load_models(flag, device, waugmentation=False, size=224, model_backbone='resnet18')
model = models[0]
model.eval()

# Get predictions
def get_predictions(dataset, model, device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_scores = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            # Check if binary (1 output) or multiclass (2+ outputs)
            if outputs.shape[1] == 1:
                # Binary: use sigmoid
                scores = torch.sigmoid(outputs).cpu().numpy()
            else:
                # Multiclass: use softmax
                scores = torch.softmax(outputs, dim=1).cpu().numpy()
            all_scores.append(scores)
    return np.vstack(all_scores)

print("\nGetting model predictions...")
calib_scores = get_predictions(calib_dataset, model, device)
test_scores = get_predictions(test_dataset, model, device)

# Check output shape
print(f"\nModel output shapes:")
print(f"  Calib scores: {calib_scores.shape}")
print(f"  Test scores: {test_scores.shape}")

# Analyze uncalibrated predictions
if calib_scores.shape[1] == 2:
    # 2-class output
    calib_pred_pos = calib_scores[:, 1]  # Positive class probability
    test_pred_pos = test_scores[:, 1]
else:
    # Single output (already positive class prob)
    calib_pred_pos = calib_scores[:, 0]
    test_pred_pos = test_scores[:, 0]

print(f"\nUncalibrated predictions (positive class):")
print(f"  Calib: min={calib_pred_pos.min():.4f}, max={calib_pred_pos.max():.4f}, mean={calib_pred_pos.mean():.4f}")
print(f"  Test: min={test_pred_pos.min():.4f}, max={test_pred_pos.max():.4f}, mean={test_pred_pos.mean():.4f}")

# Check prediction distribution
for name, preds in [('Calib', calib_pred_pos), ('Test', test_pred_pos)]:
    very_low = np.sum(preds < 0.2)
    low = np.sum((preds >= 0.2) & (preds < 0.4))
    mid = np.sum((preds >= 0.4) & (preds < 0.6))
    high = np.sum((preds >= 0.6) & (preds < 0.8))
    very_high = np.sum(preds >= 0.8)
    total = len(preds)
    print(f"  {name} distribution: <0.2: {very_low}/{total} ({very_low/total*100:.1f}%), "
          f"0.2-0.4: {low}/{total} ({low/total*100:.1f}%), "
          f"0.4-0.6: {mid}/{total} ({mid/total*100:.1f}%), "
          f"0.6-0.8: {high}/{total} ({high/total*100:.1f}%), "
          f">0.8: {very_high}/{total} ({very_high/total*100:.1f}%)")

# Compute uncalibrated metrics
calib_pred_class = np.argmax(calib_scores, axis=1)
test_pred_class = np.argmax(test_scores, axis=1)
calib_correct = (calib_pred_class == calib_labels)
test_correct = (test_pred_class == test_labels)

print(f"\nUncalibrated accuracy:")
print(f"  Calib: {calib_correct.mean():.4f}")
print(f"  Test: {test_correct.mean():.4f}")

# Check reliability: are high-confidence predictions actually correct?
for name, preds, correct in [('Calib', calib_pred_pos, calib_correct), ('Test', test_pred_pos, test_correct)]:
    confident_high = preds > 0.8
    confident_low = preds < 0.2
    acc_high = correct[confident_high].mean() if confident_high.sum() > 0 else None
    acc_low = correct[confident_low].mean() if confident_low.sum() > 0 else None
    acc_high_str = f"{acc_high:.4f}" if acc_high is not None else "N/A"
    acc_low_str = f"{acc_low:.4f}" if acc_low is not None else "N/A"
    print(f"  {name} - High conf (>0.8): {confident_high.sum()} samples, accuracy: {acc_high_str}")
    print(f"  {name} - Low conf (<0.2): {confident_low.sum()} samples, accuracy: {acc_low_str}")

# Apply Platt calibration
print("\n" + "="*60)
print("Applying Platt calibration...")
calibrated_calib_probs, platt_model = posthoc_calibration(calib_scores, calib_labels, 'platt')
print(f"Platt model coefficients: coef={platt_model.coef_}, intercept={platt_model.intercept_}")

# Apply to test set
from FailCatcher.core.utils import apply_calibration
calibrated_test_probs = apply_calibration(test_scores, platt_model, 'platt')

print(f"\nCalibrated predictions (positive class):")
print(f"  Calib: min={calibrated_calib_probs.min():.4f}, max={calibrated_calib_probs.max():.4f}, mean={calibrated_calib_probs.mean():.4f}")
print(f"  Test: min={calibrated_test_probs.min():.4f}, max={calibrated_test_probs.max():.4f}, mean={calibrated_test_probs.mean():.4f}")

# Check how many predictions are now near 0.5
for name, preds in [('Calib', calibrated_calib_probs), ('Test', calibrated_test_probs)]:
    very_low = np.sum(preds < 0.2)
    low = np.sum((preds >= 0.2) & (preds < 0.4))
    mid = np.sum((preds >= 0.4) & (preds < 0.6))
    high = np.sum((preds >= 0.6) & (preds < 0.8))
    very_high = np.sum(preds >= 0.8)
    total = len(preds)
    print(f"  {name} distribution: <0.2: {very_low}/{total} ({very_low/total*100:.1f}%), "
          f"0.2-0.4: {low}/{total} ({low/total*100:.1f}%), "
          f"0.4-0.6: {mid}/{total} ({mid/total*100:.1f}%), "
          f"0.6-0.8: {high}/{total} ({high/total*100:.1f}%), "
          f">0.8: {very_high}/{total} ({very_high/total*100:.1f}%)")

# Compare uncertainty metrics
uncalib_uncertainty = 0.5 - np.abs(test_pred_pos - 0.5)  # Distance to 0.5
calib_uncertainty = 0.5 - np.abs(calibrated_test_probs - 0.5)  # Distance to 0.5

print(f"\nUncertainty statistics (distance to 0.5):")
print(f"  Uncalibrated: min={uncalib_uncertainty.min():.4f}, max={uncalib_uncertainty.max():.4f}, mean={uncalib_uncertainty.mean():.4f}")
print(f"  Calibrated: min={calib_uncertainty.min():.4f}, max={calib_uncertainty.max():.4f}, mean={calib_uncertainty.mean():.4f}")

# Check AUROC for failure detection
test_incorrect = ~test_correct
if test_incorrect.sum() > 0:
    try:
        auroc_uncalib = roc_auc_score(test_incorrect, uncalib_uncertainty)
        auroc_calib = roc_auc_score(test_incorrect, calib_uncertainty)
        print(f"\nAUROC for failure detection:")
        print(f"  Uncalibrated: {auroc_uncalib:.4f}")
        print(f"  Calibrated: {auroc_calib:.4f}")
        print(f"  Difference: {auroc_calib - auroc_uncalib:+.4f}")
    except Exception as e:
        print(f"\nCould not compute AUROC: {e}")

# Visualize calibration curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Uncalibrated vs Calibrated predictions
axes[0].scatter(test_pred_pos, calibrated_test_probs, alpha=0.3, s=10)
axes[0].plot([0, 1], [0, 1], 'r--', label='No change')
axes[0].set_xlabel('Uncalibrated probability')
axes[0].set_ylabel('Calibrated probability')
axes[0].set_title('Platt Calibration Effect')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Uncertainty comparison
axes[1].scatter(uncalib_uncertainty[test_correct], calib_uncertainty[test_correct], 
                alpha=0.3, s=10, label='Correct', c='green')
axes[1].scatter(uncalib_uncertainty[~test_correct], calib_uncertainty[~test_correct], 
                alpha=0.8, s=30, label='Incorrect', c='red', marker='x')
axes[1].plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5, label='No change')
axes[1].set_xlabel('Uncalibrated uncertainty')
axes[1].set_ylabel('Calibrated uncertainty')
axes[1].set_title('Uncertainty Comparison (correct vs incorrect)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = project_root / 'debug_scripts' / 'platt_calibration_analysis.png'
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
