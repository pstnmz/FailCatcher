"""
Test temperature scaling implementation on organamnist (multiclass).
"""
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

# Load organamnist calibration data
cache_dir = project_root / 'uq_benchmark_results' / 'cache'
calib_cache = np.load(cache_dir / 'organamnist_resnet18_calib_results.npz')

logits_calib = calib_cache['logits']  # [N, C]
y_true_calib = calib_cache['y_true']

print("="*70)
print("TEMPERATURE SCALING ANALYSIS - OrganaMNIST")
print("="*70)
print(f"Calibration set: {len(y_true_calib)} samples")
print(f"Logits shape: {logits_calib.shape}")
print(f"Num classes: {logits_calib.shape[1]}")
print(f"Label distribution: {np.bincount(y_true_calib.astype(int))}")
print()

# Test different configurations
from FailCatcher.methods.distance import fit_temperature_scaling

# Current implementation
print("1. CURRENT IMPLEMENTATION (init=1.5, weighted, constrained [0.5, 5.0])")
print("-"*70)
model_current = fit_temperature_scaling(logits_calib, y_true_calib, max_iter=1000)
temp_current = torch.exp(model_current.log_temperature).item()
print()

# Standard implementation
class StandardTemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # Init at 1.0
    
    def forward(self, logits):
        return logits / self.temperature

print("2. STANDARD IMPLEMENTATION (init=1.0, unweighted, unconstrained)")
print("-"*70)
logits_t = torch.from_numpy(logits_calib).float()
labels_t = torch.from_numpy(y_true_calib).long()

model_standard = StandardTemperatureScaler()
criterion = nn.CrossEntropyLoss()  # No class weighting
optimizer = optim.LBFGS([model_standard.temperature], lr=0.01, max_iter=1000)

print(f"Initial temperature: {model_standard.temperature.item():.4f}")

def closure():
    optimizer.zero_grad()
    loss = criterion(model_standard(logits_t), labels_t)
    loss.backward()
    return loss

optimizer.step(closure)
temp_standard = model_standard.temperature.item()
print(f"Optimized temperature: {temp_standard:.4f}")
print()

# With wider bounds
class BoundedTemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # Init at log(1)=0
    
    def forward(self, logits):
        temp = torch.exp(self.log_temperature)
        temp = torch.clamp(temp, min=0.1, max=10.0)  # Wider bounds
        return logits / temp

print("3. WIDER BOUNDS (init=1.0, unweighted, constrained [0.1, 10.0])")
print("-"*70)
model_wide = BoundedTemperatureScaler()
optimizer = optim.LBFGS([model_wide.log_temperature], lr=0.01, max_iter=1000)

print(f"Initial temperature: {torch.exp(model_wide.log_temperature).item():.4f}")

def closure():
    optimizer.zero_grad()
    loss = criterion(model_wide(logits_t), labels_t)
    loss.backward()
    return loss

optimizer.step(closure)
temp_wide = torch.exp(model_wide.log_temperature).item()
print(f"Optimized temperature: {temp_wide:.4f}")
print()

# Compare calibration quality
print("="*70)
print("CALIBRATION QUALITY COMPARISON")
print("="*70)

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(mask) > 0:
            avg_conf = np.mean(confidences[mask])
            avg_acc = np.mean(accuracies[mask])
            ece += np.abs(avg_conf - avg_acc) * np.sum(mask)
    
    return ece / len(labels)

# Uncalibrated
probs_uncalib = torch.softmax(logits_t, dim=1).numpy()
ece_uncalib = compute_ece(probs_uncalib, y_true_calib)

# Current
logits_current = model_current(logits_t).detach()
probs_current = torch.softmax(logits_current, dim=1).numpy()
ece_current = compute_ece(probs_current, y_true_calib)

# Standard  
logits_standard = model_standard(logits_t).detach()
probs_standard = torch.softmax(logits_standard, dim=1).numpy()
ece_standard = compute_ece(probs_standard, y_true_calib)

# Wide
logits_wide = model_wide(logits_t).detach()
probs_wide = torch.softmax(logits_wide, dim=1).numpy()
ece_wide = compute_ece(probs_wide, y_true_calib)

print(f"Expected Calibration Error (lower is better):")
print(f"  Uncalibrated: {ece_uncalib:.4f}")
print(f"  Current impl: {ece_current:.4f} (T={temp_current:.2f})")
print(f"  Standard:     {ece_standard:.4f} (T={temp_standard:.2f})")
print(f"  Wide bounds:  {ece_wide:.4f} (T={temp_wide:.2f})")
print()

print("="*70)
print("RECOMMENDATIONS")
print("="*70)
print("""
1. Remove class weighting - temperature scaling should use unweighted CE loss
2. Initialize at T=1.0 instead of T=1.5
3. Consider wider bounds [0.1, 10.0] or remove bounds entirely
4. Increase learning rate to 0.01 for LBFGS
5. Optionally: add validation monitoring to prevent overfitting

The differences may be small for well-calibrated models but matter for
poorly calibrated ones or when the optimal T is outside [0.5, 5.0].
""")
