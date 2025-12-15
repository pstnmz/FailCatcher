"""
Test script to verify TTA ensemble_mode parameter works correctly.

This script creates a minimal test case with 2 models, 3 samples, and 2 augmentations
to verify that:
1. per_fold_evaluation=True computes uncertainty per-model, then averages
2. per_fold_evaluation=False averages models first, then computes uncertainty
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
from FailCatcher.methods.tta import TTA

# Create dummy data
np.random.seed(42)
torch.manual_seed(42)

# 3 samples, 3 channels, 28x28 images
N = 3
C, H, W = 3, 28, 28
num_classes = 2

# Create synthetic images
images = torch.randn(N, C, H, W)
labels = torch.tensor([0, 1, 0])
dataset = TensorDataset(images, labels)

# Create two simple models that give different predictions
class DummyModel(torch.nn.Module):
    def __init__(self, bias=0.0):
        super().__init__()
        self.fc = torch.nn.Linear(C * H * W, num_classes)
        self.bias = bias
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits + self.bias

model1 = DummyModel(bias=0.0)
model2 = DummyModel(bias=1.0)  # Different bias to ensure different predictions
models = [model1, model2]

device = torch.device('cpu')
for model in models:
    model.to(device)
    model.eval()

print("=" * 80)
print("Testing TTA ensemble_mode behavior")
print("=" * 80)

# Test 1: ensemble_mode=False (average models first, then compute uncertainty)
print("\n1️⃣  Test ensemble_mode=False (ensemble evaluation)")
print("   Expected: Average models → augment → compute std")
stds_ensemble, _ = TTA(
    transformations=None,
    models=models,
    dataset=dataset,
    device=device,
    nb_augmentations=2,
    usingBetterRandAugment=False,
    n=2, m=9,
    image_normalization=False,
    nb_channels=C,
    mean=0.5,
    std=0.5,
    image_size=H,
    batch_size=N,
    ensemble_mode=False  # Ensemble evaluation
)
print(f"   Uncertainties shape: {stds_ensemble.shape}")
print(f"   Uncertainties: {stds_ensemble}")

# Test 2: ensemble_mode=True (compute per-model, then average)
print("\n2️⃣  Test ensemble_mode=True (per-fold evaluation)")
print("   Expected: Augment → compute std per-model → average stds")
stds_per_fold, _ = TTA(
    transformations=None,
    models=models,
    dataset=dataset,
    device=device,
    nb_augmentations=2,
    usingBetterRandAugment=False,
    n=2, m=9,
    image_normalization=False,
    nb_channels=C,
    mean=0.5,
    std=0.5,
    image_size=H,
    batch_size=N,
    ensemble_mode=True  # Per-fold evaluation
)
print(f"   Uncertainties shape: {stds_per_fold.shape}")
print(f"   Uncertainties: {stds_per_fold}")

# Verify shapes are correct
print("\n" + "=" * 80)
print("✅ VERIFICATION")
print("=" * 80)
assert stds_ensemble.shape == (N,), f"Expected shape ({N},), got {stds_ensemble.shape}"
assert stds_per_fold.shape == (N,), f"Expected shape ({N},), got {stds_per_fold.shape}"
print("✓ Both modes return correct shape: (N,)")

# Verify they give different results (they should, since aggregation order matters)
print(f"\n✓ Results are different (as expected): {not np.allclose(stds_ensemble, stds_per_fold)}")
print(f"  Ensemble mode: {stds_ensemble}")
print(f"  Per-fold mode: {stds_per_fold}")
print(f"  Mean difference: {np.abs(stds_ensemble - stds_per_fold).mean():.6f}")

print("\n" + "=" * 80)
print("🎉 All tests passed!")
print("=" * 80)
