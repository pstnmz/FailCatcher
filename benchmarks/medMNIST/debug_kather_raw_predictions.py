"""
Debug script to see raw 9-class predictions on Kather WITHOUT collapsing.
This will show which PathMNIST classes the model is actually predicting.
"""
import torch
import torch.nn as nn
from torchvision import models as torch_models
from pathlib import Path
import numpy as np
from collections import Counter

# Import Kather dataset
import sys
sys.path.insert(0, str(Path(__file__).parent / 'Data'))
from kather_texture_dataset_6class import KatherTexture2016, KATHER_TO_6CLASS

# PathMNIST 9-class names
PATHMNIST_9CLASS = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# Load Kather dataset
data_root = Path(__file__).parent / 'Data' / 'kather_texture' / 'Kather_texture_2016_image_tiles_5000'
dataset = KatherTexture2016(str(data_root), target_size=224)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# Load PathMNIST model
device = torch.device('cuda')
model = torch_models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)
model.load_state_dict(torch.load('models/224*224/pathmnist_resnet18_224_randaug0_fold_0.pt'))
model = model.to(device)
model.eval()

# Get predictions WITHOUT collapsing
print("Running inference on Kather dataset...")
all_preds_9class = []
all_labels_6class = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds_9class.extend(preds.cpu().numpy())
        all_labels_6class.extend(labels.numpy())

all_preds_9class = np.array(all_preds_9class)
all_labels_6class = np.array(all_labels_6class)

# Analyze predictions
print("\n" + "="*80)
print("Raw 9-class predictions on Kather dataset (NO COLLAPSE)")
print("="*80)

print("\nOverall prediction distribution:")
pred_counts = Counter(all_preds_9class)
for i in range(9):
    count = pred_counts[i]
    pct = 100 * count / len(all_preds_9class)
    print(f"  {i}: {PATHMNIST_9CLASS[i]:8} = {count:5} ({pct:5.2f}%)")

# Analyze by Kather ground truth class
print("\n" + "="*80)
print("Predictions breakdown by Kather tissue type:")
print("="*80)

COLLAPSED_6CLASS = ['ADI', 'BACK', 'LYM', 'NORM', 'STR', 'TUM']

for kather_idx, kather_name in enumerate(COLLAPSED_6CLASS):
    mask = (all_labels_6class == kather_idx)
    if not mask.any():
        continue
    
    preds_for_class = all_preds_9class[mask]
    print(f"\nKather class: {kather_name} ({mask.sum()} samples)")
    
    pred_counter = Counter(preds_for_class)
    # Show top 3 predicted classes
    top_preds = pred_counter.most_common(5)
    for pathmnist_idx, count in top_preds:
        pct = 100 * count / len(preds_for_class)
        print(f"  → Predicts {PATHMNIST_9CLASS[pathmnist_idx]:8}: {count:4} ({pct:5.1f}%)")

print("\n" + "="*80)
print("Key insights:")
print("="*80)
print("- If DEB/MUC/MUS are frequently predicted, the collapse might be losing info")
print("- If STR is predicted for everything, that's the dominant class problem")
print("- Check if predictions make sense (e.g., Kather NORM → PathMNIST NORM)")
