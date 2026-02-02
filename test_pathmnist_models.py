"""Quick test to verify PathMNIST models work correctly on their own test set."""
import torch
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / 'benchmarks' / 'medMNIST' / 'utils'))
import train_models_load_datasets as tr
from benchmarks.medMNIST.utils import dataset_utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Loading PathMNIST models...")
models = tr.load_models('pathmnist', device=device, size=224, model_backbone='resnet18', setup='')

print("Loading PathMNIST data with standard transforms...")
color = True  # PathMNIST is color
transform, transform_tta = dataset_utils.get_transforms(color, 224)

[study_dataset, calib_dataset, test_dataset], \
[_, calib_loader, test_loader], info = \
    tr.load_datasets('pathmnist', color, 224, transform, batch_size=256)

print(f"Test set size: {len(test_dataset)}")
print(f"Number of classes: {len(info['label'])}")

# Test first batch
print("\nTesting first batch...")
for batch_data in test_loader:
    if isinstance(batch_data, (tuple, list)):
        images, labels = batch_data
    else:
        images = batch_data
        labels = None
    
    images = images.to(device)
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape if labels is not None else 'None'}")
    print(f"First 10 labels: {labels[:10].flatten() if labels is not None else 'None'}")
    
    # Test model
    model = models[0]
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    print(f"Predictions shape: {preds.shape}")
    print(f"First 10 predictions: {preds[:10]}")
    
    if labels is not None:
        labels_flat = labels.flatten().numpy() if isinstance(labels, torch.Tensor) else labels
        acc = (preds == labels_flat[:len(preds)]).mean()
        print(f"Batch accuracy: {acc:.4f}")
    
    break

# Full test set evaluation
print("\nEvaluating on full test set...")
model = models[0]
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_data in test_loader:
        if isinstance(batch_data, (tuple, list)):
            images, labels = batch_data
        else:
            images = batch_data
            labels = None
        
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.append(preds)
        
        if labels is not None:
            labels_flat = labels.flatten().numpy() if isinstance(labels, torch.Tensor) else labels
            all_labels.append(labels_flat)

all_preds = torch.cat([torch.from_numpy(p) for p in all_preds]).numpy()
if all_labels:
    all_labels = torch.cat([torch.from_numpy(l) for l in all_labels]).numpy()
    accuracy = (all_preds == all_labels).mean()
    print(f"Test set accuracy: {accuracy:.4f}")
    print(f"Expected: ~0.70-0.95 for PathMNIST models")
