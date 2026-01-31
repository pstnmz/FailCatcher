# UMAP Projections for medMNIST Datasets

This script generates UMAP visualizations for all 8 evaluated medMNIST datasets using **raw image features** (flattened pixel values), showing how different distribution shifts affect the image space.

## Overview

The script:
1. **Flattens raw images** directly (no model feature extraction)
2. **Fits UMAP on training images** 
3. **Projects different test sets** onto the training UMAP:
   - **ID (In-Distribution)**: Standard test set
   - **CS (Corruption Shift)**: Test set with random corruptions applied
   - **PS (Population Shift)**: External/different population data
   - **NCS (New Class Shift)**: Unseen classes (AMOS only)

**Note**: This approach visualizes the raw image space, not learned feature representations. For feature-based UMAPs, use model features instead.

## Dataset-Specific Scenarios

| Dataset | ID | CS | PS | NCS | Notes |
|---------|----|----|----|----|-------|
| breastmnist | ✓ | ✓ | ✗ | ✗ | Basic: ID + corruptions only |
| bloodmnist | ✓ | ✓ | ✗ | ✗ | Basic: ID + corruptions only |
| tissuemnist | ✓ | ✓ | ✗ | ✗ | Basic: ID + corruptions only |
| octmnist | ✓ | ✓ | ✗ | ✗ | Basic: ID + corruptions only |
| pneumoniamnist | ✓ | ✓ | ✗ | ✗ | Basic: ID + corruptions only |
| dermamnist-e | ✓ | ✓ | ✓ | ✗ | PS = external center |
| organamnist | ✓ | ✓ | ✓ | ✓ | PS = AMOS mapped, NCS = AMOS new classes |
| pathmnist | ✗ | ✗ | ✓ | ✗ | Only PS projection |

## Installation

Install UMAP if not already available:

```bash
pip install umap-learn
```

Other dependencies (should already be installed for UQ_Toolbox):
- torch
- numpy
- matplotlib
- seaborn
- tqdm
- scikit-learn

## Usage

### Process All Datasets

```bash
python generate_umap_projections.py --all-datasets
```

### Process Specific Datasets

```bash
python generate_umap_projections.py --datasets breastmnist organamnist dermamnist-e
```

### Control Corruption Severity

```bash
python generate_umap_projections.py --all-datasets --corruption-severity 5
```

Severity levels:
- 1: Mild covariate shift
- 2: Mild-moderate
- 3: Moderate (default)
- 4: Moderate-severe
- 5: Severe covariate shift

### Use Different Model Backbone

**Not applicable** - this version uses raw images, not model features.

### Use Different Training Setup

**Not applicable** - this version uses raw images, not model features.

### Specify Output Directory

```bash
python generate_umap_projections.py --all-datasets --output-dir ./my_umap_results
```

## Output Files

For each dataset, the script generates:

1. **`umap_{dataset}.png`**: Visualization with subplots for each test scenario
   - Train set shown in first subplot
   - Each shift type in separate subplot
   - Color-coded by class labels
   - 300 DPI for publication quality

2. **`umap_{dataset}_embeddings.npz`**: Saved embeddings for further analysis
   - Contains embeddings for all scenarios
   - Includes labels and class names
   - Can be loaded for custom visualizations

## Example Output Structure

```
umap_projections/
├── umap_breastmnist.png
├── umap_breastmnist_embeddings.npz
├── umap_bloodmnist.png
├── umap_bloodmnist_embeddings.npz
├── umap_dermamnist-e.png
├── umap_dermamnist-e_embeddings.npz
├── umap_organamnist.png
├── umap_organamnist_embeddings.npz
├── umap_pathmnist.png
├── umap_pathmnist_embeddings.npz
├── umap_tissuemnist.png
├── umap_tissuemnist_embeddings.npz
├── umap_octmnist.png
├── umap_octmnist_embeddings.npz
├── umap_pneumoniamnist.png
└── umap_pneumoniamnist_embeddings.npz
```

## UMAP Parameters

Default parameters (can be modified in the script):
- `n_neighbors=15`: Controls local vs global structure
- `min_dist=0.1`: Minimum distance between points in embedding
- `n_components=2`: 2D for visualization
- `random_state=42`: Reproducibility

Features are **standardized** before UMAP fitting using `StandardScaler`.

## Feature Extraction

- **Method**: Flattened raw images (C×H×W → C*H*W vector)
- **Preprocessing**: Images are normalized using dataset-specific transforms
- **Standardization**: Features standardized before UMAP using `StandardScaler`
- **Dimensionality**: e.g., for 224×224 RGB images: 224×224×3 = 150,528 dimensions

## Implementation Details

### Corruption Application

For CS scenarios, corruptions are applied using `medmnistc`:
- Random corruption selected per sample from available pool
- Seed=42 for reproducibility
- Cached to avoid recomputation

### Population Shift Loading

- **dermamnist-e**: External test center via `test_subset='external'`
- **organamnist**: AMOS dataset with organ mapping
- **pathmnist**: Standard test set (can be updated if external set added)

### New Class Shift (organamnist only)

- **Known classes**: Only unanimously correct predictions (all 5 folds agree)
- **New classes**: AMOS organs not in OrganaMNIST (unmapped organs)
- **Binary labels**: 0=known class (correct), 1=new class (failure)

## Interpreting Results

### Expected Patterns

1. **ID test**: Should overlap closely with train distribution
2. **CS test**: May show shift away from train clusters (covariate shift)
3. **PS test**: Distinct distribution shifts (different population)
4. **NCS test**: New classes should form separate clusters

### Good UQ Methods Should

- Detect samples far from train distribution
- Identify new class samples (high uncertainty)
- Handle covariate shifts gracefully

## Advanced Usage

### Load Saved Embeddings

```python
import numpy as np
import matplotlib.pyplot as plt

# Load embeddings
data = np.load('umap_projections/umap_organamnist_embeddings.npz')

train_emb = data['train_embedding']
train_labels = data['train_labels']
id_emb = data['id_embedding']
id_labels = data['id_labels']
ncs_emb = data['ncs_embedding']
ncs_labels = data['ncs_labels']

# Custom visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(train_emb[:, 0], train_emb[:, 1], c=train_labels, cmap='tab10', s=1)
plt.title('Train')

plt.subplot(1, 2, 2)
plt.scatter(ncs_emb[:, 0], ncs_emb[:, 1], c=ncs_labels, cmap='coolwarm', s=1)
plt.title('New Class Shift')
plt.show()
```

### Batch Processing

```bash
#!/bin/bash
# Process all datasets with different corruption levels

for sev in 1 2 3 4 5; do
    python generate_umap_projections.py --all-datasets \
        --corruption-severity $sev \
        --output-dir ./umap_projections_sev${sev}
done
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python generate_umap_projections.py --all-datasets --batch-size 256
```

### Missing AMOS Dataset

Ensure Git LFS has pulled the data:
```bash
cd benchmarks/medMNIST/Data/AMOS_2022/
git lfs pull
```

### UMAP Installation Issues

Try conda:
```bash
conda install -c conda-forge umap-learn
```

Or use pip with specific version:
```bash
pip install umap-learn==0.5.3
```

## Citation

If you use these visualizations in your work, please cite:

```bibtex
@article{medmnist,
    title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    year={2023}
}
```

## Contact

For issues or questions about UMAP projections, see the main UQ_Toolbox documentation or open an issue.
