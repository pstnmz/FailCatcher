# Fail Catcher

Post-Hoc Uncertainty quantification toolkit for pytorch deep learning models.

## Installation
```bash
pip install -e .
```

## Quick Start
```python
import UQ_toolbox as uq

# TTA uncertainty
stds, preds = uq.TTA(policies, model, dataset, device)

# GPS: discover optimal augmentations
gps = uq.GPSMethod(aug_folder, correct_calib, incorrect_calib)
gps.search_policies(num_workers=90, top_k=3)
scores = gps.compute(model, test_dataset, device)

# Ensemble uncertainty
ensemble = uq.EnsembleSTDMethod()
scores = ensemble.compute(models, test_loader, device)
```

## Modules
- `methods/` - UQ methods (TTA, GPS, ensemble, distance, latent)
- `search/` - Greedy policy search
- `visualization/` - Plotting utilities
- `core/` - Base classes and utilities

## Citation
[Your paper citation]