# FailCatcher Reorganization

## Overview
FailCatcher has been reorganized into a clean, pip-installable library with dataset-specific benchmarking utilities separated from the core library.

## New Structure

```
UQ_Toolbox/
├── FailCatcher/                    # 📦 Core library (pip-installable)
│   ├── __init__.py
│   ├── UQ_toolbox.py              # Main API exports
│   ├── failure_detection.py       # ✨ NEW: Generic failure detector
│   ├── core/                      # Base classes and utilities
│   ├── methods/                   # UQ method implementations
│   │   ├── tta.py                # TTA and GPS methods
│   │   ├── gps_augment/          # ✨ MOVED: GPS augmentation utilities
│   │   ├── latent.py
│   │   └── ...
│   ├── search/                    # Policy search algorithms
│   ├── evaluation/                # Metrics and visualization
│   └── setup.py                   # For pip install
│
├── benchmarks/                     # 📊 Dataset-specific benchmarks
│   └── medMNIST/
│       ├── __init__.py
│       └── run_medmnist_benchmark.py  # ✨ NEW: Clean medMNIST runner
│
├── medMNIST/                      # MedMNIST data, models, training scripts
│   ├── models/                    # Trained model checkpoints
│   ├── gps_augment/              # GPS augmentation predictions (data)
│   ├── utils/                     # Data loading utilities
│   ├── train_resnet18_medMNIST.py
│   └── ...
│
└── uq_benchmark_results/          # Output directory

```

## Key Changes

### 1. **Renamed to `failure_detection.py`** (was `benchmark.py`)
The new `FailureDetector` class provides a dataset-agnostic, user-friendly interface:

```python
from FailCatcher import failure_detection

# Initialize detector
detector = failure_detection.FailureDetector(
    models=models,                  # List of PyTorch models
    study_dataset=train_dataset,    # Full training data
    calib_dataset=calib_dataset,    # Calibration data
    test_dataset=test_dataset,      # Test data
    device='cuda'
)

# Run UQ methods to detect failures
uncertainties, metrics = detector.run_msr(y_scores, y_true)
uncertainties, metrics = detector.run_ensemble(indiv_scores, y_true)
uncertainties, metrics = detector.run_knn_shap(
    calib_loader, test_loader, y_true,
    cv_generator=my_cv_generator,
    flag='mydataset'
)
```

**Available methods:**
- `run_msr()` - Maximum Softmax Response
- `run_msr_calibrated()` - Calibrated MSR (temperature/platt/isotonic)
- `run_ensemble()` - Ensemble standard deviation
- `run_tta()` - Test-Time Augmentation
- `run_gps()` - Greedy Policy Search
- `run_knn_raw()` - KNN in raw latent space
- `run_knn_shap()` - KNN in SHAP-selected latent space

### 2. **Moved GPS Augmentation** (`gps_augment/` → `FailCatcher/methods/gps_augment/`)
The GPS augmentation utilities are now part of the core library since they're required for the GPS method to function:

```python
# Before: External dependency
from gps_augment.utils.randaugment import BetterRandAugment

# After: Internal module
from .gps_augment.utils.randaugment import BetterRandAugment
```

This makes FailCatcher self-contained and easier to distribute via pip.

### 3. **Dataset-Specific Benchmarks** (`benchmarks/medMNIST/`)
All medMNIST-specific code (CV splitting, transforms, data loading) is now isolated:

```bash
# Old way (mixed concerns)
python benchmark_uq_methods.py --flag breastmnist --methods MSR

# New way (clean separation)
python benchmarks/medMNIST/run_medmnist_benchmark.py --flag breastmnist --methods MSR
```

### 4. **CV Generator Pattern**
To handle cross-validation splits, benchmarks provide a generator function:

```python
def create_cv_generator(n_splits=5, seed=42):
    """Factory for CV train loaders matching training splits."""
    def cv_generator(study_dataset, models, batch_size):
        # Dataset-specific CV splitting logic
        # Returns List[DataLoader], one per model
        ...
    return cv_generator

# Pass to detector
cv_gen = create_cv_generator(n_splits=5, seed=42)
detector.run_knn_shap(..., cv_generator=cv_gen)
```

This ensures:
- ✅ Core library has **no dataset assumptions**
- ✅ CV splits match training (critical for KNN methods)
- ✅ Easy to adapt to new datasets

## Migration Guide

### For Library Users (pip install)

```bash
# Install FailCatcher
cd FailCatcher
pip install -e .

# Use in your code
from FailCatcher import failure_detection
import FailCatcher.UQ_toolbox as uq

detector = failure_detection.FailureDetector(...)
uncertainties, metrics = detector.run_msr(...)
```

### For Benchmark Users (medMNIST)

```bash
# Run benchmarks from root directory
python benchmarks/medMNIST/run_medmnist_benchmark.py \
    --flag breastmnist \
    --methods MSR Ensembling KNN_SHAP \
    --batch-size 2000
```

### For New Dataset Benchmarks

1. Create `benchmarks/my_dataset/` folder
2. Implement dataset loading and CV generator
3. Use `FailCatcher.failure_detection.FailureDetector` API
4. See `benchmarks/medMNIST/run_medmnist_benchmark.py` as template

## Benefits

### ✅ Clean Separation
- **FailCatcher/**: Zero dataset dependencies, pure library code
- **benchmarks/**: Dataset-specific utilities and scripts
- **medMNIST/**: Data, models, training scripts

### ✅ Pip-Installable & Self-Contained
```bash
pip install git+https://github.com/user/UQ_Toolbox.git#subdirectory=FailCatcher
```
All GPS dependencies now included internally.

### ✅ User-Friendly Naming
- `FailureDetector` class emphasizes practical use case
- Methods named as actions: `run_msr()`, `run_ensemble()`
- Clear intent: detect failures using uncertainty

### ✅ Reusable API
Same `FailureDetector` interface works for:
- MedMNIST datasets
- ImageNet
- Custom medical datasets
- Any PyTorch classification task

### ✅ Maintainable
- Core methods isolated from benchmark logic
- Easy to add new UQ methods
- Easy to add new dataset benchmarks
- Clear responsibility boundaries

## Example: Adding a New Dataset

```python
# benchmarks/imagenet/run_imagenet_benchmark.py
from FailCatcher import failure_detection

def create_imagenet_cv_generator():
    """ImageNet-specific CV splitting."""
    def cv_generator(study_dataset, models, batch_size):
        # Your ImageNet CV logic here
        return train_loaders
    return cv_generator

# Load ImageNet models and data
models = load_imagenet_models()
train_data, calib_data, test_data = load_imagenet_data()

# Use FailCatcher API
detector = failure_detection.FailureDetector(
    models=models,
    study_dataset=train_data,
    calib_dataset=calib_data,
    test_dataset=test_data
)

cv_gen = create_imagenet_cv_generator()
uncertainties, metrics = detector.run_knn_shap(
    calib_loader, test_loader, y_true,
    cv_generator=cv_gen,
    flag='imagenet'
)
```

## Status

### ✅ Completed
- [x] Create `FailCatcher/failure_detection.py` with generic API (renamed from benchmark.py)
- [x] Create `benchmarks/medMNIST/run_medmnist_benchmark.py` template
- [x] Export failure_detection module in `FailCatcher/__init__.py`
- [x] Move `gps_augment/` into `FailCatcher/methods/` (now self-contained)
- [x] Update import paths for GPS augmentation utilities

### 🚧 Next Steps
1. **Migrate existing benchmark logic** from `benchmark_uq_methods.py` to new structure
2. **Create `FailCatcher/setup.py`** for pip installation
3. **Update documentation** with new API examples
4. **Add tests** for failure_detection module
5. **Create command-line interface** for easy failure detection

### 📝 Design Decisions
- **Renamed to `failure_detection`**: Better reflects user-facing purpose (detecting failures)
- **Moved `gps_augment` inside library**: GPS is a core method, dependencies should be internal
- **Keep `medMNIST/` at root** (not inside `benchmarks/`) because it contains:
  - Model checkpoints (large files)
  - Training scripts
  - Data utilities used by both training and benchmarking
- **`benchmarks/medMNIST/`** contains only the runner script
- **`medMNIST/gps_augment/`** contains augmentation *predictions* (data), while
  `FailCatcher/methods/gps_augment/` contains augmentation *code* (utilities)
- **CV generator pattern** allows dataset-specific splitting without polluting library

## Backward Compatibility

The old `benchmark_uq_methods.py` remains functional. Migration is opt-in:
- **Old script**: Still works, all-in-one monolithic approach
- **New structure**: Cleaner, for production use and pip installation
