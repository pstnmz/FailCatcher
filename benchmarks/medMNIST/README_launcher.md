# Benchmark Launcher

Comprehensive automated launcher for running all medMNIST UQ benchmarks across multiple configurations.

## Features

- **Automatic configuration generation**: Handles all combinations of datasets, models, setups, and methods
- **Smart method filtering**: Automatically excludes MCDropout for setups without dropout (standard, DA)
- **Dataset-specific optimization**: Uses optimal batch sizes and GPS subsampling per dataset
- **Dry-run mode**: Preview all commands before execution
- **Flexible filtering**: Run specific subsets of configurations
- **Error handling**: Continues to next configuration if one fails

## Quick Start

```bash
# Dry run: see what would be executed
python launcher_benchmark.py --dry-run

# Run all benchmarks (full suite)
python launcher_benchmark.py --python .venv/bin/python --gpu 0

# Run specific datasets
python launcher_benchmark.py --datasets breastmnist organamnist --gpu 0

# Run only internal test sets (ID)
python launcher_benchmark.py --id-only --gpu 0

# Run only external test sets
python launcher_benchmark.py --external-only --gpu 0

# Run specific model
python launcher_benchmark.py --models resnet18 --gpu 0

# Run specific setups
python launcher_benchmark.py --setups DO DADO --gpu 0

# Exclude specific methods
python launcher_benchmark.py --exclude-methods KNN_SHAP --gpu 0
```

## Configuration Matrix

### Datasets

**Internal Test Sets (ID)**:
- `breastmnist` - Breast ultrasound (binary, 546 test)
- `pneumoniamnist` - Chest X-ray pneumonia (binary, 624 test)
- `organamnist` - Abdominal CT organs (11 classes, 17,778 test)
- `octmnist` - Retinal OCT (4 classes, 1,000 test)
- `pathmnist` - Colon pathology (9 classes, 7,180 test)
- `bloodmnist` - Blood cell microscopy (8 classes, 3,421 test)
- `tissuemnist` - Kidney cortex microscopy (8 classes, 2,639 test)
- `dermamnist-e-id` - Dermatoscopy ID centers (7 classes, 1,014 test)

**External Test Sets (OOD)**:
- `amos2022` - AMOS-2022 organs (OrganaMNIST → external, 1,077 test)
- `dermamnist-e-external` - Dermatoscopy external center (7 classes, 297 test)

### Model Backbones

- `resnet18` - ResNet-18 (512-dim features)
- `vit_b_16` - Vision Transformer Base/16 (768-dim features)

### Training Setups

- **Standard** (`''` or `'standard'`) - Baseline training
- **DA** - Data augmentation (RandAugment)
- **DO** - Dropout regularization
- **DADO** - Both DA + Dropout

### UQ Methods

All methods are run by default, with automatic filtering:

- `MSR` - Maximum Softmax Response
- `MSR_calibrated` - Temperature-scaled MSR
- `MLS` - Maximum Logit Score
- `Ensembling` - Ensemble standard deviation
- `TTA` - Test-Time Augmentation
- `GPS` - Greedy Policy Search
- `KNN_Raw` - KNN in raw latent space
- `KNN_SHAP` - KNN in SHAP-selected features
- `MCDropout` - Monte Carlo Dropout (⚠️ only for DO/DADO setups)

## Method-Setup Compatibility

The launcher automatically filters methods based on training setup:

| Method | Standard | DA | DO | DADO |
|--------|----------|----|----|------|
| MSR, MSR_calibrated, MLS | ✅ | ✅ | ✅ | ✅ |
| Ensembling, TTA, GPS | ✅ | ✅ | ✅ | ✅ |
| KNN_Raw, KNN_SHAP | ✅ | ✅ | ✅ | ✅ |
| MCDropout | ❌ | ❌ | ✅ | ✅ |

## Dataset-Specific Settings

The launcher uses optimized settings per dataset:

```python
DATASET_CONFIG = {
    'breastmnist':          {'batch_size': 4000, 'gps_subsample': None},
    'pneumoniamnist':       {'batch_size': 4000, 'gps_subsample': None},
    'organamnist':          {'batch_size': 128,  'gps_subsample': 5000},
    'octmnist':             {'batch_size': 256,  'gps_subsample': 5000},
    'pathmnist':            {'batch_size': 128,  'gps_subsample': 5000},
    'bloodmnist':           {'batch_size': 256,  'gps_subsample': 5000},
    'tissuemnist':          {'batch_size': 512,  'gps_subsample': 5000},
    'dermamnist-e-id':      {'batch_size': 256,  'gps_subsample': 5000},
    'dermamnist-e-external': {'batch_size': 256, 'gps_subsample': 5000},
    'amos2022':             {'batch_size': 128,  'gps_subsample': 5000},
}
```

- Binary datasets (breast, pneumonia): No GPS subsampling needed (small datasets)
- Multi-class datasets: GPS subsampling to 5000 samples for efficiency

## Usage Examples

### Full Benchmark Suite

Run everything (10 datasets × 2 models × 4 setups = 80 configurations):

```bash
python launcher_benchmark.py --gpu 0
```

### Targeted Benchmarking

**Compare setups on one dataset**:
```bash
python launcher_benchmark.py \
    --datasets breastmnist \
    --setups "" DA DO DADO \
    --gpu 0
```

**Benchmark all datasets with ResNet-18 only**:
```bash
python launcher_benchmark.py \
    --models resnet18 \
    --gpu 0
```

**Test external generalization only**:
```bash
python launcher_benchmark.py \
    --external-only \
    --models resnet18 vit_b_16 \
    --gpu 0
```

**Fast benchmark (exclude expensive methods)**:
```bash
python launcher_benchmark.py \
    --exclude-methods KNN_SHAP GPS \
    --datasets breastmnist pneumoniamnist \
    --gpu 0
```

### Development & Testing

**Dry run to preview commands**:
```bash
python launcher_benchmark.py \
    --datasets breastmnist \
    --models resnet18 \
    --setups "" \
    --dry-run
```

**Quiet mode (suppress command output)**:
```bash
python launcher_benchmark.py \
    --datasets breastmnist \
    --quiet \
    --gpu 0
```

## Output Structure

Results are saved per configuration:

```
uq_benchmark_results/
├── cache/
│   ├── breastmnist_resnet18_calib_results.npz
│   ├── breastmnist_resnet18_DA_calib_results.npz
│   └── ...
├── figures/
│   ├── breastmnist/
│   │   ├── 20251217_143045/
│   │   │   ├── resnet18_MSR_roc_curve.png
│   │   │   ├── resnet18_DA_MSR_roc_curve.png
│   │   │   └── ...
│   └── ...
├── all_metrics_breastmnist_resnet18_20251217_143045.npz
├── all_metrics_breastmnist_resnet18_DA_20251217_143100.npz
├── uq_benchmark_breastmnist_resnet18_20251217_143045.json
├── uq_benchmark_breastmnist_resnet18_DA_20251217_143100.json
└── ...
```

## Advanced Options

### Custom Python Environment

```bash
python launcher_benchmark.py \
    --python /path/to/custom/venv/bin/python \
    --gpu 0
```

### Custom Script Location

```bash
python launcher_benchmark.py \
    --script /path/to/custom/run_medmnist_benchmark.py \
    --gpu 0
```

### Ensemble Evaluation (Legacy Mode)

```bash
python launcher_benchmark.py \
    --ensemble-eval \
    --datasets breastmnist \
    --gpu 0
```

## Execution Details

- **Sequential execution**: Runs one configuration at a time
- **Error handling**: Failures are logged but don't stop the pipeline
- **Caching**: Leverages cached model evaluations for speed
- **Per-fold evaluation**: Default mode (use `--ensemble-eval` for legacy behavior)

## Time Estimates

Approximate runtime per configuration (depends on dataset size and GPU):

| Dataset | Model | Methods | Estimated Time |
|---------|-------|---------|----------------|
| breastmnist | resnet18 | All (no MCD) | ~5-10 min |
| breastmnist | resnet18 | All (with MCD) | ~10-15 min |
| organamnist | resnet18 | All (no MCD) | ~30-45 min |
| organamnist | vit_b_16 | All (with MCD) | ~60-90 min |

**Full suite** (all datasets, all models, all setups): **30-50 hours**

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size or exclude memory-intensive methods:
```bash
python launcher_benchmark.py \
    --exclude-methods KNN_SHAP \
    --gpu 0
```

### Slow Execution

Use faster methods only:
```bash
python launcher_benchmark.py \
    --exclude-methods GPS KNN_Raw KNN_SHAP \
    --gpu 0
```

### Missing Models

The launcher expects models in:
```
benchmarks/medMNIST/models/{dataset}/{model}_fold_{0-4}.pth
```

For setup-specific models:
```
benchmarks/medMNIST/models/{dataset}/{model}_{setup}_fold_{0-4}.pth
```

### AMOS Dataset

If you get an error about Git LFS:
```bash
git lfs pull --include="benchmarks/medMNIST/Data/AMOS_2022/amos_external_test_224.npz"
```

## See Also

- `run_medmnist_benchmark.py` - Main benchmark script
- `README_corruption.md` - Covariate shift corruption guide
- `README.md` - General project documentation
