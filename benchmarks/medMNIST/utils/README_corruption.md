# Data Corruption Utilities

## Overview
The `data_corruption_utils.py` module provides tools for applying covariate shift corruptions to medMNIST datasets using the [medmnistc library](https://github.com/francescodisalvo05/medmnistc-api).

## Features
- **Random corruption application**: Each sample gets a random corruption from the available pool
- **Deterministic selection**: Uses sample index as seed for reproducible corruption assignment
- **Caching**: Optional in-memory caching for faster repeated access
- **Severity control**: 5 severity levels (1=mild to 5=severe)

## Available Corruptions by Dataset

Run `python run_medmnist_benchmark.py --flag <dataset> --list-corruptions` to see available corruptions.

Examples:
- **breastmnist** (7 corruptions): pixelate, jpeg_compression, speckle_noise, motion_blur, brightness_up/down, contrast_down
- **dermamnist** (15 corruptions): All above + gaussian_noise, impulse_noise, shot_noise, defocus_blur, zoom_blur, contrast_up, black_corner, characters
- **organamnist** (13 corruptions): noise types, blur types, brightness/contrast/gamma adjustments

## Usage

### Command Line

```bash
# List available corruptions for a dataset
python run_medmnist_benchmark.py --flag breastmnist --list-corruptions

# Run benchmark with corrupted test set (severity 3)
python run_medmnist_benchmark.py --flag breastmnist \
    --methods MSR TTA \
    --corruption-severity 3 \
    --corrupt-test

# Run with both test and calibration sets corrupted (severity 5 = severe)
python run_medmnist_benchmark.py --flag organamnist \
    --methods GPS KNN_SHAP \
    --corruption-severity 5 \
    --corrupt-test \
    --corrupt-calib
```

### Python API

```python
from benchmarks.medMNIST.utils import data_corruption_utils as corruption_utils

# Apply random corruptions to a dataset
corrupted_dataset = corruption_utils.apply_random_corruptions(
    dataset, 
    flag='breastmnist',
    severity=3,              # 1-5 (mild to severe)
    cache=True,              # Cache corrupted images in RAM
    seed=42                  # For reproducibility
)

# Get available corruptions
corruptions = corruption_utils.get_available_corruptions('breastmnist')
print(corruptions)  # {'pixelate': <Pixelate>, 'jpeg_compression': <JPEGCompression>, ...}

# Apply specific corruption (advanced)
corrupted_dataset = corruption_utils.apply_specific_corruption(
    dataset,
    flag='dermamnist',
    corruption_type='gaussian_noise',
    severity=2,
    cache=True
)
```

## How It Works

1. **Random Selection**: For each sample index `i`, a corruption is selected deterministically using `Random(i).choice(corruptions)`
2. **Preprocessing**: Tensor is denormalized from `[-1,1]` or `[0,1]` to `[0,255]` uint8
3. **Corruption**: medmnistc corruption applied to PIL Image
4. **Postprocessing**: Corrupted image renormalized to match original range
5. **Caching**: Optionally cached in memory for repeated access

## Design Notes

- **Why random corruptions?**: Simulates realistic covariate shift where multiple types of degradation occur
- **Deterministic per-sample**: Same sample always gets same corruption (reproducible)
- **No augmentation on-the-fly**: All samples corrupted once (not multiple corruptions per sample)
- **Memory trade-off**: Caching uses RAM but speeds up multi-epoch evaluation

## Dependencies

- `medmnistc` - Install with: `pip install medmnistc`
- Requires: `medmnist`, `opencv-python`, `scikit-image`, `wand` (ImageMagick)

## Troubleshooting

If you get `ImportError: cannot import medmnistc`:
```bash
pip install medmnistc
# May also need: sudo apt-get install libmagickwand-dev
```

If corruptions are not listed for your dataset:
- Check if dataset is supported by medmnistc (see `CORRUPTIONS_DS` registry)
- Some datasets have limited corruption types based on imaging modality
