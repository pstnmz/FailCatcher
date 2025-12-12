# Path Configuration

This document describes how file paths are handled in the UQ_toolbox project to ensure portability across different environments.

## Overview

All hardcoded absolute paths have been replaced with **relative paths** based on automatic project root detection. This makes the code portable across different systems and directory structures.

## Project Root Detection

The project root is automatically detected by searching upward from the current file until a directory containing both `FailCatcher/` and `benchmarks/` is found. This works regardless of where the project is located on your filesystem.

## Default Path Structure

By default, the following relative paths are used:

```
<project_root>/
├── benchmarks/medMNIST/
│   ├── models/           # Trained model checkpoints
│   │   └── 224*224/      # Models for 224x224 images
│   ├── Data/             # Dataset files
│   │   └── AMOS_2022/    # AMOS external test set
│   └── runs/             # Training run outputs
└── FailCatcher/          # Core UQ library
```

## Environment Variables (Optional Overrides)

You can override default paths using environment variables:

### `MEDMNIST_MODELS_DIR`
Override the base directory for loading trained models.

**Example:**
```bash
export MEDMNIST_MODELS_DIR=/mnt/shared/models
python run_medmnist_benchmark.py --flag breastmnist
```

With this set, models will be loaded from:
```
/mnt/shared/models/224*224/{flag}_{backbone}_224_*.pt
```

### Future Variables (can be added as needed)
- `MEDMNIST_DATA_DIR` - Override data directory
- `MEDMNIST_RUNS_DIR` - Override training runs output directory

## Files Updated

The following files now use relative paths:

1. **`benchmarks/medMNIST/utils/train_models_load_datasets.py`**
   - `load_models()` - Uses project root + optional `MEDMNIST_MODELS_DIR`
   - `_get_project_root()` - Helper for project root detection

2. **`benchmarks/medMNIST/run_medmnist_benchmark.py`**
   - AMOS dataset loading uses relative path

3. **`benchmarks/medMNIST/utils/run_medmnist_benchmark.py`**
   - AMOS dataset loading uses relative path

4. **`benchmarks/medMNIST/trainings/launcher_resnet_training.py`**
   - Training script path is relative

5. **`benchmarks/medMNIST/trainings/launcher_vit_training.py`**
   - Training script path is relative

## Migration from Old Hardcoded Paths

If you have existing code or notebooks with hardcoded paths like:
```python
# OLD (hardcoded)
model_path = '/mnt/data/psteinmetz/.../models/224*224/model.pt'
```

Replace with:
```python
# NEW (portable)
from benchmarks.medMNIST.utils.train_models_load_datasets import _get_project_root
project_root = _get_project_root()
model_path = project_root / 'benchmarks' / 'medMNIST' / 'models' / '224*224' / 'model.pt'
```

Or simply use the provided functions like `load_models()` which handle paths automatically.

## Benefits

✅ **Portable** - Code works on any system without modification  
✅ **Flexible** - Environment variables allow custom paths when needed  
✅ **Maintainable** - No hardcoded paths to update when moving directories  
✅ **Team-friendly** - Works for all collaborators regardless of their setup
