"""
MedMNIST Benchmark Runner using FailCatcher library.

This script demonstrates how to use the generic FailCatcher.benchmark API
for dataset-specific benchmarking.

Usage:
    python run_medmnist_benchmark.py --flag breastmnist --methods MSR Ensembling
    python run_medmnist_benchmark.py --flag organamnist --all-methods
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

# Import FailCatcher library
import FailCatcher
from FailCatcher import failure_detection
from FailCatcher import UQ_toolbox as uq

# Import medMNIST-specific utilities
from benchmarks.medMNIST.utils import train_models_load_datasets as tr
from benchmarks.medMNIST.utils import dataset_utils


def run_medmnist_benchmark(flag, methods, output_dir='./uq_benchmark_results',
                           batch_size=4000, image_size=224, gpu_id=0, per_fold_eval=True,
                           model_backbone='resnet18', setup='', gps_calib_samples=None,
                           min_failure_ratio=0.3, corruption_severity=0,
                           corrupt_test=False, corrupt_calib=False, new_class_shift=False):
    """
    Run UQ benchmark on a medMNIST dataset using FailCatcher library.
    
    Args:
        flag: Dataset name (e.g., 'breastmnist', 'organamnist')
        methods: List of method names to run
        output_dir: Output directory for results
        batch_size: Batch size for inference
        image_size: Image size
        gpu_id: GPU device ID to use
        gps_calib_samples: Max samples for GPS calibration (default: None = use all)
        min_failure_ratio: Minimum target proportion of failures (default: 0.3 = 30%)
        per_fold_eval: If True, compute per-fold metrics (mean±std). If False, use ensemble-based evaluation
        model_backbone: Model architecture ('resnet18' or 'vit_b_16')
        setup: Training setup - '' (standard), 'DA', 'DO', or 'DADO'
        corruption_severity: Corruption severity (0=disabled, 1-5=mild to severe covariate shift)
                            When enabled, randomly applies available medmnistc corruptions
        corrupt_test: If True, apply corruption to test set (requires corruption_severity > 0)
        corrupt_calib: If True, apply corruption to calibration set (requires corruption_severity > 0)
        new_class_shift: If True, create artificial test set with new classes (failures) + unanimous correct predictions
                        Only supported for AMOS2022 dataset
    """
    print(f"\n{'='*80}")
    print(f"MedMNIST Benchmark: {flag}")
    print(f"Using FailCatcher v{FailCatcher.__version__}")
    if new_class_shift:
        print(f"New Class Shift: Evaluating unseen classes (artificial test set)")
        print(f"  Test = New classes (failures) + Unanimous correct predictions (known classes)")
    if corruption_severity > 0:
        print(f"Covariate Shift: Random corruptions (severity={corruption_severity}/5)")
        print(f"  Test set: {'✓ Corrupted' if corrupt_test else '✗ Clean'}")
        print(f"  Calibration set: {'✓ Corrupted' if corrupt_calib else '✗ Clean'}")
    print(f"{'='*80}\n")
    
    # Set seeds for reproducibility
    import random
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # Enable deterministic algorithms for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("🔒 Deterministic mode enabled (seed=42)\n")
    
    # Get absolute path to workspace root (UQ_Toolbox/)
    workspace_root = Path(__file__).parent.parent.parent.absolute()
    
    # Make output_dir absolute if it's relative
    if not Path(output_dir).is_absolute():
        output_dir = str(workspace_root / output_dir)
    
    # Setup
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Parse dermamnist-e variants
    base_flag = flag
    test_subset = 'all'  # Default for all datasets
    if flag == 'dermamnist-e-id':
        base_flag = 'dermamnist-e'
        test_subset = 'id'
    elif flag == 'dermamnist-e-external':
        base_flag = 'dermamnist-e'
        test_subset = 'external'
    
    # AMOS uses organamnist models and calibration → use organamnist GPS cache
    gps_cache_flag = 'organamnist' if flag in ['amos2022', 'amos_external', 'amos22'] else base_flag
    
    color = base_flag in ['dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist']
    calib_method = 'platt' if base_flag in ['breastmnist', 'pneumoniamnist'] else 'temperature'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # LOAD DATA AND MODELS (medMNIST-specific)
    # ========================================================================
    print("📦 Loading medMNIST data and models...")
    
    # Transforms
    transform, transform_tta = dataset_utils.get_transforms(color, image_size)
    
    if base_flag != 'amos2022':
        # Load datasets and models
        models = tr.load_models(base_flag, device=device, size=image_size, 
                               model_backbone=model_backbone, setup=setup)
        [study_dataset, calib_dataset, test_dataset], \
        [_, calib_loader, test_loader], info = \
            tr.load_datasets(base_flag, color, image_size, transform, batch_size, test_subset=test_subset)
        
        [_, calib_dataset_tta, test_dataset_tta], \
        [_, _, _], _ = \
            tr.load_datasets(base_flag, color, image_size, transform_tta, batch_size, test_subset=test_subset)
        
        # Apply corruptions if requested
        if corruption_severity > 0 and (corrupt_test or corrupt_calib):
            print(f"\n🔬 Applying covariate shift corruptions...")
            # Map dataset name for corruption (dermamnist-e variants use 'dermamnist')
            if 'dermamnist' in base_flag:
                corruption_flag = 'dermamnist'
            else:
                corruption_flag = base_flag
            
            if corrupt_test:
                print(f"  → Corrupting test set (severity={corruption_severity}/5)")
                test_dataset = dataset_utils.apply_random_corruptions(
                    test_dataset, corruption_flag, corruption_severity, cache=True, seed=42
                )
                # For TTA: use return_pil=True to get uint8 tensors for proper augmentation
                test_dataset_tta = dataset_utils.apply_random_corruptions(
                    test_dataset_tta, corruption_flag, corruption_severity, cache=True, seed=42, return_pil=True
                )
                # Rebuild test loader with corrupted dataset
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=4, pin_memory=True
                )
            if corrupt_calib:
                print(f"  → Corrupting calibration set (severity={corruption_severity}/5)")
                calib_dataset = dataset_utils.apply_random_corruptions(
                    calib_dataset, corruption_flag, corruption_severity, cache=True, seed=42
                )
                # For TTA: use return_pil=True to get uint8 tensors for proper augmentation
                calib_dataset_tta = dataset_utils.apply_random_corruptions(
                    calib_dataset_tta, corruption_flag, corruption_severity, cache=True, seed=42, return_pil=True
                )
                # Rebuild calib loader with corrupted dataset
                calib_loader = DataLoader(
                    calib_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=4, pin_memory=True
                )
    else:
        # Load datasets and models of organamnist and amos2022 as test set
        models = tr.load_models('organamnist', device=device, size=image_size,
                               model_backbone=model_backbone, setup=setup)
        [study_dataset, calib_dataset, _], \
        [_, calib_loader, _], info = \
            tr.load_datasets('organamnist', color, image_size, transform, batch_size)
        
        # Load calibration dataset with TTA transform (for GPS augmentation caching)
        [_, calib_dataset_tta, _], \
        [_, _, _], _ = \
            tr.load_datasets('organamnist', color, image_size, transform_tta, batch_size)
        
        # Load AMOS external test dataset
        if new_class_shift:
            # Load full AMOS dataset including unmapped classes
            test_dataset, test_loader, test_dataset_tta = dataset_utils.load_amos_for_new_class_shift(
                transform, transform_tta, models, device, batch_size,
                workspace_root=Path(__file__).resolve().parent.parent.parent
            )
        else:
            # Load standard AMOS dataset (only mapped classes)
            test_dataset, test_loader, test_dataset_tta, _, _ = dataset_utils.load_amos_dataset(
                transform, transform_tta, batch_size, workspace_root=Path(__file__).resolve().parent.parent.parent
            )
        
        # Apply corruptions if requested
        if corruption_severity > 0 and corrupt_test:
            print(f"\n🔬 Applying covariate shift corruptions...")
            print(f"  → Corrupting test set (severity={corruption_severity}/5)")
            test_dataset = dataset_utils.apply_random_corruptions(
                test_dataset, flag, corruption_severity, cache=True, seed=42
            )
            # For TTA: use return_pil=True to get uint8 tensors for proper augmentation
            test_dataset_tta = dataset_utils.apply_random_corruptions(
                test_dataset_tta, flag, corruption_severity, cache=True, seed=42, return_pil=True
            )
            # Rebuild test loader
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
    
    print(f"  Models: {len(models)} folds")
    # For organamnist: study=train (medMNIST), calib=val (medMNIST)
    # For others: study=80% of (train+val), calib=20% of (train+val)
    study_label = "Train" if flag == 'organamnist' else "Train+val"
    calib_label = "Val" if flag == 'organamnist' else "Calib"
    print(f"  {study_label}: {len(study_dataset)}, {calib_label}: {len(calib_dataset)}, Test: {len(test_dataset)}")
    print(f"  Task: {info['task']}, Classes: {len(info['label'])}")
    
    # ========================================================================
    # EVALUATE MODELS (or load from cache)
    # ========================================================================
    
    # Cache file paths - include model backbone, setup, and corruption params to avoid mixing results
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    setup_suffix = f"_{setup}" if setup else ""
    
    # Add corruption parameters to cache key
    corruption_suffix = ""
    if corruption_severity > 0:
        corruption_suffix = f"_corrupt{corruption_severity}"
        if corrupt_test:
            corruption_suffix += "_test"
        if corrupt_calib:
            corruption_suffix += "_calib"
    
    # Add new class shift to cache key
    new_class_suffix = "_new_class_shift" if new_class_shift else ""
    
    calib_cache_path = os.path.join(cache_dir, f'{flag}_{model_backbone}{setup_suffix}{corruption_suffix}{new_class_suffix}_calib_results.npz')
    test_cache_path = os.path.join(cache_dir, f'{flag}_{model_backbone}{setup_suffix}{corruption_suffix}{new_class_suffix}_test_results.npz')
    
    # Try to load cached results FIRST
    cache_loaded = False
    if os.path.exists(calib_cache_path) and os.path.exists(test_cache_path):
        print("\n📦 Loading cached evaluation results...")
        try:
            calib_cache = np.load(calib_cache_path, allow_pickle=True)
            test_cache = np.load(test_cache_path, allow_pickle=True)
            cache_loaded = True
        except (pickle.UnpicklingError, ValueError, EOFError) as e:
            print(f"  ⚠️  Cache corrupted ({e.__class__.__name__}), regenerating...")
            # Delete corrupted cache files
            try:
                os.remove(calib_cache_path)
                os.remove(test_cache_path)
            except Exception:
                pass
            cache_loaded = False
    
    if cache_loaded:
        
        # Calibration
        y_true_calib = calib_cache['y_true']
        y_scores_calib = calib_cache['y_scores']
        correct_idx_calib = calib_cache['correct_idx']
        incorrect_idx_calib = calib_cache['incorrect_idx']
        indiv_scores_calib = calib_cache['indiv_scores']  # [N_calib, K, C]
        logits_calib = calib_cache['logits']  # [N_calib, C]
        
        # Test
        y_true = test_cache['y_true']
        y_scores = test_cache['y_scores']
        correct_idx = test_cache['correct_idx']
        incorrect_idx = test_cache['incorrect_idx']
        indiv_scores = test_cache['indiv_scores']  # [N, K, C]
        logits = test_cache['logits']  # [N, C]
        
        # For new class shift: override with binary ground truth
        if new_class_shift and hasattr(test_dataset, 'binary_gt'):
            print("  ✓ Using binary ground truth for new class shift evaluation (from cache)")
            y_true = test_dataset.binary_gt
            correct_idx = np.where(y_true == 0)[0]
            incorrect_idx = np.where(y_true == 1)[0]
        
        # Check if per-fold logits are cached (new format)
        if 'indiv_logits' in test_cache.files:
            indiv_logits = test_cache['indiv_logits']  # [N, K, C]
            indiv_logits_calib = calib_cache['indiv_logits']  # [N_calib, K, C]
        else:
            # Old cache format without per-fold logits
            indiv_logits = None
            indiv_logits_calib = None
        
        # Check if per-fold predictions are cached
        if 'per_fold_predictions' in test_cache.files:
            per_fold_predictions = test_cache['per_fold_predictions']  # [K, N]
            per_fold_predictions_calib = calib_cache['per_fold_predictions']  # [K, N_calib]
            print(f"  ✓ Loaded per-fold predictions from cache")
        else:
            # Old cache format - compute from indiv_scores
            print(f"  ⚠️  Old cache format detected - computing per-fold predictions...")
            # Compute from indiv_scores (still in [N, K, C] format at this point)
            per_fold_predictions = np.argmax(indiv_scores, axis=2).T  # [N, K] → [K, N]
            per_fold_predictions_calib = np.argmax(indiv_scores_calib, axis=2).T  # [N_calib, K] → [K, N_calib]
        
        # Check if per-fold correct/incorrect indices are cached
        if 'per_fold_correct_idx' in test_cache.files:
            per_fold_correct_idx = [arr for arr in test_cache['per_fold_correct_idx']]
            per_fold_incorrect_idx = [arr for arr in test_cache['per_fold_incorrect_idx']]
            per_fold_correct_idx_calib = [arr for arr in calib_cache['per_fold_correct_idx']]
            per_fold_incorrect_idx_calib = [arr for arr in calib_cache['per_fold_incorrect_idx']]
            print(f"  ✓ Loaded per-fold correct/incorrect indices from cache")
        else:
            # Old cache format - need to compute from per_fold_predictions
            print(f"  ⚠️  Computing per-fold indices from predictions...")
            per_fold_correct_idx = []
            per_fold_incorrect_idx = []
            per_fold_correct_idx_calib = []
            per_fold_incorrect_idx_calib = []
            
            for fold_idx in range(per_fold_predictions.shape[0]):
                # For new class shift: use binary ground truth
                if new_class_shift and hasattr(test_dataset, 'binary_gt'):
                    fold_correct = np.where(y_true == 0)[0]
                    fold_incorrect = np.where(y_true == 1)[0]
                else:
                    fold_correct = np.where(per_fold_predictions[fold_idx] == y_true)[0]
                    fold_incorrect = np.where(per_fold_predictions[fold_idx] != y_true)[0]
                per_fold_correct_idx.append(fold_correct)
                per_fold_incorrect_idx.append(fold_incorrect)
                
                fold_correct_calib = np.where(per_fold_predictions_calib[fold_idx] == y_true_calib)[0]
                fold_incorrect_calib = np.where(per_fold_predictions_calib[fold_idx] != y_true_calib)[0]
                per_fold_correct_idx_calib.append(fold_correct_calib)
                per_fold_incorrect_idx_calib.append(fold_incorrect_calib)
        
        # Transpose to [K, N, C] format for per-fold evaluation
        indiv_scores = np.transpose(indiv_scores, (1, 0, 2))  # [K, N, C]
        indiv_scores_calib = np.transpose(indiv_scores_calib, (1, 0, 2))  # [K, N_calib, C]
        if indiv_logits is not None:
            indiv_logits = np.transpose(indiv_logits, (1, 0, 2))  # [K, N, C]
            indiv_logits_calib = np.transpose(indiv_logits_calib, (1, 0, 2))  # [K, N_calib, C]
        
        # Compute y_pred from cached scores (for compatibility with old cache format)
        y_pred = np.argmax(y_scores, axis=1)
        y_pred_calib = np.argmax(y_scores_calib, axis=1)
        
        # For new class shift: extract binary_gt from cache and use it as y_true for risk computation
        if new_class_shift and 'binary_gt' in test_cache.files:
            binary_gt = test_cache['binary_gt']
            print(f"  ✓ Loaded cached results (new class shift mode)")
            print(f"  Test accuracy: {len(correct_idx) / len(binary_gt):.4f} (failure rate: {np.sum(binary_gt)/len(binary_gt):.4f})")
            # Override y_true with binary_gt for proper risk computation
            y_true = binary_gt
        else:
            print(f"  ✓ Loaded cached results")
            print(f"  Test accuracy: {len(correct_idx) / len(y_true):.4f}")
    
    else:
        # No cache - evaluate models
        print("\n📊 Evaluating ensemble predictions on test set...")
        y_true, y_scores, y_pred, correct_idx, incorrect_idx, indiv_scores_raw, logits = uq.evaluate_models_on_loader(
            models, test_loader, device, return_logits=True
        )
        
        # For new class shift: replace y_true with binary ground truth for failure detection
        if new_class_shift and hasattr(test_dataset, 'binary_gt'):
            print("  ✓ Using binary ground truth for new class shift evaluation")
            y_true_original = y_true.copy()  # Keep original labels
            y_true = test_dataset.binary_gt  # Binary: 0=correct (known class), 1=failure (new class)
            
            # Recompute correct/incorrect based on binary ground truth
            # Correct = known class samples (-1 in original labels means new class)
            correct_idx = np.where(y_true == 0)[0]  # Known classes
            incorrect_idx = np.where(y_true == 1)[0]  # New classes (all failures by definition)
            print(f"  Binary GT: {len(correct_idx)} known class (correct), {len(incorrect_idx)} new class (failures)")
        
        # Calibration set
        y_true_calib, y_scores_calib, y_pred_calib, correct_idx_calib, incorrect_idx_calib, indiv_scores_calib_raw, logits_calib = \
            uq.evaluate_models_on_loader(models, calib_loader, device, return_logits=True)
        
        print(f"  Test accuracy: {len(correct_idx)/len(y_true):.4f}")
        
        # Compute per-fold logits from models
        print("\n📊 Computing per-fold logits for calibration...")
        indiv_logits_raw = []  # Will be [N, K, C]
        indiv_logits_calib_raw = []  # Will be [N_calib, K, C]
        
        for model in models:
            model.eval()
            # Test set
            test_logits_fold = []
            with torch.no_grad():
                for batch in test_loader:
                    if isinstance(batch, dict):
                        images = batch["image"].to(device)
                    else:
                        images = batch[0].to(device)
                    logits_batch = model(images)
                    test_logits_fold.append(logits_batch.cpu().numpy())
            indiv_logits_raw.append(np.concatenate(test_logits_fold, axis=0))  # [N, C]
            
            # Calibration set
            calib_logits_fold = []
            with torch.no_grad():
                for batch in calib_loader:
                    if isinstance(batch, dict):
                        images = batch["image"].to(device)
                    else:
                        images = batch[0].to(device)
                    logits_batch = model(images)
                    calib_logits_fold.append(logits_batch.cpu().numpy())
            indiv_logits_calib_raw.append(np.concatenate(calib_logits_fold, axis=0))  # [N_calib, C]
        
        # Stack to [K, N, C] and [K, N_calib, C]
        indiv_logits_raw = np.stack(indiv_logits_raw, axis=1)  # [N, K, C]
        indiv_logits_calib_raw = np.stack(indiv_logits_calib_raw, axis=1)  # [N_calib, K, C]
        
        # Transpose to [K, N, C] for per-fold evaluation
        indiv_scores = np.transpose(indiv_scores_raw, (1, 0, 2))  # [K, N, C]
        indiv_scores_calib = np.transpose(indiv_scores_calib_raw, (1, 0, 2))  # [K, N_calib, C]
        indiv_logits = np.transpose(indiv_logits_raw, (1, 0, 2))  # [K, N, C]
        indiv_logits_calib = np.transpose(indiv_logits_calib_raw, (1, 0, 2))  # [K, N_calib, C]
        
        # Compute per-fold correct/incorrect indices
        print("\n📊 Computing per-fold correct/incorrect indices...")
        per_fold_correct_idx = []
        per_fold_incorrect_idx = []
        per_fold_correct_idx_calib = []
        per_fold_incorrect_idx_calib = []
        
        for fold_idx in range(len(models)):
            # Test set
            fold_preds = np.argmax(indiv_scores[fold_idx], axis=1)  # [N]
            
            # For new class shift: use binary ground truth
            if new_class_shift and hasattr(test_dataset, 'binary_gt'):
                # Known classes (binary_gt=0): correct if model predicts correct original label
                # New classes (binary_gt=1): always incorrect (failures by definition)
                fold_correct = np.where(y_true == 0)[0]  # All known class samples
                fold_incorrect = np.where(y_true == 1)[0]  # All new class samples
            else:
                fold_correct = np.where(fold_preds == y_true)[0]
                fold_incorrect = np.where(fold_preds != y_true)[0]
            
            per_fold_correct_idx.append(fold_correct)
            per_fold_incorrect_idx.append(fold_incorrect)
            
            # Calibration set
            fold_preds_calib = np.argmax(indiv_scores_calib[fold_idx], axis=1)  # [N_calib]
            fold_correct_calib = np.where(fold_preds_calib == y_true_calib)[0]
            fold_incorrect_calib = np.where(fold_preds_calib != y_true_calib)[0]
            per_fold_correct_idx_calib.append(fold_correct_calib)
            per_fold_incorrect_idx_calib.append(fold_incorrect_calib)
            
            print(f"  Fold {fold_idx}: {len(fold_correct)} correct, {len(fold_incorrect)} incorrect (test)")
        
        # Compute per-fold predictions [M, N] for caching
        per_fold_predictions = np.argmax(indiv_scores, axis=2)  # [K, N, C] → [K, N]
        per_fold_predictions_calib = np.argmax(indiv_scores_calib, axis=2)  # [K, N_calib, C] → [K, N_calib]
        
        # Save to cache for next time
        print("\n💾 Saving evaluation results to cache...")
        np.savez_compressed(
            calib_cache_path,
            y_true=y_true_calib,
            y_scores=y_scores_calib,
            y_pred=y_pred_calib,
            correct_idx=correct_idx_calib,
            incorrect_idx=incorrect_idx_calib,
            indiv_scores=indiv_scores_calib_raw,  # Save as [N, K, C]
            logits=logits_calib,
            indiv_logits=indiv_logits_calib_raw,  # Save as [N, K, C]
            per_fold_correct_idx=np.array(per_fold_correct_idx_calib, dtype=object),  # Object array of variable-length arrays
            per_fold_incorrect_idx=np.array(per_fold_incorrect_idx_calib, dtype=object),  # Object array
            per_fold_predictions=per_fold_predictions_calib  # [K, N_calib]
        )
        # Save test cache (with binary_gt if new_class_shift)
        cache_data = dict(
            y_true=y_true,
            y_scores=y_scores,
            y_pred=y_pred,
            correct_idx=correct_idx,
            incorrect_idx=incorrect_idx,
            indiv_scores=indiv_scores_raw,  # Save as [N, K, C]
            logits=logits,
            indiv_logits=indiv_logits_raw,  # Save as [N, K, C]
            per_fold_correct_idx=np.array(per_fold_correct_idx, dtype=object),  # Object array of variable-length arrays
            per_fold_incorrect_idx=np.array(per_fold_incorrect_idx, dtype=object),  # Object array
            per_fold_predictions=per_fold_predictions  # [K, N]
        )
        if new_class_shift and hasattr(test_dataset, 'binary_gt'):
            cache_data['binary_gt'] = test_dataset.binary_gt  # Save binary ground truth for risk computation
        np.savez_compressed(test_cache_path, **cache_data)
        print(f"  ✓ Cached to {cache_dir}")
    
    # ========================================================================
    # CREATE FAILCATCHER DETECTOR
    # ========================================================================
    detector = failure_detection.FailureDetector(
        models=models,
        study_dataset=study_dataset,
        calib_dataset=calib_dataset,
        test_dataset=test_dataset,
        device=device,
        num_classes=len(info['label'])
    )
    
    # Set predictions once to avoid recomputing for each method
    # For new_class_shift: pass pre-computed correct/incorrect indices to avoid recomputation
    if new_class_shift:
        detector.set_test_predictions(y_scores, y_true, y_pred, correct_idx, incorrect_idx,
                                      per_fold_correct_idx, per_fold_incorrect_idx)
    else:
        detector.set_test_predictions(y_scores, y_true, y_pred)
    
    # Set per-fold predictions to avoid redundant vanilla inference
    # This is especially important when running multiple UQ methods
    detector.set_per_fold_predictions(per_fold_predictions)
    print("  ✓ Pre-cached per-fold predictions - vanilla inference will be skipped")
    
    # Adaptive batch size for KNN methods based on model architecture
    # KNN requires full forward passes on large datasets which can OOM
    knn_batch_size = min(batch_size, 3000)  # Conservative default for all models
    knn_test_loader = test_loader
    
    # TTA/GPS batch size - also needs reduction for ViT on large datasets
    tta_gps_batch_size = batch_size
    if model_backbone == 'vit_b_16' and batch_size > 3000:
        tta_gps_batch_size = 3000
        print(f"  ℹ️  Using reduced batch size {tta_gps_batch_size} for TTA/GPS with ViT (avoids OOM)")
    
    # Further reduce for ViT models which consume significantly more memory
    if model_backbone == 'vit_b_16':
        knn_batch_size = min(batch_size, 4000)  # Reduce to 4000 for ViT to avoid OOM
        print(f"  ℹ️  Using reduced batch size {knn_batch_size} for KNN with ViT (avoids OOM)")
    elif knn_batch_size < batch_size:
        print(f"  ℹ️  Using reduced batch size {knn_batch_size} for KNN (avoids OOM on large datasets)")
    
    # Create reduced batch size test loader if needed
    if knn_batch_size < batch_size:
        knn_test_loader = DataLoader(
            test_dataset, batch_size=knn_batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
    
    # Create CV train loaders for KNN methods
    cv_gen = dataset_utils.create_cv_generator(n_splits=5, seed=42, batch_size=knn_batch_size)
    train_loaders = cv_gen(study_dataset, models, knn_batch_size)
    
    # Create KNN-specific calibration loader with reduced batch size (for hyperparameter tuning)
    if knn_batch_size < batch_size:
        knn_calib_loader = DataLoader(
            calib_dataset, batch_size=knn_batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
    else:
        knn_calib_loader = calib_loader  # Use original loader if batch size is same
    
    # ========================================================================
    # RUN UQ METHODS using FailCatcher API
    # ========================================================================
    results = {}
    
    if 'MSR' in methods:
        print("\n🔍 Running MSR...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        uncertainties, metrics = detector.run_msr(
            y_scores, y_true, 
            indiv_scores=indiv_scores if per_fold_eval else None,
            per_fold_evaluation=per_fold_eval
        )
        results['MSR'] = metrics
        if 'auroc_f_mean' in metrics:
            print(f"  AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                  f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
        else:
            print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'MSR_calibrated' in methods:
        print(f"\n🔍 Running MSR-{calib_method}...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        uncertainties, metrics = detector.run_msr_calibrated(
            y_scores, y_true, y_scores_calib, y_true_calib,
            logits, logits_calib,
            indiv_logits_test=indiv_logits if (per_fold_eval and indiv_logits is not None) else None,
            indiv_logits_calib=indiv_logits_calib if (per_fold_eval and indiv_logits_calib is not None) else None,
            indiv_scores_test=indiv_scores if per_fold_eval else None,
            indiv_scores_calib=indiv_scores_calib if per_fold_eval else None,
            method=calib_method,
            per_fold_evaluation=per_fold_eval,
            auto_tune_platt=True,  # Enable automatic hyperparameter selection
            verbose_tuning=True    # Print tuning results
        )
        results[f'MSR_{calib_method}'] = metrics
        if 'auroc_f_mean' in metrics:
            print(f"  AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                  f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
        else:
            print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'MLS' in methods:
        print("\n🔍 Running MLS (Maximum Logit Score)...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        uncertainties, metrics = detector.run_mls(
            logits, y_true,
            indiv_logits=indiv_logits if per_fold_eval else None,
            per_fold_evaluation=per_fold_eval
        )
        results['MLS'] = metrics
        if 'auroc_f_mean' in metrics:
            print(f"  AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                  f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
        else:
            print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'Ensembling' in methods:
        print("\n🔍 Running Ensemble STD...")
        uncertainties, metrics = detector.run_ensemble(indiv_scores, y_true)
        results['Ensemble'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'TTA' in methods:
        print("\n🔍 Running TTA...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        uncertainties, metrics = detector.run_tta(
            test_dataset_tta, y_true,
            image_size=image_size,
            batch_size=tta_gps_batch_size,
            nb_augmentations=5,
            per_fold_evaluation=per_fold_eval,
            seed=42
        )
        results['TTA'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'TTA_calib' in methods:
        print("\n🔍 Running TTA Calibration Caching (BetterRandAugment)...")
        if gps_calib_samples is not None:
            print(f"  Subsampling strategy: Keep {gps_calib_samples} samples with min {min_failure_ratio:.0%} failures")
        setup_name = setup if setup else 'standard'
        # Include sample count in folder name if subsampling occurs
        folder_suffix = f'_N{gps_calib_samples}' if gps_calib_samples is not None else ''
        aug_folder = os.path.join(output_dir, 'gps_augment_cache', f'{gps_cache_flag}_{model_backbone}_{setup_name}_calibration_set{folder_suffix}')
        
        # Subsample calibration dataset (failure-aware for GPS)
        # Prioritizes failures (incorrect predictions) to maximize information density
        # Using fixed seed ensures consistency between TTA_calib and GPS search
        # CRITICAL: Pass normalized calib_dataset for accurate ensemble inference,
        #           but return subset of unnormalized calib_dataset_tta for augmentation
        # Returns: (subsampled_dataset, correct_indices, incorrect_indices)
        calib_dataset_tta_subsampled, correct_idx_calib_subsampled, incorrect_idx_calib_subsampled = \
            dataset_utils.subsample_dataset_failure_aware(
                dataset=calib_dataset_tta,
                models=models,
                device=device,
                max_samples=gps_calib_samples,
                min_failure_ratio=min_failure_ratio,
                seed=42,
                batch_size=batch_size,
                eval_dataset=calib_dataset  # Use normalized data for ensemble inference!
            )
        
        # correct_idx_calib_subsampled and incorrect_idx_calib_subsampled 
        # are already computed by subsample_dataset_failure_aware()
        # They are 0-indexed positions within the subsampled dataset
        print(f"  GPS will use: {len(correct_idx_calib_subsampled)} correct, {len(incorrect_idx_calib_subsampled)} incorrect indices")
        
        # Determine normalization parameters based on color
        # Note: nb_channels should always be 3 because models expect 3-channel input
        # (grayscale is converted via RepeatGrayToRGB in the dataset)
        nb_channels = 3
        mean = [.5, .5, .5] if color else [.5]
        std = [.5, .5, .5] if color else [.5]
        
        # Use smaller batch size for augmentation caching to avoid OOM
        # Each batch gets multiplied by num_policies, so memory usage is much higher
        aug_batch_size = min(batch_size, 256)  # Conservative for memory safety
        
        # Use MONAI cache with full rate for speed
        # Cache is stored in RAM (CPU memory), not GPU, so it's safe
        print(f"  Original calibration set size: {len(calib_dataset_tta)}")
        print(f"  Subsampled calibration set size: {len(calib_dataset_tta_subsampled)}")
        print(f"  Batch size: {aug_batch_size}")
        
        # Cache augmentation predictions on calibration dataset
        aug_folder = detector.run_augmentation_calibration_caching(
            dataset=calib_dataset_tta_subsampled,  # Use subsampled dataset!
            aug_folder=aug_folder,
            N=2,                      # Number of augmentation ops per policy
            M=45,                     # Magnitude parameter
            num_policies=500,         # Number of random policies to generate
            image_size=image_size,
            batch_size=aug_batch_size,
            nb_channels=nb_channels,
            image_normalization=True,
            mean=mean,
            std=std,
            use_monai_cache=True,
            cache_rate=1.0,  # Full cache in RAM for speed
            cache_num_workers=8,
            dataloader_workers=6,  # More workers with persistent mode for speed
            dataloader_prefetch=2  # Conservative prefetch to avoid OOM
        )
        print(f"  ✓ Augmentation predictions cached in: {aug_folder}")
        # Note: TTA_calib doesn't produce uncertainty scores, it only caches predictions
        # The cached predictions are used by GPS method
    

    if 'GPS' in methods:
        print("\n🔍 Running GPS...")
        
        # CRITICAL: Reset test_dataset_tta transform to original state
        # TTA may have modified it, and GPS needs clean dataset
        test_dataset_tta.transform = transform_tta
        
        setup_name = setup if setup else 'standard'
        # Include sample count in folder name if subsampling occurs
        folder_suffix = f'_N{gps_calib_samples}' if gps_calib_samples is not None else ''
        aug_folder = os.path.join(output_dir, 'gps_augment_cache', f'{gps_cache_flag}_{model_backbone}_{setup_name}_calibration_set{folder_suffix}')
        
        # If TTA_calib was run, use the subsampled indices
        # Otherwise, compute them now (GPS can run independently of TTA_calib)
        if 'TTA_calib' in methods:
            # Use the subsampled indices computed during TTA_calib
            gps_correct_idx = correct_idx_calib_subsampled.tolist()
            gps_incorrect_idx = incorrect_idx_calib_subsampled.tolist()
            print(f"  Using subsampled calibration indices from TTA_calib")
        else:
            # GPS running independently - need to subsample and compute indices
            print(f"  TTA_calib not run - computing subsampled calibration indices...")
            calib_dataset_tta_subsampled, gps_correct_idx, gps_incorrect_idx = \
                dataset_utils.subsample_dataset_failure_aware(
                    dataset=calib_dataset_tta,
                    models=models,
                    device=device,
                    max_samples=gps_calib_samples,
                    eval_dataset=calib_dataset,  # Use normalized data for ensemble inference!
                    min_failure_ratio=min_failure_ratio,
                    seed=42,
                    batch_size=batch_size
                )
            # Convert numpy arrays to lists for GPS
            gps_correct_idx = gps_correct_idx.tolist()
            gps_incorrect_idx = gps_incorrect_idx.tolist()

        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        uncertainties, metrics = detector.run_gps(
            test_dataset_tta, y_true,
            aug_folder=aug_folder,
            correct_idx_calib=gps_correct_idx,
            incorrect_idx_calib=gps_incorrect_idx,
            image_size=image_size,
            batch_size=tta_gps_batch_size,
            cache_dir=os.path.join(output_dir, 'gps_augment_cache'),
            per_fold_evaluation=per_fold_eval
        )
        results['GPS'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'KNN_Raw' in methods:
        print("\n🔍 Running KNN-Raw...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        
        # For new_class_shift: use fixed k=1000 (no calibration set available)
        # For other shifts: use grid search on calibration set
        if new_class_shift:
            k = 1000
            k_grid = None
            print(f"  New class shift detected: Using fixed k={k} (no calibration)")
        else:
            k = None
            k_grid = [1, 5, 10, 20, 50, 100, 200]
            print(f"  Using k grid search: {k_grid}")
        
        uncertainties, metrics = detector.run_knn_raw(
            test_loader=knn_test_loader,
            train_loaders=train_loaders,
            y_true=y_true,
            layer_name='avgpool',
            k=k,
            per_fold_evaluation=per_fold_eval,
            k_grid=k_grid,
            calib_loader=knn_calib_loader,
            y_true_calib=y_true_calib
        )
        results['KNN_Raw'] = metrics
        
        # Print results
        if 'auroc_f_mean' in metrics:
            print(f"  AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                  f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
        else:
            print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'KNN_SHAP' in methods:
        print("\n🔍 Running KNN-SHAP...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        parallel_mode = torch.cuda.device_count() >= 3
        n_jobs = 3 if parallel_mode else 1
        
        uncertainties, metrics = detector.run_knn_shap(
            calib_loader=calib_loader,
            test_loader=knn_test_loader,
            train_loaders=train_loaders,
            y_true=y_true,
            flag=flag,
            layer_name='avgpool',
            k=5,
            n_shap_features=50,
            cache_dir=os.path.join(output_dir, 'shap_cache'),
            parallel=parallel_mode,
            n_jobs=n_jobs,
            per_fold_evaluation=per_fold_eval
        )
        results['KNN_SHAP'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    # ========================================================================
    # MC DROPOUT - RUN LAST TO AVOID INTERFERING WITH OTHER METHODS
    # ========================================================================
    # MCDropout is executed last because it modifies dropout layer states
    # which can cause CUDA RNG issues with subsequent DataLoader forking
    # in methods like GPS/TTA that use multiprocessing workers
    if 'MCDropout' in methods:
        print("\n🔍 Running MC Dropout (running last to avoid interference)...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        uncertainties, metrics = detector.run_mcdropout(
            test_dataset, y_true,
            batch_size=batch_size,
            num_samples=30,
            per_fold_evaluation=per_fold_eval
        )
        results['MCDropout'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    # ========================================================================
    # SAVE RESULTS AND FIGURES (via FailureDetector)
    # ========================================================================
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build corruption info string for filenames
    corruption_info = None
    if corruption_severity > 0:
        corruption_parts = [f"severity{corruption_severity}"]
        if corrupt_test:
            corruption_parts.append("test")
        if corrupt_calib:
            corruption_parts.append("calib")
        corruption_info = "_".join(corruption_parts)
    
    # Override output directory for new class shift
    if new_class_shift:
        output_dir = os.path.join(output_dir, 'new_class_shifts')
        os.makedirs(output_dir, exist_ok=True)
    
    # Save all results using the detector's save_results method
    saved_paths = detector.save_results(
        output_dir=output_dir,
        flag=flag,
        timestamp=timestamp,
        model_backbone=model_backbone,
        setup=setup,
        corruption_info=corruption_info
    )
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"{'Method':<20} {'AUROC_f':<20} {'AURC':<20} {'AUGRC':<20} {'Accuracy':<10}")
    print("-"*100)
    for method_name, method_results in results.items():
        # Check if per-fold metrics exist (mean and std)
        if 'auroc_f_mean' in method_results and 'auroc_f_std' in method_results:
            # Per-fold evaluation: show mean±std
            auroc_str = f"{method_results['auroc_f_mean']:.4f}±{method_results['auroc_f_std']:.4f}"
            aurc_str = f"{method_results['aurc_mean']:.6f}±{method_results['aurc_std']:.6f}"
            augrc_str = f"{method_results['augrc_mean']:.6f}±{method_results['augrc_std']:.6f}"
        else:
            # Single evaluation: show just the value
            auroc_str = f"{method_results['auroc_f']:.4f}"
            aurc_str = f"{method_results['aurc']:.6f}"
            augrc_str = f"{method_results['augrc']:.6f}"
        
        print(f"{method_name:<20} "
              f"{auroc_str:<20} "
              f"{aurc_str:<20} "
              f"{augrc_str:<20} "
              f"{method_results['accuracy']:<10.4f}")
    print("="*100)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark FailCatcher UQ methods on medMNIST datasets'
    )
    parser.add_argument(
        '--flag', type=str, required=True,
        choices=['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'dermamnist-e',
                'dermamnist-e-id', 'dermamnist-e-external', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'amos2022', 'midog'],
        help='MedMNIST dataset to benchmark. For dermamnist-e, use -id for ID centers or -external for OOD center'
    )
    
    parser.add_argument(
        '--model', type=str, default='resnet18', choices=['resnet18', 'vit_b_16'],
        help='Model backbone to use (default: resnet18)'
    )
    
    parser.add_argument(
        '--setup', type=str,
        choices=['DA', 'DO', 'DADO'], default='',
        help='Load models trained under different setups (DA: data augmentation, DO: dropout, DADO: both). Default is standard training.'
    )
    
    parser.add_argument(
        '--methods', nargs='+',
        default=['MSR', 'MSR_calibrated', 'MLS', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP', 'MCDropout'],
        choices=['MSR', 'MSR_calibrated', 'MLS', 'Ensembling', 'TTA', 'GPS', 'TTA_calib', 'KNN_Raw', 'KNN_SHAP', 'MCDropout'],
        help='UQ methods to run (MCDropout runs last to avoid interference with other methods)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./uq_benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4000,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID to use (default: 0)'
    )
    parser.add_argument(
        '--per-fold-eval', action='store_true', default=False,
        help='Use per-fold evaluation (mean±std). If not set, uses ensemble-based evaluation (default: False for backward compatibility)'
    )
    parser.add_argument(
        '--ensemble-eval', dest='per_fold_eval', action='store_false',
        help='Use ensemble-based evaluation (legacy mode)'
    )
    parser.add_argument(
        '--gps-calib-samples', type=int, default=None,
        help='Maximum number of calibration samples for GPS augmentation caching (default: None = use all). Specify a number to subsample (e.g., 2000, 3000).'
    )
    parser.add_argument(
        '--min-failure-ratio', type=float, default=0.3,
        help='Minimum target proportion of failures in GPS calibration subsampling (default: 0.3 = 30%%). Will keep all available failures if less than this ratio.'
    )
    
    # Covariate shift / corruption arguments
    parser.add_argument(
        '--corruption-severity', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
        help='Apply random covariate shift corruptions. 0=disabled (clean), 1=mild to 5=severe (default: 0)'
    )
    parser.add_argument(
        '--corrupt-test', action='store_true', default=False,
        help='Apply corruption to test set (requires --corruption-severity > 0)'
    )
    parser.add_argument(
        '--corrupt-calib', action='store_true', default=False,
        help='Apply corruption to calibration set (requires --corruption-severity > 0)'
    )
    parser.add_argument(
        '--list-corruptions', action='store_true',
        help='List available corruptions for the specified dataset and exit'
    )
    parser.add_argument(
        '--new-class-shift', action='store_true', default=False,
        help='Evaluate new class shift (AMOS and MIDOG only): Create artificial test sets with new classes (failures) + unanimous correct predictions (known classes)'
    )
    
    args = parser.parse_args()
    
    # Handle --list-corruptions flag
    if args.list_corruptions:
        print(f"\nAvailable corruptions for {args.flag}:")
        corruptions = dataset_utils.list_available_corruptions(args.flag)
        if corruptions:
            print(f"  Random corruptions will be applied from this pool:")
            for c in sorted(corruptions):
                print(f"    - {c}")
            print(f"\n  Usage example (random corruptions):")
            print(f"    python run_medmnist_benchmark.py --flag {args.flag} --corruption-severity 3 --corrupt-test")
            print(f"\n  Each sample gets a random corruption from the pool at the specified severity.")
        else:
            print(f"  No corruptions available for {args.flag}")
            if not dataset_utils.MEDMNISTC_AVAILABLE:
                print(f"  (medmnistc is not installed - run: pip install medmnistc)")
        sys.exit(0)
    
    run_medmnist_benchmark(
        flag=args.flag,
        methods=args.methods,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu_id=args.gpu,
        per_fold_eval=args.per_fold_eval,
        model_backbone=args.model,
        setup=args.setup,
        gps_calib_samples=args.gps_calib_samples,
        min_failure_ratio=args.min_failure_ratio,
        corruption_severity=args.corruption_severity,
        corrupt_test=args.corrupt_test,
        corrupt_calib=args.corrupt_calib,
        new_class_shift=args.new_class_shift
    )
