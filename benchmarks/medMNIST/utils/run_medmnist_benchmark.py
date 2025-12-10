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


class RepeatGrayToRGB:
    """Transform for converting grayscale to RGB."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)


def subsample_dataset_failure_aware(dataset, models, device, max_samples=None, 
                                     min_failure_ratio=0.3, seed=42, batch_size=256):
    """
    Subsample calibration dataset prioritizing failures for GPS.
    
    Returns:
        tuple: (subsampled_dataset, correct_indices, incorrect_indices)
            - subsampled_dataset: Subset of the dataset
            - correct_indices: Indices of correctly predicted samples in the subset (0-indexed)
            - incorrect_indices: Indices of incorrectly predicted samples in the subset (0-indexed)
    
    Strategy for GPS augmentation search:
    1. Run ensemble inference to identify correct/incorrect predictions
    2. Keep ALL failures if possible (up to min_failure_ratio * max_samples)
    3. Fill remaining slots with stratified correct predictions
    4. If not enough failures, lower the ratio and add more correct samples
    
    This maximizes information density for GPS by focusing on model weaknesses
    while maintaining class distribution and reproducibility.
    
    Args:
        dataset: PyTorch dataset (calibration set)
        models: List of trained models for ensemble prediction
        device: Device for inference
        max_samples: Target sample count (default: None = use all)
        min_failure_ratio: Minimum proportion of failures to target (default: 0.3)
                          Actual ratio may be lower if not enough failures exist
        seed: Random seed for reproducibility (default: 42)
        batch_size: Batch size for inference (default: 256)
    
    Returns:
        Subset: Subsampled dataset with bias toward failures
    
    Example:
        If max_samples=2000, min_failure_ratio=0.3, and 500 failures available:
        - Keep all 500 failures (25% of final set)
        - Sample 1500 correct predictions (75% of final set)
        
        If max_samples=2000, min_failure_ratio=0.3, and 800 failures available:
        - Keep all 800 failures (40% of final set, exceeds minimum)
        - Sample 1200 correct predictions (60% of final set)
    """
    if max_samples is None or len(dataset) <= max_samples:
        # No subsampling needed - compute indices on full dataset
        print(f"  No subsampling needed (dataset size: {len(dataset)})")
        
        # Still need to compute correct/incorrect indices for GPS
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.squeeze().item() if label.numel() == 1 else label.squeeze().cpu().numpy()
            if isinstance(label, np.ndarray):
                label = label.item() if label.size == 1 else label[0]
            labels.append(int(label))
        labels = np.array(labels)
        
        # Get ensemble predictions
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
        
        all_preds = []
        for model in models:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, dict):
                        images = batch['image'].to(device)
                    else:
                        images = batch[0].to(device)
                    outputs = model(images)
                    pred = torch.argmax(outputs, dim=1).cpu().numpy()
                    preds.extend(pred)
            all_preds.append(np.array(preds))
        
        all_preds = np.array(all_preds)
        ensemble_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds
        )
        
        correct_mask = (ensemble_preds == labels)
        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(~correct_mask)[0]
        
        print(f"    Full dataset: {len(correct_indices)} correct, {len(incorrect_indices)} incorrect")
        return dataset, correct_indices, incorrect_indices
    
    print(f"  Running ensemble inference to identify failures...")
    
    # Extract labels and get predictions
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.squeeze().item() if label.numel() == 1 else label.squeeze().cpu().numpy()
        if isinstance(label, np.ndarray):
            label = label.item() if label.size == 1 else label[0]
        labels.append(int(label))
    labels = np.array(labels)
    
    print(f"  Extracted {len(labels)} labels from calibration dataset")
    
    # Get ensemble predictions
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=4, pin_memory=True)
    
    all_preds = []
    for model_idx, model in enumerate(models):
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(device)
                else:
                    images = batch[0].to(device)
                
                outputs = model(images)
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(pred)
        preds_array = np.array(preds)
        all_preds.append(preds_array)
    
    # Ensemble: majority vote
    all_preds = np.array(all_preds)  # (n_models, n_samples)
    print(f"  Ensemble predictions shape: {all_preds.shape} (models × samples)")
    
    ensemble_preds = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=all_preds
    )
    
    print(f"  Final ensemble: {len(ensemble_preds)} predictions, {len(labels)} labels")
    
    # Sanity check: lengths must match
    assert len(ensemble_preds) == len(labels), f"Shape mismatch: {len(ensemble_preds)} preds vs {len(labels)} labels"
    assert len(ensemble_preds) == len(dataset), f"Count mismatch: {len(ensemble_preds)} preds vs {len(dataset)} samples"
    
    # Identify correct and incorrect predictions
    correct_mask = (ensemble_preds == labels)
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(~correct_mask)[0]
    
    n_correct = len(correct_indices)
    n_incorrect = len(incorrect_indices)
    
    # Validation: counts must sum to dataset size
    assert n_correct + n_incorrect == len(dataset), f"Count error: {n_correct} + {n_incorrect} != {len(dataset)}"
    
    print(f"  Ensemble evaluation: {n_correct} correct, {n_incorrect} incorrect ({n_incorrect/len(dataset)*100:.1f}% failure rate)")
    
    # Strategy: Keep as many failures as possible (up to min_failure_ratio * max_samples)
    # Then fill remaining slots with correct predictions
    
    # Target at least min_failure_ratio of samples to be failures
    target_min_failures = int(max_samples * min_failure_ratio)
    
    if n_incorrect <= target_min_failures:
        # Not enough failures to reach target ratio - keep all failures
        target_incorrect = n_incorrect
        target_correct = max_samples - target_incorrect
        actual_failure_ratio = n_incorrect / max_samples
        print(f"  Not enough failures to reach target ratio ({min_failure_ratio:.1%})")
        print(f"  Keeping ALL {n_incorrect} failures ({actual_failure_ratio:.1%} of final set)")
    else:
        # Enough failures - keep all if possible, or subsample if too many
        if n_incorrect + n_correct <= max_samples:
            # Can keep everything
            target_incorrect = n_incorrect
            target_correct = n_correct
        else:
            # Keep all failures, sample from correct to fill budget
            target_incorrect = min(n_incorrect, max_samples)
            target_correct = max_samples - target_incorrect
        actual_failure_ratio = target_incorrect / max_samples
        print(f"  Target: keep {target_incorrect} failures ({actual_failure_ratio:.1%} of final set)")
    
    # Clamp to available samples
    target_incorrect = min(target_incorrect, n_incorrect)
    target_correct = min(target_correct, n_correct)
    
    # Final adjustment: use all budget
    actual_total = target_correct + target_incorrect
    if actual_total < max_samples and n_correct > target_correct:
        target_correct = min(n_correct, max_samples - target_incorrect)
    
    print(f"  Target subsampling: {target_correct} correct + {target_incorrect} incorrect = {target_correct + target_incorrect} total")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Stratified sampling from each group
    selected_indices = []
    
    # Sample incorrect predictions (stratified by class)
    if target_incorrect > 0 and n_incorrect > 0:
        incorrect_labels = labels[incorrect_indices]
        try:
            if target_incorrect >= n_incorrect:
                # Keep all failures
                sampled_incorrect = incorrect_indices
            else:
                # Stratified sample from failures
                sampled_incorrect, _ = train_test_split(
                    incorrect_indices,
                    train_size=target_incorrect,
                    stratify=incorrect_labels,
                    random_state=seed
                )
            selected_indices.extend(sampled_incorrect)
        except ValueError:
            # Fallback: random sample if stratification fails
            sampled_incorrect = np.random.choice(incorrect_indices, size=target_incorrect, replace=False)
            selected_indices.extend(sampled_incorrect)
    
    # Sample correct predictions (stratified by class)
    if target_correct > 0 and n_correct > 0:
        correct_labels = labels[correct_indices]
        try:
            if target_correct >= n_correct:
                # Keep all correct
                sampled_correct = correct_indices
            else:
                # Stratified sample from correct
                sampled_correct, _ = train_test_split(
                    correct_indices,
                    train_size=target_correct,
                    stratify=correct_labels,
                    random_state=seed
                )
            selected_indices.extend(sampled_correct)
        except ValueError:
            # Fallback: random sample if stratification fails
            sampled_correct = np.random.choice(correct_indices, size=target_correct, replace=False)
            selected_indices.extend(sampled_correct)
    
    # Sort for consistency
    selected_indices = sorted(selected_indices)
    
    # Count samples per class in final subset
    selected_labels = labels[selected_indices]
    unique, counts = np.unique(selected_labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    # Count correct/incorrect in final subset and create new index arrays
    # These are 0-indexed positions in the subsampled dataset
    final_correct_mask = (ensemble_preds[selected_indices] == selected_labels)
    final_correct_indices = np.where(final_correct_mask)[0]
    final_incorrect_indices = np.where(~final_correct_mask)[0]
    
    print(f"  Subsampled calibration: {len(dataset)} → {len(selected_indices)} samples (failure-aware)")
    print(f"    Final: {len(final_correct_indices)} correct, {len(final_incorrect_indices)} incorrect ({len(final_incorrect_indices)/len(selected_indices)*100:.1f}%)")
    print(f"    Class distribution: {class_dist}")
    
    return Subset(dataset, selected_indices), final_correct_indices, final_incorrect_indices


def create_cv_generator(n_splits=5, seed=42, batch_size=5000, num_workers=0):
    """
    Factory function to create a CV generator matching training splits.
    
    Returns a function that generates CV train loaders for models.
    """
    def cv_generator(study_dataset, models, batch_size_override=None):
        """
        Generate CV train loaders matching the splits used during training.
        
        Args:
            study_dataset: Full training dataset
            models: List of models (one per fold)
            batch_size_override: Override batch size
        
        Returns:
            List[DataLoader]: One train loader per model
        """
        bs = batch_size_override or batch_size
        
        # Get labels
        labels = [label for _, label in study_dataset]
        
        # Create same CV splits as training (CRITICAL: same seed!)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        train_loaders = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            train_subset = Subset(study_dataset, train_idx)
            train_loader = DataLoader(
                train_subset,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
            train_loaders.append(train_loader)
        
        if len(train_loaders) != len(models):
            raise ValueError(
                f"CV splits produced {len(train_loaders)} folds but got {len(models)} models"
            )
        
        return train_loaders
    
    return cv_generator


def run_medmnist_benchmark(flag, methods, output_dir='./uq_benchmark_results',
                           batch_size=4000, image_size=224, gpu_id=0, per_fold_eval=True,
                           model_backbone='resnet18', setup='', gps_calib_samples=None,
                           min_failure_ratio=0.3):
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
    """
    print(f"\n{'='*80}")
    print(f"MedMNIST Benchmark: {flag}")
    print(f"Using FailCatcher v{FailCatcher.__version__}")
    print(f"{'='*80}\n")
    
    # Get absolute path to workspace root (UQ_Toolbox/)
    workspace_root = Path(__file__).parent.parent.parent.absolute()
    
    # Make output_dir absolute if it's relative
    if not Path(output_dir).is_absolute():
        output_dir = str(workspace_root / output_dir)
    
    # Setup
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    color = flag in ['dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist']
    calib_method = 'platt' if flag in ['breastmnist', 'pneumoniamnist'] else 'temperature'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # LOAD DATA AND MODELS (medMNIST-specific)
    # ========================================================================
    print("📦 Loading medMNIST data and models...")
    
    # Transforms
    if color:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        transform_tta = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            RepeatGrayToRGB(),
        ])
        transform_tta = transforms.Compose([
            transforms.ToTensor(),
            RepeatGrayToRGB(),
        ])
    
    if flag != 'amos2022':
        # Load datasets and models
        models = tr.load_models(flag, device=device, size=image_size, 
                               model_backbone=model_backbone, setup=setup)
        [study_dataset, calib_dataset, test_dataset], \
        [_, calib_loader, test_loader], info = \
            tr.load_datasets(flag, color, image_size, transform, batch_size)
        
        [_, calib_dataset_tta, test_dataset_tta], \
        [_, _, _], _ = \
            tr.load_datasets(flag, color, image_size, transform_tta, batch_size)
    else:
        # Load datasets and models of organamnist and amos2022 as test set
        models = tr.load_models('organamnist', device=device, size=image_size,
                               model_backbone=model_backbone, setup=setup)
        [study_dataset, calib_dataset, _], \
        [_, calib_loader, _], info = \
            tr.load_datasets('organamnist', color, image_size, transform, batch_size)
        
        # Load AMOS external test dataset
        print("  Loading AMOS external test dataset...")
        amos_path = '/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data_models/AMOS_2022/amos_external_test_224.npz'
        amos_data = np.load(amos_path)
        amos_images = amos_data['test_images']  # (N, 224, 224, 1)
        amos_labels = amos_data['test_labels']  # (N, 15) - AMOS organ labels
        
        # OrganaMNIST to AMOS mapping
        amos_to_organamnist = {
            0: 10,  # spleen → spleen
            1: 5,   # right kidney → kidney-right
            2: 4,   # left kidney → kidney-left
            5: 6,   # liver → liver
            9: 9,   # pancreas → pancreas
            13: 0,  # bladder → bladder
        }
        
        # Filter to mapped organs and convert labels
        mapped_indices = []
        mapped_labels = []
        for idx in range(len(amos_labels)):
            amos_organ_id = np.argmax(amos_labels[idx])
            if amos_organ_id in amos_to_organamnist:
                mapped_indices.append(idx)
                mapped_labels.append(amos_to_organamnist[amos_organ_id])
        
        filtered_images = amos_images[mapped_indices]
        filtered_labels = np.array(mapped_labels)
        
        # Create AMOS dataset classes
        class AMOSDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                from PIL import Image as PILImage
                img = self.images[idx].squeeze()
                label = self.labels[idx]
                img_pil = PILImage.fromarray(img, mode='L')
                
                if self.transform:
                    img_tensor = self.transform(img_pil)
                else:
                    img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
                
                # Convert to 3-channel for ResNet
                if img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1)
                
                return img_tensor, torch.tensor(label, dtype=torch.long)
            
        # Create datasets and loaders
        test_dataset = AMOSDataset(filtered_images, filtered_labels, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        test_dataset_tta = AMOSDataset(filtered_images, filtered_labels, transform=transform_tta)

        print(f"  AMOS: {len(test_dataset)} samples (filtered from {len(amos_images)})")
    
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
    
    # Cache file paths - include model backbone and setup to avoid mixing different model results
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    setup_suffix = f"_{setup}" if setup else ""
    calib_cache_path = os.path.join(cache_dir, f'{flag}_{model_backbone}{setup_suffix}_calib_results.npz')
    test_cache_path = os.path.join(cache_dir, f'{flag}_{model_backbone}{setup_suffix}_test_results.npz')
    
    # Try to load cached results FIRST
    if os.path.exists(calib_cache_path) and os.path.exists(test_cache_path):
        print("\n📦 Loading cached evaluation results...")
        calib_cache = np.load(calib_cache_path, allow_pickle=True)
        test_cache = np.load(test_cache_path, allow_pickle=True)
        
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
        
        # Check if per-fold logits are cached (new format)
        if 'indiv_logits' in test_cache.files:
            indiv_logits = test_cache['indiv_logits']  # [N, K, C]
            indiv_logits_calib = calib_cache['indiv_logits']  # [N_calib, K, C]
        else:
            # Old cache format without per-fold logits
            indiv_logits = None
            indiv_logits_calib = None
        
        # Transpose to [K, N, C] format for per-fold evaluation
        indiv_scores = np.transpose(indiv_scores, (1, 0, 2))  # [K, N, C]
        indiv_scores_calib = np.transpose(indiv_scores_calib, (1, 0, 2))  # [K, N_calib, C]
        if indiv_logits is not None:
            indiv_logits = np.transpose(indiv_logits, (1, 0, 2))  # [K, N, C]
            indiv_logits_calib = np.transpose(indiv_logits_calib, (1, 0, 2))  # [K, N_calib, C]
        
        # Compute y_pred from cached scores (for compatibility with old cache format)
        y_pred = np.argmax(y_scores, axis=1)
        y_pred_calib = np.argmax(y_scores_calib, axis=1)
        
        print(f"  ✓ Loaded cached results")
        print(f"  Test accuracy: {len(correct_idx) / len(y_true):.4f}")
    
    else:
        # No cache - evaluate models
        print("\n📊 Evaluating ensemble predictions on test set...")
        y_true, y_scores, y_pred, correct_idx, incorrect_idx, indiv_scores_raw, logits = uq.evaluate_models_on_loader(
            models, test_loader, device, return_logits=True
        )
        
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
            indiv_logits=indiv_logits_calib_raw  # Save as [N, K, C]
        )
        np.savez_compressed(
            test_cache_path,
            y_true=y_true,
            y_scores=y_scores,
            y_pred=y_pred,
            correct_idx=correct_idx,
            incorrect_idx=incorrect_idx,
            indiv_scores=indiv_scores_raw,  # Save as [N, K, C]
            logits=logits,
            indiv_logits=indiv_logits_raw  # Save as [N, K, C]
        )
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
    detector.set_test_predictions(y_scores, y_true, y_pred)
    
    # Create CV train loaders for KNN methods
    cv_gen = create_cv_generator(n_splits=5, seed=42, batch_size=batch_size)
    train_loaders = cv_gen(study_dataset, models, batch_size)
    
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
            per_fold_evaluation=per_fold_eval
        )
        results[f'MSR_{calib_method}'] = metrics
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
        uncertainties, metrics = detector.run_tta(
            test_dataset_tta, y_true,
            image_size=image_size,
            batch_size=batch_size,
            nb_augmentations=5
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
        aug_folder = os.path.join(output_dir, 'gps_augment_cache', f'{flag}_{model_backbone}_{setup_name}_calibration_set{folder_suffix}')
        
        # Subsample calibration dataset (failure-aware for GPS)
        # Prioritizes failures (incorrect predictions) to maximize information density
        # Using fixed seed ensures consistency between TTA_calib and GPS search
        # Returns: (subsampled_dataset, correct_indices, incorrect_indices)
        calib_dataset_tta_subsampled, correct_idx_calib_subsampled, incorrect_idx_calib_subsampled = \
            subsample_dataset_failure_aware(
                dataset=calib_dataset_tta,
                models=models,
                device=device,
                max_samples=gps_calib_samples,
                min_failure_ratio=min_failure_ratio,
                seed=42,
                batch_size=batch_size
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
        setup_name = setup if setup else 'standard'
        aug_folder = os.path.join(output_dir, 'gps_augment_cache', f'{flag}_{model_backbone}_{setup_name}_calibration_set')
        
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
                subsample_dataset_failure_aware(
                    dataset=calib_dataset_tta,
                    models=models,
                    device=device,
                    max_samples=gps_calib_samples,
                    min_failure_ratio=min_failure_ratio,
                    seed=42,
                    batch_size=batch_size
                )
            # Convert numpy arrays to lists for GPS
            gps_correct_idx = gps_correct_idx.tolist()
            gps_incorrect_idx = gps_incorrect_idx.tolist()
            print(f"  GPS will use: {len(gps_correct_idx)} correct, {len(gps_incorrect_idx)} incorrect indices")
        
        uncertainties, metrics = detector.run_gps(
            test_dataset_tta, y_true,
            aug_folder=aug_folder,
            correct_idx_calib=gps_correct_idx,
            incorrect_idx_calib=gps_incorrect_idx,
            image_size=image_size,
            batch_size=batch_size,
            cache_dir=os.path.join(output_dir, 'gps_cache')
        )
        results['GPS'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'KNN_Raw' in methods:
        print("\n🔍 Running KNN-Raw...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        uncertainties, metrics = detector.run_knn_raw(
            test_loader=test_loader,
            train_loaders=train_loaders,
            y_true=y_true,
            layer_name='avgpool',
            k=5,
            per_fold_evaluation=per_fold_eval
        )
        results['KNN_Raw'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'KNN_SHAP' in methods:
        print("\n🔍 Running KNN-SHAP...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f"  Mode: {mode_str} evaluation")
        parallel_mode = torch.cuda.device_count() >= 3
        n_jobs = 3 if parallel_mode else 1
        
        uncertainties, metrics = detector.run_knn_shap(
            calib_loader=calib_loader,
            test_loader=test_loader,
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
    # SAVE RESULTS AND FIGURES (via FailureDetector)
    # ========================================================================
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all results using the detector's save_results method
    saved_paths = detector.save_results(
        output_dir=output_dir,
        flag=flag,
        timestamp=timestamp,
        model_backbone=model_backbone,
        setup=setup
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
                'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'amos2022'],
        help='MedMNIST dataset to benchmark'
    )
    
    parser.add_argument(
        '--model', type=str, default='resnet18 ', choices=['resnet18', 'vit_b_16'],
        help='Model backbone to use (default: resnet18)'
    )
    
    parser.add_argument(
        '--setup', type=str,
        choices=['DA', 'DO', 'DADO'], default='',
        help='Load models trained under different setups (DA: data augmentation, DO: dropout, DADO: both). Default is standard training.'
    )
    
    parser.add_argument(
        '--methods', nargs='+',
        default=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP'],
        choices=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'TTA_calib', 'KNN_Raw', 'KNN_SHAP'],
        help='UQ methods to run'
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
    
    args = parser.parse_args()
    
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
        min_failure_ratio=args.min_failure_ratio
    )
