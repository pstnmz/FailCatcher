"""
Dataset utilities for medMNIST benchmarking.

This module provides tools for loading, transforming, subsampling datasets,
and applying covariate shift corruptions using medmnistc.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
import random

# Import medmnistc for covariate shift corruptions
try:
    from medmnistc.corruptions.registry import CORRUPTIONS_DS
    MEDMNISTC_AVAILABLE = True
except ImportError:
    MEDMNISTC_AVAILABLE = False


class RepeatGrayToRGB:
    """Transform for converting grayscale to RGB by repeating channels."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)


class AMOSDataset(torch.utils.data.Dataset):
    """
    Custom dataset for AMOS-2022 external test data.
    Handles loading from .npz files and applying transforms.
    """
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


def get_transforms(color, image_size=224):
    """
    Get standard transforms for medMNIST datasets.
    
    Args:
        color: Whether dataset is color (True) or grayscale (False)
        image_size: Target image size (default: 224)
    
    Returns:
        tuple: (transform, transform_tta) - normalized and unnormalized transforms
    """
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
    
    return transform, transform_tta


def load_amos_dataset(transform, transform_tta, batch_size=256, workspace_root=None):
    """
    Load AMOS-2022 external test dataset.
    
    Args:
        transform: Transform for normalized data
        transform_tta: Transform for unnormalized data (TTA)
        batch_size: Batch size for DataLoader
        workspace_root: Path to workspace root (optional, auto-detected if None)
    
    Returns:
        tuple: (test_dataset, test_loader, test_dataset_tta, filtered_images, filtered_labels)
    """
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    
    print("  Loading AMOS external test dataset...")
    amos_path = workspace_root / 'benchmarks' / 'medMNIST' / 'Data' / 'AMOS_2022' / 'amos_external_test_224.npz'
    
    # Check if file is a Git LFS pointer (not the actual data)
    if amos_path.stat().st_size < 1000:  # Real file should be ~133MB
        raise FileNotFoundError(
            f"\n❌ AMOS dataset file appears to be a Git LFS pointer.\n"
            f"   Please download the actual file using:\n"
            f"   git lfs pull\n"
            f"   Or manually download from the repository."
        )
    
    try:
        amos_data = np.load(str(amos_path), allow_pickle=True)
    except Exception as e:
        raise RuntimeError(
            f"\n❌ Failed to load AMOS dataset from {amos_path}\n"
            f"   Error: {e}\n"
            f"   File size: {amos_path.stat().st_size} bytes\n"
            f"   If this is a Git LFS pointer, run: git lfs pull"
        ) from e
    
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
    
    # Create datasets and loaders
    test_dataset = AMOSDataset(filtered_images, filtered_labels, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_dataset_tta = AMOSDataset(filtered_images, filtered_labels, transform=transform_tta)
    
    print(f"  AMOS: {len(test_dataset)} samples (filtered from {len(amos_images)})")
    
    return test_dataset, test_loader, test_dataset_tta, filtered_images, filtered_labels


def subsample_dataset_failure_aware(dataset, models, device, max_samples=None, 
                                     min_failure_ratio=0.3, seed=42, batch_size=256,
                                     eval_dataset=None):
    """
    Subsample calibration dataset prioritizing failures for GPS.
    
    **CRITICAL**: Models expect NORMALIZED inputs! If `dataset` is unnormalized (e.g., for TTA),
    pass the normalized version via `eval_dataset` to ensure accurate failure detection.
    
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
        eval_dataset: Normalized version of dataset for evaluation (optional)
    
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
    # Import here to avoid circular dependency
    from FailCatcher import UQ_toolbox as uq
    
    if max_samples is None or len(dataset) <= max_samples:
        # No subsampling needed - compute indices on full dataset
        print(f"  No subsampling needed (dataset size: {len(dataset)})")
        
        # Use eval_dataset if provided (for normalized inference), otherwise use dataset
        inference_dataset = eval_dataset if eval_dataset is not None else dataset
        
        # Use evaluate_models_on_loader for consistent ensemble inference
        loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
        
        y_true, y_scores, ensemble_preds, correct_indices, incorrect_indices, _ = \
            uq.evaluate_models_on_loader(models, loader, device)
        
        print(f"    Full dataset: {len(correct_indices)} correct, {len(incorrect_indices)} incorrect")
        return dataset, np.array(correct_indices), np.array(incorrect_indices)
    
    print(f"  Running ensemble inference to identify failures...")
    
    # Use eval_dataset if provided (for normalized inference), otherwise use dataset
    inference_dataset = eval_dataset if eval_dataset is not None else dataset
    print(f"  Inference dataset type: {type(inference_dataset)}, length: {len(inference_dataset)}")
    
    # Use evaluate_models_on_loader for consistent ensemble inference (soft voting)
    loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=4, pin_memory=True)
    
    labels, y_scores, ensemble_preds, correct_indices, incorrect_indices, _ = \
        uq.evaluate_models_on_loader(models, loader, device)
    
    correct_indices = np.array(correct_indices)
    incorrect_indices = np.array(incorrect_indices)
    
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
        # Enough failures - cap at min_failure_ratio (interpreted as maximum)
        if n_incorrect + n_correct <= max_samples:
            # Can keep everything
            target_incorrect = n_incorrect
            target_correct = n_correct
        else:
            # Cap failures at min_failure_ratio * max_samples, fill rest with correct
            max_failures_allowed = int(max_samples * min_failure_ratio)
            target_incorrect = min(n_incorrect, max_failures_allowed)
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
    
    Args:
        n_splits: Number of CV folds (default: 5)
        seed: Random seed for splits (default: 42)
        batch_size: Batch size for loaders (default: 5000)
        num_workers: Number of DataLoader workers (default: 0)
    
    Returns:
        function: CV generator function
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

# =============================================================================
# Corruption Utilities
# =============================================================================

class CorruptedDataset(Dataset):
    """
    Wrapper that applies medmnistc corruptions to a dataset on-the-fly.
    
    Args:
        base_dataset: Original PyTorch dataset
        corruption_funcs: Either a single corruption object or a list of corruption objects
                         from medmnistc (e.g., CORRUPTIONS_DS['breastmnist']['pixelate'])
        severity: Corruption severity level (1-5)
        random_corruption: If True and multiple corruption_funcs provided, randomly select one per sample
        cache_corruptions: If True, cache corrupted images in memory (faster but uses more RAM)
        seed: Random seed for reproducible corruption selection (default: None)
    """
    def __init__(self, base_dataset, corruption_funcs, severity, 
                 random_corruption=True, cache_corruptions=True, seed=None):
        self.base_dataset = base_dataset
        
        # Convert single corruption to list
        if not isinstance(corruption_funcs, list):
            corruption_funcs = [corruption_funcs]
        
        self.corruption_funcs = corruption_funcs
        self.severity = severity
        self.random_corruption = random_corruption
        self.cache_corruptions = cache_corruptions
        self.cache = {} if cache_corruptions else None
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _select_corruption(self, idx):
        """Select corruption function for this index."""
        if len(self.corruption_funcs) == 1:
            return self.corruption_funcs[0]
        
        if self.random_corruption:
            # Use index combined with global seed for deterministic per-sample selection
            # This ensures same index always gets same corruption across runs
            sample_rng = random.Random(idx)
            return sample_rng.choice(self.corruption_funcs)
        else:
            # Cycle through corruptions
            return self.corruption_funcs[idx % len(self.corruption_funcs)]
    
    def __getitem__(self, idx):
        # Check cache first
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        # Get original image and label
        img, label = self.base_dataset[idx]
        
        # Select corruption for this sample
        corruption_func = self._select_corruption(idx)
        
        # Convert tensor to PIL Image for corruption
        # Handle both normalized and unnormalized inputs
        if isinstance(img, torch.Tensor):
            # Denormalize if needed (assume [-1, 1] or [0, 1] range)
            img_np = img.cpu().numpy()
            
            # Convert from CxHxW to HxWxC
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # Denormalize to [0, 255]
            if img_np.min() < 0:  # Normalized to [-1, 1]
                img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
            elif img_np.max() <= 1.0:  # Normalized to [0, 1]
                img_np = (img_np * 255).astype(np.uint8)
            else:  # Already in [0, 255]
                img_np = img_np.astype(np.uint8)
            
            # Handle grayscale (single channel)
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            
            # Convert to PIL
            from PIL import Image as PILImage
            if img_np.ndim == 2:  # Grayscale
                pil_img = PILImage.fromarray(img_np, mode='L')
            else:  # RGB
                pil_img = PILImage.fromarray(img_np, mode='RGB')
        else:
            pil_img = img
        
        # Apply corruption
        corrupted_pil = corruption_func.apply(pil_img, severity=self.severity)
        
        # Convert back to tensor with same format as original
        corrupted_np = np.array(corrupted_pil)
        if corrupted_np.ndim == 2:  # Grayscale
            corrupted_np = corrupted_np[:, :, np.newaxis]
        
        # Normalize back to original range
        corrupted_tensor = torch.from_numpy(corrupted_np).float()
        if corrupted_tensor.ndim == 3:
            corrupted_tensor = corrupted_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Renormalize to match original preprocessing
        if isinstance(img, torch.Tensor) and img.min() < 0:
            corrupted_tensor = (corrupted_tensor / 255.0) * 2 - 1  # [0, 255] -> [-1, 1]
        elif isinstance(img, torch.Tensor) and img.max() <= 1.0:
            corrupted_tensor = corrupted_tensor / 255.0  # [0, 255] -> [0, 1]
        
        result = (corrupted_tensor, label)
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = result
        
        return result


def get_available_corruptions(flag):
    """
    Get all available corruptions for a given dataset.
    
    Args:
        flag: Dataset name (e.g., 'breastmnist')
    
    Returns:
        dict: Dictionary of {corruption_name: corruption_object} or empty dict if none available
    """
    if not MEDMNISTC_AVAILABLE:
        return {}
    
    # Handle dermamnist-e variants
    base_flag = flag.replace('-id', '').replace('-external', '')
    
    if base_flag == 'amos2022':
        base_flag = 'organamnist'
    
    if base_flag not in CORRUPTIONS_DS:
        return {}
    
    return dict(CORRUPTIONS_DS[base_flag])


def apply_random_corruptions(dataset, flag, severity, cache=True, seed=42):
    """
    Apply random corruptions from available medmnistc corruptions to a dataset.
    Each sample will be randomly corrupted using one of the available corruptions.
    
    Args:
        dataset: PyTorch dataset to corrupt
        flag: Dataset name (e.g., 'breastmnist')
        severity: Corruption severity (1-5)
        cache: Whether to cache corrupted images
        seed: Random seed for reproducible corruption selection
    
    Returns:
        CorruptedDataset wrapper or original dataset if corruption not available
    """
    if not MEDMNISTC_AVAILABLE:
        print(f"  ⚠️  Cannot apply corruptions - medmnistc not installed")
        return dataset
    
    corruptions = get_available_corruptions(flag)
    
    if not corruptions:
        # Handle dermamnist-e variants
        base_flag = flag.replace('-id', '').replace('-external', '')
        if base_flag == 'amos2022':
            base_flag = 'organamnist'
        
        available_datasets = list(CORRUPTIONS_DS.keys())
        print(f"  ⚠️  No corruptions defined for '{flag}' in medmnistc")
        print(f"      Available datasets: {available_datasets}")
        return dataset
    
    corruption_funcs = list(corruptions.values())
    corruption_names = list(corruptions.keys())
    
    print(f"  ✓ Applying random corruptions (severity={severity}) to {flag} dataset")
    print(f"    Available corruptions ({len(corruption_names)}): {', '.join(corruption_names)}")
    
    return CorruptedDataset(
        dataset, 
        corruption_funcs, 
        severity, 
        random_corruption=True,
        cache_corruptions=cache,
        seed=seed
    )


def apply_specific_corruption(dataset, flag, corruption_type, severity, cache=True):
    """
    Apply a specific corruption to a dataset.
    
    Args:
        dataset: PyTorch dataset to corrupt
        flag: Dataset name (e.g., 'breastmnist')
        corruption_type: Name of corruption (e.g., 'pixelate', 'gaussian_noise')
        severity: Corruption severity (1-5)
        cache: Whether to cache corrupted images
    
    Returns:
        CorruptedDataset wrapper or original dataset if corruption not available
    """
    if not MEDMNISTC_AVAILABLE:
        print(f"  ⚠️  Cannot apply corruption '{corruption_type}' - medmnistc not installed")
        return dataset
    
    corruptions = get_available_corruptions(flag)
    
    if not corruptions:
        base_flag = flag.replace('-id', '').replace('-external', '')
        if base_flag == 'amos2022':
            base_flag = 'organamnist'
        
        available_datasets = list(CORRUPTIONS_DS.keys())
        print(f"  ⚠️  No corruptions defined for '{flag}' in medmnistc")
        print(f"      Available datasets: {available_datasets}")
        return dataset
    
    if corruption_type not in corruptions:
        available_corruptions = list(corruptions.keys())
        print(f"  ⚠️  Corruption '{corruption_type}' not available for {flag}")
        print(f"      Available corruptions: {available_corruptions}")
        return dataset
    
    corruption_func = corruptions[corruption_type]
    print(f"  ✓ Applying '{corruption_type}' (severity={severity}) to {flag} dataset")
    
    return CorruptedDataset(
        dataset, 
        corruption_func, 
        severity, 
        cache_corruptions=cache
    )


def list_available_corruptions(flag):
    """
    List all available corruptions for a given dataset.
    
    Args:
        flag: Dataset name (e.g., 'breastmnist')
    
    Returns:
        List of corruption names or empty list if none available
    """
    corruptions = get_available_corruptions(flag)
    return list(corruptions.keys())