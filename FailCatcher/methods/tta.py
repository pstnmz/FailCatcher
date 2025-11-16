"""
Test-Time Augmentation (TTA) methods for uncertainty quantification.
Includes both functional API (TTA, apply_augmentations) and class-based API (TTAMethod, GPSMethod).
"""
import os
import re
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from gps_augment.utils.randaugment import BetterRandAugment

from ..core.base import UQMethod
from ..core.utils import (
    get_batch_predictions,
    average_predictions,
    compute_stds,
    build_monai_cache_dataset,
    EnsurePIL,
    _CachedRandAugDataset,
    _dl_worker_init,
    to_3_channels,
    to_1_channel
)


# ============================================================================
# CLASS-BASED API (new, recommended)
# ============================================================================

class TTAMethod(UQMethod):
    """
    TTA uncertainty quantification using fixed or random augmentations.
    """
    def __init__(self, transformations=None, n=2, m=45, nb_augmentations=10, **kwargs):
        """
        Args:
            transformations: List of augmentation policies or None (random if None)
            n: Number of ops per augmentation (BetterRandAugment)
            m: Magnitude of ops (BetterRandAugment)
            nb_augmentations: Number of augmentations (if transformations=None)  # ← ADD
            **kwargs: Additional TTA parameters (image_size, mean, std, etc.)
        """
        super().__init__("TTA")
        self.transformations = transformations
        self.n = n
        self.m = m
        self.nb_augmentations = nb_augmentations
        self.kwargs = kwargs
    
    def compute(self, models, dataset, device, **run_kwargs):
        """Compute per-sample std across augmentations."""
        stds, _ = TTA(
            self.transformations, models, dataset, device,
            nb_augmentations=self.nb_augmentations,  # ← ADD THIS
            usingBetterRandAugment=False,
            n=self.n, m=self.m,
            **{**self.kwargs, **run_kwargs}
        )
        return np.array(stds)


class GPSMethod(UQMethod):
    """
    Greedy Policy Search (GPS) for TTA-based uncertainty quantification.
    
    Discovers optimal augmentation policies on calibration set, then applies to test set.
    """
    def __init__(self, aug_folder, correct_calib, incorrect_calib, max_iter=50):
        """
        Args:
            aug_folder: Directory with pre-computed augmentation predictions (.npz)
            correct_calib: Indices of correct calibration predictions
            incorrect_calib: Indices of incorrect calibration predictions
            max_iter: Max policies to select per greedy search
        """
        super().__init__("GPS")
        self.aug_folder = aug_folder
        self.correct_calib = correct_calib
        self.incorrect_calib = incorrect_calib
        self.max_iter = max_iter
        self.policies = None
    
    def search_policies(self, **kwargs):
        """
        Search for optimal augmentation policies on calibration set.
        
        Args:
            **kwargs: Arguments for perform_greedy_policy_search
                (num_workers, top_k, num_searches, seed, etc.)
        
        Returns:
            list: Selected policy filenames (groups)
        """
        from ..search.greedy import perform_greedy_policy_search
        self.policies = perform_greedy_policy_search(
            self.aug_folder, self.correct_calib, self.incorrect_calib,
            max_iterations=self.max_iter, **kwargs
        )
        return self.policies
    
    def compute(self, models, dataset, device, n=2, m=45, **kwargs):
        """Compute uncertainty using discovered policies."""
        if self.policies is None:
            raise RuntimeError("Call search_policies() before compute()")
        
        # Extract policies from search results
        transformation_pipeline = []
        for group in self.policies:
            n_group, m_group, transformations_group = extract_gps_augmentations_info(group)
            transformation_pipeline.append(transformations_group)
        
        if not transformation_pipeline:
            raise RuntimeError("No valid policies extracted from search results")
        
        # Call TTA in GPS mode
        stds, _ = TTA(
            transformations=transformation_pipeline,
            models=models,
            dataset=dataset,
            device=device,
            usingBetterRandAugment=True,
            n=n, m=m,
            is_gps_mode=True,
            average_groups=True,
            **kwargs
        )
        
        return np.array(stds)


# ============================================================================
# FUNCTIONAL API (existing, for backward compatibility)
# ============================================================================

def _batched_augmentation_inference(augmented_inputs, models, device, batch_size):
    """
    Process multiple augmentations in a single forward pass (faster).
    
    Args:
        augmented_inputs: [K, N, C, H, W] where K=num_augmentations, N=num_samples
        models: List of models
        device: torch.device
        batch_size: Max batch size
    
    Returns:
        torch.Tensor: Predictions [K, N, num_classes]
    """
    K, N, C, H, W = augmented_inputs.shape
    
    # Check if we can fit all augmentations in memory
    total_samples = K * N
    max_gpu_batch = 10000  # Conservative estimate - adjust based on GPU memory
    
    if total_samples <= max_gpu_batch:
        # Fast path: batch all augmentations together
        print(f"  Using batched inference ({K} augmentations × {N} samples)")
        
        # Reshape: [K, N, C, H, W] → [K*N, C, H, W]
        imgs_batched = augmented_inputs.reshape(total_samples, C, H, W)
        
        # Single forward pass
        batch_preds = get_batch_predictions(models, imgs_batched, device)  # [K*N, num_classes]
        avg_preds = average_predictions(batch_preds)  # [K*N, num_classes]
        
        # Reshape back: [K*N, num_classes] → [K, N, num_classes]
        predictions = avg_preds.reshape(K, N, -1)
        
        return predictions
    else:
        # Fallback: process each augmentation separately
        print(f"  Using sequential inference (total samples {total_samples} > {max_gpu_batch})")
        preds_list = []
        
        for k in range(K):
            dataset_aug = TensorDataset(augmented_inputs[k])
            loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
            
            all_preds = []
            for batch in loader:
                imgs = batch[0].to(device)
                batch_predictions = get_batch_predictions(models, imgs, device)
                avg_preds = average_predictions(batch_predictions)
                all_preds.append(avg_preds)
            
            preds_list.append(torch.cat(all_preds, dim=0))  # [N, num_classes]
        
        predictions = torch.stack(preds_list, dim=0)  # [K, N, num_classes]
        return predictions
    
def extract_gps_augmentations_info(policies):
    """
    Parse N, M and policy ops from filenames of form:
      N{N}M{M}__{A}_np.float64_{magA}__{B}_np.float64_{magB}__.npz
    Returns: N (int), M (int), [ [ (op_index, magnitude), ... ], ... ]
    """
    if not policies:
        return None, None, []
    if isinstance(policies, str):
        policies = [policies]

    # Determine valid op index range from BetterRandAugment
    try:
        _probe = BetterRandAugment(n=2, m=45, rand_m=True, resample=False, image_size=51)
        max_valid_index = len(_probe.augment_list) - 1
    except Exception:
        max_valid_index = None  # fall back to accepting provided indices

    # Regex components for both styles (with or without underscore between N and M)
    nm_regex = re.compile(r'N(?P<N>\d+)_?M(?P<M>\d+)')
    pair_regex = re.compile(r'__(?P<idx>\d+)_np\.float64_(?P<mag>-?\d+(?:\.\d+)?)')

    formatted_policies = []
    N_global = None
    M_global = None

    for fname in policies:
        base = os.path.basename(fname)
        stem = base[:-4] if base.endswith('.npz') else base

        # Extract N and M
        nm_match = nm_regex.search(stem)
        if nm_match:
            N_val = int(nm_match.group('N'))
            M_val = int(nm_match.group('M'))
            if N_global is None:  # record from first file
                N_global, M_global = N_val, M_val
        else:
            # Default if missing
            if N_global is None:
                N_global, M_global = 2, 45

        # Extract all (idx, magnitude) pairs
        pairs = []
        for m in pair_regex.finditer(stem):
            idx_raw = int(m.group('idx'))
            mag_raw = float(m.group('mag'))
            mag_cast = int(mag_raw) if abs(mag_raw - int(mag_raw)) < 1e-6 else mag_raw

            if max_valid_index is not None and (idx_raw < 0 or idx_raw > max_valid_index):
                # Strict: raise or clamp. Here we clamp and warn.
                idx_clamped = idx_raw % (max_valid_index + 1)
                print(f"Policy index {idx_raw} out of range [0,{max_valid_index}]; remapped to {idx_clamped}")
                idx_raw = idx_clamped

            pairs.append((idx_raw, mag_cast))

        if not pairs:
            print(f"Warning: could not parse policy ops from '{base}'. Expected pattern N*M*__A_np.float64_mag__B_np.float64_mag__")
        formatted_policies.append(pairs)

    # Ensure global N/M defaults
    if N_global is None:
        N_global = 2
    if M_global is None:
        M_global = 45
    return N_global, M_global, formatted_policies


def TTA(transformations, models, dataset, device, nb_augmentations=10, 
        usingBetterRandAugment=False, n=2, m=45, image_normalization=False, 
        nb_channels=1, mean=None, std=None, image_size=51, batch_size=None, 
        use_monai_cache=False, cache_rate=1.0, cache_num_workers=0, 
        dataloader_workers=None, is_gps_mode=False, average_groups=True):
    """
    Perform Test-Time Augmentation (TTA) on a batch of images.
    
    Args:
        ...existing args...
        is_gps_mode: If True, expect transformations = [[group1], [group2], [group3]]
        average_groups: If True (default), average std across groups in GPS mode
    
    Returns:
        tuple: (stds, averaged_predictions)
            - If GPS mode with average_groups=True: stds is [N] averaged across groups
            - If GPS mode with average_groups=False: stds is [G, N] per-group stds
            - Otherwise: stds is [N] for standard TTA
    """
    if usingBetterRandAugment and transformations is not None and not isinstance(transformations, list):
        raise ValueError("Transformations must be a list (or None for random policies) when usingBetterRandAugment.")
    
    # Determine number of augmentations
    if usingBetterRandAugment and transformations is not None:
        nb_augmentations = len(transformations)
    
    # Build cached dataset if requested
    cached_dataset = None
    if use_monai_cache:
        cached_dataset = build_monai_cache_dataset(dataset, cache_rate=cache_rate, num_workers=cache_num_workers)
    
    with torch.no_grad():
        predictions = []
        if is_gps_mode:
            # GPS mode: transformations is [[group1_policies], [group2_policies], [group3_policies]]
            all_groups_stds = []
            for group_idx, policy_group in enumerate(transformations):
                print(f"  Applying policy group {group_idx + 1}/{len(transformations)}...")
                
                # Get augmented inputs: [K, N, C, H, W]
                augmented_inputs, _ = apply_augmentations(
                    dataset, len(policy_group), usingBetterRandAugment, n, m, 
                    image_normalization, nb_channels, mean, std, image_size, 
                    policy_group, batch_size=batch_size
                )
                
                # NEW: Use batched inference if possible
                predictions = _batched_augmentation_inference(
                    augmented_inputs, models, device, batch_size
                )  # [K, N, num_classes]
                
                # Compute std across K augmentations
                std_per_class = torch.std(predictions, dim=0)  # [N, num_classes]
                
                if std_per_class.shape[1] == 1:
                    group_stds = std_per_class.squeeze(1)
                else:
                    group_stds = torch.mean(std_per_class, dim=1)
                
                all_groups_stds.append(group_stds.cpu().numpy())
            
            # Average across groups if requested
            if average_groups:
                stds = np.mean(all_groups_stds, axis=0)  # [N]
            else:
                stds = all_groups_stds  # [G, N] - return per-group stds
            
            averaged_predictions = None # not computed in GPS mode
        else:
            # Single policy or standard TTA: process each augmentation one by one
            for aug_idx in range(nb_augmentations):
                augmented_inputs = apply_augmentations(
                    dataset, 1, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformations, batch_size=batch_size, cached_dataset=cached_dataset, dataloader_workers=dataloader_workers
                )
                # augmented_inputs shape: [1, batch_size, C, H, W]
                dataset_aug = TensorDataset(augmented_inputs[0])
                loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
                all_preds = []
                for batch in loader:
                    batch_predictions = get_batch_predictions(models, batch[0], device)
                    avg_preds = average_predictions(batch_predictions)
                    all_preds.append(avg_preds)
                predictions.append(torch.cat(all_preds, dim=0))
            # Stack predictions: [nb_augmentations, batch_size, num_classes]
            averaged_predictions = torch.stack(predictions, dim=0).permute(1, 0, 2)  # [batch_size, nb_augmentations, num_classes]
            stds = compute_stds(averaged_predictions)
    
    return stds, averaged_predictions


def apply_augmentations(dataset, nb_augmentations, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformations=None, batch_size=None, cached_dataset=None, dataloader_workers=None):
    """
    Apply augmentations to the images.

    Args:
        images (torch.Tensor): Batch of images.
        transformations (callable or list): Transformations to apply to each image.
        nb_augmentations (int): Number of augmentations to apply per image.
        usingBetterRandAugment (bool): If True, use BetterRandAugment with provided policies.
        n (int): Number of augmentation transformations to apply when using BetterRandAugment.
        m (int): Magnitude of the augmentation transformations when using BetterRandAugment.
        batch_norm (bool): Whether to use batch normalization.
        nb_channels (int): Number of channels in the input images.
        mean (list or None): Mean for normalization.
        std (list or None): Standard deviation for normalization.
        image_size (int): Size of the input images.
        cached_dataset (CacheDataset or None): Pre-cached dataset to speed up augmentation.
        dataloader_workers (int or None): Number of workers for the augmentation DataLoader.

    Returns:
        torch.Tensor: Augmented images.
    """
    augmented_inputs = []
    worker_count = 4 if dataloader_workers is None else dataloader_workers

    def _get_loader(augmentation):
        if cached_dataset is not None:
            aug_dataset = _CachedRandAugDataset(cached_dataset, augmentation)
            return DataLoader(
                dataset=aug_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=worker_count,
                pin_memory=True,
            )
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'datasets'):
            for subds in dataset.dataset.datasets:
                subds.transform = augmentation
        else:
            dataset.transform = augmentation
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=True,
        )
    
    if usingBetterRandAugment:
        if isinstance(transformations, list):
            # Case 1: Explicit policies provided
            rand_aug_policies = [BetterRandAugment(n=n, m=m, resample=False, transform=policy, verbose=True, randomize_sign=False, image_size=image_size) for policy in transformations]
        
        # Case 2: Random policies (transformations is None or False)
        else:
            rand_aug_policies = [BetterRandAugment(n, m, True, False, randomize_sign=False, image_size=image_size) for _ in range(nb_augmentations)] 

        # Ensure input is PIL, run randaugment (expects PIL), then convert to tensor and normalize.
        augmentations = [transforms.Compose([
                    EnsurePIL(),                                 # convert tensor/ndarray -> PIL if needed
                    transforms.Lambda(lambda img: img.convert("RGB")),  # ensure RGB for randaug
                    # removed redundant to_3_channels() - convert("RGB") already ensures RGB
                    rand_aug,                                            # BetterRandAugment expects PIL Image
                    *([to_1_channel] if nb_channels == 1 else []),       # convert back to single channel if needed
                    transforms.PILToTensor(),                            # PIL -> Tensor (uint8 -> [0,255] -> Tensor)
                    transforms.ConvertImageDtype(torch.float),          # ensure float Tensor
                    *([transforms.Normalize(mean=mean, std=std)] if image_normalization else [])
                ]) for rand_aug in rand_aug_policies]
        

        for i, augmentation in enumerate(augmentations):
            augmented_inputs_batch = []
            print(f"Applying augmentation n : {i}")
            data_loader = _get_loader(augmentation)
            for batch in data_loader:
                augmented_images = batch[0]
                augmented_inputs_batch.append(augmented_images)
            augmented_inputs.append(torch.cat(augmented_inputs_batch, dim=0))
        augmented_inputs = torch.stack(augmented_inputs, dim=0)  # Shape: [ num_augmentations, batch_size, C, H, W]
    
    else:
        # Standard TTA with torchvision RandAugment
        # Build complete augmentation pipeline that outputs tensors
        base_augmentations = [
            transforms.Compose([
                EnsurePIL(),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.RandAugment(num_ops=n, magnitude=m),  # torchvision RandAugment
                *([to_1_channel] if nb_channels == 1 else []),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                *([transforms.Normalize(mean=mean, std=std)] if image_normalization else [])
            ]) for _ in range(nb_augmentations)
        ]
        for i, augmentation in enumerate(base_augmentations):
            augmented_inputs_batch = []
            print(f"Applying augmentation {i+1}/{nb_augmentations}")
            data_loader = _get_loader(augmentation)
            for batch in data_loader:
                augmented_images = batch[0]
                augmented_inputs_batch.append(augmented_images)
            augmented_inputs.append(torch.cat(augmented_inputs_batch, dim=0))
        augmented_inputs = torch.stack(augmented_inputs, dim=0)  # Shape: [num_augmentations, batch_size, C, H, W]
    
    if usingBetterRandAugment : 
        return augmented_inputs, augmentations
    else:
        return augmented_inputs


def apply_randaugment_and_store_results(
    dataset, models, N, M, num_policies, device, folder_name='savedpolicies',
    image_normalization=False, mean=False, std=False, nb_channels=1, image_size=51,
    batch_size=None, use_monai_cache=False, cache_rate=1.0, cache_num_workers=0,
    dataloader_workers=None, dataloader_prefetch=8
):
    """
    Apply random augmentation policies and save predictions to disk (for GPS pre-computation).
    
    Generates `num_policies` random augmentation policies, applies each to the dataset,
    computes model predictions, and saves to .npz files for later greedy policy search.
    
    Args:
        dataset: PyTorch dataset
        models: Single model or list of models
        N: Number of ops per augmentation (BetterRandAugment)
        M: Magnitude parameter (BetterRandAugment)
        num_policies: Number of random policies to generate
        device: torch.device
        folder_name: Output directory for .npz files
        image_normalization: Whether to normalize images
        mean: Normalization mean
        std: Normalization std
        nb_channels: Number of channels (1 or 3)
        image_size: Input image size
        batch_size: Batch size for DataLoader
        use_monai_cache: Whether to use MONAI CacheDataset for faster loading
        cache_rate: Fraction of dataset to cache (0-1)
        cache_num_workers: Workers for cache building
        dataloader_workers: Workers for DataLoader
        dataloader_prefetch: Prefetch factor for DataLoader
    
    Returns:
        None (saves .npz files to disk)
    
    Example:
        >>> apply_randaugment_and_store_results(
        ...     calibration_dataset, models, N=2, M=45, num_policies=500,
        ...     device=device, folder_name='gps_augment/breastmnist_calib',
        ...     use_monai_cache=True, batch_size=128
        ... )
    """
    os.makedirs(folder_name, exist_ok=True)
    cached_dataset = None
    
    if use_monai_cache:
        cached_dataset = build_monai_cache_dataset(
            dataset, cache_rate=cache_rate, num_workers=cache_num_workers
        )

    # Streaming mode with cached dataset (memory-efficient)
    if cached_dataset is not None:
        print(f"Streaming {num_policies} policies in chunks to avoid RAM overload")
        policy_chunk_size = min(25, num_policies)
        num_samples = len(cached_dataset)

        def build_augmentations_chunk(k_chunk):
            """Build k_chunk random BetterRandAugment policies."""
            rand_aug_policies = [
                BetterRandAugment(N, M, True, False, randomize_sign=False, image_size=image_size)
                for _ in range(k_chunk)
            ]
            return [
                transforms.Compose([
                    EnsurePIL(),
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    *([to_3_channels] if nb_channels == 1 else []),
                    rand_aug,
                    *([to_1_channel] if nb_channels == 1 else []),
                    transforms.ToTensor(),
                    *([transforms.Normalize(mean=mean, std=std)] if image_normalization else [])
                ]) for rand_aug in rand_aug_policies
            ]

        # Infer num_classes once
        with torch.no_grad():
            dummy = torch.zeros(1, nb_channels, image_size, image_size, device=device)
            try:
                dp = get_batch_predictions(models, dummy, device)
                nc = average_predictions(dp).shape[1]
            except Exception:
                nc = 1

        # Process policies in chunks
        for start in range(0, num_policies, policy_chunk_size):
            end = min(start + policy_chunk_size, num_policies)
            K_chunk = end - start
            print(f"Processing policy chunk {start}-{end-1} (K={K_chunk})")
            
            # Build augmentations and DataLoader
            augmentations = build_augmentations_chunk(K_chunk)
            worker_count = dataloader_workers or 0
            aug_dataset = _CachedRandAugDataset(cached_dataset, augmentations)
            loader = DataLoader(
                dataset=aug_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=worker_count,
                pin_memory=True,
                worker_init_fn=_dl_worker_init,
                persistent_workers=False,
                prefetch_factor=(int(dataloader_prefetch) if worker_count > 0 else None),
            )

            # Get first batch to determine actual K
            it = iter(loader)
            try:
                first_batch = next(it)
            except StopIteration:
                raise RuntimeError("Augmentation loader yielded no data")
            
            imgs_stacked = first_batch[0]  # [B, K, C, H, W]
            B0, K_actual, C, H, W = imgs_stacked.shape
            
            if K_actual != K_chunk:
                print(f"Warning: K_actual ({K_actual}) != K_chunk ({K_chunk}); using K_actual")

            # Create memory-mapped files for each policy
            memmaps, memmap_paths = [], []
            for idx in range(K_actual):
                mmap_path = os.path.join(folder_name, f"tmp_policy_{start+idx}.mmap")
                mem = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(num_samples, nc))
                memmaps.append(mem)
                memmap_paths.append(mmap_path)

            # Stream batches and fill memmaps
            sample_ptr = 0
            
            def _process_batch(imgs_stacked_local):
                """Process a batch and return predictions."""
                B, K, C, H, W = imgs_stacked_local.shape
                imgs_flat = imgs_stacked_local.reshape(B * K, C, H, W).float()
                batch_predictions = get_batch_predictions(models, imgs_flat, device, use_amp=True)
                avg_preds = average_predictions(batch_predictions).view(B, K, -1).cpu().numpy()
                return B, K, avg_preds

            # Process first batch
            B, K, avg_preds = _process_batch(imgs_stacked)
            for local_k in range(K):
                memmaps[local_k][sample_ptr:sample_ptr + B, :] = avg_preds[:, local_k, :]
            sample_ptr += B
            
            # Process remaining batches
            for batch in it:
                imgs_stacked = batch[0]
                B, K, avg_preds = _process_batch(imgs_stacked)
                for local_k in range(K):
                    memmaps[local_k][sample_ptr:sample_ptr + B, :] = avg_preds[:, local_k, :]
                sample_ptr += B

            if sample_ptr != num_samples:
                raise RuntimeError(f"Expected {num_samples} samples but wrote {sample_ptr}")

            # Compress each memmap to .npz and cleanup
            for local_k, (mem, mempath) in enumerate(zip(memmaps, memmap_paths)):
                # Extract policy key for filename
                rand_transform = None
                try:
                    for t in augmentations[local_k].transforms:
                        if isinstance(t, BetterRandAugment):
                            rand_transform = t
                            break
                except Exception:
                    pass

                if rand_transform is not None and hasattr(rand_transform, 'get_transform_str'):
                    policy_key = rand_transform.get_transform_str()
                else:
                    policy_key = f"policy_{start + local_k}"

                safe_key = re.sub(r'[^A-Za-z0-9_.-]', '_', policy_key)
                out_fname = os.path.join(folder_name, f'N{N}_M{M}_{safe_key}.npz')
                
                arr = np.asarray(mem)
                np.savez_compressed(out_fname, predictions=arr)
                print(f"Saved policy {start + local_k} -> {out_fname}")

            # Cleanup memmaps
            try:
                for mempath in memmap_paths:
                    os.remove(mempath)
            except Exception as e:
                print(f"Warning: Failed to remove memmap files: {e}")
            
            del memmaps

    else:
        # Fallback: no cache (process one policy at a time)
        print("Processing policies sequentially (no cache)")
        for i in range(num_policies):
            print(f"Applying policy {i+1}/{num_policies}")
            
            augmented_inputs, augmentations = apply_augmentations(
                dataset, 1, True, N, M, image_normalization, nb_channels,
                mean, std, image_size, batch_size=batch_size
            )
            
            augmented_input = augmented_inputs[0]  # [N, C, H, W]
            dataset_aug = TensorDataset(augmented_input)
            loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
            
            all_preds = []
            for batch in loader:
                batch_predictions = get_batch_predictions(models, batch[0], device)
                avg_preds = average_predictions(batch_predictions)
                all_preds.append(avg_preds)
            
            averaged_predictions = torch.cat(all_preds, dim=0)
            
            # Extract policy key
            rand_transform = None
            for t in augmentations[0].transforms:
                if isinstance(t, BetterRandAugment):
                    rand_transform = t
                    break

            if rand_transform is not None and hasattr(rand_transform, 'get_transform_str'):
                policy_key = rand_transform.get_transform_str()
            else:
                policy_key = f"policy_{i}"

            safe_key = re.sub(r'[^A-Za-z0-9_.-]', '_', policy_key)
            filename = os.path.join(folder_name, f'N{N}_M{M}_{safe_key}.npz')
            
            np.savez_compressed(filename, predictions=averaged_predictions.numpy())
            print(f"Saved predictions to {filename}")