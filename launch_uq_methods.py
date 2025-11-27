"""
Benchmark script for UQ methods on medMNIST datasets.
Runs all UQ methods and logs execution time + resource usage.

Usage:
    python run_uq_benchmark.py --flag breastmnist --methods MSR Ensembling TTA GPS
    python run_uq_benchmark.py --flag organamnist --all-methods
    python run_uq_benchmark.py --flag dermamnist --output-dir ./results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import pickle as pkl
from sklearn.model_selection import StratifiedKFold

import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Import UQ_Toolbox
import FailCatcher.UQ_toolbox as uq
from medMNIST.utils import train_load_datasets_resnet as tr


# ============================================================================
# HELPER CLASSES
# ============================================================================

GPS_CONFIG = {
    'num_workers': 90,
    'num_searches': 30,
    'top_k': 3,
    'seed': 64,
    'mean': 0.5,
    'std': 0.5
}

TTA_CONFIG = {
    'nb_augmentations': 5,
    'n': 2,  # RandAugment num_ops
    'm': 9,  # RandAugment magnitude
}

class RepeatGrayToRGB:
    """Transform for converting grayscale to RGB."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed:.2f}s")


# ============================================================================
# CV SPLIT HELPERS (matching train_resnet18_medMNIST.py)
# ============================================================================

def get_cv_train_loaders_for_models(study_dataset, models, batch_size=5000, 
                                   n_splits=5, seed=42, num_workers=0):
    """
    Create per-fold training loaders matching the CV splits used during training.
    
    Returns:
        List[DataLoader]: One train_loader per model (fold)
    """
    from torch.utils.data import DataLoader, Subset
    
    # Get labels from dataset
    labels = [label for _, label in study_dataset]
    
    # Create same CV splits as training (CRITICAL: same seed!)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    train_loaders = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
        
        # Create training subset for THIS fold
        train_subset = Subset(study_dataset, train_idx)
        
        # Create loader
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        train_loaders.append(train_loader)
    
    if len(train_loaders) != len(models):
        raise ValueError(
            f"CV splits produced {len(train_loaders)} folds but got {len(models)} models. "
            f"Check n_splits={n_splits} and seed={seed} match training!"
        )
    
    return train_loaders


# ============================================================================
# UQ METHOD IMPLEMENTATIONS
# ============================================================================

def compute_msr(y_prob, y_true):
    """Compute Maximum Softmax Response / Distance to Hard Labels."""
    metric = uq.distance_to_hard_labels_computation(y_prob)
    return metric


def compute_msr_calibrated(y_scores_test, y_true_test, y_scores_calib, y_true_calib,
                           logits_test=None, logits_calib=None, method='temperature'):
    """
    Compute calibrated MSR using post-hoc calibration.
    
    Args:
        y_scores_test: Test probabilities [N, C]
        y_true_test: Test labels [N]
        y_scores_calib: Calibration probabilities [N_calib, C]
        y_true_calib: Calibration labels [N_calib]
        logits_test: Test logits (for temperature) [N, C]
        logits_calib: Calibration logits (for temperature) [N_calib, C]
        method: 'temperature', 'platt', or 'isotonic'
    
    Returns:
        np.ndarray: MSR scores for calibrated predictions
    """
    from FailCatcher.methods.distance import posthoc_calibration
    from FailCatcher.core.utils import apply_calibration
    
    # Step 1: Fit calibration on calibration set
    if method == 'temperature':
        if logits_calib is None:
            raise ValueError("Temperature scaling requires logits. Use return_logits=True in evaluate_models_on_loader.")
        _, calibration_model = posthoc_calibration(logits_calib, y_true_calib, method)
    else:
        _, calibration_model = posthoc_calibration(y_scores_calib, y_true_calib, method)
    
    # Step 2: Apply calibration to test set
    if method == 'temperature':
        calibrated_test_scores = apply_calibration(
            y_scores_test, calibration_model, method, logits=logits_test
        )
    else:
        calibrated_test_scores = apply_calibration(
            y_scores_test, calibration_model, method
        )
    
    # Step 3: Compute distance to hard labels on calibrated scores
    metric = uq.distance_to_hard_labels_computation(calibrated_test_scores)
    
    return metric


def compute_ensemble_std(indiv_scores):
    """Compute ensemble standard deviation."""
    metric = uq.ensembling_stds_computation(indiv_scores)
    return metric


def compute_tta(models, test_dataset, device, image_size, 
                batch_size=4000, config=None):
    """Compute TTA using class-based API."""
    cfg = {**TTA_CONFIG, **(config or {})}
    
    tta = uq.TTAMethod(
        transformations=None,  # Random policies
        n=cfg['n'],
        m=cfg['m'],
        nb_augmentations=cfg['nb_augmentations'],
        nb_channels=3,
        image_size=image_size,
        image_normalization=True,
        mean=0.5,
        std=0.5,
        batch_size=batch_size
    )
    
    metric = tta.compute(models, test_dataset, device)
    return metric.tolist()


def compute_gps(models, test_dataset, device, image_size,
                aug_folder, correct_idx_calib, incorrect_idx_calib,
                max_iterations=5, batch_size=4000, config=None, cache_dir='./uq_benchmark_results/gps_cache'):
    """Compute GPS using class-based API."""
    cfg = {**GPS_CONFIG, **(config or {})}
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on aug_folder, indices, and search params
    import hashlib
    cache_key_parts = [
        aug_folder,
        str(sorted(correct_idx_calib)),
        str(sorted(incorrect_idx_calib)),
        str(max_iterations),
        str(cfg['num_workers']),
        str(cfg['num_searches']),
        str(cfg['top_k']),
        str(cfg['seed'])
    ]
    cache_key = hashlib.md5('_'.join(cache_key_parts).encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f'gps_policies_{cache_key}.pkl')
    
    # Step 1: Initialize GPS method
    gps = uq.GPSMethod(
        aug_folder=aug_folder,
        correct_calib=correct_idx_calib,
        incorrect_calib=incorrect_idx_calib,
        max_iter=max_iterations
    )
    
    # Step 2: Search for policies (or load from cache)
    if os.path.exists(cache_file):
        print(f"  Loading cached GPS policies from {os.path.basename(cache_file)}...")
        import pickle
        with open(cache_file, 'rb') as f:
            gps.policies = pickle.load(f)
        print(f"  ✓ Loaded {len(gps.policies)} policy groups from cache")
    else:
        print("  Performing greedy policy search...")
        gps.search_policies(
            num_workers=cfg['num_workers'],
            num_searches=cfg['num_searches'],
            top_k=cfg['top_k'],
            seed=cfg['seed']
        )
        # Save to cache
        print(f"  Saving GPS policies to cache: {os.path.basename(cache_file)}")
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(gps.policies, f)
        print(f"  ✓ Cached {len(gps.policies)} policy groups")
    
    # Step 3: Compute metric
    print(f"  Applying GPS augmentations to test set...")
    metric = gps.compute(
        models, test_dataset, device,
        n=2, m=45,
        nb_channels=3,  # Always use 3 channels for ResNet models
        image_size=image_size,
        image_normalization=True,
        mean=cfg['mean'],
        std=cfg['std'],
        batch_size=batch_size
    )
    
    return metric.tolist()

def compute_knn_raw(models, study_dataset, test_loader, device, 
                   layer_name='avgpool', k=5, batch_size=4000, 
                   cv_seed=42, n_splits=5):
    """
    Compute KNN in raw latent space using correct CV splits per model.
    """
    print(f"  Reconstructing CV splits (seed={cv_seed}, n_splits={n_splits})...")
    train_loaders = get_cv_train_loaders_for_models(
        study_dataset, models, batch_size, n_splits, cv_seed
    )
    
    # Pass loaders directly to the method
    knn_method = uq.KNNLatentMethod(layer_name=layer_name, k=k)
    knn_method.fit(models, train_loaders, device)
    metric = knn_method.compute(models, test_loader, device)
    return metric


def compute_knn_shap(models, study_dataset, calib_loader, test_loader, device, flag, layer_name='avgpool', k=5, n_shap_features=50, batch_size=4000, cv_seed=42, n_splits=5, cache_dir='./uq_benchmark_results/shap_cache', parallel=False, n_jobs=2):
    """
    Compute KNN in SHAP-selected latent space using correct CV splits.
    
    Args:
        models: List of models
        study_dataset: Full study dataset (for CV splitting)
        calib_loader: Pre-built calibration loader (for SHAP)
        test_loader: Pre-built test loader
        device: torch.device
        layer_name: Layer name for feature extraction
        k: Number of nearest neighbors
        n_shap_features: Number of top SHAP features to select
        batch_size: Batch size for CV train loaders
        cv_seed: CV random seed (must match training)
        n_splits: Number of CV folds (must match training)
        cache_dir: Directory to cache SHAP results
        parallel: Enable parallel processing across folds
        n_jobs: Number of parallel workers (default: 2)
    """
    print(f"  Reconstructing CV splits (seed={cv_seed}, n_splits={n_splits})...")
    train_loaders = get_cv_train_loaders_for_models(
        study_dataset, models, batch_size, n_splits, cv_seed
    )
    
    # Create method with parallelization enabled
    knn_method = uq.KNNLatentSHAPMethod(
        layer_name=layer_name, 
        k=k, 
        n_shap_features=n_shap_features,
        cache_dir=cache_dir,
        parallel=parallel,  # ← Enable parallel mode
        n_jobs=n_jobs       # ← Number of workers
    )
    
    # fit() will now run in parallel if enabled
    knn_method.fit(models, train_loaders, calib_loader, device, flag=flag)
    metric = knn_method.compute(models, test_loader, device)
    return metric



# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def find_best_threshold(metric, correct_idx, metric_name='balanced_accuracy'):
    """Find optimal threshold for UQ metric."""
    def compute_metrics(values, labels, threshold):
        predictions = (values <= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return specificity, sensitivity, balanced_acc
    
    labels = np.array([1 if i in correct_idx else 0 for i in range(len(metric))])
    thresholds = np.linspace(min(metric), max(metric), 1000)
    
    best_threshold = thresholds[0]
    best_value = 0
    
    for threshold in thresholds:
        spec, sens, bal_acc = compute_metrics(metric, labels, threshold)
        
        if metric_name == 'balanced_accuracy' and bal_acc > best_value:
            best_value = bal_acc
            best_threshold = threshold
    
    # Compute final metrics
    spec, sens, bal_acc = compute_metrics(metric, labels, best_threshold)
    return {
        'threshold': best_threshold,
        'balanced_accuracy': bal_acc,
        'specificity': spec,
        'sensitivity': sens
    }


def evaluate_uq_method(metric, correct_idx, incorrect_idx, predictions=None, labels=None):
    """
    Compute all UQ evaluation metrics.
    
    Args:
        metric: Uncertainty scores [N]
        correct_idx: Indices of correct predictions
        incorrect_idx: Indices of incorrect predictions
        predictions: Optional predicted labels [N] (for AURC/AUGRC)
        labels: Optional true labels [N] (for AURC/AUGRC)
    
    Returns:
        dict: Metrics including AUROC, threshold metrics, and optionally AURC/AUGRC
    """
    # ROC AUC (for failure prediction)
    fpr, tpr, auc = uq.roc_curve_UQ_method_computation(
        [metric[i] for i in correct_idx],
        [metric[i] for i in incorrect_idx]
    )
    
    # Optimal threshold metrics
    threshold_metrics = find_best_threshold(metric, correct_idx)
    
    # Base results
    results = {
        'auroc': float(auc),  # Renamed from 'auc' for clarity
        'threshold': float(threshold_metrics['threshold']),
        'balanced_accuracy': float(threshold_metrics['balanced_accuracy']),
        'specificity': float(threshold_metrics['specificity']),
        'sensitivity': float(threshold_metrics['sensitivity'])
    }
    
    # Compute AURC and AUGRC if predictions and labels provided
    if predictions is not None and labels is not None:
        aurc, _ = uq.compute_aurc(metric, predictions, labels)
        augrc, _ = uq.compute_augrc(metric, predictions, labels)
        
        results.update({
            'aurc': float(aurc),
            'augrc': float(augrc)
        })
    
    return results


# ============================================================================
# MAIN BENCHMARK FUNCTION
# ============================================================================

def run_uq_benchmark(flag, methods, output_dir, max_gps_iterations=5, 
                     batch_size=4000, image_size=224, use_cache=True):
    """
    Run UQ methods benchmark for a given dataset.
    
    Args:
        flag: Dataset name (e.g., 'breastmnist')
        methods: List of method names to run
        output_dir: Directory to save results
        max_gps_iterations: Max iterations for GPS search
        batch_size: Batch size for inference
        image_size: Image size
    
    Returns:
        dict: Results dictionary with timings and metrics
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking UQ Methods: {flag}")
    print(f"{'='*80}\n")
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    color = flag in ['dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist']
    # Use temperature for most, platt for breastmnist (binary)
    calib_method = 'platt' if flag == 'breastmnist' or flag == 'pneumoniamnist' else 'temperature'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # LOAD DATA AND MODELS
    # ========================================================================
    computed_metrics = {}
    print("📦 Loading data and models...")
    with Timer("Data loading"):
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
        
        # Load models and datasets
        if flag != 'amos22':
            models = tr.load_models(flag, device=device)
            [study_dataset, calib_dataset, test_dataset], \
            [study_loader, calib_loader, test_loader], info = \
                tr.load_datasets(flag, color, image_size, transform, batch_size)
            
            [_, calib_dataset_tta, test_dataset_tta], \
            [_, calib_loader_tta, test_loader_tta], _ = \
                tr.load_datasets(flag, color, image_size, transform_tta, batch_size)
            
            task_type = info['task']
            num_classes = len(info['label'])
        else:
            # Load OrganaMNIST models and calibration data
            models = tr.load_models('organamnist', device=device)
            [study_dataset, calib_dataset, _], \
            [study_loader, calib_loader, _], info = \
                tr.load_datasets('organamnist', color, image_size, transform, batch_size)
            
            [_, calib_dataset_tta, _], \
            [_, calib_loader_tta, _], _ = \
                tr.load_datasets('organamnist', color, image_size, transform_tta, batch_size)
            
            # Load AMOS external test dataset
            print("  Loading AMOS external test dataset...")
            amos_path = '/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/AMOS_2022/amos_external_test_224.npz'
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
            test_loader_tta = torch.utils.data.DataLoader(
                test_dataset_tta, batch_size=batch_size, shuffle=False, num_workers=4
            )
            
            task_type = info['task']
            num_classes = len(info['label'])
            print(f"  AMOS: {len(test_dataset)} samples (filtered from {len(amos_images)})")
    print(f"  Models: {len(models)}")
    print(f"  Train+val: {len(study_dataset)}, Calib: {len(calib_dataset)}, Test: {len(test_dataset)}")
    print(f"  Task: {task_type}, Classes: {num_classes}")
    print(f"  Calibration method: {calib_method}")
    
    # ========================================================================
    # EVALUATE MODELS (or load from cache)
    # ========================================================================
    
    # Cache file paths
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    calib_cache_path = os.path.join(cache_dir, f'{flag}_calib_results.npz')
    test_cache_path = os.path.join(cache_dir, f'{flag}_test_results.npz')
    
    # Try to load cached results FIRST
    if use_cache and os.path.exists(calib_cache_path) and os.path.exists(test_cache_path):
        print("\n📦 Loading cached evaluation results...")
        with Timer("Cache loading"):
            calib_cache = np.load(calib_cache_path, allow_pickle=True)
            test_cache = np.load(test_cache_path, allow_pickle=True)
            
            # Calibration
            y_true_calib = calib_cache['y_true']
            y_scores_calib = calib_cache['y_scores']
            correct_idx_calib = calib_cache['correct_idx']
            incorrect_idx_calib = calib_cache['incorrect_idx']
            indiv_scores_calib = calib_cache['indiv_scores']
            logits_calib = calib_cache['logits']
            
            # Test
            y_true = test_cache['y_true']
            y_scores = test_cache['y_scores']
            correct_idx = test_cache['correct_idx']
            incorrect_idx = test_cache['incorrect_idx']
            indiv_scores = test_cache['indiv_scores']
            logits_test = test_cache['logits']
        
        print(f"  ✓ Loaded cached results")
        print(f"  Test accuracy: {len(correct_idx) / len(y_true):.3f}")
        print(f"  Test: {len(correct_idx)} correct, {len(incorrect_idx)} incorrect")
        print(f"  Calib: {len(correct_idx_calib)} correct, {len(incorrect_idx_calib)} incorrect")
    
    else:
        # No cache - evaluate models
        print("\n🔍 Evaluating models...")
        with Timer("Model evaluation"):
            y_true, y_scores, digits, correct_idx, incorrect_idx, indiv_scores, logits_test = \
                uq.evaluate_models_on_loader(models, test_loader, device, return_logits=True)
            
            y_true_calib, y_scores_calib, digits_calib, correct_idx_calib, \
            incorrect_idx_calib, indiv_scores_calib, logits_calib = \
                uq.evaluate_models_on_loader(models, calib_loader, device, return_logits=True)
        
        print(f"  Test accuracy: {len(correct_idx) / len(y_true):.3f}")
        print(f"  Test: {len(correct_idx)} correct, {len(incorrect_idx)} incorrect")
        print(f"  Calib: {len(correct_idx_calib)} correct, {len(incorrect_idx_calib)} incorrect")
        
        # Save to cache for next time
        if use_cache:
            print("\n💾 Saving evaluation results to cache...")
            np.savez_compressed(
                calib_cache_path,
                y_true=y_true_calib,
                y_scores=y_scores_calib,
                correct_idx=correct_idx_calib,
                incorrect_idx=incorrect_idx_calib,
                indiv_scores=indiv_scores_calib,
                logits=logits_calib
            )
            np.savez_compressed(
                test_cache_path,
                y_true=y_true,
                y_scores=y_scores,
                correct_idx=correct_idx,
                incorrect_idx=incorrect_idx,
                indiv_scores=indiv_scores,
                logits=logits_test
            )
            print(f"  ✓ Cached to {cache_dir}")
            
    # ========================================================================
    # RUN UQ METHODS
    # ========================================================================
    
    # Compute predictions for AURC/AUGRC metrics
    y_pred = np.argmax(y_scores, axis=1)
    
    results = {
        'flag': flag,
        'timestamp': timestamp,
        'task_type': task_type,
        'num_classes': num_classes,
        'test_size': len(test_dataset),
        'calib_size': len(calib_dataset),
        'test_accuracy': len(correct_idx) / len(y_true),
        'calibration_method': calib_method,
        'methods': {}
    }
    
    # MSR (Maximum Softmax Response)
    if 'MSR' in methods:
        print("\n🔬 Running MSR...")
        with Timer("MSR") as timer:
            metric = compute_msr(y_scores, y_true)
        computed_metrics['MSR'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx, y_pred, y_true)
        results['methods']['MSR'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUROC: {metrics['auroc']:.4f}, AURC: {metrics.get('aurc', 'N/A')}, AUGRC: {metrics.get('augrc', 'N/A')}")
    
    # MSR with Calibration (Temperature/Platt/Isotonic)
    if 'MSR_calibrated' in methods:
        print(f"\n🔬 Running MSR + {calib_method.upper()}...")
        with Timer(f"MSR_{calib_method}") as timer:
            metric = compute_msr_calibrated(
                y_scores, y_true, y_scores_calib, y_true_calib,
                logits_test, logits_calib, calib_method
            )
        computed_metrics[f'MSR_{calib_method}'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx, y_pred, y_true)
        results['methods'][f'MSR_{calib_method}'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUROC: {metrics['auroc']:.4f}, AURC: {metrics.get('aurc', 'N/A')}, AUGRC: {metrics.get('augrc', 'N/A')}")
    
    # Ensemble STD
    if 'Ensembling' in methods:
        print("\n🔬 Running Ensemble STD...")
        with Timer("Ensembling") as timer:
            metric = compute_ensemble_std(indiv_scores)
        computed_metrics['Ensembling'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx, y_pred, y_true)
        results['methods']['Ensembling'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUROC: {metrics['auroc']:.4f}, AURC: {metrics.get('aurc', 'N/A')}, AUGRC: {metrics.get('augrc', 'N/A')}")
    
    # TTA (Test-Time Augmentation)
    if 'TTA' in methods:
        print("\n🔬 Running TTA...")
        with Timer("TTA") as timer:
            metric = compute_tta(
                models, test_dataset_tta, device, image_size, batch_size
            )
        computed_metrics['TTA'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx, y_pred, y_true)
        results['methods']['TTA'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUROC: {metrics['auroc']:.4f}, AURC: {metrics.get('aurc', 'N/A')}, AUGRC: {metrics.get('augrc', 'N/A')}")
    
    # GPS (Greedy Policy Search)
    if 'GPS' in methods:
        print("\n🔬 Running GPS...")
        if flag != 'amos22':
            aug_folder = f'/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/gps_augment/{image_size}*{image_size}/{flag}_calibration_set'
        else:
            aug_folder = f'/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/gps_augment/{image_size}*{image_size}/organamnist_calibration_set'
        
        if not os.path.exists(aug_folder):
            print(f"  ⚠️  GPS augmentation folder not found: {aug_folder}")
            print(f"  Skipping GPS for {flag}")
        else:
            with Timer("GPS") as timer:
                metric = compute_gps(
                    models, test_dataset, device, image_size,
                    aug_folder, correct_idx_calib, incorrect_idx_calib,
                    max_gps_iterations, batch_size,
                    cache_dir=os.path.join(output_dir, 'gps_cache')
                )
            metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx, y_pred, y_true)
            computed_metrics['GPS'] = metric
            results['methods']['GPS'] = {
                'time_seconds': timer.elapsed,
                **metrics
            }
            print(f"  AUROC: {metrics['auroc']:.4f}, AURC: {metrics.get('aurc', 'N/A')}, AUGRC: {metrics.get('augrc', 'N/A')}")
    
    if 'KNN_Raw' in methods:
        print("\n🔬 Running KNN (Raw Latent)...")
        with Timer("KNN_Raw") as timer:
            metric = compute_knn_raw(
                models, 
                study_dataset,  # Full study dataset for CV splits
                test_loader,    # Already-built test loader
                device,
                layer_name='avgpool',
                k=5,
                batch_size=batch_size,
                cv_seed=42,      # MUST match training
                n_splits=5       # MUST match training
            )
        computed_metrics['KNN_Raw'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx, y_pred, y_true)
        results['methods']['KNN_Raw'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUROC: {metrics['auroc']:.4f}, AURC: {metrics.get('aurc', 'N/A')}, AUGRC: {metrics.get('augrc', 'N/A')}")

    # KNN SHAP
    if 'KNN_SHAP' in methods:
        print("\n🔬 Running KNN (SHAP-Selected Latent)...")
        n_gpus = torch.cuda.device_count()
    
        if n_gpus >= 3:
            print(f"  Using {n_gpus} GPUs for parallel processing")
            parallel_mode = True
            n_jobs = 3  # Use 3 GPUs (one per fold, with 5 folds = some will reuse GPUs)
        else:
            print(f"  Only {n_gpus} GPU(s) available, using sequential mode")
            parallel_mode = False
            n_jobs = 1
        
        with Timer("KNN_SHAP") as timer:
            metric = compute_knn_shap(
                models, 
                study_dataset,
                calib_loader,
                test_loader,
                device,  # Main device (for non-parallel parts)
                flag,
                layer_name='avgpool',
                k=5,
                n_shap_features=50,
                batch_size=batch_size,
                cv_seed=42,
                n_splits=5,
                cache_dir=os.path.join(output_dir, 'shap_cache'),
                parallel=parallel_mode,
                n_jobs=n_jobs  # Number of parallel workers (≤ num GPUs)
            )
        computed_metrics['KNN_SHAP'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx, y_pred, y_true)
        results['methods']['KNN_SHAP'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUROC: {metrics['auroc']:.4f}, AURC: {metrics.get('aurc', 'N/A')}, AUGRC: {metrics.get('augrc', 'N/A')}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    # Save all metric values
    all_metrics_file = os.path.join(output_dir, f'all_metrics_{flag}_{timestamp}.npz')
    if computed_metrics:
        np.savez_compressed(all_metrics_file, **computed_metrics)
        results['all_metrics_file'] = all_metrics_file
        print(f"\n💾 All metric values saved to: {all_metrics_file}")
    
    # Generate and save evaluation plots for each method
    figures_dir = os.path.join(output_dir, 'figures', flag, timestamp)
    os.makedirs(figures_dir, exist_ok=True)
    results['figures'] = {}
    
    print(f"\n📊 Generating evaluation plots...")
    for method_name, metric_values in computed_metrics.items():
        try:
            fig_paths = uq.save_all_evaluation_plots(
                uncertainties=metric_values,
                predictions=y_pred,
                labels=y_true,
                method_name=method_name,
                output_dir=figures_dir
            )
            results['figures'][method_name] = fig_paths
        except Exception as e:
            print(f"  ⚠️  Failed to generate plots for {method_name}: {e}")
    
    print(f"✓ Figures saved to: {figures_dir}")
        
    # Save JSON global results
    output_file = os.path.join(output_dir, f'uq_benchmark_{flag}_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Time (s)':<12} {'AUROC_f':<10} {'AUGRC':<10} {'Bal. Acc':<10}")
    print("-"*80)
    for method_name, method_results in results['methods'].items():
        auroc = method_results.get('auroc', method_results.get('auc', 'N/A'))
        aurc = method_results.get('aurc', 'N/A')
        augrc = method_results.get('augrc', 'N/A')
        bal_acc = method_results.get('balanced_accuracy', 'N/A')
        
        auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) else str(auroc)
        aurc_str = f"{aurc:.4f}" if isinstance(aurc, float) else str(aurc)
        augrc_str = f"{augrc:.4f}" if isinstance(augrc, float) else str(augrc)
        bal_acc_str = f"{bal_acc:.4f}" if isinstance(bal_acc, float) else str(bal_acc)
        
        print(f"{method_name:<20} {method_results['time_seconds']:<12.2f} "
              f"{auroc_str:<10} {aurc_str:<10} {augrc_str:<10} {bal_acc_str:<10}")
    print("="*80)
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark UQ methods on medMNIST datasets'
    )
    parser.add_argument(
        '--flag', type=str,
        choices=['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'dermamnist-e',
                 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'amos22'],
        help='Dataset to benchmark'
    )
    parser.add_argument(
        '--all-flags', action='store_true',
        help='Run all available flags'
    )
    parser.add_argument(
        '--methods', nargs='+', 
        default=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP'],
        choices=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP'],
        help='UQ methods to run (default: all)'
    )
    parser.add_argument(
        '--all-methods', action='store_true',
        help='Run all available methods'
    )
    parser.add_argument(
        '--output-dir', type=str, 
        default='./uq_benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max-gps-iterations', type=int, default=5,
        help='Maximum iterations for GPS search'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4000,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--image-size', type=int, default=224,
        help='Image size'
    )
    
    args = parser.parse_args()
    if args.all_flags:
        flags = ['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'dermamnist-e',
                 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'amos22']
    else:
        flags = [args.flag]
        
    if args.all_methods:
        methods = ['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP']
    else:
        methods = args.methods
    
    for flag in flags:
        print(f"\n==============================")
        print(f" Running benchmark for: {flag} ")
        print(f"==============================")
        
        # Run benchmark
        results = run_uq_benchmark(
            flag=flag,
            methods=methods,
            output_dir=args.output_dir,
            max_gps_iterations=args.max_gps_iterations,
            batch_size=args.batch_size,
            image_size=args.image_size
        )
        
        print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()