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

def get_cv_train_loaders_for_models(study_dataset, models, batch_size=4000, 
                                   n_splits=5, seed=42, num_workers=4):
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
            shuffle=True,  # No shuffle needed for KNN fitting
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
                max_iterations=5, batch_size=4000, config=None):
    """Compute GPS using class-based API."""
    cfg = {**GPS_CONFIG, **(config or {})}
    
    # Step 1: Initialize GPS method
    gps = uq.GPSMethod(
        aug_folder=aug_folder,
        correct_calib=correct_idx_calib,
        incorrect_calib=incorrect_idx_calib,
        max_iter=max_iterations
    )
    
    # Step 2: Search for policies
    print("  Performing greedy policy search...")
    gps.search_policies(
        num_workers=cfg['num_workers'],
        num_searches=cfg['num_searches'],
        top_k=cfg['top_k'],
        seed=cfg['seed']
    )
    
    # Step 3: Compute metric
    print(f"  Applying GPS augmentations to test set...")
    metric = gps.compute(
        models, test_dataset, device,
        n=2, m=45,
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


def compute_knn_shap(models, study_dataset, calib_loader, test_loader, 
                    device, layer_name='avgpool', k=5, n_shap_features=50,
                    batch_size=4000, cv_seed=42, n_splits=5):
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
    """
    print(f"  Reconstructing CV splits (seed={cv_seed}, n_splits={n_splits})...")
    train_loaders = get_cv_train_loaders_for_models(
        study_dataset, models, batch_size, n_splits, cv_seed
    )
    
    # Pass existing calib_loader directly to the method
    knn_method = uq.KNNLatentSHAPMethod(
        layer_name=layer_name, k=k, n_shap_features=n_shap_features
    )
    knn_method.fit(models, train_loaders, calib_loader, device)
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


def evaluate_uq_method(metric, correct_idx, incorrect_idx):
    """Compute ROC AUC and optimal threshold metrics."""
    # ROC AUC
    fpr, tpr, auc = uq.roc_curve_UQ_method_computation(
        [metric[i] for i in correct_idx],
        [metric[i] for i in incorrect_idx]
    )
    
    # Optimal threshold metrics
    threshold_metrics = find_best_threshold(metric, correct_idx)
    
    # Convert numpy types to Python types for JSON serialization
    return {
        'auc': float(auc),  # Convert numpy.float64 -> Python float
        'threshold': float(threshold_metrics['threshold']),
        'balanced_accuracy': float(threshold_metrics['balanced_accuracy']),
        'specificity': float(threshold_metrics['specificity']),
        'sensitivity': float(threshold_metrics['sensitivity'])
    }


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
        models = tr.load_models(flag, device=device)
        [study_dataset, calib_dataset, test_dataset], \
        [study_loader, calib_loader, test_loader], info = \
            tr.load_datasets(flag, color, image_size, transform, batch_size)
        
        [_, calib_dataset_tta, test_dataset_tta], \
        [_, calib_loader_tta, test_loader_tta], _ = \
            tr.load_datasets(flag, color, image_size, transform_tta, batch_size)
        
        task_type = info['task']
        num_classes = len(info['label'])
    
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
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
        results['methods']['MSR'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # MSR with Calibration (Temperature/Platt/Isotonic)
    if 'MSR_calibrated' in methods:
        print(f"\n🔬 Running MSR + {calib_method.upper()}...")
        with Timer(f"MSR_{calib_method}") as timer:
            metric = compute_msr_calibrated(
                y_scores, y_true, y_scores_calib, y_true_calib,
                logits_test, logits_calib, calib_method
            )
        computed_metrics[f'MSR_{calib_method}'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
        results['methods'][f'MSR_{calib_method}'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # Ensemble STD
    if 'Ensembling' in methods:
        print("\n🔬 Running Ensemble STD...")
        with Timer("Ensembling") as timer:
            metric = compute_ensemble_std(indiv_scores)
        computed_metrics['Ensembling'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
        results['methods']['Ensembling'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # TTA (Test-Time Augmentation)
    if 'TTA' in methods:
        print("\n🔬 Running TTA...")
        with Timer("TTA") as timer:
            metric = compute_tta(
                models, test_dataset_tta, device, image_size, batch_size
            )
        computed_metrics['TTA'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
        results['methods']['TTA'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # GPS (Greedy Policy Search)
    if 'GPS' in methods:
        print("\n🔬 Running GPS...")
        aug_folder = f'/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/gps_augment/{image_size}*{image_size}/{flag}_calibration_set'
        
        if not os.path.exists(aug_folder):
            print(f"  ⚠️  GPS augmentation folder not found: {aug_folder}")
            print(f"  Skipping GPS for {flag}")
        else:
            with Timer("GPS") as timer:
                metric = compute_gps(
                    models, test_dataset, device, image_size,
                    aug_folder, correct_idx_calib, incorrect_idx_calib,
                    max_gps_iterations, batch_size
                )
            metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
            computed_metrics['GPS'] = metric
            results['methods']['GPS'] = {
                'time_seconds': timer.elapsed,
                **metrics
            }
            print(f"  AUC: {metrics['auc']:.4f}")
    
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
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
        results['methods']['KNN_Raw'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUC: {metrics['auc']:.4f}")

    # KNN SHAP
    if 'KNN_SHAP' in methods:
        print("\n🔬 Running KNN (SHAP-Selected Latent)...")
        with Timer("KNN_SHAP") as timer:
            metric = compute_knn_shap(
                models, 
                study_dataset,   # Full study dataset for CV splits
                calib_loader,    # Already-built calib loader (reuse!)
                test_loader,     # Already-built test loader
                device,
                layer_name='avgpool',
                k=5,
                n_shap_features=50,
                batch_size=batch_size,
                cv_seed=42,      # MUST match training
                n_splits=5       # MUST match training
            )
        computed_metrics['KNN_SHAP'] = metric
        metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
        results['methods']['KNN_SHAP'] = {
            'time_seconds': timer.elapsed,
            **metrics
        }
        print(f"  AUC: {metrics['auc']:.4f}")
        

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    # Save all metric values
    all_metrics_file = os.path.join(output_dir, f'all_metrics_{flag}_{timestamp}.npz')
    if computed_metrics:
        np.savez_compressed(all_metrics_file, **computed_metrics)
        results['all_metrics_file'] = all_metrics_file
        print(f"\n💾 All metric values saved to: {all_metrics_file}")
        
    # Save JSON global results
    output_file = os.path.join(output_dir, f'uq_benchmark_{flag}_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Time (s)':<12} {'AUC':<10} {'Bal. Acc':<10}")
    print("-"*80)
    for method_name, method_results in results['methods'].items():
        print(f"{method_name:<20} {method_results['time_seconds']:<12.2f} "
              f"{method_results['auc']:<10.4f} "
              f"{method_results['balanced_accuracy']:<10.4f}")
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
        '--flag', type=str, required=True,
        choices=['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'dermamnist-e',
                 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist'],
        help='Dataset to benchmark'
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
    
    if args.all_methods:
        methods = ['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP']
    else:
        methods = args.methods
    
    # Run benchmark
    results = run_uq_benchmark(
        flag=args.flag,
        methods=methods,
        output_dir=args.output_dir,
        max_gps_iterations=args.max_gps_iterations,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()