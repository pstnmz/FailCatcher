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


def compute_tta(models, test_dataset, device, num_classes, image_size, 
                color=False, batch_size=4000, nb_augmentations=5):
    """Compute TTA with regular RandAugment (torchvision)."""
    metric, avg_preds = uq.TTA(
        transformations=None,  # Use random policies
        models=models,
        dataset=test_dataset,
        device=device,
        nb_augmentations=nb_augmentations,
        usingBetterRandAugment=False,  # Use regular RandAugment
        n=2,
        m=9,  # Standard magnitude for torchvision RandAugment
        image_normalization=True,
        nb_channels=3,
        mean=0.5,
        std=0.5,
        image_size=image_size,
        batch_size=batch_size
    )
    return metric


def compute_gps(models, test_dataset, device, num_classes, image_size,
                aug_folder, correct_idx_calib, incorrect_idx_calib,
                max_iterations=5, batch_size=4000):
    """Compute GPS with BetterRandAugment (greedy policy search)."""
    # Step 1: Greedy search on calibration set
    print("  Performing greedy policy search...")
    best_aug = uq.perform_greedy_policy_search(
        aug_folder, 
        correct_idx_calib, 
        incorrect_idx_calib,
        num_workers=90,
        max_iterations=max_iterations,
        num_searches=30,
        top_k=3,
        plot=False,
        seed=24
    )
    
    # Step 2: Extract augmentation policies - pass entire group at once
    # best_aug is [[group1_files], [group2_files], [group3_files]]
    # where each group has 5 policy files
    if isinstance(best_aug, list) and all(isinstance(policy, list) for policy in best_aug):
        transformation_pipeline = []
        for aug in best_aug:
            n, m, transformations = uq.extract_gps_augmentations_info(aug)
            transformation_pipeline.append(transformations)
    else:
        raise ValueError("GPS search did not return valid policies")
    
    # Step 3: Apply TTA to the ENTIRE transformation_pipeline at once
    # This matches the notebook behavior: TTA receives [[5 policies], [5 policies], [5 policies]]
    print(f"  Applying GPS augmentations to test set...")
    metric, _ = uq.TTA(
        transformation_pipeline, 
        models, 
        test_dataset, 
        device,
        usingBetterRandAugment=True, 
        n=n, 
        m=m, 
        nb_channels=3, 
        image_size=image_size,
        image_normalization=True, 
        mean=0.5, 
        std=0.5, 
        batch_size=batch_size
)
    
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
                     batch_size=4000, image_size=224):
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
        [train_dataset, calib_dataset, test_dataset], \
        [train_loader, calib_loader, test_loader], info = \
            tr.load_datasets(flag, color, image_size, transform, batch_size)
        
        [_, calib_dataset_tta, test_dataset_tta], \
        [_, calib_loader_tta, test_loader_tta], _ = \
            tr.load_datasets(flag, color, image_size, transform_tta, batch_size)
        
        task_type = info['task']
        num_classes = len(info['label'])
    
    print(f"  Models: {len(models)}")
    print(f"  Train: {len(train_dataset)}, Calib: {len(calib_dataset)}, Test: {len(test_dataset)}")
    print(f"  Task: {task_type}, Classes: {num_classes}")
    print(f"  Calibration method: {calib_method}")
    
    # ========================================================================
    # EVALUATE MODELS (with logits for temperature scaling)
    # ========================================================================
    
    print("\n🔍 Evaluating models...")
    with Timer("Model evaluation"):
        # Test set (with logits)
        y_true, y_scores, digits, correct_idx, incorrect_idx, indiv_scores, logits_test = \
            uq.evaluate_models_on_loader(models, test_loader, device, return_logits=True)
        
        # Calibration set (with logits)
        y_true_calib, y_scores_calib, digits_calib, correct_idx_calib, \
        incorrect_idx_calib, indiv_scores_calib, logits_calib = \
            uq.evaluate_models_on_loader(models, calib_loader, device, return_logits=True)
    
    print(f"  Test accuracy: {len(correct_idx) / len(y_true):.3f}")
    print(f"  Test: {len(correct_idx)} correct, {len(incorrect_idx)} incorrect")
    print(f"  Calib: {len(correct_idx_calib)} correct, {len(incorrect_idx_calib)} incorrect")
    
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
                models, test_dataset_tta, device, num_classes, 
                image_size, color, batch_size
            )
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
                    models, test_dataset, device, num_classes, image_size,
                    aug_folder, correct_idx_calib, incorrect_idx_calib,
                    max_gps_iterations, batch_size
                )
            metrics = evaluate_uq_method(metric, correct_idx, incorrect_idx)
            results['methods']['GPS'] = {
                'time_seconds': timer.elapsed,
                **metrics
            }
            print(f"  AUC: {metrics['auc']:.4f}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
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
        default=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS'],
        choices=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS'],
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
        methods = ['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS']
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