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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

# Import FailCatcher library
import FailCatcher
from FailCatcher import failure_detection
from FailCatcher import UQ_toolbox as uq

# Import medMNIST-specific utilities
from utils import train_load_datasets_resnet as tr


class RepeatGrayToRGB:
    """Transform for converting grayscale to RGB."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)


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
                           batch_size=4000, image_size=224):
    """
    Run UQ benchmark on a medMNIST dataset using FailCatcher library.
    
    Args:
        flag: Dataset name (e.g., 'breastmnist', 'organamnist')
        methods: List of method names to run
        output_dir: Output directory for results
        batch_size: Batch size for inference
        image_size: Image size
    """
    print(f"\n{'='*80}")
    print(f"MedMNIST Benchmark: {flag}")
    print(f"Using FailCatcher v{FailCatcher.__version__}")
    print(f"{'='*80}\n")
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        models = tr.load_models(flag, device=device)
        [study_dataset, calib_dataset, test_dataset], \
        [_, calib_loader, test_loader], info = \
            tr.load_datasets(flag, color, image_size, transform, batch_size)
        
        [_, _, test_dataset_tta], \
        [_, _, _], _ = \
            tr.load_datasets(flag, color, image_size, transform_tta, batch_size)
    else:
        # Load datasets and models of organamnist and amos2022 as test set
        models = tr.load_models('organamnist', device=device)
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
    print(f"  Train+val: {len(study_dataset)}, Calib: {len(calib_dataset)}, Test: {len(test_dataset)}")
    print(f"  Task: {info['task']}, Classes: {len(info['label'])}")
    
    # ========================================================================
    # EVALUATE MODELS (or load from cache)
    # ========================================================================
    
    # Cache file paths
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    calib_cache_path = os.path.join(cache_dir, f'{flag}_calib_results.npz')
    test_cache_path = os.path.join(cache_dir, f'{flag}_test_results.npz')
    
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
        indiv_scores_calib = calib_cache['indiv_scores']
        logits_calib = calib_cache['logits']
        
        # Test
        y_true = test_cache['y_true']
        y_scores = test_cache['y_scores']
        correct_idx = test_cache['correct_idx']
        incorrect_idx = test_cache['incorrect_idx']
        indiv_scores = test_cache['indiv_scores']
        logits = test_cache['logits']
        
        # Compute y_pred from cached scores (for compatibility with old cache format)
        y_pred = np.argmax(y_scores, axis=1)
        y_pred_calib = np.argmax(y_scores_calib, axis=1)
        
        print(f"  ✓ Loaded cached results")
        print(f"  Test accuracy: {len(correct_idx) / len(y_true):.4f}")
    
    else:
        # No cache - evaluate models
        print("\n📊 Evaluating ensemble predictions on test set...")
        y_true, y_scores, y_pred, correct_idx, incorrect_idx, indiv_scores, logits = uq.evaluate_models_on_loader(
            models, test_loader, device, return_logits=True
        )
        
        # Calibration set
        y_true_calib, y_scores_calib, y_pred_calib, correct_idx_calib, incorrect_idx_calib, indiv_scores_calib, logits_calib = \
            uq.evaluate_models_on_loader(models, calib_loader, device, return_logits=True)
        
        print(f"  Test accuracy: {len(correct_idx)/len(y_true):.4f}")
        
        # Save to cache for next time
        print("\n💾 Saving evaluation results to cache...")
        np.savez_compressed(
            calib_cache_path,
            y_true=y_true_calib,
            y_scores=y_scores_calib,
            y_pred=y_pred_calib,
            correct_idx=correct_idx_calib,
            incorrect_idx=incorrect_idx_calib,
            indiv_scores=indiv_scores_calib,
            logits=logits_calib
        )
        np.savez_compressed(
            test_cache_path,
            y_true=y_true,
            y_scores=y_scores,
            y_pred=y_pred,
            correct_idx=correct_idx,
            incorrect_idx=incorrect_idx,
            indiv_scores=indiv_scores,
            logits=logits
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
        uncertainties, metrics = detector.run_msr(y_scores, y_true)
        results['MSR'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'MSR_calibrated' in methods:
        print(f"\n🔍 Running MSR-{calib_method}...")
        uncertainties, metrics = detector.run_msr_calibrated(
            y_scores, y_true, y_scores_calib, y_true_calib,
            logits, logits_calib, method=calib_method
        )
        results[f'MSR_{calib_method}'] = metrics
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
    
    if 'GPS' in methods:
        print("\n🔍 Running GPS...")
        aug_folder = f'./medMNIST/gps_augment/{image_size}*{image_size}/{flag}_calibration_set'
        uncertainties, metrics = detector.run_gps(
            test_dataset_tta, y_true,
            aug_folder=aug_folder,
            correct_idx_calib=correct_idx_calib.tolist(),
            incorrect_idx_calib=incorrect_idx_calib.tolist(),
            image_size=image_size,
            batch_size=batch_size,
            cache_dir=os.path.join(output_dir, 'gps_cache')
        )
        results['GPS'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'KNN_Raw' in methods:
        print("\n🔍 Running KNN-Raw...")
        uncertainties, metrics = detector.run_knn_raw(
            test_loader=test_loader,
            train_loaders=train_loaders,
            y_true=y_true,
            layer_name='avgpool',
            k=5
        )
        results['KNN_Raw'] = metrics
        print(f"  AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")
    
    if 'KNN_SHAP' in methods:
        print("\n🔍 Running KNN-SHAP...")
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
            n_jobs=n_jobs
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
        timestamp=timestamp
    )
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'AUROC_f':<10} {'AURC':<10} {'AUGRC':<10} {'Accuracy':<10}")
    print("-"*80)
    for method_name, method_results in results.items():
        print(f"{method_name:<20} "
              f"{method_results['auroc_f']:<10.4f} "
              f"{method_results['aurc']:<10.6f} "
              f"{method_results['augrc']:<10.6f} "
              f"{method_results['accuracy']:<10.4f}")
    print("="*80)
    
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
        '--methods', nargs='+',
        default=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP'],
        choices=['MSR', 'MSR_calibrated', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP'],
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
    
    args = parser.parse_args()
    
    run_medmnist_benchmark(
        flag=args.flag,
        methods=args.methods,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
