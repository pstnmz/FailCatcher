"""
Failure detection and uncertainty quantification runner for FailCatcher.

This module provides a clean, dataset-agnostic API for detecting failures
and quantifying uncertainty in PyTorch models using various UQ methods.

Example:
    >>> from FailCatcher import failure_detection
    >>> 
    >>> detector = failure_detection.FailureDetector(
    ...     models=models,
    ...     study_dataset=train_dataset,
    ...     calib_dataset=calib_dataset,
    ...     test_dataset=test_dataset,
    ...     device='cuda'
    ... )
    >>> 
    >>> # Run MSR
    >>> results = detector.run_msr(y_scores_test, y_true_test)
    >>> 
    >>> # Run ensemble
    >>> results = detector.run_ensemble(indiv_scores_test)
    >>> 
    >>> # Cache augmentations for GPS
    >>> aug_folder = detector.run_augmentation_calibration_caching(
    ...     aug_folder='gps_augment/dataset_calib',
    ...     N=2, M=45, num_policies=500
    ... )
    >>> 
    >>> # Run GPS with cached augmentations
    >>> results = detector.run_gps(
    ...     test_dataset, y_true, aug_folder, 
    ...     correct_idx_calib, incorrect_idx_calib
    ... )
    >>> 
    >>> # Run KNN-SHAP
    >>> results = detector.run_knn_shap(
    ...     calib_loader=calib_loader,
    ...     test_loader=test_loader,
    ...     train_loaders=train_loaders
    ... )
"""

import os
import numpy as np
import torch
from typing import List, Callable, Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader
import time

from . import UQ_toolbox as uq


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
        if self.name:
            print(f"⏱️  {self.name}: {self.elapsed:.2f}s")


class FailureDetector:
    """
    Failure detection and uncertainty quantification runner.
    
    This class provides a dataset-agnostic interface for running and evaluating
    uncertainty quantification methods on PyTorch models to detect failures.
    
    Args:
        models: List of PyTorch models (e.g., CV fold models)
        study_dataset: Full training dataset (for CV splitting)
        calib_dataset: Calibration dataset
        test_dataset: Test dataset
        device: torch.device or str
        num_classes: Number of classes in classification task
        
    Example:
        >>> detector = FailureDetector(
        ...     models=fold_models,
        ...     study_dataset=train_data,
        ...     calib_dataset=calib_data,
        ...     test_dataset=test_data,
        ...     device='cuda'
        ... )
    """
    
    def __init__(
        self,
        models: List[torch.nn.Module],
        study_dataset: torch.utils.data.Dataset,
        calib_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        device: torch.device = None,
        num_classes: int = None
    ):
        self.models = models
        self.study_dataset = study_dataset
        self.calib_dataset = calib_dataset
        self.test_dataset = test_dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        # Cache for predictions (computed once)
        self._test_predictions_cache = None
        self._test_loader_cache = None
        
        # Storage for computed uncertainties and predictions
        self._uncertainties = {}
        self._predictions_per_fold = {}  # Store per-fold predictions for accurate ROC curves
        self._results = {}
    
    def set_test_predictions(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None
    ):
        """
        Set pre-computed test predictions to avoid recomputing.
        
        Args:
            y_scores: Ensemble probability scores [N, num_classes]
            y_true: True labels [N]
            y_pred: Predicted labels [N] (optional, will compute from y_scores if not provided)
        """
        if y_pred is None:
            y_pred = np.argmax(y_scores, axis=1)
        
        correct_idx = np.where(y_pred == y_true)[0]
        incorrect_idx = np.where(y_pred != y_true)[0]
        
        self._test_predictions_cache = {
            'y_scores': y_scores,
            'y_true': y_true,
            'y_pred': y_pred,
            'correct_idx': correct_idx,
            'incorrect_idx': incorrect_idx
        }
    
    def run_msr(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        indiv_scores: Optional[np.ndarray] = None,
        per_fold_evaluation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Maximum Softmax Response (MSR).
        
        Args:
            y_scores: Ensemble probability scores [N, num_classes]
            y_true: True labels [N]
            indiv_scores: Optional individual model scores [num_models, N, num_classes]
            per_fold_evaluation: If True and indiv_scores provided, compute per-fold metrics.
                                If False, use ensemble predictions (legacy behavior)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True and indiv_scores provided: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N]
        """
        timer = Timer("MSR computation")
        with timer:
            if indiv_scores is not None and per_fold_evaluation:
                # Compute MSR per-fold
                uncertainties_per_fold = []
                predictions_per_fold = []
                for fold_idx in range(indiv_scores.shape[0]):
                    fold_scores = indiv_scores[fold_idx]  # [N, num_classes]
                    fold_uncertainties = uq.distance_to_hard_labels_computation(fold_scores)
                    fold_predictions = np.argmax(fold_scores, axis=1)
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
            else:
                # Ensemble-based: single uncertainty from ensemble
                uncertainties = uq.distance_to_hard_labels_computation(y_scores)
                predictions_per_fold = None
        
        # Use cached predictions if available
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true, 
                                       predictions_per_fold=predictions_per_fold)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble (for reference)
            self._uncertainties['MSR'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['MSR_per_fold'] = uncertainties  # [num_folds, N] for plotting all curves
            self._uncertainties['MSR_ensemble'] = uq.distance_to_hard_labels_computation(y_scores)  # [N]
            self._predictions_per_fold['MSR'] = predictions_per_fold  # [num_folds, N] for accurate ROC curves
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['MSR'] = uncertainties
        self._results['MSR'] = metrics
        
        return uncertainties, metrics
    
    def run_mls(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        indiv_logits: Optional[np.ndarray] = None,
        per_fold_evaluation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Maximum Logit Score (MLS) - negative of maximum unnormalized logit.
        
        Args:
            logits: Ensemble logits [N, C] (unnormalized scores)
            y_true: True labels [N]
            indiv_logits: Optional individual model logits [num_models, N, C]
            per_fold_evaluation: If True and indiv_logits provided, compute per-fold metrics.
                                If False, use ensemble predictions (legacy behavior)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True and indiv_logits provided: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N]
        """
        from .methods.distance import maximum_logit_score_computation
        
        timer = Timer("MLS computation")
        with timer:
            if indiv_logits is not None and per_fold_evaluation:
                # Compute MLS per-fold
                uncertainties_per_fold = []
                predictions_per_fold = []
                for fold_idx in range(indiv_logits.shape[0]):
                    fold_logits = indiv_logits[fold_idx]  # [N, C]
                    fold_uncertainties = maximum_logit_score_computation(fold_logits)
                    # Predictions from logits
                    if fold_logits.ndim == 1 or fold_logits.shape[1] == 1:
                        # Binary: threshold at 0
                        fold_predictions = (fold_logits.ravel() > 0).astype(int)
                    else:
                        # Multiclass: argmax
                        fold_predictions = np.argmax(fold_logits, axis=1)
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
            else:
                # Ensemble-based: single uncertainty from ensemble
                uncertainties = maximum_logit_score_computation(logits)
                predictions_per_fold = None
        
        # Use cached predictions if available
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            # Compute predictions from logits
            if logits.ndim == 1 or logits.shape[1] == 1:
                y_pred = (logits.ravel() > 0).astype(int)
            else:
                y_pred = np.argmax(logits, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true, 
                                       predictions_per_fold=predictions_per_fold)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble (for reference)
            self._uncertainties['MLS'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['MLS_per_fold'] = uncertainties  # [num_folds, N] for plotting all curves
            self._uncertainties['MLS_ensemble'] = maximum_logit_score_computation(logits)  # [N]
            self._predictions_per_fold['MLS'] = predictions_per_fold  # [num_folds, N] for accurate ROC curves
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['MLS'] = uncertainties
        self._results['MLS'] = metrics
        
        return uncertainties, metrics
    
    def run_msr_calibrated(
        self,
        y_scores_test: np.ndarray,
        y_true_test: np.ndarray,
        y_scores_calib: np.ndarray,
        y_true_calib: np.ndarray,
        logits_test: Optional[np.ndarray] = None,
        logits_calib: Optional[np.ndarray] = None,
        indiv_logits_test: Optional[np.ndarray] = None,
        indiv_logits_calib: Optional[np.ndarray] = None,
        indiv_scores_test: Optional[np.ndarray] = None,
        indiv_scores_calib: Optional[np.ndarray] = None,
        method: str = 'temperature',
        per_fold_evaluation: bool = True,
        auto_tune_platt: bool = False,
        verbose_tuning: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run calibrated MSR using post-hoc calibration.
        
        Args:
            y_scores_test: Ensemble test probabilities [N, C]
            y_true_test: Test labels [N]
            y_scores_calib: Ensemble calibration probabilities [N_calib, C]
            y_true_calib: Calibration labels [N_calib]
            logits_test: Ensemble test logits (for temperature) [N, C]
            logits_calib: Ensemble calibration logits (for temperature) [N_calib, C]
            indiv_logits_test: Per-fold test logits [num_folds, N, C]
            indiv_logits_calib: Per-fold calibration logits [num_folds, N_calib, C]
            indiv_scores_test: Per-fold test scores [num_folds, N, C]
            indiv_scores_calib: Per-fold calibration scores [num_folds, N_calib, C]
            method: 'temperature', 'platt', or 'isotonic'
            per_fold_evaluation: If True and indiv_* provided, compute per-fold metrics.
                                If False, use ensemble predictions (legacy behavior)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True and indiv_* provided: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N]
        """
        from .methods.distance import posthoc_calibration
        from .core.utils import apply_calibration
        
        timer = Timer(f"MSR-{method} calibration")
        with timer:
            if per_fold_evaluation and method == 'temperature' and indiv_logits_test is not None and indiv_logits_calib is not None:
                # Per-fold calibration with temperature scaling
                uncertainties_per_fold = []
                predictions_per_fold = []
                num_folds = indiv_logits_test.shape[0]
                
                for fold_idx in range(num_folds):
                    fold_logits_calib = indiv_logits_calib[fold_idx]
                    fold_logits_test = indiv_logits_test[fold_idx]
                    
                    # Fit calibration on this fold's calibration set
                    _, calibration_model = posthoc_calibration(
                        fold_logits_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    
                    # Get fold's test scores
                    fold_scores_test = indiv_scores_test[fold_idx] if indiv_scores_test is not None else None
                    if fold_scores_test is None:
                        # Compute from logits if not provided
                        import torch
                        fold_scores_test = torch.softmax(torch.from_numpy(fold_logits_test), dim=1).numpy()
                    
                    # Apply calibration
                    calibrated_scores = apply_calibration(
                        fold_scores_test, calibration_model, method, logits=fold_logits_test
                    )
                    
                    # Compute uncertainty and predictions
                    fold_uncertainties = uq.distance_to_hard_labels_computation(calibrated_scores)
                    # Handle both binary (1D) and multiclass (2D) calibrated scores
                    if calibrated_scores.ndim == 1:
                        # Binary: calibrated_scores is [N] (prob of positive class)
                        fold_predictions = (calibrated_scores > 0.5).astype(int)
                    else:
                        # Multiclass: calibrated_scores is [N, C]
                        fold_predictions = np.argmax(calibrated_scores, axis=1)
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
                
            elif per_fold_evaluation and method != 'temperature' and indiv_scores_test is not None and indiv_scores_calib is not None:
                # Per-fold calibration with Platt/Isotonic
                uncertainties_per_fold = []
                predictions_per_fold = []
                num_folds = indiv_scores_test.shape[0]
                
                for fold_idx in range(num_folds):
                    fold_scores_calib = indiv_scores_calib[fold_idx]
                    fold_scores_test = indiv_scores_test[fold_idx]
                    
                    # Fit calibration on this fold's calibration set
                    _, calibration_model = posthoc_calibration(
                        fold_scores_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    
                    # Apply calibration
                    calibrated_scores = apply_calibration(
                        fold_scores_test, calibration_model, method
                    )
                    
                    # Compute uncertainty and predictions
                    fold_uncertainties = uq.distance_to_hard_labels_computation(calibrated_scores)
                    # Handle both binary (1D) and multiclass (2D) calibrated scores
                    if calibrated_scores.ndim == 1:
                        # Binary: calibrated_scores is [N] (prob of positive class)
                        fold_predictions = (calibrated_scores > 0.5).astype(int)
                    else:
                        # Multiclass: calibrated_scores is [N, C]
                        fold_predictions = np.argmax(calibrated_scores, axis=1)
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
                
            else:
                # Ensemble mode: calibrate the ENSEMBLE predictions (not individual folds)
                # This is the correct approach for deep ensembles - treat ensemble as a single model
                if method == 'temperature':
                    if logits_calib is None:
                        raise ValueError("Temperature scaling requires logits")
                    _, calibration_model = posthoc_calibration(
                        logits_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    calibrated_scores = apply_calibration(y_scores_test, calibration_model, method, logits=logits_test)
                else:
                    _, calibration_model = posthoc_calibration(
                        y_scores_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    calibrated_scores = apply_calibration(y_scores_test, calibration_model, method)
                
                # Compute uncertainty on calibrated ensemble predictions
                uncertainties = uq.distance_to_hard_labels_computation(calibrated_scores)
                predictions_per_fold = None
        
        # Handle both 1D (binary) and 2D (multiclass) calibrated scores for y_pred for y_pred
        # Use ensemble predictions for computing correct/incorrect indices
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred_calibrated = self._test_predictions_cache['y_pred']
        else:
            if y_scores_test.ndim == 1:
                # Binary classification: y_scores_test is [N] (prob of positive class)
                y_pred_calibrated = (y_scores_test > 0.5).astype(int)
            else:
                # Multiclass: y_scores_test is [N, C]
                y_pred_calibrated = np.argmax(y_scores_test, axis=1)
            
            correct_idx = np.where(y_pred_calibrated == y_true_test)[0]
            incorrect_idx = np.where(y_pred_calibrated != y_true_test)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx,
                                       y_pred_calibrated, y_true_test,
                                       predictions_per_fold=predictions_per_fold)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble (for reference)
            self._uncertainties['MSR_calibrated'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['MSR_calibrated_per_fold'] = uncertainties  # [num_folds, N]
            self._predictions_per_fold['MSR_calibrated'] = predictions_per_fold  # [num_folds, N] for accurate ROC curves
            
            # Compute ensemble calibration for reference
            from .methods.distance import posthoc_calibration
            from .core.utils import apply_calibration
            
            if method == 'temperature':
                if logits_calib is None:
                    raise ValueError("Temperature scaling requires ensemble logits")
                _, calib_model = posthoc_calibration(
                    logits_calib, y_true_calib, method,
                    auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                )
                calibrated_ensemble = apply_calibration(y_scores_test, calib_model, method, logits=logits_test)
            else:
                _, calib_model = posthoc_calibration(
                    y_scores_calib, y_true_calib, method,
                    auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                )
                calibrated_ensemble = apply_calibration(y_scores_test, calib_model, method)
            
            self._uncertainties['MSR_calibrated_ensemble'] = uq.distance_to_hard_labels_computation(calibrated_ensemble)
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['MSR_calibrated'] = uncertainties
        self._results['MSR_calibrated'] = metrics
        
        return uncertainties, metrics
    
    def run_ensemble(
        self,
        indiv_scores: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run ensemble standard deviation.
        
        Args:
            indiv_scores: Individual model scores [num_models, N, num_classes]
            y_true: True labels [N]
        
        Returns:
            tuple: (uncertainties, metrics_dict)
        """
        timer = Timer("Ensemble STD computation")
        with timer:
            uncertainties = uq.ensembling_stds_computation(indiv_scores)
        
        # Use cached predictions if available
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_scores = np.mean(indiv_scores, axis=0)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        self._uncertainties['Ensembling'] = uncertainties
        self._results['Ensembling'] = metrics
        
        return uncertainties, metrics
    
    def run_tta(
        self,
        test_dataset: torch.utils.data.Dataset,
        y_true: np.ndarray,
        image_size: int = 224,
        batch_size: int = 4000,
        nb_augmentations: int = 5,
        n: int = 2,
        m: int = 9,
        nb_channels: int = 3,
        mean: float = 0.5,
        std: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Test-Time Augmentation (TTA).
        
        Args:
            test_dataset: Test dataset (with transforms applied)
            y_true: True labels [N]
            image_size: Image size
            batch_size: Batch size for inference
            nb_augmentations: Number of augmentations
            n: RandAugment num_ops
            m: RandAugment magnitude
            nb_channels: Number of image channels
            mean: Normalization mean
            std: Normalization std
        
        Returns:
            tuple: (uncertainties, metrics_dict)
        """
        timer = Timer("TTA computation")
        with timer:
            tta = uq.TTAMethod(
                transformations=None,  # Random policies
                n=n,
                m=m,
                nb_augmentations=nb_augmentations,
                nb_channels=nb_channels,
                image_size=image_size,
                image_normalization=True,
                mean=mean,
                std=std,
                batch_size=batch_size
            )
            
            uncertainties = tta.compute(self.models, test_dataset, self.device)
        
        # Use cached predictions if available, otherwise compute
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        self._uncertainties['TTA'] = uncertainties
        self._results['TTA'] = metrics
        
        return uncertainties, metrics
    
    def run_augmentation_calibration_caching(
        self,
        dataset: torch.utils.data.Dataset,
        aug_folder: str,
        N: int = 2,
        M: int = 45,
        num_policies: int = 500,
        image_size: int = 224,
        batch_size: int = 128,
        nb_channels: int = 3,
        image_normalization: bool = True,
        mean: float = 0.5,
        std: float = 0.5,
        use_monai_cache: bool = True,
        cache_rate: float = 1.0,
        cache_num_workers: int = 8,
        dataloader_workers: int = 8,
        dataloader_prefetch: int = 8
    ) -> str:
        """
        Run BetterRandAugment on calibration dataset and cache results for GPS.
        
        This function applies random augmentation policies to the calibration dataset,
        computes model predictions for each augmented version, and saves them to disk
        as .npz files. These cached predictions are then used by GPS for greedy policy search.
        
        Args:
            dataset: Calibration dataset (should use transform WITHOUT normalization, e.g., calib_dataset_tta)
            aug_folder: Output directory for cached augmentation predictions
            N: Number of augmentation operations per policy (BetterRandAugment)
            M: Magnitude parameter for augmentations (0-100 typical)
            num_policies: Number of random augmentation policies to generate
            image_size: Input image size
            batch_size: Batch size for inference
            nb_channels: Number of image channels (1 for grayscale, 3 for RGB)
            image_normalization: Whether to apply normalization
            mean: Normalization mean (single value or per-channel)
            std: Normalization std (single value or per-channel)
            use_monai_cache: Whether to use MONAI CacheDataset for faster loading
            cache_rate: Fraction of dataset to cache in memory (0-1)
            cache_num_workers: Number of workers for cache building
            dataloader_workers: Number of workers for DataLoader
            dataloader_prefetch: Prefetch factor for DataLoader
        
        Returns:
            str: Path to the output folder containing cached predictions
        
        Example:
            >>> detector = FailureDetector(models, train_data, calib_data, test_data, device)
            >>> aug_folder = detector.run_augmentation_calibration_caching(
            ...     dataset=calib_dataset_tta,  # Use unnormalized dataset!
            ...     aug_folder='gps_augment/breastmnist_calib',
            ...     N=2, M=45, num_policies=500,
            ...     use_monai_cache=True, batch_size=128
            ... )
            >>> # Later use for GPS:
            >>> uncertainties, metrics = detector.run_gps(
            ...     test_dataset, y_true, aug_folder, correct_idx, incorrect_idx
            ... )
        """
        print(f"\n🔄 Caching augmentation predictions for GPS...")
        print(f"  Output folder: {aug_folder}")
        print(f"  Calibration set size: {len(dataset)}")
        print(f"  Policies: {num_policies}, N={N}, M={M}")
        print(f"  MONAI cache: {use_monai_cache}, Cache rate: {cache_rate}")
        
        timer = Timer("Augmentation caching")
        with timer:
            uq.apply_randaugment_and_store_results(
                dataset=dataset,
                models=self.models,
                N=N,
                M=M,
                num_policies=num_policies,
                device=self.device,
                folder_name=aug_folder,
                image_normalization=image_normalization,
                mean=mean,
                std=std,
                nb_channels=nb_channels,
                image_size=image_size,
                batch_size=batch_size,
                use_monai_cache=use_monai_cache,
                cache_rate=cache_rate,
                cache_num_workers=cache_num_workers,
                dataloader_workers=dataloader_workers,
                dataloader_prefetch=dataloader_prefetch
            )
        
        # Verify files were created
        import glob
        npz_files = glob.glob(os.path.join(aug_folder, f"N{N}_M{M}_*.npz"))
        print(f"✓ Cached {len(npz_files)} augmentation prediction files")
        
        if len(npz_files) == 0:
            raise RuntimeError(f"No .npz files created in {aug_folder}")
        
        return aug_folder
    
    def run_gps(
        self,
        test_dataset: torch.utils.data.Dataset,
        y_true: np.ndarray,
        aug_folder: str,
        correct_idx_calib: List[int],
        incorrect_idx_calib: List[int],
        image_size: int = 224,
        batch_size: int = 4000,
        max_iterations: int = 5,
        num_workers: int = 90,
        num_searches: int = 30,
        top_k: int = 3,
        seed: int = 64,
        nb_channels: int = 3,
        mean: float = 0.5,
        std: float = 0.5,
        cache_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Greedy Policy Search (GPS).
        
        Args:
            test_dataset: Test dataset
            y_true: True labels [N]
            aug_folder: Folder containing pre-computed augmentation predictions
            correct_idx_calib: Indices of correctly classified calibration samples
            incorrect_idx_calib: Indices of incorrectly classified calibration samples
            image_size: Image size
            batch_size: Batch size
            max_iterations: Max GPS iterations
            num_workers: Number of parallel workers for search
            num_searches: Number of searches per iteration
            top_k: Top-k policies to select
            seed: Random seed
            nb_channels: Number of image channels
            mean: Normalization mean
            std: Normalization std
            cache_dir: Directory to cache policies
        
        Returns:
            tuple: (uncertainties, metrics_dict)
        """
        import os
        import pickle
        import hashlib
        
        timer = Timer("GPS computation")
        with timer:
            # Initialize GPS
            gps = uq.GPSMethod(
                aug_folder=aug_folder,
                correct_calib=correct_idx_calib,
                incorrect_calib=incorrect_idx_calib,
                max_iter=max_iterations
            )
            
            # Cache logic
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_key_parts = [
                    aug_folder,
                    str(sorted(correct_idx_calib)),
                    str(sorted(incorrect_idx_calib)),
                    str(max_iterations),
                    str(num_workers),
                    str(num_searches),
                    str(top_k),
                    str(seed)
                ]
                cache_key = hashlib.md5('_'.join(cache_key_parts).encode()).hexdigest()
                cache_file = os.path.join(cache_dir, f'gps_policies_{cache_key}.pkl')
                
                if os.path.exists(cache_file):
                    print(f"  Loading cached GPS policies...")
                    with open(cache_file, 'rb') as f:
                        gps.policies = pickle.load(f)
                    print(f"  ✓ Loaded {len(gps.policies)} policy groups from cache")
                else:
                    gps.search_policies(
                        num_workers=num_workers,
                        num_searches=num_searches,
                        top_k=top_k,
                        seed=seed
                    )
                    with open(cache_file, 'wb') as f:
                        pickle.dump(gps.policies, f)
                    print(f"  ✓ Cached {len(gps.policies)} policy groups")
            else:
                gps.search_policies(
                    num_workers=num_workers,
                    num_searches=num_searches,
                    top_k=top_k,
                    seed=seed
                )
            
            # Compute metric
            uncertainties = gps.compute(
                self.models, test_dataset, self.device,
                n=2, m=45,
                nb_channels=nb_channels,
                image_size=image_size,
                image_normalization=True,
                mean=mean,
                std=std,
                batch_size=batch_size
            )
        
        # Use cached predictions if available, otherwise compute
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        self._uncertainties['GPS'] = uncertainties
        self._results['GPS'] = metrics
        
        return uncertainties, metrics
    
    def run_knn_raw(
        self,
        test_loader: DataLoader,
        train_loaders: List[DataLoader],
        y_true: np.ndarray,
        layer_name: str = 'avgpool',
        k: int = 5,
        per_fold_evaluation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run KNN in raw latent space.
        
        Args:
            test_loader: Test data loader (or single DataLoader if 1 model)
            train_loaders: List of train data loaders (one per model fold)
                          Or single DataLoader if models is a single model
            y_true: True labels [N]
            layer_name: Layer name for feature extraction
            k: Number of nearest neighbors
            per_fold_evaluation: If True, compute per-fold metrics. If False, average uncertainties
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N] (averaged)
        """
        timer = Timer("KNN-Raw computation")
        with timer:
            # Fit and compute uncertainties
            knn_method = uq.KNNLatentMethod(layer_name=layer_name, k=k)
            knn_method.fit(self.models, train_loaders, self.device)
            uncertainties = knn_method.compute(self.models, test_loader, self.device, 
                                              return_per_fold=per_fold_evaluation)
        
        # Use cached predictions if available, otherwise compute
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble averaged
            self._uncertainties['KNN_Raw'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['KNN_Raw_per_fold'] = uncertainties  # [num_folds, N]
            # Note: KNN_Raw doesn't compute ensemble separately, so no _ensemble key
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['KNN_Raw'] = uncertainties
        self._results['KNN_Raw'] = metrics
        
        return uncertainties, metrics
    
    def run_knn_shap(
        self,
        calib_loader: DataLoader,
        test_loader: DataLoader,
        train_loaders: List[DataLoader],
        y_true: np.ndarray,
        flag: str = 'dataset',
        layer_name: str = 'avgpool',
        k: int = 5,
        n_shap_features: int = 50,
        cache_dir: Optional[str] = None,
        parallel: bool = False,
        n_jobs: int = 2,
        per_fold_evaluation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run KNN in SHAP-selected latent space.
        
        Args:
            calib_loader: Calibration data loader (for SHAP)
            test_loader: Test data loader (or single DataLoader if 1 model)
            train_loaders: List of train data loaders (one per model fold)
                          Or single DataLoader if models is a single model
            y_true: True labels [N]
            flag: Dataset identifier (for caching)
            layer_name: Layer name for feature extraction
            k: Number of nearest neighbors
            n_shap_features: Number of top SHAP features
            cache_dir: Directory to cache SHAP values
            parallel: Enable parallel processing across folds
            n_jobs: Number of parallel workers
            per_fold_evaluation: If True, compute per-fold metrics. If False, average uncertainties
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N] (averaged)
        """
        timer = Timer("KNN-SHAP computation")
        with timer:
            # Create method with parallelization
            knn_method = uq.KNNLatentSHAPMethod(
                layer_name=layer_name,
                k=k,
                n_shap_features=n_shap_features,
                cache_dir=cache_dir,
                parallel=parallel,
                n_jobs=n_jobs
            )
            
            # Fit and compute uncertainties
            knn_method.fit(self.models, train_loaders, calib_loader, self.device, flag=flag)
            uncertainties = knn_method.compute(self.models, test_loader, self.device,
                                              return_per_fold=per_fold_evaluation)
        
        # Use cached predictions if available, otherwise compute
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots)
            self._uncertainties['KNN_SHAP'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['KNN_SHAP_per_fold'] = uncertainties  # [num_folds, N]
            # Note: KNN_SHAP doesn't compute ensemble separately, so no _ensemble key
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['KNN_SHAP'] = uncertainties
        self._results['KNN_SHAP'] = metrics
        
        return uncertainties, metrics
    
    def _get_ensemble_predictions(self, data_loader: DataLoader) -> np.ndarray:
        """Helper to get ensemble predictions from data loader."""
        all_preds = []
        
        for model in self.models:
            model.eval()
            preds_fold = []
            
            with torch.no_grad():
                for batch in data_loader:
                    if isinstance(batch, dict):
                        images = batch['image']
                    else:
                        images = batch[0]
                    
                    images = images.to(self.device)
                    logits = model(images)
                    
                    # Handle both binary and multiclass
                    if logits.shape[1] == 1:
                        # Binary classification with single output
                        probs = torch.sigmoid(logits)  # [B, 1]
                        # Convert to 2-class format: [B, 2]
                        probs = torch.cat([1 - probs, probs], dim=1)
                    else:
                        # Multi-class
                        probs = torch.softmax(logits, dim=1)  # [B, C]
                    
                    preds_fold.append(probs.cpu().numpy())
            
            all_preds.append(np.concatenate(preds_fold, axis=0))
        
        # Average across models
        ensemble_preds = np.mean(all_preds, axis=0)
        return ensemble_preds
    
    def _compute_metrics(
        self,
        uncertainties: np.ndarray,
        correct_idx: np.ndarray,
        incorrect_idx: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        predictions_per_fold: np.ndarray = None
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        # Check if uncertainties are per-fold [num_folds, N] or averaged [N]
        if uncertainties.ndim == 2:
            # Per-fold uncertainties: compute metrics per fold, then aggregate
            metrics = uq.compute_all_metrics_per_fold(
                uncertainties, predictions, labels, predictions_per_fold=predictions_per_fold
            )
        else:
            # Single uncertainty array: compute standard metrics
            metrics = uq.compute_all_metrics(
                uncertainties, predictions, labels, correct_idx, incorrect_idx
            )
        return metrics
    
    def save_results(
        self, 
        output_dir: str,
        flag: str = None,
        timestamp: str = None,
        model_backbone: str = None,
        setup: str = None
    ) -> Dict[str, str]:
        """
        Save all computed uncertainties, evaluation plots, and results summary.
        
        Args:
            output_dir: Base directory for saving results
            flag: Dataset/experiment identifier (optional, for filenames)
            timestamp: Timestamp string (optional, will generate if not provided)
            model_backbone: Model architecture identifier (e.g., 'resnet18', 'vit_b_16')
            setup: Training setup identifier (e.g., '', 'DA', 'DO', 'DADO')
            
        Returns:
            Dictionary with paths to saved files:
                - 'metrics_file': Path to .npz file with all uncertainties
                - 'figures_dir': Directory containing evaluation plots
                - 'summary_file': Path to JSON results summary
        """
        from datetime import datetime
        import json
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if flag is None:
            flag = 'results'
        
        # Build filename suffix with model configuration
        config_suffix = ''
        if model_backbone:
            config_suffix += f'_{model_backbone}'
        if setup:
            config_suffix += f'_{setup}'
        
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        # Save all metric values (uncertainties) to .npz
        if self._uncertainties:
            metrics_file = os.path.join(output_dir, f'all_metrics_{flag}{config_suffix}_{timestamp}.npz')
            np.savez_compressed(metrics_file, **self._uncertainties)
            saved_paths['metrics_file'] = metrics_file
            print(f"\n💾 All metric values saved to: {metrics_file}")
        
        # Generate and save evaluation plots for each method
        if self._uncertainties and self._test_predictions_cache is not None:
            figures_dir = os.path.join(output_dir, 'figures', flag, timestamp)
            os.makedirs(figures_dir, exist_ok=True)
            
            y_pred = self._test_predictions_cache['y_pred']
            y_true = self._test_predictions_cache['y_true']
            
            print(f"\n📊 Generating evaluation plots...")
            for method_name, metric_values in self._uncertainties.items():
                # Skip per-fold entries (they're paired with main entries)
                if method_name.endswith('_per_fold') or method_name.endswith('_ensemble'):
                    continue
                
                try:
                    # Check if per-fold data exists
                    per_fold_key = f'{method_name}_per_fold'
                    ensemble_key = f'{method_name}_ensemble'
                    
                    uncertainties_per_fold = None
                    ensemble_uncertainties = None
                    predictions_per_fold = None
                    
                    if per_fold_key in self._uncertainties:
                        uncertainties_per_fold = self._uncertainties[per_fold_key]
                    
                    if ensemble_key in self._uncertainties:
                        ensemble_uncertainties = self._uncertainties[ensemble_key]
                    
                    if method_name in self._predictions_per_fold:
                        predictions_per_fold = self._predictions_per_fold[method_name]
                    
                    fig_paths = uq.save_all_evaluation_plots(
                        uncertainties=metric_values,
                        predictions=y_pred,
                        labels=y_true,
                        method_name=method_name,
                        output_dir=figures_dir,
                        uncertainties_per_fold=uncertainties_per_fold,
                        ensemble_uncertainties=ensemble_uncertainties,
                        predictions_per_fold=predictions_per_fold
                    )
                    print(f"  ✓ {method_name}: {len(fig_paths)} plots saved")
                except Exception as e:
                    print(f"  ⚠️  Failed to generate plots for {method_name}: {e}")
            
            saved_paths['figures_dir'] = figures_dir
            print(f"✓ Figures saved to: {figures_dir}")
        
        # Save JSON summary
        if self._results and self._test_predictions_cache is not None:
            y_true = self._test_predictions_cache['y_true']
            correct_idx = self._test_predictions_cache['correct_idx']
            
            summary = {
                'flag': flag,
                'model_backbone': model_backbone,
                'setup': setup,
                'timestamp': timestamp,
                'test_accuracy': float(len(correct_idx) / len(y_true)),
                'test_size': len(y_true),
                'methods': self._results,
                'metrics_file': saved_paths.get('metrics_file'),
                'figures_dir': saved_paths.get('figures_dir')
            }
            
            summary_file = os.path.join(output_dir, f'uq_benchmark_{flag}{config_suffix}_{timestamp}.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            saved_paths['summary_file'] = summary_file
            print(f"\n💾 Results summary saved to: {summary_file}")
        
        return saved_paths
