"""
Latent space-based uncertainty quantification methods.
Includes SHAP importance, KNN distances, feature engineering, and hyperplane distance analysis.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from ..core.base import UQMethod


# ============================================================================
# HELPER FUNCTION FOR LAYER EXTRACTION
# ============================================================================

def get_layer_from_model(model, layer_name='avgpool'):
    """
    Extract a layer from a model by name.
    
    Args:
        model: PyTorch model
        layer_name: Name or pattern of layer to extract
    
    Returns:
        torch.nn.Module: The layer module
    
    Supported patterns:
        - 'avgpool': Global average pooling
        - 'layer4': Last conv layer (ResNet)
        - 'fc': Fully connected layer
        - 'head': Classifier head (ViT)
        - Custom: Direct attribute access (e.g., 'features.denseblock4')
    
    Example:
        >>> layer = get_layer_from_model(model, 'avgpool')
        >>> layer = get_layer_from_model(model, 'layer4')
    """
    # Try direct attribute access first
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    
    # Pattern-based search
    if layer_name == 'avgpool':
        # Try common pooling layer names
        for attr in ['avgpool', 'global_pool', 'avg_pool']:
            if hasattr(model, attr):
                return getattr(model, attr)
        # Fallback: search for adaptive pooling in modules
        for name, module in model.named_modules():
            if 'pool' in name.lower() and 'adaptive' in str(type(module)).lower():
                return module
    
    elif layer_name == 'layer4':
        if hasattr(model, 'layer4'):
            return model.layer4
        # For DenseNet: last dense block
        if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
            return model.features.denseblock4
    
    elif layer_name == 'fc':
        for attr in ['fc', 'classifier', 'head']:
            if hasattr(model, attr):
                return getattr(model, attr)
    
    # Nested attribute access (e.g., 'features.denseblock4')
    if '.' in layer_name:
        obj = model
        for attr in layer_name.split('.'):
            obj = getattr(obj, attr)
        return obj
    
    raise ValueError(
        f"Could not find layer '{layer_name}' in model. "
        f"Available top-level attributes: {[name for name, _ in model.named_children()]}"
    )
# ============================================================================
# CLASS-BASED API (new, recommended)
# ============================================================================

class ClassifierHeadWrapper(nn.Module):
    """
    Generic wrapper for classifier head extraction.
    Works with ResNets, ViTs, EfficientNets, etc.
    """
    def __init__(self, model, layer_name='avgpool'):
        """
        Args:
            model: Full PyTorch model
            layer_name: Name of layer to hook (e.g., 'avgpool', 'layer4')
        """
        super().__init__()
        self.model = model
        self.layer = get_layer_from_model(model, layer_name)
        
    def forward(self, x):
        """
        Forward pass from latent features to predictions.
        
        Args:
            x: Latent features (B, feature_dim) - already flattened
        
        Returns:
            Predictions (logits or probabilities)
        """
        # For ResNets: avgpool -> flatten -> fc
        if hasattr(self.model, 'fc'):
            return self.model.fc(x)
        
        # For ViTs: typically has 'head' or 'heads'
        elif hasattr(self.model, 'head'):
            return self.model.head(x)
        elif hasattr(self.model, 'heads'):
            return self.model.heads(x)
        
        # For EfficientNet/MobileNet: usually 'classifier'
        elif hasattr(self.model, 'classifier'):
            return self.model.classifier(x)
        
        # Fallback: try to find any Linear layer at the end
        else:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Linear):
                    return module(x)
            
            raise RuntimeError(
                "Could not identify classifier head. "
                "Please manually specify the classifier in ClassifierHeadWrapper."
            )


class KNNLatentMethod(UQMethod):
    """
    Uncertainty quantification via KNN distance in RAW latent space.
    Supports CV ensembles with per-fold training data.
    """
    def __init__(self, layer_name='avgpool', k=5, pca_variance=0.9):
        super().__init__("KNN-Latent-Raw")
        self.layer_name = layer_name
        self.k = k
        self.pca_variance = pca_variance
        self.fitted_models = []
    
    def fit(self, models, train_loaders, device):
        """
        Fit KNN on training data latent space for each model.
        
        Args:
            models: List of models (one per fold)
            train_loaders: List of train DataLoaders (one per fold) OR single loader
            device: torch.device
        """
        if not isinstance(models, list):
            models = [models]
        
        # Handle both single loader and list of loaders
        if not isinstance(train_loaders, list):
            train_loaders = [train_loaders] * len(models)
        
        if len(models) != len(train_loaders):
            raise ValueError(
                f"Number of models ({len(models)}) must match "
                f"number of train loaders ({len(train_loaders)})"
            )
        
        self.fitted_models = []
        
        for idx, (model, train_loader) in enumerate(zip(models, train_loaders)):
            print(f"  Fold {idx}: Fitting KNN on its training set...")
            
            # Extract layer from this model
            layer = get_layer_from_model(model, self.layer_name)
            
            # Extract features from THIS fold's training data
            features, labels, _, _ = extract_latent_space_and_compute_shap_importance(
                model, train_loader, device, layer, importance=False
            )
            
            # Standardize and apply PCA
            scaler = StandardScaler()
            features_std = scaler.fit_transform(features.numpy())
            
            pca = PCA(n_components=self.pca_variance)
            features_pca = pca.fit_transform(features_std)
            
            # Fit KNN
            knn = NearestNeighbors(n_neighbors=self.k)
            knn.fit(features_pca)
            
            self.fitted_models.append({
                'knn': knn,
                'scaler': scaler,
                'pca': pca,
                'layer': layer
            })
            
            print(f"    {len(features)} train samples, {features.shape[1]} → {features_pca.shape[1]} PCA dims")
        
        print(f"  ✓ Fitted {len(models)} fold(s)")
        return self
    
    def compute(self, models, data_loader, device):
        """Compute KNN distances, averaged across folds."""
        if not self.fitted_models:
            raise RuntimeError("Call fit() before compute()")
        
        if not isinstance(models, list):
            models = [models]
        
        if len(models) != len(self.fitted_models):
            raise ValueError(f"Expected {len(self.fitted_models)} models, got {len(models)}")
        
        all_distances = []
        
        for idx, (model, fitted) in enumerate(zip(models, self.fitted_models)):
            # Extract test features using the saved layer
            features, _, _, _ = extract_latent_space_and_compute_shap_importance(
                model, data_loader, device, fitted['layer'], importance=False
            )
            
            # Transform test features
            features_std = fitted['scaler'].transform(features.numpy())
            features_pca = fitted['pca'].transform(features_std)
            
            # Compute distances
            distances, _ = fitted['knn'].kneighbors(features_pca)
            avg_distances = distances.mean(axis=1)
            all_distances.append(avg_distances)
        
        # Average across folds
        final_distances = np.mean(all_distances, axis=0)
        
        print(f"  ✓ Computed KNN distances (averaged over {len(models)} folds)")
        return final_distances


class KNNLatentSHAPMethod(UQMethod):
    """
    Uncertainty quantification via KNN distance in SHAP-SELECTED latent space.
    
    **Critical Process:**
    For EACH model with its CV training split:
      1. Compute SHAP on calibration → select top-k features per class
      2. For each test sample:
         - Get predicted class
         - Extract latent features, keep only top-k for that class
         - Compute KNN distance to training samples (true label = predicted class)
      3. Average distances across all models
    """
    def __init__(self, layer_name='avgpool', k=5, n_shap_features=50, max_background_samples=1000):
        super().__init__("KNN-Latent-SHAP")
        self.layer_name = layer_name
        self.k = k
        self.n_shap_features = n_shap_features
        self.max_background = max_background_samples
        
        self.fitted_models = []
        self.num_classes = None
    
    def fit(self, models, train_loaders, calib_loader, device):
        """
        Fit KNN PER MODEL using:
        - SHAP computed on calibration (per model)
        - KNN fitted on training (per model's CV split)
        """
        if not isinstance(models, list):
            models = [models]
        
        if not isinstance(train_loaders, list):
            train_loaders = [train_loaders] * len(models)
        
        if len(models) != len(train_loaders):
            raise ValueError(f"Models ({len(models)}) != train_loaders ({len(train_loaders)})")
        
        self.fitted_models = []
        
        # =======================================================================
        # FOR EACH MODEL: Compute SHAP + Fit KNN
        # =======================================================================
        for fold_idx, (model, train_loader) in enumerate(zip(models, train_loaders)):
            print(f"\n  Model {fold_idx+1}/{len(models)}: Computing SHAP + fitting KNN...")
            
            # Extract layer for this model
            layer = get_layer_from_model(model, self.layer_name)
            classifierhead = ClassifierHeadWrapper(model, self.layer_name)
            
            # STEP 1: Compute SHAP on CALIBRATION for THIS model
            print(f"    Step 1: Computing SHAP on calibration...")
            shap_values, features_calib, labels_calib, _ = \
                extract_latent_space_and_compute_shap_importance(
                    model, calib_loader, device, layer,
                    importance=True,
                    classifierheadwrapper=classifierhead,
                    max_background_samples=self.max_background
                )
            
            # STEP 2: Get mean SHAP importances per class
            print(f"    Step 2: Computing mean SHAP per class...")
            mean_shap_results = compute_mean_shap_values(
                shap_values, fold=fold_idx, true_labels=labels_calib, 
                nb_features=self.n_shap_features
            )
            
            if self.num_classes is None:
                self.num_classes = len(mean_shap_results)
                print(f"      Detected {self.num_classes} classes")
            
            # STEP 3: Extract top-k SHAP features per class
            selected_features_per_class = []
            for class_idx in range(self.num_classes):
                mean_shap_series = mean_shap_results[class_idx][2]
                top_features = mean_shap_series.nlargest(self.n_shap_features).index.tolist()
                selected_features_per_class.append(top_features)
            
            # STEP 4: Extract TRAINING features for THIS model
            print(f"    Step 3: Extracting training features...")
            features_train, labels_train, _, _ = extract_latent_space_and_compute_shap_importance(
                model, train_loader, device, layer, importance=False
            )
            
            train_df = pd.DataFrame(
                features_train.numpy(),
                columns=[f"Feature_{i}" for i in range(features_train.shape[1])]
            )
            
            # STEP 5: Fit KNN per class using THIS MODEL's SHAP features
            print(f"    Step 4: Fitting KNN per class...")
            model_knns = []
            
            for class_idx in range(self.num_classes):
                # Filter by TRUE class label
                class_mask = (labels_train == class_idx)
                train_class_df = train_df[class_mask]
                
                if len(train_class_df) == 0:
                    print(f"      Class {class_idx}: No training samples")
                    model_knns.append(None)
                    continue
                
                # Use THIS MODEL's SHAP features
                selected_features = selected_features_per_class[class_idx]
                train_selected = train_class_df[selected_features].values
                
                # Fit scaler + PCA on TRAINING data
                scaler = StandardScaler()
                train_std = scaler.fit_transform(train_selected)
                
                pca = PCA(n_components=0.9)
                train_pca = pca.fit_transform(train_std)
                
                # Fit KNN on TRAINING data
                knn = NearestNeighbors(n_neighbors=min(self.k, len(train_pca)))
                knn.fit(train_pca)
                
                model_knns.append({
                    'knn': knn,
                    'scaler': scaler,
                    'pca': pca,
                    'selected_features': selected_features,
                    'layer': layer,
                    'n_samples': len(train_pca)
                })
                
                print(f"      Class {class_idx}: {len(train_pca)} train samples, "
                      f"{len(selected_features)} SHAP → {train_pca.shape[1]} PCA")
            
            self.fitted_models.append(model_knns)
        
        print(f"\n  ✓ Fitted {len(models)} models (each with its own SHAP + KNN)")
        return self
    
    def compute(self, models, data_loader, device):
        """
        Compute KNN distances for test samples.
        
        For each test sample:
        - Get predicted class from model
        - Use that model's SHAP-selected features for that class
        - Compute distance to training samples (true label = predicted class)
        """
        if not self.fitted_models:
            raise RuntimeError("Call fit() before compute()")
        
        if not isinstance(models, list):
            models = [models]
        
        if len(models) != len(self.fitted_models):
            raise ValueError(f"Expected {len(self.fitted_models)} models, got {len(models)}")
        
        all_distances = []
        
        for model_idx, (model, model_knns) in enumerate(zip(models, self.fitted_models)):
            model.eval()
            
            # Get layer from first non-None class
            layer = None
            for knn_data in model_knns:
                if knn_data is not None:
                    layer = knn_data['layer']
                    break
            
            if layer is None:
                raise RuntimeError("No valid class KNN data found")
            
            # =======================================================================
            # EXTRACT TEST FEATURES + PREDICTIONS
            # =======================================================================
            
            all_labels = []
            features_test, labels_test, _, predicted_classes = extract_latent_space_and_compute_shap_importance(
                model, data_loader, device, layer, importance=False
            )
            
            # Concatenate features
            labels_test = np.array(labels_test, dtype=int)  # ← Ensure int type
            predicted_classes = np.array(predicted_classes, dtype=int)  # ← Ensure int type
            
            print(f"\n  Model {model_idx+1}: Processing {len(predicted_classes)} test samples")
            print(f"    Predicted: {np.bincount(labels_test)} | True: {np.bincount(labels_test)}")
            
            test_df = pd.DataFrame(
                features_test.numpy(),
                columns=[f"Feature_{i}" for i in range(features_test.shape[1])]
            )
            
            distances_per_sample = np.zeros(len(test_df))
            
            # =======================================================================
            # COMPUTE DISTANCES PER PREDICTED CLASS
            # =======================================================================
            for class_idx in range(self.num_classes):
                if model_knns[class_idx] is None:
                    continue
                
                fitted = model_knns[class_idx]
                
                # Test samples PREDICTED as this class
                class_mask = (labels_test == class_idx)
                n_samples_class = class_mask.sum()
                
                if n_samples_class == 0:
                    continue
                
                test_class_df = test_df[class_mask]
                
                # Use THIS MODEL's SHAP features for this class
                test_selected = test_class_df[fitted['selected_features']].values
                
                # Transform using THIS MODEL's fitted scaler + PCA
                test_std = fitted['scaler'].transform(test_selected)
                test_pca = fitted['pca'].transform(test_std)
                
                # Compute KNN distances to TRAINING data
                distances, _ = fitted['knn'].kneighbors(test_pca)
                avg_distances = distances.mean(axis=1)
                
                # Debug
                class_labels = labels_test[class_mask]
                n_correct = (class_labels == class_idx).sum()
                n_incorrect = (class_labels != class_idx).sum()
                
                print(f"    Class {class_idx}: {n_samples_class} pred ({n_correct} ✓, {n_incorrect} ✗)")
                if n_correct > 0:
                    correct_dists = avg_distances[class_labels == class_idx]
                    print(f"      Correct: {correct_dists.mean():.3f}±{correct_dists.std():.3f}")
                if n_incorrect > 0:
                    incorrect_dists = avg_distances[class_labels != class_idx]
                    print(f"      Incorrect: {incorrect_dists.mean():.3f}±{incorrect_dists.std():.3f}")
                
                # Store distances
                indices = np.where(class_mask)[0]
                distances_per_sample[indices] = avg_distances
            
            all_distances.append(distances_per_sample)
        
        # Step 3: Average distances across all models
        final_distances = np.mean(all_distances, axis=0)
        
        print(f"\n  ✓ Final averaged distances: {final_distances.mean():.3f}±{final_distances.std():.3f}")
        
        return final_distances

class HyperplaneDistanceMethod(UQMethod):
    """
    Uncertainty quantification via distance to SVM hyperplane in latent space.
    """
    def __init__(self, layer_to_hook):
        super().__init__("Hyperplane-Distance")
        self.layer = layer_to_hook
        self.svm = None
        self.scaler = None
    
    def fit(self, model, train_loader, device):
        """
        Train SVM on training latent space.
        """
        features, labels, _, _ = extract_latent_space_and_compute_shap_importance(
            model, train_loader, device, self.layer, importance=False
        )
        
        self.scaler = StandardScaler()
        features_std = self.scaler.fit_transform(features.numpy())
        
        self.svm = SVC(kernel="linear")
        self.svm.fit(features_std, labels)
        
        return self
    
    def compute(self, model, data_loader, device):
        """
        Compute signed distances to hyperplane.
        
        Returns:
            np.ndarray: Distances (N,) - absolute value = uncertainty
        """
        if self.svm is None:
            raise RuntimeError("Call fit() before compute()")
        
        features, labels, success, _ = extract_latent_space_and_compute_shap_importance(
            model, data_loader, device, self.layer, importance=False
        )
        
        features_std = self.scaler.transform(features.numpy())
        distances = self.svm.decision_function(features_std)
        
        # Absolute distance = uncertainty
        return np.abs(distances)


# ============================================================================
# FUNCTIONAL API (existing, for backward compatibility)
# ============================================================================

def extract_latent_space_and_compute_shap_importance(
    model, data_loader, device, layer_to_be_hooked,
    importance=True, classifierheadwrapper=None, max_background_samples=1000
):
    """
    Extract latent features and optionally compute SHAP values.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: torch.device
        layer_to_be_hooked: Layer to hook (e.g., model.avgpool)
        importance: Whether to compute SHAP values
        classifierheadwrapper: Wrapped classifier head for SHAP
        max_background_samples: Max samples for SHAP background

    Returns:
        If importance=True: (shap_values, features, labels, success_flags)
        If importance=False: (features, labels, success_flags, predictions)
    
    Example:
        >>> # Extract features only
        >>> features, labels, success, preds = extract_latent_space_and_compute_shap_importance(
        ...     model, test_loader, device, model.avgpool, importance=False
        ... )
        
        >>> # Compute SHAP values
        >>> shap_vals, features, labels, success = extract_latent_space_and_compute_shap_importance(
        ...     model, test_loader, device, model.avgpool,
        ...     importance=True, classifierheadwrapper=classifier_head
        ... )
    """
    model.eval()

    penultimate_features = []
    all_labels = []
    success_flags = []
    predictions = []
    
    def hook(module, input, output):
        # Flatten (B, C, 1, 1) → (B, C)
        penultimate_features.append(output.detach().flatten(1))

    hook_handle = layer_to_be_hooked.register_forward_hook(hook)

    with torch.no_grad():
        is_binary = None
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = (batch['image'], batch['label'])

            images = batch[0].to(device, non_blocking=True)
            labels_t = batch[1].to(device, non_blocking=True)
            
            labels_flat = labels_t.view(-1).long()
            all_labels.extend(labels_flat.cpu().numpy().tolist())

            logits = model(images)
            
            if is_binary is None:
                is_binary = (logits.shape[1] == 1)

            if is_binary:
                probs = torch.sigmoid(logits).squeeze(1)
                preds_cls = (probs > 0.5).long()
                success_flags.extend((preds_cls == labels_flat).cpu().numpy().astype(int).tolist())
                predictions.extend(probs.cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=1)
                preds_cls = probs.argmax(dim=1)
                success_flags.extend((preds_cls == labels_flat).cpu().numpy().astype(int).tolist())
                predictions.extend(probs.cpu().numpy())

    hook_handle.remove()

    features = torch.cat(penultimate_features).cpu()
    labels = np.array(all_labels)
    success_flags = np.array(success_flags)
    background_features = features.to(device)

    if len(background_features) > max_background_samples:
        background_features = background_features[:max_background_samples]

    if importance:
        if classifierheadwrapper is None:
            raise ValueError("classifierheadwrapper required when importance=True")
        
        explainer = shap.DeepExplainer(classifierheadwrapper, background_features.clone())
        shap_values = explainer.shap_values(features.clone().detach())
        return shap_values, features, labels, success_flags
    else:
        return features, labels, success_flags, predictions


def compute_mean_shap_values(shap_values, fold, true_labels=None, nb_features=50):
    """
    Compute mean absolute SHAP values per class.

    Args:
        shap_values: SHAP values array (2D or 3D)
        fold: Fold index for labeling
        true_labels: True labels for binary classification class filtering
        nb_features: Number of top features to keep

    Returns:
        list: [(fold, class_idx, shap_importance_series), ...]
    
    Example:
        >>> mean_shap = compute_mean_shap_values(shap_vals, fold=0, true_labels=labels)
        >>> # [(0, 0, Series([0.5, 0.3, ...])), (0, 1, Series([0.4, 0.2, ...]))]
    """
    mean_shap_fold = []
    print(f"SHAP Feature Importances Computation (Fold {fold})")

    if shap_values.ndim == 3 and true_labels is not None and len(np.unique(true_labels)) == 2:
        shap_values = shap_values.squeeze(-1)

    if shap_values.ndim == 3:
        num_samples, num_features, num_classes = shap_values.shape
    elif shap_values.ndim == 2:
        num_samples, num_features = shap_values.shape
        num_classes = 2
    else:
        raise ValueError("Expected 2D or 3D SHAP values array")

    for class_idx in range(num_classes):
        print(f"  Class {class_idx}: Computing SHAP importances")

        if shap_values.ndim == 3:
            class_shap_values = shap_values[:, :, class_idx]
        else:
            class_shap_values = shap_values[true_labels == class_idx, :]

        shap_df = pd.DataFrame(
            class_shap_values,
            columns=[f"Feature_{i}" for i in range(num_features)]
        )

        mean_abs_shap = shap_df.abs().mean(axis=0)
        top_n_features = mean_abs_shap.nlargest(nb_features).index
        shap_df_top_n = shap_df[top_n_features]

        shap_importance = display_shap_values(shap_df_top_n)
        mean_shap_fold.append((fold, class_idx, shap_importance))

    return mean_shap_fold


def display_shap_values(shap_df):
    """
    Compute mean absolute SHAP values for display.

    Args:
        shap_df: DataFrame of SHAP values (samples x features)

    Returns:
        pd.Series: Mean absolute SHAP values per feature (sorted descending)
    """
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)
    return shap_importance

def feature_engineering_pipeline(mean_shap_df, latent_space, shap_threshold=0.05, corr_threshold=0.8):
    """
    DEPRECATED: Use n_shap_features parameter in KNNLatentSHAPMethod instead.
    
    Simplified version: just select top features by SHAP importance.
    PCA will handle redundancy automatically.
    """
    import warnings
    warnings.warn(
        "feature_engineering_pipeline is deprecated. "
        "Use KNNLatentSHAPMethod(n_shap_features=N) instead.",
        DeprecationWarning
    )
    
    # Just select features above threshold
    retained_features = mean_shap_df[mean_shap_df > shap_threshold].index
    retained_features = retained_features.intersection(latent_space.columns)
    
    retained_latent_space = latent_space[retained_features]
    final_features = retained_features.tolist()
    
    print(f"Selected {len(final_features)} features above SHAP threshold {shap_threshold}")
    
    return retained_latent_space, final_features


def analyze_hyperplane_distance(train_latent, train_labels, eval_latent, eval_success, display_distrib=False):
    """
    Train SVM and compute distances to hyperplane.

    Args:
        train_latent: Training features (samples x features)
        train_labels: Training labels
        eval_latent: Evaluation features
        eval_success: Success flags for evaluation set
        display_distrib: Whether to plot distributions

    Returns:
        np.ndarray: Signed distances for evaluation set
    """
    svm = SVC(kernel="linear")
    svm.fit(train_latent, train_labels)

    eval_distances = svm.decision_function(eval_latent)

    success_distances = eval_distances[eval_success == 1]
    failure_distances = eval_distances[eval_success == 0]
    
    scaler = StandardScaler()
    normalized_distances = scaler.fit_transform(eval_distances.reshape(-1, 1)).flatten()
    
    success_distances_norm = normalized_distances[eval_success == 1]
    failure_distances_norm = normalized_distances[eval_success == 0]
    
    if display_distrib:
        plt.figure(figsize=(8, 6))
        sns.histplot(success_distances_norm, color='green', label='Success', kde=True, stat="count")
        sns.histplot(failure_distances_norm, color='red', label='Failure', kde=True, stat="count")
        plt.axvline(0, color='black', linestyle='dashed', label='Decision Boundary')
        plt.xlabel("Normalized Distance to Hyperplane")
        plt.ylabel("Count")
        plt.title("Distance to Hyperplane (Success vs Failure)")
        plt.legend()
        plt.show()
    
    return eval_distances


def compute_knn_distances_to_train_data(
    model, train_loader, test_loader, layer, device,
    latent_spaces, mean_shap_importances, num_classes
):
    """
    Compute KNN distances to training data per class in latent space.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        layer: Layer to hook for features
        device: torch.device
        latent_spaces: DataFrame of latent features
        mean_shap_importances: List of (fold, class, importance_series)
        num_classes: Number of classes

    Returns:
        tuple: (knn_distances, success_flags) both as np.ndarray (N,)
    
    Example:
        >>> distances, success = compute_knn_distances_to_train_data(
        ...     model, train_loader, test_loader, model.avgpool, device,
        ...     latent_df, mean_shap, num_classes=2
        ... )
    """
    # Extract train features
    latent_space_training, labels_training, _, _ = extract_latent_space_and_compute_shap_importance(
        model, train_loader, device, layer, importance=False
    )
    
    # Extract test features
    latent_space_test, labels_test, success_test, _ = extract_latent_space_and_compute_shap_importance(
        model, test_loader, device, layer, importance=False
    )
    
    train_latent_space = pd.DataFrame(latent_space_training.numpy(), columns=latent_spaces.columns)
    test_latent_space = pd.DataFrame(latent_space_test.numpy(), columns=latent_spaces.columns)
    
    knn_distances_all = np.zeros(len(test_latent_space))
    successes_all = np.zeros(len(test_latent_space))
    
    for i in range(num_classes):
        print(f'Processing class {i}')
        
        # Get SHAP-selected features for this class
        important_features = mean_shap_importances[i][2].keys()
        train_latent_class = train_latent_space[important_features]
        
        # Filter by class
        mask_train = labels_training == i
        train_filtered = train_latent_class[mask_train]
        print(f'  Train samples: {len(train_filtered)}')
        
        test_latent_class = test_latent_space[important_features]
        mask_test = labels_test == i
        test_filtered = test_latent_class[mask_test]
        print(f'  Test samples: {len(test_filtered)}')
        
        success_filtered = success_test[mask_test.flatten()]
        indices_filtered = np.where(mask_test.flatten())[0]
        
        # Standardize and PCA
        scaler = StandardScaler()
        train_std = scaler.fit_transform(train_filtered)
        
        pca = PCA(n_components=0.9)
        train_pca = pca.fit_transform(train_std)
        
        test_std = scaler.transform(test_filtered)
        test_pca = pca.transform(test_std)
        
        # KNN
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(train_pca)
        distances, _ = knn.kneighbors(test_pca)
        avg_distances = distances.mean(axis=1)
        
        knn_distances_all[indices_filtered] = avg_distances
        successes_all[indices_filtered] = success_filtered

    return knn_distances_all, successes_all