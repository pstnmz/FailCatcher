"""
Latent space-based uncertainty quantification methods.
Includes SHAP importance, KNN distances, feature engineering, and hyperplane distance analysis.
"""
import numpy as np
import pandas as pd
import torch
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
# CLASS-BASED API (new, recommended)
# ============================================================================

class SHAPLatentMethod(UQMethod):
    """
    Uncertainty quantification using SHAP importance on latent features.
    """
    def __init__(self, layer_to_hook, classifierhead_wrapper, max_background_samples=1000):
        """
        Args:
            layer_to_hook: Layer to extract features from (e.g., model.avgpool)
            classifierhead_wrapper: Wrapped classifier head for SHAP computation
            max_background_samples: Max samples for SHAP background
        """
        super().__init__("SHAP-Latent")
        self.layer = layer_to_hook
        self.classifierhead = classifierhead_wrapper
        self.max_background = max_background_samples
    
    def compute(self, model, data_loader, device):
        """
        Compute SHAP values for latent features.
        
        Returns:
            tuple: (shap_values, features, labels, success_flags)
        """
        return extract_latent_space_and_compute_shap_importance(
            model, data_loader, device, self.layer,
            importance=True,
            classifierheadwrapper=self.classifierhead,
            max_background_samples=self.max_background
        )


class KNNLatentMethod(UQMethod):
    """
    Uncertainty quantification via KNN distance to training data in latent space.
    """
    def __init__(self, layer_to_hook, k=5):
        """
        Args:
            layer_to_hook: Layer to extract features from
            k: Number of nearest neighbors
        """
        super().__init__("KNN-Latent")
        self.layer = layer_to_hook
        self.k = k
        self.train_features = None
        self.knn = None
        self.scaler = None
        self.pca = None
    
    def fit(self, model, train_loader, device, pca_variance=0.9):
        """
        Fit KNN on training data latent space.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            device: torch.device
            pca_variance: Variance to retain in PCA (0-1)
        """
        features, labels, _, _ = extract_latent_space_and_compute_shap_importance(
            model, train_loader, device, self.layer, importance=False
        )
        
        # Standardize and apply PCA
        self.scaler = StandardScaler()
        features_std = self.scaler.fit_transform(features.numpy())
        
        self.pca = PCA(n_components=pca_variance)
        features_pca = self.pca.fit_transform(features_std)
        
        # Fit KNN
        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(features_pca)
        
        return self
    
    def compute(self, model, data_loader, device):
        """
        Compute KNN distances for test data.
        
        Returns:
            np.ndarray: Mean distance to k nearest neighbors (N,)
        """
        if self.knn is None:
            raise RuntimeError("Call fit() before compute()")
        
        features, labels, success, _ = extract_latent_space_and_compute_shap_importance(
            model, data_loader, device, self.layer, importance=False
        )
        
        # Transform test features
        features_std = self.scaler.transform(features.numpy())
        features_pca = self.pca.transform(features_std)
        
        # Compute distances
        distances, _ = self.knn.kneighbors(features_pca)
        avg_distances = distances.mean(axis=1)
        
        return avg_distances


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
    Feature selection pipeline based on SHAP and correlation.

    Args:
        mean_shap_df: Mean absolute SHAP values (Series)
        latent_space: DataFrame of latent features (samples x features)
        shap_threshold: Min SHAP value to retain feature
        corr_threshold: Max correlation allowed between retained features

    Returns:
        tuple: (retained_latent_space_df, final_feature_list)
    
    Pipeline:
        1. Filter by SHAP threshold
        2. Cluster correlated features
        3. Keep most important feature per cluster
        4. Iteratively remove correlated pairs
    """
    # Step 1: SHAP filtering
    retained_features = mean_shap_df[mean_shap_df > shap_threshold].index
    retained_features = retained_features.intersection(latent_space.columns)
    print(f"Retained {len(retained_features)} features after SHAP filtering (threshold={shap_threshold})")

    retained_latent_space = latent_space[retained_features]
    correlation_matrix = retained_latent_space.corr()
    abs_correlation_matrix = np.abs(correlation_matrix)

    # Visualize with dendrogram
    linkage_matrix = linkage(squareform(1 - abs_correlation_matrix), method="ward")
    sns.clustermap(
        abs_correlation_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="coolwarm",
        vmin=0, vmax=1,
        figsize=(12, 12),
        annot=True
    )
    plt.title("Clustered Correlation Heatmap After SHAP Filtering")
    plt.show()

    # Step 2: Identify clusters
    clusters = fcluster(linkage_matrix, t=1 - corr_threshold, criterion="distance")
    cluster_groups = {cluster: [] for cluster in np.unique(clusters)}
    
    for feature, cluster in zip(abs_correlation_matrix.columns, clusters):
        cluster_groups[cluster].append(feature)

    print(f"Identified {len(cluster_groups)} clusters")

    # Step 3: Keep most important from each cluster
    final_features = []
    for cluster, features in cluster_groups.items():
        if len(features) > 1:
            most_important = max(features, key=lambda f: mean_shap_df[f])
            final_features.append(most_important)
        else:
            final_features.extend(features)

    # Step 4: Resolve remaining high correlations
    retained_latent_space = latent_space[final_features]
    correlation_matrix = retained_latent_space.corr()
    abs_correlation_matrix = np.abs(correlation_matrix)

    while True:
        correlated_pairs = [
            (i, j)
            for i in abs_correlation_matrix.columns
            for j in abs_correlation_matrix.columns
            if i != j and abs_correlation_matrix.loc[i, j] > corr_threshold
        ]
        if not correlated_pairs:
            break

        features_to_remove = set()
        for i, j in correlated_pairs:
            less_important = i if mean_shap_df[i] < mean_shap_df[j] else j
            features_to_remove.add(less_important)

        final_features = [f for f in final_features if f not in features_to_remove]
        retained_latent_space = latent_space[final_features]
        correlation_matrix = retained_latent_space.corr()
        abs_correlation_matrix = np.abs(correlation_matrix)

    print(f"Retained {len(final_features)} features after correlation filtering (threshold={corr_threshold})")

    # Final heatmap
    final_corr_matrix = abs_correlation_matrix.loc[final_features, final_features]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        final_corr_matrix,
        xticklabels=final_features,
        yticklabels=final_features,
        cmap="coolwarm",
        vmin=0, vmax=1,
        annot=True,
        cbar=True
    )
    plt.title("Final Retained Features Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

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