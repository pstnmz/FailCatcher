"""
SHAP visualization utilities for uncertainty quantification.

SHAP (SHapley Additive exPlanations) shows which input features/pixels 
most influence model predictions. Used for interpretability and debugging.
"""
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list


# ============================================================================
# IMAGE-LEVEL SHAP (Input Attribution)
# ============================================================================

def visualize_input_shap_overlayed_multimodel(
    models, eval_dataloader, device, success_indices, failure_indices, 
    sample_size=5, max_background_samples=1000
):
    """
    Visualize SHAP heatmaps overlayed on original images for multiple models.
    
    Purpose:
        - See which pixels/regions models focus on
        - Compare attention across different models
        - Debug why predictions succeed/fail
    
    Args:
        models: List of trained PyTorch models (e.g., 5-fold ensemble)
        eval_dataloader: DataLoader for test data
        device: torch.device
        success_indices: Indices where all models predicted correctly
        failure_indices: Indices where models failed
        sample_size: How many examples to visualize per category
        max_background_samples: Background data for SHAP baseline
    
    Visualization:
        - Each row = one image
        - Column 1 = Original image
        - Columns 2+ = SHAP overlay for each model
        - Red regions = increase prediction score
        - Blue regions = decrease prediction score
    
    Example:
        >>> # After training 5 models
        >>> visualize_input_shap_overlayed_multimodel(
        ...     models, test_loader, device, 
        ...     correct_idx, incorrect_idx, sample_size=3
        ... )
        >>> # Shows 3 successes + 3 failures with SHAP overlays
    """
    for model in models:
        model.eval()

    # Select random samples from each category
    np.random.seed(433)
    success_sample = np.random.choice(success_indices, min(sample_size, len(success_indices)), replace=False)
    failure_sample = np.random.choice(failure_indices, min(sample_size, len(failure_indices)), replace=False)
    selected_indices = np.concatenate([success_sample, failure_sample])

    print(f"Visualizing SHAP for {len(selected_indices)} samples ({len(success_sample)} success, {len(failure_sample)} failure)")

    # Extract images and labels
    images_to_explain = []
    labels_to_explain = []
    cases = []  # "Success" or "Failure"
    background_images = []
    
    with torch.no_grad():
        sample_count = 0
        for i, batch in enumerate(eval_dataloader):
            # Handle different data formats
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch.get("label", batch.get("shape", torch.zeros(len(images))))
            else:
                images, labels = batch[0].to(device), batch[1]
            
            # Collect background data (for SHAP baseline)
            background_images.append(images)
            
            # Find selected samples in this batch
            batch_start_idx = i * eval_dataloader.batch_size
            for local_idx, (img, lbl) in enumerate(zip(images, labels)):
                global_idx = batch_start_idx + local_idx
                
                if global_idx in selected_indices:
                    images_to_explain.append(img)
                    labels_to_explain.append(lbl.item() if hasattr(lbl, 'item') else lbl)
                    cases.append("Success" if global_idx in success_sample else "Failure")
                    sample_count += 1

            if sample_count >= len(selected_indices):
                break

    if len(images_to_explain) == 0:
        raise ValueError("No images found at selected indices. Check dataloader and indices.")

    images_to_explain = torch.stack(images_to_explain).to(device)
    labels_to_explain = np.array(labels_to_explain)
    background_images = torch.cat(background_images).to(device)

    # Limit background size (SHAP is expensive)
    if len(background_images) > max_background_samples:
        background_images = background_images[:max_background_samples]

    print(f"Computing SHAP values with {len(background_images)} background samples...")

    # Create grid: rows=samples, cols=original + (one per model)
    num_images = len(images_to_explain)
    num_models = len(models)
    fig, axes = plt.subplots(
        num_images, num_models + 1, 
        figsize=(4 * (num_models + 1), 4 * num_images)
    )
    
    # Handle single-row case
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # Column 0: Original image
        original_image = images_to_explain[i].cpu().numpy()
        if original_image.ndim == 3:
            original_image = original_image.transpose(1, 2, 0)  # CHW -> HWC
            if original_image.shape[2] == 1:
                original_image = original_image.squeeze(-1)
        
        axes[i, 0].imshow(original_image, cmap="gray" if original_image.ndim == 2 else None)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"Original\n{cases[i]} (Label={labels_to_explain[i]})", 
            fontsize=10, fontweight='bold'
        )

        # Columns 1+: SHAP overlay for each model
        for j, model in enumerate(models):
            explainer = shap.GradientExplainer(model, background_images)
            
            # Compute SHAP values (may take ~1-2 sec per image)
            shap_values = explainer.shap_values(images_to_explain[i : i + 1])
            
            # Handle binary vs multiclass
            if isinstance(shap_values, list):
                shap_value = shap_values[0].squeeze()  # Binary: take positive class
            else:
                shap_value = shap_values.squeeze()
            
            # Get prediction
            with torch.no_grad():
                logits = model(images_to_explain[i : i + 1])
                if logits.shape[1] == 1:
                    prediction = torch.sigmoid(logits).item()
                else:
                    prediction = torch.softmax(logits, dim=1).max().item()

            # Plot SHAP overlay
            axes[i, j + 1].imshow(original_image, cmap="gray" if original_image.ndim == 2 else None)
            
            # Overlay SHAP heatmap (red=positive, blue=negative)
            if shap_value.ndim == 3:
                shap_value = shap_value.mean(axis=0)  # Average across channels
            
            im = axes[i, j + 1].imshow(
                shap_value, cmap="seismic", alpha=0.5, 
                vmin=-np.abs(shap_value).max(), vmax=np.abs(shap_value).max()
            )
            axes[i, j + 1].axis("off")
            axes[i, j + 1].set_title(
                f"Model {j+1}\nPred={prediction:.3f}", 
                fontsize=10
            )

    plt.suptitle(
        "SHAP Attribution: Red = Increases Prediction, Blue = Decreases Prediction",
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# FEATURE-LEVEL SHAP (Latent Space)
# ============================================================================

def plot_shap_importance(shap_importance, fold, feature_names=None, top_k=50):
    """
    Plot bar chart of SHAP feature importance.
    
    Use case: After extracting latent features (e.g., from model.avgpool),
    this shows which features matter most for predictions.
    
    Args:
        shap_importance: pd.Series of mean |SHAP value| per feature (sorted descending)
        fold: Fold index for title
        feature_names: Optional list to filter specific features
        top_k: Show only top K features (avoid cluttered plot)
    
    Example:
        >>> # After computing mean SHAP (see methods/latent.py)
        >>> shap_importance = display_shap_values(shap_df)
        >>> plot_shap_importance(shap_importance, fold=0, top_k=20)
    """
    if feature_names is not None:
        shap_importance = shap_importance[shap_importance.index.isin(feature_names)]
    
    # Limit to top_k
    shap_importance = shap_importance.head(top_k)

    plt.figure(figsize=(12, max(6, len(shap_importance) * 0.3)))
    shap_importance.plot(kind="barh", color='steelblue')
    plt.title(f"Top {len(shap_importance)} SHAP Feature Importances (Fold {fold})", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_clustered_feature_heatmap(features, fold, feature_names=None):
    """
    Plot hierarchical clustered correlation heatmap of latent features.
    
    Purpose: Identify groups of correlated features (redundancy) for feature selection.
    
    Args:
        features: Feature array (samples x features) or DataFrame
        fold: Fold index for title
        feature_names: Optional list to label features
    
    Example:
        >>> # After extracting latent features
        >>> plot_clustered_feature_heatmap(latent_features, fold=0)
    """
    # Compute correlation matrix
    if isinstance(features, pd.DataFrame):
        correlation_matrix = features.corr().abs()
        labels = features.columns.tolist()
    else:
        correlation_matrix = np.abs(np.corrcoef(features, rowvar=False))
        labels = feature_names if feature_names else [f"F{i}" for i in range(features.shape[1])]

    # Hierarchical clustering (group similar features)
    linkage_matrix = linkage(correlation_matrix.values if isinstance(correlation_matrix, pd.DataFrame) 
                             else correlation_matrix, method='ward')
    clustered_order = leaves_list(linkage_matrix)

    # Reorder matrix
    if isinstance(correlation_matrix, pd.DataFrame):
        clustered_corr = correlation_matrix.iloc[clustered_order, clustered_order]
    else:
        clustered_corr = correlation_matrix[clustered_order][:, clustered_order]
    
    clustered_labels = [labels[i] for i in clustered_order]

    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        clustered_corr,
        xticklabels=clustered_labels,
        yticklabels=clustered_labels,
        cmap="coolwarm",
        vmin=0, vmax=1,
        cbar_kws={'label': '|Correlation|'},
        square=True
    )
    plt.title(f"Clustered Feature Correlation (Fold {fold})", fontsize=16, fontweight='bold')
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


# ============================================================================
# UMAP VISUALIZATION (Latent Space Projection)
# ============================================================================

def visualize_umap_with_labels(umap_train, umap_test, success, labels, fold=0):
    """
    Visualize UMAP-reduced latent space with train/test separation.
    
    Purpose: 
        - Check if test data falls within training distribution (OOD detection)
        - Visualize where predictions fail (red X's) vs succeed (green X's)
    
    Args:
        umap_train: UMAP-reduced train features (N_train, 2)
        umap_test: UMAP-reduced test features (N_test, 2)
        success: Binary flags for test set (1=correct, 0=incorrect)
        labels: Train set labels (for coloring classes)
        fold: Fold index for title
    
    Example:
        >>> from umap import UMAP
        >>> # After extracting latent features (train + test)
        >>> reducer = UMAP(n_components=2)
        >>> umap_train = reducer.fit_transform(train_features)
        >>> umap_test = reducer.transform(test_features)
        >>> visualize_umap_with_labels(umap_train, umap_test, success_flags, train_labels)
    """
    # Separate train data by class
    unique_classes = np.unique(labels)
    
    # Separate test data by success/failure
    success_mask = np.array(success) == 1
    failure_mask = ~success_mask

    plt.figure(figsize=(12, 8))

    # Background density for train data (shows class distribution)
    sns.kdeplot(
        x=umap_train[:, 0], y=umap_train[:, 1],
        cmap="Blues", fill=True, alpha=0.2, levels=10, warn_singular=False
    )

    # Train data scatter (colored by class)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    for class_idx, color in zip(unique_classes, colors):
        mask = np.array(labels) == class_idx
        plt.scatter(
            umap_train[mask, 0], umap_train[mask, 1],
            label=f"Class {class_idx} (Train)", 
            alpha=0.3, color=color, marker="o", s=30
        )

    # Test data: Success (green X)
    plt.scatter(
        umap_test[success_mask, 0], umap_test[success_mask, 1],
        label="Correct Predictions", 
        alpha=0.8, color="green", marker="x", s=100, linewidths=2
    )

    # Test data: Failure (red X)
    plt.scatter(
        umap_test[failure_mask, 0], umap_test[failure_mask, 1],
        label="Incorrect Predictions", 
        alpha=0.8, color="red", marker="x", s=100, linewidths=2
    )

    plt.title(f"UMAP Latent Space Visualization (Fold {fold})", fontsize=16, fontweight="bold")
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)
    plt.legend(fontsize=10, loc="best", frameon=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()