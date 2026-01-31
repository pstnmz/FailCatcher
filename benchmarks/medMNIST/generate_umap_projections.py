import argparse
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA

# UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not installed. Install with: pip install umap-learn")
    sys.exit(1)

# Import medMNIST-specific utilities
from benchmarks.medMNIST.utils import train_models_load_datasets as tr
from benchmarks.medMNIST.utils import dataset_utils


# Dataset configurations
DATASET_CONFIGS = {
    'breastmnist': {
        'test_scenarios': ['id', 'cs'],  # ID and corruption shift only
        'color': False,
        'has_ps': False,
        'has_ncs': False
    },
    'bloodmnist': {
        'test_scenarios': ['id', 'cs'],
        'color': True,
        'has_ps': False,
        'has_ncs': False
    },
    'tissuemnist': {
        'test_scenarios': ['id', 'cs'],
        'color': False,
        'has_ps': False,
        'has_ncs': False
    },
    'octmnist': {
        'test_scenarios': ['id', 'cs'],
        'color': False,
        'has_ps': False,
        'has_ncs': False
    },
    'pneumoniamnist': {
        'test_scenarios': ['id', 'cs'],
        'color': False,
        'has_ps': False,
        'has_ncs': False
    },
    'dermamnist-e': {
        'test_scenarios': ['id', 'cs', 'ps'],  # ID, corruption, and external (PS)
        'color': True,
        'has_ps': True,
        'has_ncs': False,
        'ps_flag': 'dermamnist-e-external'
    },
    'organamnist': {
        'test_scenarios': ['id', 'cs', 'ps', 'ncs'],  # ID, corruption, AMOS (PS), AMOS new class (NCS)
        'color': False,
        'has_ps': True,
        'has_ncs': True,
        'ps_flag': 'amos2022'
    },
    'pathmnist': {
        'test_scenarios': ['ps'],  # Only PS mentioned in requirements
        'color': True,
        'has_ps': True,
        'has_ncs': False,
        'ps_flag': 'pathmnist'  # pathmnist uses external test set
    }
}


def extract_image_features(dataloader, device):
    """
    Extract flattened image features directly from raw images.
    
    Args:
        dataloader: DataLoader for dataset
        device: torch device (not used, kept for compatibility)
    
    Returns:
        tuple: (features, labels) as numpy arrays
    """
    features_list = []
    labels_list = []
    
    for batch in tqdm(dataloader, desc="Extracting image features", leave=False):
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch
        
        # Flatten images: (B, C, H, W) -> (B, C*H*W)
        batch_features = images.view(images.size(0), -1).cpu().numpy()
        
        features_list.append(batch_features)
        labels_list.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Ensure labels are 1D
    if labels.ndim > 1:
        labels = labels.ravel()
    
    return features, labels


def fit_umap_on_train(train_features, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, 
                     use_scaler=False, pca_variance=0.99):
    """
    Fit UMAP on training features with optional PCA preprocessing.
    
    Args:
        train_features: Training features (N_train, D)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_components: Number of UMAP components (2 for visualization)
        random_state: Random seed
        use_scaler: Whether to use StandardScaler (default: False for raw images)
        pca_variance: Fraction of variance to keep in PCA (default: 0.95)
                     Set to None to skip PCA
    
    Returns:
        tuple: (umap_model, scaler, pca_model, train_embedding)
    """
    print(f"  Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, scaling={use_scaler}, PCA={pca_variance})...")
    
    # Optionally standardize features before PCA/UMAP
    if use_scaler:
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
    else:
        scaler = None
        train_features_scaled = train_features
    
    # Apply PCA for dimensionality reduction (important for large feature spaces)
    if pca_variance is not None:
        n_samples, n_features = train_features_scaled.shape
        matrix_size = n_samples * n_features
        
        # Use IncrementalPCA for very large matrices to avoid LAPACK overflow
        # Threshold: 2B elements (int32 limit in LAPACK)
        if matrix_size > 2e9:
            print(f"  Large matrix detected ({matrix_size:.2e} elements), using IncrementalPCA...")
            # For IncrementalPCA, n_components must be fixed (can't use variance fraction)
            # Choose a reasonable fixed number: min of (n_samples-1, 500)
            n_components_fixed = min(n_samples - 1, 500)
            batch_size = max(n_components_fixed + 10, n_samples // 20)  # Ensure batch_size > n_components
            
            pca = IncrementalPCA(n_components=n_components_fixed, batch_size=batch_size)
            train_features_pca = pca.fit_transform(train_features_scaled)
            
            # Check variance captured
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            variance_captured = cumsum_var[-1]
            
            # If we didn't capture enough variance, warn user
            if variance_captured < pca_variance:
                print(f"  ⚠️  PCA: {n_features} → {n_components_fixed} dims "
                      f"({variance_captured:.3f} variance retained, target {pca_variance} not reached)")
                print(f"      Consider using a smaller dataset or accepting lower variance retention.")
            else:
                # Trim to target variance if we exceeded it
                n_components_keep = np.searchsorted(cumsum_var, pca_variance) + 1
                if n_components_keep < n_components_fixed:
                    train_features_pca = train_features_pca[:, :n_components_keep]
                    pca.n_components_keep_ = n_components_keep
                    print(f"  PCA: {n_features} → {n_components_keep} dims "
                          f"({cumsum_var[n_components_keep-1]:.3f} variance retained)")
                else:
                    pca.n_components_keep_ = n_components_fixed
                    print(f"  PCA: {n_features} → {n_components_fixed} dims "
                          f"({variance_captured:.3f} variance retained)")
        else:
            # Standard PCA for smaller matrices
            pca = PCA(n_components=pca_variance, random_state=42)
            train_features_pca = pca.fit_transform(train_features_scaled)
            pca.n_components_keep_ = train_features_pca.shape[1]  # No trimming needed
            print(f"  PCA: {train_features_scaled.shape[1]} → {train_features_pca.shape[1]} dims "
                  f"({pca.explained_variance_ratio_.sum():.3f} variance retained)")
    else:
        pca = None
        train_features_pca = train_features_scaled
    
    # Fit UMAP (no random_state for parallel processing - much faster)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        n_jobs=-1,  # Use all CPUs
        verbose=False
    )
    
    train_embedding = reducer.fit_transform(train_features_pca)
    
    return reducer, scaler, pca, train_embedding


def project_features(umap_model, scaler, pca, features):
    """
    Project features using fitted UMAP (with optional PCA preprocessing).
    
    Args:
        umap_model: Fitted UMAP model
        scaler: Fitted StandardScaler (or None if not using scaling)
        pca: Fitted PCA model (or None if not using PCA)
        features: Features to project (N, D)
    
    Returns:
        np.ndarray: Projected features (N, 2)
    """
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    if pca is not None:
        features_pca = pca.transform(features_scaled)
        # Apply component trimming if IncrementalPCA was used
        if hasattr(pca, 'n_components_keep_'):
            features_pca = features_pca[:, :pca.n_components_keep_]
    else:
        features_pca = features_scaled
    
    embedding = umap_model.transform(features_pca)
    return embedding


def create_umap_plot(embeddings_dict, labels_dict, dataset_name, output_path, class_names=None):
    """
    Create UMAP visualization with all shift types overlaid on a single plot.
    Colors represent shift types (train, ID, CS, PS, NCS), not class labels.
    
    Args:
        embeddings_dict: Dictionary of {scenario_name: embedding}
        labels_dict: Dictionary of {scenario_name: labels} (not used, kept for compatibility)
        dataset_name: Name of dataset
        output_path: Path to save figure
        class_names: Optional list of class names (not used)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors and order for shift types
    shift_colors = {
        'train': '#1f77b4',  # Blue
        'id': '#2ca02c',     # Green
        'cs_sev1': '#ff7f0e', # Orange
        'cs_sev2': '#ff7f0e',
        'cs_sev3': '#ff7f0e',
        'cs_sev4': '#ff7f0e',
        'cs_sev5': '#ff7f0e',
        'ps': '#d62728',     # Red
        'ncs': '#9467bd'     # Purple
    }
    
    shift_labels = {
        'train': 'Train',
        'id': 'ID (In-Distribution)',
        'cs_sev1': 'CS (Corruption Shift)',
        'cs_sev2': 'CS (Corruption Shift)',
        'cs_sev3': 'CS (Corruption Shift)',
        'cs_sev4': 'CS (Corruption Shift)',
        'cs_sev5': 'CS (Corruption Shift)',
        'ps': 'PS (Population Shift)',
        'ncs': 'NCS (New Class Shift)'
    }
    
    # Plot order: train first (background), then others
    plot_order = ['train', 'id', 'ps', 'ncs'] + [k for k in embeddings_dict.keys() if k.startswith('cs_')]
    
    for scenario_name in plot_order:
        if scenario_name not in embeddings_dict:
            continue
        
        embedding = embeddings_dict[scenario_name]
        color = shift_colors.get(scenario_name, '#7f7f7f')
        label = shift_labels.get(scenario_name, scenario_name.upper())
        
        # Adjust alpha and size based on shift type
        if scenario_name == 'train':
            alpha = 0.3
            size = 5
            zorder = 1
        else:
            alpha = 0.6
            size = 15
            zorder = 2
        
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=color,
            label=label,
            alpha=alpha,
            s=size,
            edgecolors='none',
            zorder=zorder
        )
    
    ax.set_title(f"UMAP Projection: {dataset_name}", fontsize=16, fontweight='bold')
    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.legend(loc='best', fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to {output_path}")


def generate_umap_for_dataset(flag, device, output_dir, corruption_severity=3, 
                              batch_size=512, image_size=224):
    """
    Generate UMAP projections for a single dataset using raw images.
    
    Args:
        flag: Dataset name
        device: torch device (not used, kept for compatibility)
        output_dir: Output directory for figures
        corruption_severity: Severity for corruption shift (1-5)
        batch_size: Batch size for feature extraction
        image_size: Image size
    """
    print(f"\n{'='*80}")
    print(f"Processing: {flag}")
    print(f"{'='*80}")
    
    config = DATASET_CONFIGS[flag]
    color = config['color']
    
    # Get transforms
    transform, transform_tta = dataset_utils.get_transforms(color, image_size)
    
    # ========================================================================
    # LOAD TRAIN SET (merge train + calibration for more data)
    # ========================================================================
    print("\n📦 Loading training data...")
    # Use transform_tta (no normalization) to preserve raw pixel values
    [study_dataset, calib_dataset, _], [_, _, _], info = tr.load_datasets(
        flag, color, image_size, transform_tta, batch_size
    )
    
    # Merge train and calibration for UMAP fitting
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([study_dataset, calib_dataset])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    print(f"  Train set: {len(study_dataset)} samples")
    print(f"  Calibration set: {len(calib_dataset)} samples")
    print(f"  Combined (train+calib): {len(train_dataset)} samples")
    print(f"  Classes: {len(info['label'])}")
    
    # ========================================================================
    # EXTRACT TRAIN FEATURES & FIT UMAP
    # ========================================================================
    print("\n🔬 Extracting training image features (raw images)...")
    train_features, train_labels = extract_image_features(train_loader, device)
    
    print(f"  Train features shape: {train_features.shape} (flattened images)")
    print(f"  Train statistics: mean={train_features.mean():.4f}, std={train_features.std():.4f}")
    print(f"  Train labels distribution: {np.bincount(train_labels.astype(int))}")
    
    # Fit UMAP on training features (no scaling to preserve corruption signal, PCA to reduce dims)
    umap_model, scaler, pca, train_embedding = fit_umap_on_train(
        train_features, n_neighbors=15, min_dist=0.1, random_state=42, 
        use_scaler=False, pca_variance=0.99
    )
    
    print(f"  ✓ UMAP fitted on training set")
    
    # ========================================================================
    # COLLECT ALL TEST SCENARIOS
    # ========================================================================
    embeddings_dict = {'train': train_embedding}
    labels_dict = {'train': train_labels}
    
    scenarios = config['test_scenarios']
    
    # ID Test Set
    if 'id' in scenarios:
        print("\n📊 Processing ID test set...")
        [_, _, test_dataset], [_, _, test_loader], _ = tr.load_datasets(
            flag, color, image_size, transform_tta, batch_size
        )
        
        print(f"  ID test set: {len(test_dataset)} samples")
        test_features, test_labels = extract_image_features(test_loader, device)
        
        # Sanity check: verify train/test separation
        print(f"  ID statistics: mean={test_features.mean():.4f}, std={test_features.std():.4f}")
        print(f"  ID labels distribution: {np.bincount(test_labels.astype(int))}")
        
        test_embedding = project_features(umap_model, scaler, pca, test_features)
        
        embeddings_dict['id'] = test_embedding
        labels_dict['id'] = test_labels
        print(f"  ✓ ID test projected")
    
    # Corruption Shift (CS)
    if 'cs' in scenarios:
        print(f"\n📊 Processing Corruption Shift (severity={corruption_severity})...")
        [_, _, test_dataset_cs], [_, _, _], _ = tr.load_datasets(
            flag, color, image_size, transform_tta, batch_size
        )
        
        # Apply corruptions
        corruption_flag = 'dermamnist' if 'dermamnist' in flag else flag
        test_dataset_cs = dataset_utils.apply_random_corruptions(
            test_dataset_cs, corruption_flag, corruption_severity, 
            cache=True, seed=42, return_pil=False
        )
        
        test_loader_cs = DataLoader(
            test_dataset_cs, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        print(f"  CS test set: {len(test_dataset_cs)} samples")
        test_features_cs, test_labels_cs = extract_image_features(test_loader_cs, device)
        
        # Sanity check: verify corruption changed the data
        print(f"  CS statistics: mean={test_features_cs.mean():.4f}, std={test_features_cs.std():.4f}")
        if 'id' in embeddings_dict:
            # Compare to uncorrupted test data (mean absolute difference per pixel)
            diff = np.abs(test_features_cs - test_features).mean()
            max_diff = np.abs(test_features_cs - test_features).max()
            print(f"  Mean |CS - ID| per pixel: {diff:.4f}, Max: {max_diff:.4f} (should be > 0)")
        
        test_embedding_cs = project_features(umap_model, scaler, pca, test_features_cs)
        
        embeddings_dict[f'cs_sev{corruption_severity}'] = test_embedding_cs
        labels_dict[f'cs_sev{corruption_severity}'] = test_labels_cs
        print(f"  ✓ CS test projected")
    
    # Population Shift (PS)
    if 'ps' in scenarios:
        print("\n📊 Processing Population Shift...")
        
        if flag == 'organamnist':
            # Load AMOS as population shift
            ps_dataset, ps_loader, _, _, _ = dataset_utils.load_amos_dataset(
                transform, transform_tta, batch_size, 
                workspace_root=Path(__file__).parent.parent.parent
            )
            print(f"  PS (AMOS) test set: {len(ps_dataset)} samples")
            
        elif flag == 'dermamnist-e':
            # Load external test set as population shift
            [_, _, ps_dataset], [_, _, ps_loader], _ = tr.load_datasets(
                'dermamnist-e', color, image_size, transform_tta, batch_size, 
                test_subset='external'
            )
            print(f"  PS (external) test set: {len(ps_dataset)} samples")
            
        elif flag == 'pathmnist':
            # PathMNIST uses its own external test set
            # For now, use the standard test set (can be updated if separate external set exists)
            [_, _, ps_dataset], [_, _, ps_loader], _ = tr.load_datasets(
                flag, color, image_size, transform_tta, batch_size
            )
            print(f"  PS test set: {len(ps_dataset)} samples")
        
        ps_features, ps_labels = extract_image_features(ps_loader, device)
        ps_embedding = project_features(umap_model, scaler, pca, ps_features)
        
        embeddings_dict['ps'] = ps_embedding
        labels_dict['ps'] = ps_labels
        print(f"  ✓ PS test projected")
    
    # New Class Shift (NCS) - only for organamnist
    if 'ncs' in scenarios and flag == 'organamnist':
        print("\n📊 Processing New Class Shift (AMOS unmapped organs)...")
        
        # For raw image projection, we don't need model-based filtering
        # Load standard AMOS and separate by mapped vs unmapped organs
        print("  Loading full AMOS dataset (mapped + unmapped organs)...")
        workspace_root = Path(__file__).parent.parent.parent
        amos_path = workspace_root / 'benchmarks' / 'medMNIST' / 'Data' / 'AMOS_2022' / 'amos_external_test_224.npz'
        
        if amos_path.stat().st_size < 1000:
            raise FileNotFoundError("AMOS dataset appears to be Git LFS pointer. Run: git lfs pull")
        
        amos_data = np.load(str(amos_path), allow_pickle=True)
        amos_images = amos_data['test_images']
        amos_labels = amos_data['test_labels']
        
        # OrganaMNIST to AMOS mapping
        amos_to_organamnist = {
            0: 10,  # spleen
            1: 5,   # right kidney
            2: 4,   # left kidney
            5: 6,   # liver
            9: 9,   # pancreas
            13: 0,  # bladder
        }
        
        # Separate mapped vs unmapped organs
        ncs_images_list = []
        ncs_binary_labels = []  # 0=known (mapped), 1=new (unmapped)
        
        for idx in range(len(amos_labels)):
            amos_organ_id = np.argmax(amos_labels[idx])
            if amos_organ_id in amos_to_organamnist:
                ncs_images_list.append(amos_images[idx])
                ncs_binary_labels.append(0)  # Known class
            else:
                ncs_images_list.append(amos_images[idx])
                ncs_binary_labels.append(1)  # New class
        
        ncs_images_array = np.stack(ncs_images_list, axis=0)
        ncs_binary_labels = np.array(ncs_binary_labels)
        
        # Create dataset (use transform_tta for no normalization)
        ncs_dataset = dataset_utils.AMOSDataset(
            ncs_images_array, ncs_binary_labels, transform=transform_tta
        )
        ncs_loader = DataLoader(
            ncs_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        print(f"  NCS test set: {len(ncs_dataset)} samples")
        print(f"    Known classes: {np.sum(ncs_binary_labels == 0)}, New classes: {np.sum(ncs_binary_labels == 1)}")
        ncs_features, _ = extract_image_features(ncs_loader, device)
        
        # Use binary labels for coloring: 0=known class, 1=new class
        ncs_labels = ncs_binary_labels
        
        ncs_embedding = project_features(umap_model, scaler, pca, ncs_features)
        
        embeddings_dict['ncs'] = ncs_embedding
        labels_dict['ncs'] = ncs_labels
        print(f"  ✓ NCS test projected")
    
    # ========================================================================
    # CREATE VISUALIZATION
    # ========================================================================
    print("\n📈 Creating UMAP visualization...")
    
    output_path = os.path.join(output_dir, f'umap_{flag}.png')
    
    # Get class names from info
    class_names = info['label'] if 'label' in info else None
    
    create_umap_plot(
        embeddings_dict, labels_dict, flag, output_path, class_names
    )
    
    # ========================================================================
    # SAVE EMBEDDINGS FOR FURTHER ANALYSIS
    # ========================================================================
    print("\n💾 Saving embeddings...")
    embeddings_path = os.path.join(output_dir, f'umap_{flag}_embeddings.npz')
    
    np.savez_compressed(
        embeddings_path,
        **{f'{k}_embedding': v for k, v in embeddings_dict.items()},
        **{f'{k}_labels': v for k, v in labels_dict.items()},
        class_names=class_names,
        train_features=train_features
    )
    
    print(f"  ✓ Embeddings saved to {embeddings_path}")
    print(f"\n{'='*80}")
    print(f"✓ Completed: {flag}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate UMAP projections for medMNIST datasets'
    )
    
    parser.add_argument(
        '--datasets', nargs='+', 
        choices=list(DATASET_CONFIGS.keys()),
        help='Specific datasets to process'
    )
    parser.add_argument(
        '--all-datasets', action='store_true',
        help='Process all 8 datasets'
    )
    parser.add_argument(
        '--corruption-severity', type=int, default=3, choices=[1, 2, 3, 4, 5],
        help='Corruption severity for CS test sets (default: 3)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./umap_projections',
        help='Output directory for UMAP visualizations'
    )
    parser.add_argument(
        '--batch-size', type=int, default=512,
        help='Batch size for feature extraction (default: 512)'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID (default: 0) - not used for raw images but kept for compatibility'
    )
    
    args = parser.parse_args()
    
    # Check UMAP availability
    if not UMAP_AVAILABLE:
        print("❌ UMAP not available. Install with: pip install umap-learn")
        sys.exit(1)
    
    # Determine datasets to process
    if args.all_datasets:
        datasets_to_process = list(DATASET_CONFIGS.keys())
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        print("❌ Please specify --datasets or --all-datasets")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"UMAP Projection Generation (Raw Images)")
    print(f"{'='*80}")
    print(f"Datasets: {', '.join(datasets_to_process)}")
    print(f"Corruption severity: {args.corruption_severity}")
    print(f"Using: Flattened raw images (no model features)")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    for flag in datasets_to_process:
        try:
            # Generate UMAP projections (no models needed - using raw images)
            generate_umap_for_dataset(
                flag, device, args.output_dir,
                corruption_severity=args.corruption_severity,
                batch_size=args.batch_size,
                image_size=224
            )
            
        except Exception as e:
            print(f"\n❌ Error processing {flag}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✓ All datasets processed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
