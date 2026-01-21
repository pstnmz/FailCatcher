"""
Create large 8x8 montages combining multiple datasets for benchmark visualization.

Usage:
    python create_large_montages.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_utils import apply_random_corruptions


def collect_images_from_dataset(dataset_name, shift='id', num_images=8):
    """
    Collect random images from a single dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Dataset name
    shift : str
        Shift type: 'id', 'cs', 'ps', 'ncs'
    num_images : int
        Number of images to collect
        
    Returns:
    --------
    images : np.ndarray
        Array of images with shape (num_images, C, H, W)
    """
    # Map dataset names
    dataset_map = {
        'dermamnist-e-id': 'dermamnist',
        'dermamnist-e-ext': 'dermamnist',
    }
    
    actual_flag = dataset_map.get(dataset_name, dataset_name)
    
    # Handle special cases
    if dataset_name == 'amos2022':
        # Load AMOS dataset
        possible_paths = [
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/amos_external_test_224.npz'),
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data/AMOS_2022/amos_external_test_224.npz'),
        ]
        
        amos_path = None
        for path in possible_paths:
            if path.exists() and path.stat().st_size > 1000:
                amos_path = path
                break
        
        if amos_path is None:
            raise FileNotFoundError("AMOS dataset not found")
        
        data = np.load(str(amos_path), allow_pickle=True)
        images = data['test_images']
        labels = data['test_labels']
        amos_organ_ids = np.argmax(labels, axis=1)
        
        if shift == 'ps':
            # Mapped classes (6 organs in OrganAMNIST)
            mapped_classes = [0, 1, 2, 5, 9, 13]
            mask = np.isin(amos_organ_ids, mapped_classes)
            images = images[mask]
        elif shift == 'ncs':
            # New/unseen classes (9 novel organs)
            unmapped_classes = [3, 4, 6, 7, 8, 10, 11, 12, 14]
            mask = np.isin(amos_organ_ids, unmapped_classes)
            images = images[mask]
        
        indices = np.random.choice(len(images), min(num_images, len(images)), replace=False)
        selected = images[indices]
        
        # Convert to CHW format if needed
        if selected.ndim == 4 and selected.shape[-1] in [1, 3]:
            selected = np.transpose(selected, (0, 3, 1, 2))
        
        # Normalize to [0, 1] range for display (AMOS images are in [0, 255])
        if selected.dtype == np.uint8 or selected.max() > 1.0:
            selected = selected.astype(np.float32) / 255.0
        
        return selected
        
    elif dataset_name == 'dermamnist-e-ext':
        # External split
        info = INFO['dermamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=transforms.ToTensor(), download=True, size=224)
        
        indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            selected_images.append(img.numpy())
        
        return np.array(selected_images)
        
    else:
        # Regular medMNIST dataset
        info = INFO[actual_flag]
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=transforms.ToTensor(), download=True, size=224)
        
        # Apply corruption if needed
        if shift == 'cs':
            dataset = apply_random_corruptions(dataset, actual_flag, severity=4, cache=False, seed=None, return_pil=False)
        
        indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
        selected_images = []
        
        for idx in indices:
            img, _ = dataset[idx]
            
            if not isinstance(img, np.ndarray):
                img = img.numpy() if hasattr(img, 'numpy') else np.array(img)
            
            # Ensure CHW format
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            elif img.ndim == 3 and img.shape[0] not in [1, 3]:
                img = np.transpose(img, (2, 0, 1))
            
            # Normalize channel count
            if img.shape[0] == 3 and info['n_channels'] == 1:
                img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                img = img[np.newaxis, :, :]
            elif img.shape[0] == 1 and info['n_channels'] == 3:
                img = np.repeat(img, 3, axis=0)
            
            selected_images.append(img)
        
        return np.stack(selected_images, axis=0)


def create_4x4_montage(datasets_config, output_path, title=None):
    """
    Create a 4x4 montage from multiple datasets.
    
    Parameters:
    -----------
    datasets_config : list of tuples
        List of (dataset_name, shift, num_images) tuples
        num_images should sum to 16 for 4x4 grid
    output_path : str or Path
        Where to save the montage
    title : str, optional
        Title for the montage
    """
    all_images = []
    
    print(f"Collecting images for {output_path.name}...")
    for dataset_name, shift, num_images in datasets_config:
        print(f"  - {dataset_name} ({shift}): {num_images} images")
        images = collect_images_from_dataset(dataset_name, shift, num_images)
        
        # Normalize to RGB (3 channels) for consistency
        if images.shape[1] == 1:  # Grayscale
            images = np.repeat(images, 3, axis=1)
        
        all_images.append(images)
    
    # Concatenate all images
    all_images = np.concatenate(all_images, axis=0)
    
    # Should have exactly 16 images
    if len(all_images) != 16:
        print(f"  ⚠️  Warning: Expected 16 images, got {len(all_images)}")
        # Pad or truncate
        if len(all_images) < 16:
            # Repeat some images to fill
            needed = 16 - len(all_images)
            indices = np.random.choice(len(all_images), needed, replace=True)
            all_images = np.concatenate([all_images, all_images[indices]], axis=0)
        else:
            all_images = all_images[:16]
    
    # Shuffle to mix datasets
    np.random.shuffle(all_images)
    
    # Create 4x4 montage - 224*4 = 896x896 pixels at 100 DPI
    fig, axes = plt.subplots(4, 4, figsize=(8.96, 8.96), dpi=100)
    
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        ax = axes[i, j]
        
        img = all_images[idx]
        
        # Convert CHW to HWC for display
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img)
        
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    
    # Remove all spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Save with exact dimensions
    fig.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0)
    print(f"  ✓ Saved: {output_path} (896x896 pixels)")
    plt.close(fig)


def generate_all_large_montages(output_dir='uq_benchmark_results/figures/dataset_montages'):
    """
    Generate all 4x4 montages.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ID datasets (7 datasets × 2 images each + 2 extra = 16)
    id_datasets = [
        'dermamnist-e-id', 'organamnist', 'breastmnist', 
        'pneumoniamnist', 'octmnist', 'bloodmnist', 'tissuemnist'
    ]
    
    print("\n=== Creating ID montage ===")
    id_config = [(ds, 'id', 2) for ds in id_datasets]
    id_config[-1] = (id_config[-1][0], id_config[-1][1], 4)  # Last dataset gets 4 to reach 16
    create_4x4_montage(id_config, output_path / 'id_4x4.png', 'In-Distribution')
    
    print("\n=== Creating CS (Corruption Shift) montage ===")
    cs_config = [(ds, 'cs', 2) for ds in id_datasets]
    cs_config[-1] = (cs_config[-1][0], cs_config[-1][1], 4)
    create_4x4_montage(cs_config, output_path / 'cs_4x4.png', 'Corruption Shift')
    
    print("\n=== Creating PS+NCS (Population + New Class Shift) montage ===")
    # pathmnist (ps): 4 images
    # dermamnist-e-ext (ps): 4 images
    # amos2022 ps: 4 images
    # amos2022 ncs: 4 images
    # Total: 16 images
    ps_ncs_config = [
        ('pathmnist', 'ps', 4),
        ('dermamnist-e-ext', 'ps', 4),
        ('amos2022', 'ps', 4),
        ('amos2022', 'ncs', 4),
    ]
    create_4x4_montage(ps_ncs_config, output_path / 'ps_ncs_4x4.png', 'Population + New Class Shift')
    
    print(f"\n✓ All 4x4 montages saved to: {output_path}")


if __name__ == "__main__":
    generate_all_large_montages()
