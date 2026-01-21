"""
Create 3x3 montages for all benchmark datasets and distribution shifts.

Usage:
    python create_dataset_montages.py
    
Or import and use the function:
    from create_dataset_montages import display_montage
    fig = display_montage('organamnist', 'id')
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


def display_montage(dataset_name, shift='id', save_path=None):
    """
    Display a 3x1 montage of 224x224 images.
    
    Parameters:
    -----------
    dataset_name : str
        One of: 'organamnist', 'amos2022', 'dermamnist-e-id', 'dermamnist-e-ext', 
                'pathmnist', 'tissuemnist', 'breastmnist', 'pneumoniamnist', 
                'octmnist', 'bloodmnist'
    shift : str
        One of: 'id' (in-distribution), 'cs' (corruption shift), 
                'ps' (population shift), 'ncs' (new class shift)
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Map dataset names
    dataset_map = {
        'dermamnist-e-id': 'dermamnist',
        'dermamnist-e-ext': 'dermamnist',
    }
    
    # Get actual medmnist flag
    actual_flag = dataset_map.get(dataset_name, dataset_name)
    
    # Handle special cases
    if dataset_name == 'amos2022':
        # Load AMOS dataset - try multiple possible paths
        possible_paths = [
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/amos_external_test_224.npz'),
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data/AMOS_2022/amos_external_test_224.npz'),
        ]
        
        amos_path = None
        for path in possible_paths:
            if path.exists() and path.stat().st_size > 1000:  # Real file should be ~133MB, not a pointer
                amos_path = path
                break
        
        if amos_path is None:
            raise FileNotFoundError(
                "❌ AMOS dataset not found or is a Git LFS pointer.\n"
                "   Please run: git lfs pull\n"
                f"   Checked paths: {[str(p) for p in possible_paths]}"
            )
        
        print(f"  Loading AMOS from: {amos_path}")
        data = np.load(str(amos_path), allow_pickle=True)
        images = data['test_images']
        labels = data['test_labels']  # (N, 15) one-hot encoded
        amos_organ_ids = np.argmax(labels, axis=1)  # Get organ ID for each sample
        
        if shift == 'ps':
            # Filter for mapped classes ONLY (6 organs that match OrganAMNIST)
            # AMOS classes IN OrganAMNIST:
            # 0=spleen, 1=right kidney, 2=left kidney, 5=liver, 9=pancreas, 13=bladder
            mapped_classes = [0, 1, 2, 5, 9, 13]
            mask = np.isin(amos_organ_ids, mapped_classes)
            
            if mask.sum() > 0:
                images = images[mask]
                print(f"    PS: Filtered to {mask.sum()} mapped class samples (6 organs in OrganAMNIST)")
            else:
                print(f"    ⚠️  No mapped class samples found!")
        
        elif shift == 'ncs':
            # Filter for new/unseen classes (unmapped AMOS organs)
            # AMOS classes NOT in OrganAMNIST (9 novel organs):
            # 3=gall bladder, 4=esophagus, 6=stomach, 7=aorta, 8=postcava,
            # 10=right adrenal gland, 11=left adrenal gland, 12=duodenum, 14=prostate/uterus
            unmapped_classes = [3, 4, 6, 7, 8, 10, 11, 12, 14]
            mask = np.isin(amos_organ_ids, unmapped_classes)
            
            if mask.sum() > 0:
                images = images[mask]
                print(f"    NCS: Filtered to {mask.sum()} new/unseen class samples (9 novel organs)")
            else:
                print(f"    ⚠️  No new class samples found!")
        
        # Select 3 random images
        indices = np.random.choice(len(images), min(3, len(images)), replace=False)
        selected_images = images[indices]
        
    elif dataset_name == 'dermamnist-e-ext':
        # Load external split
        info = INFO['dermamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=transforms.ToTensor(), download=True, size=224)
        
        indices = np.random.choice(len(dataset), 3, replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            selected_images.append(img.numpy())
        selected_images = np.array(selected_images)
        
    else:
        # Regular medMNIST dataset
        info = INFO[actual_flag]
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=transforms.ToTensor(), download=True, size=224)
        
        # Apply medMNIST-C corruption if needed
        if shift == 'cs':
            print(f"  Applying medMNIST-C corruptions to {actual_flag}...")
            # Use severity=4 (max supported by medmnistc) and remove seed for varied corruptions each run
            dataset = apply_random_corruptions(dataset, actual_flag, severity=4, cache=False, seed=None, return_pil=False)
        
        indices = np.random.choice(len(dataset), 3, replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            # Ensure consistent tensor format
            if not isinstance(img, np.ndarray):
                img = img.numpy() if hasattr(img, 'numpy') else np.array(img)
            # Ensure consistent shape [C, H, W]
            if img.ndim == 2:  # [H, W]
                img = img[np.newaxis, :, :]  # [1, H, W]
            elif img.ndim == 3:
                # Ensure CHW format
                if img.shape[0] not in [1, 3]:  # Not in CHW format
                    img = np.transpose(img, (2, 0, 1))
            
            # Normalize channel count: convert all to grayscale if original is grayscale
            if img.shape[0] == 3 and info['n_channels'] == 1:
                # Convert RGB to grayscale using standard weights
                img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                img = img[np.newaxis, :, :]  # Add channel dimension back
            elif img.shape[0] == 1 and info['n_channels'] == 3:
                # Convert grayscale to RGB by repeating
                img = np.repeat(img, 3, axis=0)
            
            selected_images.append(img)
        
        # Stack with explicit shape checking
        selected_images = np.stack(selected_images, axis=0)
    
    # Create 3x1 montage - exact size for 224x224*3 = 224x672 pixels at 100 DPI
    fig, axes = plt.subplots(3, 1, figsize=(2.24, 6.72), dpi=100)
    
    for idx in range(3):
        # 3 rows, 1 column
        ax = axes[idx]
        
        img = selected_images[idx]
        
        # Handle different image formats
        if img.shape[0] in [1, 3]:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        
        if img.shape[-1] == 1:  # Grayscale
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img)
        
        ax.axis('off')
        # Remove all padding and margins
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    
    # Remove all spacing and padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    if save_path:
        # Save with exact dimensions - 224x672 pixels (224*1 x 224*3)
        fig.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
        print(f"Saved to {save_path}")
    
    return fig


def generate_all_montages(output_dir='uq_benchmark_results/figures/dataset_montages'):
    """
    Generate all montages for the benchmark.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save all generated montages
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define all datasets and their applicable shifts
    datasets_shifts = {
        # ID datasets
        'dermamnist-e-id': ['id', 'cs'],
        'organamnist': ['id', 'cs'],
        'tissuemnist': ['id', 'cs'],
        'octmnist': ['id', 'cs'],
        'pneumoniamnist': ['id', 'cs'],
        'breastmnist': ['id', 'cs'],
        'bloodmnist': ['id', 'cs'],
        
        # Population shifts
        'dermamnist-e-ext': ['ps'],
        'pathmnist': ['ps'],
        'amos2022': ['ps', 'ncs'],
    }
    
    print("Generating all montages...")
    for dataset_name, shifts in datasets_shifts.items():
        for shift in shifts:
            print(f"  {dataset_name} - {shift}")
            fig = display_montage(dataset_name, shift)
            
            # Save the figure - 224x672 pixels (224*1 x 224*3)
            filename = f"{dataset_name}_{shift}.png"
            fig.savefig(output_path / filename, dpi=100, bbox_inches=None, pad_inches=0)
            plt.close(fig)
    
    print(f"\nAll montages saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) == 3:
        # Command line usage: python create_dataset_montages.py organamnist id
        dataset = sys.argv[1]
        shift = sys.argv[2]
        
        print(f"\nGenerating montage for {dataset} - {shift}...")
        fig = display_montage(dataset, shift)
        
        # Save the figure
        output_dir = Path('uq_benchmark_results/figures/dataset_montages')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{dataset}_{shift}_montage.png'
        
        fig.savefig(output_file, dpi=100, bbox_inches=None, pad_inches=0)
        print(f"\n✓ Saved montage to: {output_file} (224x672 pixels)")
        
        # Also try to display (may not work in all environments)
        try:
            plt.show()
        except:
            pass
            
    elif len(sys.argv) == 1:
        # No arguments: generate all montages
        generate_all_montages()
    else:
        print("Usage:")
        print("  python create_dataset_montages.py                    # Generate all")
        print("  python create_dataset_montages.py <dataset> <shift>  # Generate one")
        print("\nDatasets: organamnist, amos2022, dermamnist-e-id, dermamnist-e-ext,")
        print("          pathmnist, tissuemnist, breastmnist, pneumoniamnist, octmnist, bloodmnist")
        print("Shifts: id, cs, ps, ncs")
