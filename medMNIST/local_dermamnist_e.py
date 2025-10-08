import medmnist
from medmnist.dataset import MedMNIST
from pathlib import Path
import numpy as np
import torch
from PIL import Image

DERMAMNIST_E_INFO = {
    'python_class': 'DermaMNIST_E',
    'description': (
        'The DermaMNIST-E dataset is an enhanced and corrected version of DermaMNIST, '
        'proposed by Abhishek et al. (2025) to address data leakage and duplication issues in the '
        'original MedMNIST DermaMNIST benchmark. Like its predecessor, it is based on the '
        'HAM10000 dataset of 10,015 dermatoscopic images from two clinical sites in Austria and '
        'Australia, representing 7 diagnostic categories of common pigmented skin lesions. '
        'DermaMNIST-E ensures strict partitioning by lesion identity (no overlap of lesion IDs '
        'across train, validation, and test sets) and higher-quality image resizing for improved '
        'visual fidelity. Images are resized to MNIST-like dimensions (3×28×28) for lightweight '
        'benchmarking, with corrected splits and updated metadata. It is part of the DermaMNIST-C/E '
        'revisions introduced to improve data integrity and benchmark reliability.'
    ),
    'url_224': 'https://zenodo.org/records/12739457/files/dermamnist_e_224.npz?download=1',
    'MD5_224': '2b778311816a55b580227fa7ea8c9ce9',
    'task': 'multi-class',
    'label': {
        '0': 'actinic keratoses and intraepithelial carcinoma',
        '1': 'basal cell carcinoma',
        '2': 'benign keratosis-like lesions',
        '3': 'dermatofibroma',
        '4': 'melanoma',
        '5': 'melanocytic nevi',
        '6': 'vascular lesions'
    },
    'n_channels': 3,
    'n_samples': {
        'train': 10015,
        'val': 193,
        'test': 1511
    },
    'license': 'CC BY-NC 4.0'
    }

class DermaMNIST_E(MedMNIST):
    INFO = DERMAMNIST_E_INFO
    flag = 'dermamnist-e'
    available_sizes = (224,)  # only 224 supported
    # If your cached file is not ~/.medmnist/dermamnist_e.npz, set:
    # filename = 'dermamnist_extended_224_wsitesources.npz'
    def __init__(self, split='train', transform=None, target_transform=None,
                 download=False, as_rgb=True, size=224, **kwargs):
        super().__init__(split=split, transform=transform, target_transform=target_transform,
                         download=download, as_rgb=as_rgb, size=size, **kwargs)

        # after base init, read any extra arrays you embedded in the NPZ
        try:
            npz_path = Path.home() / ".medmnist" / f"{self.flag}_{self.size}.npz"
            with np.load(npz_path, allow_pickle=True) as z:
                # Only attach for test split (or attach all and index per split if you prefer)
                if self.split == 'test' and 'test_centers' in z.files:
                    # store as a numpy array attribute
                    self.test_centers = z['test_centers']
                else:
                    self.test_centers = None
        except Exception:
            # keep training robust even if the metadata is missing
            self.test_centers = None
    def __len__(self):
        # Prefer explicit length attributes if present
        if hasattr(self, "data"):
            return len(self.data)
        for name in ("imgs", "images", "x", "X"):
            if hasattr(self, name):
                return len(getattr(self, name))
        raise AttributeError("No image array attribute found (expected data/imgs/images/x).")

    def _get_arrays(self):
        # Figure out image array
        if hasattr(self, "data"):
            imgs = self.data
        elif hasattr(self, "imgs"):
            imgs = self.imgs
        elif hasattr(self, "images"):
            imgs = self.images
        elif hasattr(self, "x"):
            imgs = self.x
        else:
            raise AttributeError("Could not find images (data/imgs/images/x).")

        # Figure out label array
        if hasattr(self, "targets"):
            labels = self.targets
        elif hasattr(self, "labels"):
            labels = self.labels
        elif hasattr(self, "y"):
            labels = self.y
        else:
            raise AttributeError("Could not find labels (targets/labels/y).")
        return imgs, labels

    def __getitem__(self, index):
        imgs, labels = self._get_arrays()
        img = imgs[index]
        label = labels[index]

        # Convert label to plain int
        if isinstance(label, (np.ndarray, np.generic)):
            label = int(label)
        elif torch.is_tensor(label):
            label = int(label.item())

        # Ensure numpy array then to PIL if needed for transforms
        if torch.is_tensor(img):
            img = img.cpu().numpy()

        if isinstance(img, np.ndarray):
            # If shape (H,W) make it 3-channel
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            # If shape (C,H,W) transpose to (H,W,C)
            if img.ndim == 3 and img.shape[0] in (1,3) and img.shape[-1] != 3:
                # assume CHW
                img = np.transpose(img, (1,2,0))
            img = Image.fromarray(img.astype(np.uint8))

        if hasattr(self, "transform") and self.transform:
            img = self.transform(img)
        if hasattr(self, "target_transform") and self.target_transform:
            label = self.target_transform(label)

        return img, label

# Register into medmnist module
setattr(medmnist, 'DermaMNIST_E', DermaMNIST_E)
try:
    medmnist.INFO['dermamnist-e'] = DERMAMNIST_E_INFO
except Exception:
    from medmnist.info import INFO as _INFO
    _INFO['dermamnist-e'] = DERMAMNIST_E_INFO
