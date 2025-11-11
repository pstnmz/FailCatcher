import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from monai.data import CacheDataset
   
class AddBatchDimension:
    def __call__(self, image):
        # Ensure the image is a tensor and add batch dimension
        if isinstance(image, torch.Tensor):
            return image.unsqueeze(0).float()
        raise TypeError("Input should be a torch Tensor")
    
# add helper to ensure PIL input
class EnsurePIL:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return transforms.ToPILImage()(img.detach().cpu())
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

class _CachedRandAugDataset(Dataset):
    """
    Wrap a MONAI CacheDataset and apply a list of torchvision augmentation pipelines
    to each cached image, returning a stacked tensor of shape (K, C, H, W).
    """
    def __init__(self, cache_dataset, augmentations):
        self.cache_dataset = cache_dataset
        # allow either a single augmentation or a list
        self.augmentations = augmentations if isinstance(augmentations, (list, tuple)) else [augmentations]

    def __len__(self):
        return len(self.cache_dataset)

    def __getitem__(self, index):
        sample = self.cache_dataset[index]
        if isinstance(sample, dict):
            img = sample.get("image")
            label = sample.get("label", None)
        else:
            img, label = sample

        # normalize input type -> PIL Image expected by augmentation pipeline
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            img = transforms.ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        outs = []
        for aug in self.augmentations:
            out = aug(img)  # augmentation pipeline should output a Tensor (C,H,W) or PIL
            if not isinstance(out, torch.Tensor):
                out = transforms.PILToTensor()(out)
            out = out.float()
            outs.append(out)

        # stacked: K x C x H x W
        stacked = torch.stack(outs, dim=0)
        return stacked, label

    # New: allow swapping the augmentation list without recreating the DataLoader
    def set_augmentations(self, augmentations):
        self.augmentations = augmentations if isinstance(augmentations, (list, tuple)) else [augmentations]
    

def evaluate_models_on_loader(models, data_loader, device, numpy_av=True):
    """
    Evaluate ensemble of models on a DataLoader.
    
    Args:
        models: List of PyTorch models
        data_loader: DataLoader for evaluation
        device: torch.device
        numpy_av: If True, use numpy for final averaging (matches tr.evaluate_model exactly)
    
    Returns:
        tuple: (y_true, y_scores, predicted_classes, correct_idx, incorrect_idx, individual_scores)
    
    Example:
        >>> y_true, y_scores, digits, correct_idx, incorrect_idx, indiv_scores = \
        ...     evaluate_models_on_loader(models, test_loader, device)
        >>> print(f"Accuracy: {len(correct_idx) / len(y_true):.3f}")
    """
    for model in models:
        model.eval()
    
    all_labels = []
    all_predictions = []  # List of [B, K, C] tensors
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch.get("label", batch.get("shape"))
            else:
                images, labels = batch[0].to(device), batch[1]
            
            # Get predictions from all models for this batch
            batch_preds = get_batch_predictions(models, images, device)  # [B, K, C]
            all_predictions.append(batch_preds)
            all_labels.append(labels)
    
    # Concatenate along batch dimension (dimension 0)
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, K, C]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    
    if numpy_av:
        # Exact match with tr.evaluate_model: use numpy for averaging
        all_predictions_np = all_predictions.cpu().numpy()  # [N, K, C]
        avg_probs_np = np.mean(all_predictions_np, axis=1)  # [N, C] - average over models
        predicted_classes_np = np.argmax(avg_probs_np, axis=1)  # [N]
        
        y_true = all_labels.cpu().numpy().ravel()
        y_scores = avg_probs_np
        predicted_classes = predicted_classes_np
        individual_scores = all_predictions_np
    else:
        # Original torch-based method
        avg_probs = average_predictions(all_predictions)  # [N, C]
        predicted_classes = torch.argmax(avg_probs, dim=1)
        
        y_true = all_labels.cpu().numpy().ravel()
        y_scores = avg_probs.cpu().numpy()
        predicted_classes = predicted_classes.cpu().numpy().ravel()
        individual_scores = all_predictions.cpu().numpy()
    
    correct_idx = np.where(predicted_classes == y_true)[0].tolist()
    incorrect_idx = np.where(predicted_classes != y_true)[0].tolist()
    
    return y_true, y_scores, predicted_classes, correct_idx, incorrect_idx, individual_scores


def build_monai_cache_dataset(dataset, cache_rate=1.0, num_workers=0):
    if CacheDataset is None:
        raise ImportError("MONAI is required to build a cache dataset. Please install monai.")
    data = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, dict):
            img = sample.get("image")
            label = sample.get("label")
        else:
            img, label = sample
        # Convert once to PIL to avoid doing it per-augmentation in workers
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img.detach().cpu())
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        # keep label as small tensor/py type
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu()
        data.append({"image": img, "label": label})
    return CacheDataset(data=data, transform=None, cache_rate=cache_rate, num_workers=num_workers)

def get_prediction(model, image, device, use_amp=True):
    """
    Generates a prediction from a given model and image.
    NOTE: model should already be moved to `device` and set to eval() by the caller.
    """
    # Do NOT call model.to(device) here — caller should handle device placement.
    image = image.to(device, non_blocking=True)
    model.eval()  # ensure eval, cheap
    with torch.no_grad():
        if use_amp and getattr(device, "type", None) == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                prediction = model(image)
        else:
            prediction = model(image)
    # keep predictions on device; caller can .detach().cpu() as needed
    return prediction

def get_batch_predictions(models, images, device):
    """
    Get predictions from multiple models for a batch of images.
    
    Args:
        models: List of PyTorch models or single model
        images: Batch of images [B, C, H, W]
        device: torch.device
    
    Returns:
        torch.Tensor: Predictions of shape [B, K, C] where:
            B = batch size
            K = number of models
            C = number of classes
    
    Example:
        >>> models = [model1, model2, model3]
        >>> images = torch.randn(32, 3, 224, 224)
        >>> preds = get_batch_predictions(models, images, device)
        >>> preds.shape  # torch.Size([32, 3, num_classes])
    """
    # Handle single model case
    if not isinstance(models, list):
        models = [models]
    
    batch_predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            logits = model(images)
            
            # Convert logits to probabilities
            if logits.shape[1] == 1:
                # Binary classification with single output
                probs = torch.sigmoid(logits)  # [B, 1]
                # Convert to 2-class format: [B, 2]
                probs = torch.cat([1 - probs, probs], dim=1)
            else:
                # Multi-class
                probs = torch.softmax(logits, dim=1)  # [B, C]
            
            batch_predictions.append(probs)
    
    # Stack predictions: [K, B, C] -> [B, K, C]
    batch_predictions = torch.stack(batch_predictions, dim=0)  # [K, B, C]
    batch_predictions = batch_predictions.permute(1, 0, 2)  # [B, K, C]
    
    return batch_predictions

def average_predictions(batch_predictions):
    """
    Average predictions across models by first converting per-model logits to
    probabilities (sigmoid for binary, softmax for multiclass) then averaging
    the probabilities. Simpler API: activation is inferred automatically.

    Args:
        batch_predictions (torch.Tensor or array-like): shape [M, N, C] or [M, N]
            where M = number of models, N = number of samples, C = num classes

    Returns:
        torch.Tensor: Averaged probabilities, shape [N, C] (or [N,1] for binary).
    """
    if not torch.is_tensor(batch_predictions):
        batch_predictions = torch.as_tensor(batch_predictions)

    # normalize shape -> [M, N, C]
    if batch_predictions.dim() == 2:
        batch_predictions = batch_predictions.unsqueeze(-1)
    if batch_predictions.dim() != 3:
        raise ValueError(f"Expected batch_predictions to be 3D [M,N,C], got {tuple(batch_predictions.shape)}")

    M, N, C = batch_predictions.shape
    # infer activation: binary -> sigmoid, multiclass -> softmax
    if C == 1:
        probs = torch.sigmoid(batch_predictions)
    else:
        probs = torch.softmax(batch_predictions, dim=-1)
    averaged = probs.mean(dim=0)  # [N, C]
    return averaged


def compute_stds(averaged_predictions):
    """
    Compute standard deviations for the predictions.

    Args:
        averaged_predictions (torch.Tensor): Averaged predictions.

    Returns:
        list: List of standard deviations for each sample.
    """
    if averaged_predictions.ndim == 2 or averaged_predictions.shape[2] == 1:
        stds = torch.std(averaged_predictions, dim=1).squeeze().tolist()  # Binary classification: shape (num_models, num_samples)
    elif averaged_predictions.ndim == 3:
        stds_per_class = torch.std(averaged_predictions, dim=1).squeeze()  # Multiclass classification: shape (num_models, num_samples, num_classes)
        stds = torch.mean(stds_per_class, dim=1).tolist()
    return stds

def _dl_worker_init(_):
    # prevent oversubscription per worker process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    
def to_3_channels(img):
    if img.mode == 'L':  # Grayscale image
        img = img.convert('RGB')  # Convert to 3 channels by duplicating
    return img

def to_1_channel(img):
    img = img.convert('L')  # Convert back to grayscale
    return img