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
    """Convert tensor or numpy array to PIL Image, handling different value ranges."""
    def __call__(self, img):
        # If already PIL, return as-is
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            # Handle different tensor ranges
            img_tensor = img.detach().cpu()
            
            # Check tensor type and range
            if img_tensor.dtype == torch.uint8:
                # Already uint8 [0, 255] - perfect for PIL
                return transforms.ToPILImage()(img_tensor)
            elif img_tensor.dtype == torch.float32 or img_tensor.dtype == torch.float64:
                if img_tensor.max() <= 1.0:
                    # Convert [0, 1] float to [0, 255] uint8 for PIL
                    img_tensor = (img_tensor * 255).to(torch.uint8)
                # else: assume already in [0, 255] range
            
            return transforms.ToPILImage()(img_tensor)
        if isinstance(img, np.ndarray):
            # Handle numpy arrays - ensure uint8 [0, 255]
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
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
    

def evaluate_models_on_loader(models, data_loader, device, numpy_av=True, return_logits=False):
    """
    Evaluate ensemble of models on a DataLoader.
    
    Args:
        models: List of PyTorch models
        data_loader: DataLoader for evaluation
        device: torch.device
        numpy_av: If True, use numpy for final averaging (matches tr.evaluate_model exactly)
        return_logits: If True, also return raw logits for temperature scaling
    
    Returns:
        tuple: (y_true, y_scores, predicted_classes, correct_idx, incorrect_idx, individual_scores)
               If return_logits=True: also includes logits as 7th element
    
    Example:
        >>> y_true, y_scores, digits, correct_idx, incorrect_idx, indiv_scores = \
        ...     evaluate_models_on_loader(models, test_loader, device)
        >>> print(f"Accuracy: {len(correct_idx) / len(y_true):.3f}")
        
        >>> # With logits for temperature scaling
        >>> y_true, y_scores, digits, correct_idx, incorrect_idx, indiv_scores, logits = \
        ...     evaluate_models_on_loader(models, test_loader, device, return_logits=True)
    """
    for model in models:
        model.eval()
    
    all_labels = []
    all_predictions = []  # List of [B, K, C] tensors (probabilities)
    all_logits = [] if return_logits else None  # List of [B, K, C] tensors (raw logits)
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch.get("label", batch.get("shape"))
            else:
                images, labels = batch[0].to(device), batch[1]
            
            # Get predictions from all models for this batch
            batch_preds, batch_logits = get_batch_predictions(models, images, device, return_logits=True)  # [B, K, C]
            all_predictions.append(batch_preds)
            if return_logits:
                all_logits.append(batch_logits)
            all_labels.append(labels)
    
    # Concatenate along batch dimension (dimension 0)
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, K, C]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    if return_logits:
        all_logits = torch.cat(all_logits, dim=0)  # [N, K, C]
    
    if numpy_av:
        # Exact match with tr.evaluate_model: use numpy for averaging
        all_predictions_np = all_predictions.cpu().numpy()  # [N, K, C]
        avg_probs_np = np.mean(all_predictions_np, axis=1)  # [N, C] - average over models
        predicted_classes_np = np.argmax(avg_probs_np, axis=1)  # [N]
        
        y_true = all_labels.cpu().numpy().ravel()
        y_scores = avg_probs_np
        predicted_classes = predicted_classes_np
        individual_scores = all_predictions_np
        
        if return_logits:
            logits_np = all_logits.cpu().numpy()  # [N, K, C]
            avg_logits_np = np.mean(logits_np, axis=1)  # [N, C]
    else:
        # Original torch-based method
        avg_probs = average_predictions(all_predictions)  # [N, C]
        predicted_classes = torch.argmax(avg_probs, dim=1)
        
        y_true = all_labels.cpu().numpy().ravel()
        y_scores = avg_probs.cpu().numpy()
        predicted_classes = predicted_classes.cpu().numpy().ravel()
        individual_scores = all_predictions.cpu().numpy()
        
        if return_logits:
            avg_logits_np = all_logits.mean(dim=1).cpu().numpy()  # [N, C]
    
    correct_idx = np.where(predicted_classes == y_true)[0].tolist()
    incorrect_idx = np.where(predicted_classes != y_true)[0].tolist()
    
    if return_logits:
        return y_true, y_scores, predicted_classes, correct_idx, incorrect_idx, individual_scores, avg_logits_np
    else:
        return y_true, y_scores, predicted_classes, correct_idx, incorrect_idx, individual_scores


def apply_calibration(y_scores, calibration_model, method='platt', logits=None):
    """
    Apply fitted calibration model to new predictions.
    
    Args:
        y_scores: Predicted probabilities [N, C] or [N]
        calibration_model: Fitted calibration model (from posthoc_calibration)
        method: 'platt', 'isotonic', or 'temperature'
        logits: Raw logits [N, C] or [N] (required for temperature scaling)
    
    Returns:
        np.ndarray: Calibrated probabilities [N] or [N, C]
    
    Example:
        >>> # Fit on calibration set
        >>> from UQ_Toolbox.methods.distance import posthoc_calibration
        >>> _, calib_model = posthoc_calibration(y_calib, labels_calib, 'platt')
        
        >>> # Apply to test set
        >>> calibrated_test = apply_calibration(y_test, calib_model, 'platt')
        
        >>> # Temperature scaling (requires logits)
        >>> _, temp_model = posthoc_calibration(logits_calib, labels_calib, 'temperature')
        >>> calibrated_test = apply_calibration(y_test, temp_model, 'temperature', logits=logits_test)
    """
    if method == 'temperature':
        if logits is None:
            raise ValueError("Temperature scaling requires raw logits. Pass logits=... argument.")
        
        # Temperature scaling: apply TemperatureScaler to logits
        logits_tensor = torch.from_numpy(logits).float()
        if logits_tensor.ndim == 1:
            logits_tensor = logits_tensor.unsqueeze(1)
        
        calibrated_logits = calibration_model(logits_tensor).detach()
        
        # Convert to probabilities
        if calibrated_logits.ndim == 1 or calibrated_logits.shape[1] == 1:
            # Binary
            calibrated_scores = torch.sigmoid(calibrated_logits).numpy().squeeze()
        else:
            # Multiclass
            calibrated_scores = torch.softmax(calibrated_logits, dim=1).numpy()
    
    elif method == 'platt':
        # Platt scaling: apply LogisticRegression
        if y_scores.ndim == 1:
            y_scores_input = y_scores.reshape(-1, 1)
        elif y_scores.shape[1] == 2:
            # Binary: use probability of positive class
            y_scores_input = y_scores[:, 1].reshape(-1, 1)
        else:
            # Multiclass: use max probability
            y_scores_input = np.max(y_scores, axis=1).reshape(-1, 1)
        
        calibrated_scores = calibration_model.predict_proba(y_scores_input)[:, 1]
    
    elif method == 'isotonic':
        # Isotonic regression: apply IsotonicRegression
        if y_scores.ndim == 1:
            y_scores_input = y_scores
        elif y_scores.shape[1] == 2:
            # Binary: use probability of positive class
            y_scores_input = y_scores[:, 1]
        else:
            # Multiclass: use max probability
            y_scores_input = np.max(y_scores, axis=1)
        
        calibrated_scores = calibration_model.predict(y_scores_input)
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return calibrated_scores


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

def get_batch_predictions(models, images, device, return_logits=False):
    """
    Get predictions from multiple models for a batch of images.
    
    Args:
        models: List of PyTorch models or single model
        images: Batch of images [B, C, H, W]
        device: torch.device
        return_logits: If True, also return raw logits
    
    Returns:
        torch.Tensor: Predictions of shape [B, K, C] where:
            B = batch size
            K = number of models
            C = number of classes
        If return_logits=True: tuple of (probs, logits)
    
    Example:
        >>> models = [model1, model2, model3]
        >>> images = torch.randn(32, 3, 224, 224)
        >>> preds = get_batch_predictions(models, images, device)
        >>> preds.shape  # torch.Size([32, 3, num_classes])
        
        >>> preds, logits = get_batch_predictions(models, images, device, return_logits=True)
    """
    # Handle single model case
    if not isinstance(models, list):
        models = [models]
    
    batch_predictions = []
    batch_logits = [] if return_logits else None
    
    for model in models:
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            logits = model(images)
            
            if return_logits:
                batch_logits.append(logits)
            
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
    
    if return_logits:
        batch_logits = torch.stack(batch_logits, dim=0)  # [K, B, C]
        batch_logits = batch_logits.permute(1, 0, 2)  # [B, K, C]
        return batch_predictions, batch_logits
    else:
        return batch_predictions

def average_predictions(batch_predictions):
    """
    Average predictions across models.

    Args:
        batch_predictions (torch.Tensor or array-like): shape [B, K, C]
            where B = batch size, K = number of models, C = num classes
            Should already be probabilities (from get_batch_predictions)

    Returns:
        torch.Tensor: Averaged probabilities, shape [B, C]
    """
    if not torch.is_tensor(batch_predictions):
        batch_predictions = torch.as_tensor(batch_predictions)

    if batch_predictions.dim() == 2:
        # [B, C] - single model, return as-is
        return batch_predictions
    elif batch_predictions.dim() == 3:
        # [B, K, C] - average across models (dim=1)
        averaged = batch_predictions.mean(dim=1)  # [B, C]
        return averaged
    else:
        raise ValueError(f"Expected batch_predictions to be 2D or 3D, got {tuple(batch_predictions.shape)}")


def compute_stds(averaged_predictions):
    """
    Compute standard deviations for the predictions.

    Args:
        averaged_predictions (torch.Tensor): Predictions across augmentations.
            Expected shape: [N, K, C] where N=samples, K=augmentations, C=classes

    Returns:
        list: List of standard deviations for each sample (length N).
    """
    # averaged_predictions should be [N, K, C]
    if not isinstance(averaged_predictions, torch.Tensor):
        averaged_predictions = torch.as_tensor(averaged_predictions)
    
    if averaged_predictions.ndim == 2:
        # [N, K] - binary or per-class already averaged
        stds = torch.std(averaged_predictions, dim=1).tolist()
    elif averaged_predictions.ndim == 3:
        # [N, K, C] - compute std across augmentations (dim=1), then average across classes
        stds_per_class = torch.std(averaged_predictions, dim=1)  # [N, C]
        stds = torch.mean(stds_per_class, dim=1).tolist()  # [N]
    else:
        raise ValueError(f"Expected averaged_predictions to have 2 or 3 dimensions, got {averaged_predictions.ndim}")
    
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