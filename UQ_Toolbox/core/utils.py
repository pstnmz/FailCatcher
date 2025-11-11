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
    

def evaluate_models_on_loader(models, data_loader, device, use_amp=True):
    """
    Evaluate a single model or an ensemble on a DataLoader and collect:
      - y_true: ground-truth labels (N,)
      - y_scores: probabilities (N,) for binary or (N, C) for multiclass
      - digits: raw logits (N,) for binary or (N, C) for multiclass
      - correct_idx: indices of correct predictions (relative to concatenated order)
      - incorrect_idx: indices of incorrect predictions
    If `models` is a list/tuple, also returns:
      - indiv_scores: list of length num_models with per-model probabilities
                      (each np.ndarray of shape (N,) for binary or (N, C) for multiclass)

    Args:
        models: torch.nn.Module or list/tuple of modules
        data_loader: DataLoader yielding (images, labels) or dicts with keys 'image' and 'label'
        device: torch.device
        use_amp: bool, enable CUDA AMP

    Returns:
        If single model:
            y_true, y_scores, digits, correct_idx, incorrect_idx
        If ensemble:
            y_true, y_scores, digits, correct_idx, incorrect_idx, indiv_scores
    """
    is_ensemble = isinstance(models, (list, tuple))
    model_list = list(models) if is_ensemble else [models]

    # Eval mode
    for m in model_list:
        m.eval()

    y_true_list = []
    avg_prob_batches = []
    avg_logit_batches = []
    indiv_scores = [[] for _ in range(len(model_list))] if is_ensemble else None

    def _to_probs(logits_t):
        # logits_t: [B, C] with C==1 for binary or >1 for multiclass
        if logits_t.shape[1] == 1:
            return torch.sigmoid(logits_t)  # [B,1]
        return torch.softmax(logits_t, dim=1)  # [B,C]

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                images = batch.get('image')
                labels = batch.get('label')
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).to(device, non_blocking=True).long()

            # Collect per-model logits
            logits_list = []
            for m in model_list:
                if use_amp and getattr(device, "type", None) == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        logits = m(images)
                else:
                    logits = m(images)
                logits_list.append(logits)

            # Per-model probs for indiv_scores
            if is_ensemble:
                for i, logits in enumerate(logits_list):
                    probs_i = _to_probs(logits).detach().cpu().numpy()
                    indiv_scores[i].append(probs_i)

            # Average logits and probs
            logits_stack = torch.stack(logits_list, dim=0)  # [M, B, C]
            avg_logits = logits_stack.mean(dim=0)           # [B, C]
            avg_probs = _to_probs(avg_logits)               # [B, C]

            y_true_list.append(labels.detach().cpu().numpy())
            avg_logit_batches.append(avg_logits.detach().cpu().numpy())
            avg_prob_batches.append(avg_probs.detach().cpu().numpy())

    # Concatenate across batches
    y_true = np.concatenate(y_true_list, axis=0)
    digits = np.concatenate(avg_logit_batches, axis=0)  # logits
    y_scores = np.concatenate(avg_prob_batches, axis=0) # probs

    # Flatten binary to (N,)
    if y_scores.shape[1] == 1:
        y_scores = y_scores.ravel()
        digits = digits.ravel()

    # Predictions and correctness
    if y_scores.ndim == 1:
        y_pred = (y_scores >= 0.5).astype(np.int64)
    else:
        y_pred = np.argmax(y_scores, axis=1).astype(np.int64)

    correct_idx = np.where(y_pred == y_true)[0]
    incorrect_idx = np.where(y_pred != y_true)[0]

    # Finalize indiv_scores
    if is_ensemble:
        indiv_scores = [np.concatenate(slices, axis=0) for slices in indiv_scores]
        # Flatten binary indiv scores to (N,)
        if indiv_scores and indiv_scores[0].ndim == 2 and indiv_scores[0].shape[1] == 1:
            indiv_scores = [arr.ravel() for arr in indiv_scores]
        return y_true, y_scores, digits, correct_idx, incorrect_idx, indiv_scores

    return y_true, y_scores, digits, correct_idx, incorrect_idx


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

def get_batch_predictions(models, augmented_inputs, device, use_amp=True):
    """
    Get predictions for a flattened augmented_inputs tensor.
    - models: single model or list of models (already moved to device and eval())
    - augmented_inputs: Tensor on CPU (N, C, H, W) or (B*K, C, H, W)
    Returns:
      tensor shape [num_models, batch_len, num_classes] on CPU (detached)
    """
    # Ensure list
    model_list = models if isinstance(models, (list, tuple)) else [models]

    # Move input once to device
    inputs = augmented_inputs.to(device, non_blocking=True)

    preds = []
    # run each model (already on device) producing logits on device
    with torch.no_grad():
        for m in model_list:
            m.eval()
            if use_amp and getattr(device, "type", None) == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    out = m(inputs)
            else:
                out = m(inputs)
            preds.append(out.detach().cpu())  # detach and move to CPU to free GPU memory early

    # Stack: [num_models, batch_len, num_classes]
    batch_predictions = torch.stack(preds, dim=0)
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