import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.nn.functional import sigmoid, softmax
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchvision.models import resnet18, ResNet18_Weights
import medmnist
from medmnist import INFO
from .local_dermamnist_e import DERMAMNIST_E_INFO
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import random
import gc
import numpy as np
import os, json, time
from monai.data import CacheDataset as MONAI_CacheDataset
MONAI_AVAILABLE = True

torch.backends.cudnn.benchmark=True

def _clear_cache_dataset(ds):
    if ds is None:
        return
    if hasattr(ds, "clear_cache"):
        try:
            ds.clear_cache()
        except Exception:
            pass
    for attr in ("_cache", "_cached", "cache", "data"):
        if hasattr(ds, attr):
            setattr(ds, attr, None)

def _append_log(path, text):
    with open(path, 'a') as f:
        f.write(text.rstrip() + '\n')

# --- New: simple patience-based early stopper ---
class EarlyStopper:
    def __init__(self, mode='min', patience=10, min_delta=0.0):
        """
        mode: 'min' for loss, 'max' for accuracy
        patience: epochs to wait without improvement before stopping
        min_delta: minimum change to qualify as improvement
        """
        assert mode in ('min', 'max')
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_epochs = 0

    def _improved(self, current):
        if self.best is None:
            return True
        if self.mode == 'min':
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def step(self, current):
        improved = self._improved(current)
        if improved:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    def should_stop(self):
        return self.bad_epochs >= self.patience
    
# Prefetcher: moves batches to device asynchronously to overlap CPU/GPU work
class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        # expose common attributes for compatibility (e.g. `.dataset`, `.batch_size`, ...)
        self.dataset = getattr(loader, "dataset", None)
        self.batch_size = getattr(loader, "batch_size", None)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        return self

    def __next__(self):
        batch = next(self._iter)  # may raise StopIteration
        if self.stream is None:
            x, y = batch
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        # async copy on separate stream
        with torch.cuda.stream(self.stream):
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        # ensure main stream waits for the prefetch stream
        torch.cuda.current_stream().wait_stream(self.stream)
        return x, y

    def __getattr__(self, name):
        # Delegate unknown attributes to the underlying loader (keeps compatibility)
        return getattr(self.loader, name)

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def _append_log(path, text):
    with open(path, 'a') as f:
        f.write(text.rstrip() + '\n')


def get_datasets(data_flag, download=True, random_seed=None, im_size=28, color=False, transform=None, transform_test=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if data_flag == 'dermamnist-e':
        # use the local INFO we registered above
        info = DERMAMNIST_E_INFO
        DataClass = getattr(medmnist, info['python_class'])
    else : 
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
    if transform is None:
        if color:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        else:
            # For grayscale images, repeat the single channel to make it compatible with ResNet
            # ResNet expects 3 channels, so we repeat the single channel image
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=[.5], std=[.5])
            ])

    train_dataset = DataClass(split='train', transform=transform, size=im_size, download=download)
    val_dataset = DataClass(split='val', transform=transform, size=im_size, download=download)
    if transform_test is None:
        transform_test = transform
    test_dataset = DataClass(split='test', transform=transform_test, size=im_size, download=download)

    return [train_dataset, val_dataset, test_dataset], info


def get_dataloaders(datasets, batch_size=32, num_workers=None, use_cache_test=False, cache_backend='monai', cache_rate=1.0):
    """
    Build dataloaders for (train, calibration, test) datasets.

    Args:
      datasets: (train_dataset, calib_dataset, test_dataset)
      batch_size: int
      num_workers: int or None (auto heuristic)
      use_monai: bool - if True and MONAI available, use MONAI ThreadDataLoader
      monai_params: optional dict passed to MONAI ThreadDataLoader
      use_cache: bool - if True and MONAI available, build MONAI CacheDataset (cache_backend='monai')
      cache_rate: fraction (0..1) to cache in MONAI CacheDataset
      train_augment_transform: callable applied on-the-fly to training items (keeps RandAugment random)
    Returns:
      train_loader, calib_loader, test_loader
    """
    # default: conservative shared‑machine heuristic (allow override via NUM_WORKERS)
    if num_workers is None:
        try:
            env_n = os.environ.get("NUM_WORKERS")
            if env_n is not None:
                num_workers = int(env_n)
            else:
                n_cpu = os.cpu_count() or 1
                n_users = int(os.environ.get("SHARED_USERS", "4"))
                per_user = max(2, n_cpu // (n_users * 8))
                num_workers = int(min(max(per_user, 2), 16))
        except Exception:
            num_workers = 4

    train_dataset, calib_dataset, test_dataset = datasets

    # MONAI CacheDataset backend (preferred when use_cache=True)
    if use_cache_test and cache_backend == 'monai' and MONAI_AVAILABLE:
        try:
            # build MONAI-style data list (keep tensors as tensors to avoid needless numpy<->tensor roundtrip)
            def _build_data_list(ds):
                data_list = []
                if isinstance(ds, torch.utils.data.Subset):
                    base = ds.dataset
                    indices = ds.indices
                else:
                    base = ds
                    indices = range(len(ds))
                for i in indices:
                    item = base[i]
                    if isinstance(item, dict) and 'image' in item and 'label' in item:
                        img, lbl = item['image'], item['label']
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        img, lbl = item[0], item[1]
                    else:
                        raise RuntimeError("Unsupported dataset item format for MONAI caching.")
                    if torch.is_tensor(img):
                        # keep tensor (detached on CPU) to avoid round-trip
                        data_list.append({'image_tensor': img.detach().cpu(), 'label': int(lbl)})
                    else:
                        # keep numpy array
                        data_list.append({'image_numpy': np.asarray(img), 'label': int(lbl)})
                return data_list
            
            test_list = _build_data_list(test_dataset)

            # transform that returns (tensor, label) and handles both stored tensor or numpy
            def _to_tensor_tuple(d):
                if 'image_tensor' in d:
                    img = d['image_tensor']
                    if not isinstance(img, torch.Tensor):
                        img = torch.as_tensor(img)
                    img = img.float()
                else:
                    img = torch.from_numpy(d['image_numpy']).float()
                lbl = torch.tensor(int(d['label']), dtype=torch.long)
                return img, lbl

            test_cache_ds  = MONAI_CacheDataset(data=test_list,  transform=_to_tensor_tuple, cache_rate=float(cache_rate))

            persistent = True if (num_workers and num_workers>0) else False

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            calib_loader = DataLoader(dataset=calib_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_cache_ds, batch_size=batch_size, shuffle=False, prefetch_factor=3, num_workers=num_workers,
                                    pin_memory=True, persistent_workers=persistent)
            print(f"Using MONAI CacheDataset (cache_rate={cache_rate}) for test set with {len(test_cache_ds)} items.")
            return train_loader, calib_loader, test_loader
        except Exception as e:
            print("MONAI CacheDataset construction failed, falling back to non-cached loaders:", e)

    else:
        # Default: standard torch DataLoader
        persistent = True if (num_workers and num_workers > 0) else False
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                prefetch_factor=3, num_workers=num_workers,
                                pin_memory=True, persistent_workers=persistent)
        calib_loader = DataLoader(dataset=calib_dataset, batch_size=batch_size, shuffle=False,
                                prefetch_factor=3, num_workers=num_workers,
                                pin_memory=True, persistent_workers=persistent)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                prefetch_factor=3, num_workers=num_workers,
                                pin_memory=True, persistent_workers=persistent)

        return train_loader, calib_loader, test_loader
    

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        # Check the criterion type and adjust the target size accordingly
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            # Ensure both output and target are (N,1)
            target_t = target.float().view(-1, 1)
            loss = criterion(output, target_t)
        else:
            # CrossEntropyLoss: targets shape (N,)
            target_t = target.view(-1).long()
            loss = criterion(output, target_t)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return epoch_loss / len(train_loader)


def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    n_samples = len(val_loader.dataset)
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_t = target.float().view(-1, 1)
                val_loss += criterion(output, target_t).item()
                pred = (output > 0).float()
                correct += pred.eq(target_t).sum().item()
            else:
                target_t = target.view(-1).long()
                val_loss += criterion(output, target_t).item()
                pred = output.argmax(dim=1)
                correct += (pred == target_t).sum().item()

    val_loss /= len(val_loader)  # average per batch
    val_acc = correct / float(n_samples)
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{n_samples} ({100. * val_acc:.0f}%)\n')
    return val_loss, val_acc

def _compute_class_weights_from_loader(train_loader, num_classes):
    """
    Returns:
      class_weight (torch.Tensor or None) for CE
      pos_weight (torch.Tensor or None) for BCEWithLogits
    """
    counts = np.zeros(int(num_classes), dtype=np.int64)
    with torch.no_grad():
        for _, target in train_loader:
            t = target.detach().cpu().numpy().reshape(-1)
            # targets come as ints [0..C-1] (binary: 0/1)
            for c in t:
                counts[int(c)] += 1

    total = counts.sum()
    # Avoid div-by-zero
    counts = np.clip(counts, 1, None)

    if num_classes == 2:
        neg, pos = counts[0], counts[1]
        # pos_weight = neg/pos
        pos_weight = torch.tensor([float(neg) / float(pos)], dtype=torch.float32)
        return None, pos_weight
    else:
        # CE class weights: inverse frequency, normalized to mean 1.0
        w = 1.0 / counts.astype(np.float64)
        w = w / (w.mean() + 1e-12)
        class_weight = torch.tensor(w, dtype=torch.float32)
        return class_weight, None

def train_resnet18(data_flag, info, num_epochs=10, learning_rate=0.001, device=None,
                   train_loader=None, val_loader=None, test_loader=None, random_seed=None, output_dir=None, run_name="run", scheduler=False, early_stop=True, monitor='val_loss', patience=10, min_delta=0.001, restore_best=True, checkpoint_best=False, class_weighting=True):
        # Optional seeding
    if random_seed is not None:
        import random
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(info['label'])
    
    # Compute weights BEFORE wrapping loaders for CUDA prefetch
    ce_weight_cpu, bce_pos_weight_cpu = (None, None)
    if class_weighting:
        try:
            ce_weight_cpu, bce_pos_weight_cpu = _compute_class_weights_from_loader(train_loader, num_classes)
            if ce_weight_cpu is not None:
                print(f"Using CE class weights (mean=1): {ce_weight_cpu.tolist()}")
            if bce_pos_weight_cpu is not None:
                print(f"Using BCE pos_weight: {bce_pos_weight_cpu.item():.4f}")
        except Exception as e:
            print("Warning: failed to compute class weights, continuing unweighted:", e)

    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Wrap loaders with PrefetchLoader to overlap copies (only for CUDA)
    if device and 'cuda' in str(device).lower():
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)
        test_loader = PrefetchLoader(test_loader, device)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features

    if num_classes == 2:
            model.fc = torch.nn.Linear(in_features, 1)
                    # Move pos_weight to device if available
            if bce_pos_weight_cpu is not None:
                criterion = BCEWithLogitsLoss(pos_weight=bce_pos_weight_cpu.to(device))
            else:
                criterion = BCEWithLogitsLoss()
    else:
        model.fc = torch.nn.Linear(in_features, num_classes)
        # Move CE weights to device if available
        if ce_weight_cpu is not None:
            criterion = CrossEntropyLoss(weight=ce_weight_cpu.to(device))
        else:
            criterion = CrossEntropyLoss()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if scheduler is True:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    train_losses = []
    val_losses = []
    val_accs = []  # new: keep val_acc history
    epoch_times = []
    if output_dir:
        figs_dir = os.path.join(output_dir, "figs")
        _ensure_dir(figs_dir)
        log_path = os.path.join(output_dir, "metrics.log")
        _append_log(log_path, f"=== {run_name} start: epochs={num_epochs}, lr={learning_rate} ===")

    # New: setup early stopper
    stopper = None
    best_state = None
    best_epoch = -1
    best_metric_val = None
    metric_mode = 'min' if monitor == 'val_loss' else 'max'
    if early_stop:
        stopper = EarlyStopper(mode=metric_mode, patience=patience, min_delta=min_delta)
    best_ckpt_path = os.path.join(output_dir, f"best_{run_name}.pt") if (output_dir and checkpoint_best) else None

    run_t0 = time.time()
    for epoch in range(num_epochs):
        ep_t0 = time.time()
        t0_train = time.perf_counter()
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        if torch.cuda.is_available() and ('cuda' in str(device).lower()):
            torch.cuda.synchronize()
        t1_train = time.perf_counter()

        if torch.cuda.is_available() and ('cuda' in str(device).lower()):
            torch.cuda.synchronize()
        t_before_val = time.perf_counter()

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        if torch.cuda.is_available() and ('cuda' in str(device).lower()):
            torch.cuda.synchronize()
        t_after_val = time.perf_counter()

        print(f"[timing] epoch={epoch} train_exec_s={(t1_train - t0_train):.3f} transition_s={(t_before_val - t1_train):.3f} val_exec_s={(t_after_val - t_before_val):.3f}")
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        ep_dur = time.time() - ep_t0
        epoch_times.append(ep_dur)
        if output_dir:
            _append_log(log_path, f"{run_name} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f} lr={current_lr:.6e} epoch_time_s={ep_dur:.2f}")
        
        if scheduler is not None:
            scheduler.step()

        print(f"{run_name} | epoch {epoch}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f} | val_acc {val_acc:.4f}")

        # --- Early stopping check ---
        if stopper is not None:
            metric_value = val_loss if monitor == 'val_loss' else val_acc
            improved = stopper.step(metric_value)
            if improved:
                best_metric_val = metric_value
                best_epoch = epoch
                # keep an in-memory copy and optionally checkpoint
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if best_ckpt_path is not None:
                    torch.save(model.state_dict(), best_ckpt_path)
            if stopper.should_stop():
                print(f"Early stopping at epoch {epoch} (best {monitor}={best_metric_val:.4f} at epoch {best_epoch})")
                if output_dir:
                    _append_log(log_path, f"{run_name} early_stop epoch={epoch} best_epoch={best_epoch} best_{monitor}={best_metric_val:.6f}")
                break
    
    # Restore best weights if requested
    if restore_best and best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            # fallback to disk if needed
            if best_ckpt_path and os.path.isfile(best_ckpt_path):
                model.load_state_dict(torch.load(best_ckpt_path, map_location='cpu'))
        print(f"Restored best model from epoch {best_epoch} ({monitor}={best_metric_val:.4f})")

    total_train_time = time.time() - run_t0
    # Save loss curve + history
    if output_dir:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Losses - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figs", f"loss_curve_{run_name}.png"), dpi=200)
        plt.close()

        # also save val_acc curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'Val Acc - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figs", f"val_acc_{run_name}.png"), dpi=200)
        plt.close()

        history = {
            "run_name": run_name,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "epoch_times_sec": epoch_times,
            "total_train_sec": total_train_time,
            "early_stop": {
                "enabled": bool(early_stop),
                "monitor": monitor,
                "best_epoch": int(best_epoch),
                "best_value": float(best_metric_val) if best_metric_val is not None else None
            }
        }
        _save_json(history, os.path.join(output_dir, f"history_{run_name}.json"))
        _append_log(log_path, f"{run_name} total_train_sec={total_train_time:.2f}")

    # Final test evaluation
    eval_result = evaluate_model(model, test_loader, data_flag, device=device,
                                 output_dir=output_dir, prefix=f"{run_name}_test")

    return model, {
        "run_name": run_name,
        "history": {"train_losses": train_losses, "val_losses": val_losses, "val_accs": val_accs, "epoch_times_sec": epoch_times},
        "timing": {"total_train_sec": total_train_time},
        "test": eval_result["metrics"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "early_stop": {
            "enabled": bool(early_stop),
            "monitor": monitor,
            "best_epoch": int(best_epoch),
            "best_value": float(best_metric_val) if best_metric_val is not None else None
        }
    }


def evaluate_model(model, test_loader, data_flag, device=None, output_dir=None, prefix="test"):
    if data_flag == 'dermamnist-e':
        info = DERMAMNIST_E_INFO
    else:
        info = INFO[data_flag]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = list(info['label'].values())
    num_classes = len(class_names)
    is_binary = (num_classes == 2)

    if output_dir:
        figs_dir = os.path.join(output_dir, "figs")
        _ensure_dir(figs_dir)
        log_path = os.path.join(output_dir, "metrics.log")

    # Normalize to list for ensemble averaging
    models = model if isinstance(model, list) else [model]
    for m in models:
        m.eval()

    y_true = []
    y_probs = []  # shape (N, C) for multiclass; (N, 1) for binary
    t0_eval = time.time()  # timing start
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_device = y.to(device) if isinstance(y, torch.Tensor) else y

            # collect per-model probabilities then average
            probs_accum = []
            for m in models:
                logits = m(x)
                if is_binary:
                    p = sigmoid(logits).view(-1, 1)  # (B, 1)
                else:
                    p = softmax(logits, dim=1)       # (B, C)
                probs_accum.append(p.detach().cpu().numpy())

            probs_avg = np.mean(np.stack(probs_accum, axis=0), axis=0)  # (B, C) or (B, 1)

            # move labels to host safely and append
            y_true.append(y_device.detach().cpu().numpy())
            y_probs.append(probs_avg)

    y_true = np.concatenate(y_true, axis=0)
    y_probs = np.concatenate(y_probs, axis=0)

    if is_binary:
        y_score = y_probs.ravel()                         # (N,)
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_score = y_probs                                 # (N, C)
        y_pred = np.argmax(y_score, axis=1)

    eval_wall = time.time() - t0_eval
    n_samples = int(len(y_true))

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        if is_binary:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    result = {
        "data_flag": data_flag,
        "num_classes": num_classes,
        "class_names": class_names,
        "is_ensemble": isinstance(model, list),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "auc": auc
        },
        "confusion_matrix": cm.tolist(),
        "counts": {
            "n_samples": int(len(y_true))
        },
        "timing": {
            "eval_wall_sec": float(eval_wall),
            "throughput_img_per_s": float(n_samples / eval_wall) if eval_wall > 0 else float('inf'),
            "latency_ms_per_img": float(1000.0 * eval_wall / n_samples) if n_samples > 0 else float('nan')
        }
    }

    # Save confusion matrix figure
    if output_dir:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({prefix})")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "figs", f"confusion_matrix_{prefix}.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()

        # Save metrics JSON and append log
        _save_json(result, os.path.join(output_dir, f"metrics_{prefix}.json"))
        _append_log(log_path, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix} "
                              f"acc={acc:.4f} bal_acc={bal_acc:.4f} auc={auc:.4f}")

    # Minimal print
    print(f"[{prefix}] acc={acc:.3f} bal_acc={bal_acc:.3f} auc={auc:.3f}")
    return result

def save_model(model, path):
    """
    Save the PyTorch model to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_models(flag, device, waugmentation=False, size=224):

    # Load organAMNIST dataset
    data_flag = flag
    info = INFO[data_flag]
    num_classes = len(info['label'])
    # Load saved models
    models = []
    for i in range(5):
        # Initialize the model
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if num_classes == 2:
            model.fc = nn.Linear(model.fc.in_features, 1)  # Output 1 value for binary classification
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)  # Output logits for each class
        
        # Load the state dictionary
        if waugmentation:
            state_dict = torch.load(f'/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/models/{size}*{size}/resnet18_{flag}_{size}_{i}_augmented.pt')
        else:
            # Load the state dictionary
            state_dict = torch.load(f'/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/models/{size}*{size}/resnet18_{flag}_{size}_{i}.pt')
            
        # Remove the 'model.' prefix from the state_dict keys if necessary
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models

def load_datasets(dataflag, color, im_size, transform, batch_size, cache_test=False, transform_test=None):    
    datasets, info = get_datasets(dataflag, im_size=im_size, color=color, transform=transform, transform_test=transform_test)
    # Combine train_dataset and val_dataset
    combined_train_dataset = ConcatDataset([datasets[0], datasets[1]])

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Calculate the sizes for training and calibration datasets
    train_size = int(0.8 * len(combined_train_dataset))
    calibration_size = len(combined_train_dataset) - train_size

    # Split the combined_train_dataset into training and calibration datasets
    train_dataset, calibration_dataset = random_split(combined_train_dataset, [train_size, calibration_size])
    test_dataset = datasets[2]  # Use the test dataset as is

    dataloaders = get_dataloaders([train_dataset, calibration_dataset, test_dataset], batch_size=batch_size, use_cache_test=cache_test)

    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Calibration dataset size: {len(calibration_dataset)}')
    
    return [train_dataset, calibration_dataset, test_dataset], dataloaders, info

def CV_train_val_loaders(study_dataset_aug, study_dataset_plain, batch_size,
                         n_splits=5, seed=42, use_monai=False, cache_rate=1.0, train_augment_transform=None, num_workers=None, pin_memory=True, prewarm_cache=False):
    """
    Create CV train/val DataLoaders with optional MONAI ThreadDataLoader and CacheDataset support.
    - use_monai: prefer MONAI ThreadDataLoader (if available)
    - use_cache: build MONAI CacheDataset per-fold (only when MONAI available)
    - cache_rate: fraction to cache in MONAI CacheDataset
    - train_augment_transform: callable applied on-the-fly to training items (RandAugment)
    """
    # decide num_workers if not provided
    if num_workers is None:
        try:
            env_n = os.environ.get("NUM_WORKERS")
            if env_n is not None:
                num_workers = int(env_n)
            else:
                n_cpu = os.cpu_count() or 1
                n_users = int(os.environ.get("SHARED_USERS", "4"))
                per_user = max(2, n_cpu // (n_users * 8))
                num_workers = int(min(max(per_user, 2), 16))
        except Exception:
            num_workers = 4
        print(f"CV loaders: using num_workers={num_workers}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Labels from the plain (non-augmented) view
    labels = [label for _, label in study_dataset_plain]

    train_loaders = []
    val_loaders = []
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    use_monai_local = bool(use_monai) and MONAI_AVAILABLE

    def _build_data_list_from_subset(ds):
        data_list = []
        if isinstance(ds, torch.utils.data.Subset):
            base = ds.dataset
            indices = ds.indices
        else:
            base = ds
            indices = range(len(ds))
        for i in indices:
            item = base[i]
            if isinstance(item, dict) and 'image' in item and 'label' in item:
                img, lbl = item['image'], item['label']
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                img, lbl = item[0], item[1]
            else:
                raise RuntimeError("Unsupported dataset item format for caching.")
            if torch.is_tensor(img):
                data_list.append({'image_tensor': img.detach().cpu(), 'label': int(lbl)})
            else:
                data_list.append({'image_numpy': np.asarray(img), 'label': int(lbl)})
        return data_list

    for train_index, val_index in skf.split(np.zeros(len(labels)), labels):
        # Build subsets (indices are relative to the combined train set)
        if study_dataset_aug is not None:
            train_subset = torch.utils.data.Subset(study_dataset_aug, train_index)
        else:
            train_subset = torch.utils.data.Subset(study_dataset_plain, train_index)
        val_subset = torch.utils.data.Subset(study_dataset_plain, val_index)

        # If caching with MONAI is requested and available, build CacheDataset per-fold
        if use_monai_local:
            try:
                train_list = _build_data_list_from_subset(train_subset)
                val_list = _build_data_list_from_subset(val_subset)

                def _to_tensor_tuple(d):
                    # handle cached tensor or numpy entry without unnecessary roundtrip
                    if 'image_tensor' in d:
                        img = d['image_tensor']
                        if not isinstance(img, torch.Tensor):
                            img = torch.as_tensor(img)
                        img = img.float()
                    else:
                        img = torch.from_numpy(d['image_numpy']).float()
                    # when training uses RandAugment we want val cached as normalized tensors
                    if train_augment_transform is not None:
                        img = normalize(img)
                    lbl = torch.tensor(int(d['label']), dtype=torch.long)
                    return img, lbl
                def _to_train_cached(d):
                    # produce cached item for training: prefer to keep tensor when possible,
                    # but convert to PIL if we want to cache PIL for faster augment use
                    if 'image_tensor' in d:
                        img = d['image_tensor']
                        if not isinstance(img, torch.Tensor):
                            img = torch.as_tensor(img)
                        img = img.float()
                    else:
                        img = torch.from_numpy(d['image_numpy']).float()

                    if train_augment_transform is not None:
                        # cache as PIL to avoid repeated to_pil_image at runtime (optional)
                        img_pil = to_pil_image(torch.clamp(img, 0., 1.))
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img_pil, lbl
                    else:
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img, lbl
                train_cache_ds = MONAI_CacheDataset(data=train_list, transform=_to_train_cached, cache_rate=float(cache_rate))
                val_cache_ds = MONAI_CacheDataset(data=val_list, transform=_to_tensor_tuple, cache_rate=float(cache_rate))

                # wrap training cached dataset with augment transform if requested (augment should include Normalize)
                if train_augment_transform is not None:
                    class AugmentCachedDataset(torch.utils.data.Dataset):
                        """
                        Wrap a cached dataset and apply a (PIL-based) augment transform at runtime.
                        
                        wrapper will augment -> tensor -> normalize.
                        """
                        def __init__(self, cache_ds, augment):
                            self.cache_ds = cache_ds
                            self.augment = augment

                        def __len__(self):
                            return len(self.cache_ds)

                        def __getitem__(self, idx):
                            x, y = self.cache_ds[idx]
                            if train_augment_transform is not None and isinstance(x, Image.Image):
                                aug = self.augment(x)
                                x_aug = aug if torch.is_tensor(aug) else T.ToTensor()(aug)
                            else:
                                x = x.detach().cpu().float()
                                try:
                                    print("")
                                    x_aug = self.augment(x)
                                    if not torch.is_tensor(x_aug):
                                        x_aug = T.ToTensor()(x_aug)
                                except Exception:
                                    x_aug = self.augment(to_pil_image(torch.clamp(x, 0., 1.)))
                                    if not torch.is_tensor(x_aug):
                                        x_aug = T.ToTensor()(x_aug)
                            if train_augment_transform is not None:
                                x_aug = x_aug.float()
                            return x_aug, y
                    train_ds_wrapped = AugmentCachedDataset(train_cache_ds, train_augment_transform)
                else:    
                    train_ds_wrapped = train_cache_ds

                
                val_ds_wrapped = val_cache_ds
                persistent = True if (num_workers and num_workers>0) else False
                train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory,
                                          persistent_workers=persistent, prefetch_factor=2, drop_last=True)
                
                val_loader = DataLoader(val_ds_wrapped, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, persistent_workers=persistent, prefetch_factor=3)
                print('train/val loaders using torch DataLoader for cached val dataset')

                if prewarm_cache and use_monai_local:
                    try:
                        t0 = time.time()
                        print("Pre-warming MONAI cache for this fold (this may take some time)...")
                        for _ in train_loader:
                            pass
                        print(f"Cache pre-warm done ({time.time()-t0:.1f}s)")
                    except Exception as e:
                        print("Pre-warm failed or was interrupted:", e)
            except Exception as e:
                print("MONAI CacheDataset failed for fold:", e, "- falling back to torch DataLoader.")
                persistent = True if (num_workers and num_workers > 0) else False
                train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
                val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                        num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)
        else:
            # no caching: optionally wrap train subset with runtime augment transform
            if train_augment_transform is not None:
                class AugmentDataset(torch.utils.data.Dataset):
                    def __init__(self, ds, augment):
                        self.ds = ds
                        self.augment = augment
                    def __len__(self):
                        return len(self.ds)
                    def __getitem__(self, idx):
                        item = self.ds[idx]
                        if isinstance(item, dict) and 'image' in item and 'label' in item:
                            x, y = item['image'], item['label']
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            x, y = item[0], item[1]
                        else:
                            raise RuntimeError("Unsupported dataset item format in AugmentDataset.")
                        try:
                            x_aug = self.augment(x)
                        except Exception:
                            from torchvision.transforms.functional import to_pil_image, to_tensor
                            if torch.is_tensor(x):
                                x_pil = to_pil_image(x)
                            else:
                                x_pil = x
                            x_aug = self.augment(x_pil)
                            if not torch.is_tensor(x_aug):
                                x_aug = to_tensor(x_aug)
                        return x_aug, y
                train_ds_wrapped = AugmentDataset(train_subset, train_augment_transform)
            else:
                train_ds_wrapped = train_subset

            # DataLoader for normal (non-cache) case
            persistent = True if (num_workers and num_workers > 0) else False
            train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
            val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)


        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    loader_type = "DataLoader w persistent workers + MONAI CacheDataset" if use_monai_local else "torch DataLoader"
    print(f"CV loaders created: {n_splits} folds using {loader_type} (num_workers={num_workers}, cache_rate={cache_rate})")
    return train_loaders, val_loaders

def CV_fold_generator(study_dataset_aug, study_dataset_plain, batch_size,
                      n_splits=5, seed=42, use_monai=False, cache_rate=1.0,
                      train_augment_transform=None, num_workers=None, pin_memory=True, prewarm_cache=False):
    """
    Generator that yields (train_loader, val_loader, fold_index) for each CV fold.
    Build and return one fold at a time so caller can free memory after training that fold.
    Same parameters/behavior as CV_train_val_loaders but lazily constructs per-fold loaders.
    """
    if num_workers is None:
        try:
            env_n = os.environ.get("NUM_WORKERS")
            if env_n is not None:
                num_workers = int(env_n)
            else:
                n_cpu = os.cpu_count() or 1
                n_users = int(os.environ.get("SHARED_USERS", "4"))
                per_user = max(2, n_cpu // (n_users * 8))
                num_workers = int(min(max(per_user, 2), 16))
        except Exception:
            num_workers = 4

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [label for _, label in study_dataset_plain]
    use_monai_local = bool(use_monai) and MONAI_AVAILABLE
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])

    def _build_data_list_from_subset(ds):
        data_list = []
        if isinstance(ds, torch.utils.data.Subset):
            base = ds.dataset
            indices = ds.indices
        else:
            base = ds
            indices = range(len(ds))
        for i in indices:
            item = base[i]
            if isinstance(item, dict) and 'image' in item and 'label' in item:
                img, lbl = item['image'], item['label']
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                img, lbl = item[0], item[1]
            else:
                raise RuntimeError("Unsupported dataset item format for caching.")
            if torch.is_tensor(img):
                data_list.append({'image_tensor': img.detach().cpu(), 'label': int(lbl)})
            else:
                data_list.append({'image_numpy': np.asarray(img), 'label': int(lbl)})
        return data_list

    for fold_idx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_loader = None
        val_loader = None
        train_cache_ds = None
        val_cache_ds = None
        train_list = None
        val_list = None
        try:
            # build subsets
            if study_dataset_aug is not None:
                train_subset = torch.utils.data.Subset(study_dataset_aug, train_index)
            else:
                train_subset = torch.utils.data.Subset(study_dataset_plain, train_index)
            val_subset = torch.utils.data.Subset(study_dataset_plain, val_index)

            # Build loaders for this fold (reuse the same logic as CV_train_val_loaders)
            if use_monai_local:
                try:
                    train_list = _build_data_list_from_subset(train_subset)
                    val_list = _build_data_list_from_subset(val_subset)

                    def _to_tensor_tuple(d):
                        if 'image_tensor' in d:
                            img = d['image_tensor']
                            if not isinstance(img, torch.Tensor):
                                img = torch.as_tensor(img)
                            img = img.float()
                        else:
                            img = torch.from_numpy(d['image_numpy']).float()
                        if train_augment_transform is not None:
                            img = normalize(img)
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img, lbl

                    def _to_train_cached(d):
                        if 'image_tensor' in d:
                            img = d['image_tensor']
                            if not isinstance(img, torch.Tensor):
                                img = torch.as_tensor(img)
                            img = img.float()
                        else:
                            img = torch.from_numpy(d['image_numpy']).float()

                        if train_augment_transform is not None:
                            img_pil = to_pil_image(torch.clamp(img, 0., 1.))
                            lbl = torch.tensor(int(d['label']), dtype=torch.long)
                            return img_pil, lbl
                        else:
                            lbl = torch.tensor(int(d['label']), dtype=torch.long)
                            return img, lbl

                    train_cache_ds = MONAI_CacheDataset(data=train_list, transform=_to_train_cached, cache_rate=float(cache_rate))
                    val_cache_ds = MONAI_CacheDataset(data=val_list, transform=_to_tensor_tuple, cache_rate=float(cache_rate))

                    # runtime augment wrapper if requested
                    if train_augment_transform is not None:
                        class AugmentCachedDataset(torch.utils.data.Dataset):
                            def __init__(self, cache_ds, augment):
                                self.cache_ds = cache_ds
                                self.augment = augment

                            def __len__(self):
                                return len(self.cache_ds)

                            def __getitem__(self, idx):
                                x, y = self.cache_ds[idx]
                                if train_augment_transform is not None and isinstance(x, Image.Image):
                                    aug = self.augment(x)
                                    x_aug = aug if torch.is_tensor(aug) else T.ToTensor()(aug)
                                else:
                                    x = x.detach().cpu().float()
                                    try:
                                        x_aug = self.augment(x)
                                        if not torch.is_tensor(x_aug):
                                            x_aug = T.ToTensor()(x_aug)
                                    except Exception:
                                        x_aug = self.augment(to_pil_image(torch.clamp(x, 0., 1.)))
                                        if not torch.is_tensor(x_aug):
                                            x_aug = T.ToTensor()(x_aug)
                                if train_augment_transform is not None:
                                    x_aug = x_aug.float()
                                return x_aug, y
                        train_ds_wrapped = AugmentCachedDataset(train_cache_ds, train_augment_transform)
                    else:
                        train_ds_wrapped = train_cache_ds

                    val_ds_wrapped = val_cache_ds
                    persistent = True if (num_workers and num_workers > 0) else False
                    train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=pin_memory,
                                              persistent_workers=persistent, prefetch_factor=2, drop_last=True)
                    val_loader = DataLoader(dataset=val_ds_wrapped, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, prefetch_factor=3)

                    if prewarm_cache:
                        try:
                            for _ in train_loader:
                                pass
                        except Exception:
                            pass

                except Exception:
                    # fallback to plain DataLoader
                    persistent = True if (num_workers and num_workers > 0) else False
                    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
                    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                            num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)
                    train_cache_ds = None
                    val_cache_ds = None
            else:
                if train_augment_transform is not None:
                    class AugmentDataset(torch.utils.data.Dataset):
                        def __init__(self, ds, augment):
                            self.ds = ds
                            self.augment = augment
                        def __len__(self):
                            return len(self.ds)
                        def __getitem__(self, idx):
                            item = self.ds[idx]
                            if isinstance(item, dict) and 'image' in item and 'label' in item:
                                x, y = item['image'], item['label']
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                x, y = item[0], item[1]
                            else:
                                raise RuntimeError("Unsupported dataset item format in AugmentDataset.")
                            try:
                                x_aug = self.augment(x)
                            except Exception:
                                from torchvision.transforms.functional import to_pil_image, to_tensor
                                if torch.is_tensor(x):
                                    x_pil = to_pil_image(x)
                                else:
                                    x_pil = x
                                x_aug = self.augment(x_pil)
                                if not torch.is_tensor(x_aug):
                                    x_aug = to_tensor(x_aug)
                            return x_aug, y
                    train_ds_wrapped = AugmentDataset(train_subset, train_augment_transform)
                else:
                    train_ds_wrapped = train_subset

                persistent = True if (num_workers and num_workers > 0) else False
                train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
                val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                        num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)

            yield fold_idx, train_loader, val_loader
        finally:
            for dl in (train_loader, val_loader):
                if dl is None:
                    continue
                it = getattr(dl, "_iterator", None)
                if it is not None:
                    try:
                        it._shutdown_workers()
                    except Exception:
                        pass
                _clear_cache_dataset(getattr(dl, "dataset", dl))

            for ds in (train_cache_ds, val_cache_ds):
                _clear_cache_dataset(ds)

            del train_loader, val_loader, train_cache_ds, val_cache_ds, train_list, val_list
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        