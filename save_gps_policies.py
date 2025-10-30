import UQ_toolbox as uq
from medMNIST.utils import train_load_datasets_resnet as tr
import torch
import torchvision.transforms as transforms
import os
import time
import threading
import json
try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# -------------------- Resource profiling helpers --------------------
def _get_cuda_index(device):
    if isinstance(device, torch.device) and device.type == 'cuda':
        return 0 if device.index is None else device.index
    if isinstance(device, str) and device.startswith('cuda'):
        parts = device.split(':')
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return None

class ResourceProfiler:
    def __init__(self, device=None, label=""):
        self.device = device
        self.label = label
        self.records = {}
        self._gpu_thread = None
        self._gpu_utils = []
        self._stop_evt = threading.Event()
        self._nvml_handle = None
        self.cuda_idx = _get_cuda_index(device)

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.proc = psutil.Process(os.getpid()) if psutil else None
        if self.proc:
            self.cpu_times0 = self.proc.cpu_times()
            self.rss0 = self.proc.memory_info().rss
        if torch.cuda.is_available() and self.cuda_idx is not None:
            torch.cuda.reset_peak_memory_stats(self.cuda_idx)
            if _NVML_AVAILABLE:
                try:
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.cuda_idx)
                    def _poll():
                        while not self._stop_evt.is_set():
                            try:
                                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
                                self._gpu_utils.append(util)
                            except Exception:
                                pass
                            time.sleep(0.2)
                    self._gpu_thread = threading.Thread(target=_poll, daemon=True)
                    self._gpu_thread.start()
                except Exception:
                    pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._gpu_thread:
            self._stop_evt.set()
            self._gpu_thread.join()
        t1 = time.perf_counter()
        rec = {"label": self.label, "wall_time_s": t1 - self.t0}

        if self.proc:
            cpu_times1 = self.proc.cpu_times()
            rss1 = self.proc.memory_info().rss
            rec.update({
                "cpu_user_s": getattr(cpu_times1, "user", 0.0) - getattr(self.cpu_times0, "user", 0.0),
                "cpu_system_s": getattr(cpu_times1, "system", 0.0) - getattr(self.cpu_times0, "system", 0.0),
                "rss_mb_start": self.rss0 / (1024**2),
                "rss_mb_end": rss1 / (1024**2),
            })

        if torch.cuda.is_available() and self.cuda_idx is not None:
            rec.update({
                "cuda_idx": self.cuda_idx,
                "gpu_mem_mb_peak": torch.cuda.max_memory_allocated(self.cuda_idx) / (1024**2),
                "gpu_mem_mb_end": torch.cuda.memory_allocated(self.cuda_idx) / (1024**2),
                "gpu_util_avg": (sum(self._gpu_utils) / len(self._gpu_utils)) if self._gpu_utils else None,
                "gpu_util_max": max(self._gpu_utils) if self._gpu_utils else None,
            })
        self.records = rec
# -------------------- end profiling helpers --------------------
        
#flags = ['bloodmnist', 'bloodmnist', 'octmnist', 'octmnist']#
flags=['pathmnist', 'pathmnist', 'dermamnist-e', 'organamnist', 'tissuemnist']
flags=['breastmnist']
#colors = [True, True, False, False]
colors = [True, True, True, False, False]
colors = [False]
#activations = ['softmax', 'softmax', 'softmax', 'softmax']
activations=['softmax', 'softmax', 'softmax', 'softmax', 'softmax']
activations=['sigmoid']
#model_augmentations = [False, True, False, True]#, 
model_augmentations = [False, True, True, True, True]
model_augmentations = [False]
#color = True # True for color, False for grayscale
#activation = 'softmax'  # 'sigmoid' for binary-class, 'softmax' for multi-class
batch_size = 4000
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
size = 224  # Image size for the models
#model_augmentation = False  # Whether the models were trained with data augmentation
randaugment_ops = 2
randaugment_mag = 45
max_iterations = 500
nb_channels = 3
for model_augmentation, dataflag, color, activation in zip(model_augmentations, flags, colors, activations):
    # Loop over different datasets and settings
    if model_augmentation:
        output_dir = f'/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/gps_augment/{size}*{size}/{dataflag}_wdataaug_calibration_set_test'
    else:
        output_dir = f'/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/gps_augment/{size}*{size}/{dataflag}_calibration_set_test'
    os.makedirs(output_dir, exist_ok=True)


    print(f"Processing {dataflag} with color={color} and activation={activation}")
    if color is True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        
        transform_tta = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        # For grayscale images, repeat the single channel to make it compatible with ResNet
        # ResNet expects 3 channels, so we repeat the single channel image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        
        transform_tta = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
    models = tr.load_models(dataflag, waugmentation=model_augmentation, device=device)
    _, _, info = tr.load_datasets(dataflag, color, size, transform, batch_size)
    task_type = info['task']  # Determine the task type (binary-class or multi-class)
    num_classes = len(info['label'])  # Number of classes
    [_, calibration_dataset_tta, _], [_, _, _], _ = tr.load_datasets(dataflag, color, size, transform_tta, batch_size)

    cache_workers = max(os.cpu_count() - 1, 0) if os.cpu_count() else 0

    with ResourceProfiler(device=device, label="GPS_calibration_randaugment") as rp:
        uq.apply_randaugment_and_store_results(
            calibration_dataset_tta,
            models,
            randaugment_ops,
            randaugment_mag,
            max_iterations,
            device,
            folder_name=output_dir,
            image_normalization=True,
            mean=[.5],
            std=[.5],
            image_size=size,
            nb_channels=nb_channels,
            output_activation=activation,
            batch_size=batch_size,
            use_monai_cache=True,
            cache_num_workers=cache_workers,
            dataloader_workers=0
        )

    # Save JSON log
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(output_dir, f"gps_resource_log_{dataflag}_{ts}.json")
    rec = dict(rp.records)
    rec.update({
        "flag": dataflag,
        "color": color,
        "device": str(device),
        "im_size": size,
        "batch_size": batch_size,
        "activation": activation,
        "randaugment_ops": randaugment_ops,
        "randaugment_mag": randaugment_mag,
        "max_iterations": max_iterations,
        "nb_channels": nb_channels,
        "num_models": len(models),
        "num_calibration_samples": len(calibration_dataset_tta),
    })
    with open(log_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved GPS resource log to: {log_path}")