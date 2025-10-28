from utils import train_load_datasets_resnet as tr
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from monai.data import CacheDataset as MONAI_CacheDataset, ThreadDataLoader as MONAI_ThreadDataLoader
from torch.utils.data import DataLoader
import numpy as np
import torch
import os, json, time

flags = ['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'dermamnist-e', 'breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'dermamnist-e']
colors = [False, False, False, True, False, True, True, False, True, False, False, False, True, False, True, True, False, True]  # Colors for the flags
#batch_sizes = [32, 640, 128, 128, 640, 640, 640, 640, 128]  # Batch sizes for the flags
batch_sizes = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]  # Batch sizes for the flags
use_randaugments = [False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True]         # <- enable/disable RandAugment here

num_epochs = 100

cuda = 'cuda:2'
for flag, color, batch_size, use_randaugment in zip(flags, colors, batch_sizes, use_randaugments):
    print(f"Training on {flag} with color={color} and batch_size={batch_size}")
    
    randaugment_ops = 2            # number of ops per image
    randaugment_mag = 9            # magnitude (0-10 typical)
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    size = 224  # Image size for the models

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join("/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/runs", flag, f"resnet18_{size}_{timestamp}_randaug{int(use_randaugment)}_numepochs{num_epochs}_bs{batch_size}")
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)

    if color is True:
        normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        if use_randaugment:
            transform_base = transforms.Compose([transforms.ToTensor()])
            transform_train = transforms.Compose([
                transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag),
                transforms.ToTensor(),
                normalize,
            ])
            transform_eval = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            # cache normalized tensors (normalization performed once at cache time)
            transform_base = transforms.Compose([transforms.ToTensor(), normalize])
            transform_train = None
            transform_eval = transform_base

    else:
        normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        if use_randaugment:
            transform_base = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1))
            ])
            # runtime training augment only (no Normalize here)
            transform_train = transforms.Compose([
                transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag),
                transforms.ToTensor(),
                normalize
            ])
            transform_eval = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                normalize
            ])
        else:
            # cache normalized tensors (repeat then normalize once)
            transform_base = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1)), normalize])
            transform_train = None
            transform_eval = transform_base


    # Build datasets using transform_base (cached transform). When use_randaugment == False
    # transform_base already includes Normalize so cache will be normalized once.
    [study_dataset_plain, _, test_dataset], [_, _, test_loader], info = tr.load_datasets(flag, color, size, transform_base, batch_size, use_monai=True, cache_test=True, transform_test=transform_eval)

    # Decide loader/cache strategy
    use_monai_loader = True       # set False to force torch DataLoader
    use_cache = True              # set False to disable MONAI CacheDataset
    cache_rate = 1.0              # fraction of items to cache (0..1)
    num_workers = num_workers_val= 8            # or set explicit int / use NUM_WORKERS env var

    if use_randaugment:
        print(f'Using RandAugment with {randaugment_ops} ops and magnitude {randaugment_mag}')
        # cache is unnormalized; pass augment (which includes Normalize) and do NOT set normalize_transform
        train_loaders, val_loaders = tr.CV_train_val_loaders(
            None,
            study_dataset_plain,
            batch_size=batch_size,
            use_monai=use_monai_loader,
            use_cache=use_cache,
            cache_rate=cache_rate,
            train_augment_transform=transform_train,  # augment+Normalize at runtime
            num_workers=num_workers,
            pin_memory=True,
            prewarm_cache=True,         # pre-warm per-fold caches to avoid stalls during epochs                         
        )

    else:
        print('Not using RandAugment')
        # cache already normalized; no runtime augment or normalize needed
        train_loaders, val_loaders = tr.CV_train_val_loaders(
            None,
            study_dataset_plain,
            batch_size=batch_size,
            use_monai=use_monai_loader,
            use_cache=use_cache,
            cache_rate=cache_rate,
            train_augment_transform=None,
            num_workers=num_workers,
            pin_memory=True,
            prewarm_cache=True                         # pre-warm per-fold caches to avoid stalls during epochs
        )

    models = []
    results = []
    for i in range(5):
        print('MODEL ' + str(i))
        model, res = tr.train_resnet18(
            flag,
            train_loader=train_loaders[i],
            val_loader=val_loaders[i],
            test_loader=test_loader,
            num_epochs=num_epochs,
            learning_rate=0.001,
            device=cuda,
            random_seed=42,
            output_dir=exp_dir,
            run_name=f"fold_{i}",
            scheduler=True    
        )
        models.append(model)
        results.append(res)

    # Save per-fold results summary
    with open(os.path.join(exp_dir, "results_folds.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Evaluate ensemble and save
    ensemble_res = tr.evaluate_model(model=models, test_loader=test_loader, data_flag=flag, device=cuda, output_dir=exp_dir, prefix="ensemble")
    with open(os.path.join(exp_dir, "results_ensemble.json"), "w") as f:
        json.dump(ensemble_res, f, indent=2)

    # Save models
    for i, model in enumerate(models):
        path = os.path.join(exp_dir, f'resnet18_{flag}_224_augmented{i}.pt')
        tr.save_model(model, path=path)
        print(f"Saved: {path}")