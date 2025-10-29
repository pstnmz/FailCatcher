from utils import train_load_datasets_resnet as tr
from torchvision import transforms
import torch
import os, json, time
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# CLI args
parser = argparse.ArgumentParser(description="Train ResNet18 on medMNIST dataset")
parser.add_argument("--flag", type=str, default=None, help="Dataset flag to run (e.g. pneumoniamnist). If omitted runs default list)")
parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False, help="Use color images (True/False)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--use_randaugment", type=str2bool, nargs='?', const=True, default=False, help="Use RandAugment (True/False)")
parser.add_argument("--cuda", type=str, default="cuda:2", help="CUDA device string")
parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
args = parser.parse_args()

# override defaults with CLI args
cuda = args.cuda
num_epochs = args.num_epochs

default_flags = ['pneumoniamnist', 'breastmnist']
default_colors = [False, False]
default_batch_sizes = [128, 128]
default_use_randaugments = [False, False]
default_num_epochs = 100
default_cuda = "cuda:2"

# override defaults with CLI args
cuda = args.cuda
num_epochs = args.num_epochs
batch_size_arg = args.batch_size
use_randaugment_arg = args.use_randaugment
color_arg = args.color


# if --flag passed, run only that dataset in this process
if args.flag:
    flags = [args.flag]
    colors = [color_arg]
    batch_sizes = [batch_size_arg]
    use_randaugments = [use_randaugment_arg]
    num_epochs = num_epochs
else:
    flags = default_flags
    colors = default_colors
    batch_sizes = default_batch_sizes
    use_randaugments = default_use_randaugments
    num_epochs = default_num_epochs
    cuda = default_cuda

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
    [study_dataset_plain, _, test_dataset], [_, _, test_loader], info = tr.load_datasets(flag, color, size, transform_base, batch_size, cache_test=True, transform_test=transform_eval)

    # Decide loader/cache strategy
    use_monai = True       # set False to disable MONAI CacheDataset
    cache_rate = 1.0              # fraction of items to cache (0..1)
    num_workers = num_workers_val= 8            # or set explicit int / use NUM_WORKERS env var

    if use_randaugment:
        print(f'Using RandAugment with {randaugment_ops} ops and magnitude {randaugment_mag}')
        # cache is unnormalized; pass augment (which includes Normalize) and do NOT set normalize_transform
        train_loaders, val_loaders = tr.CV_train_val_loaders(
            None,
            study_dataset_plain,
            batch_size=batch_size,
            use_monai=use_monai,
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
            use_monai=use_monai,
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