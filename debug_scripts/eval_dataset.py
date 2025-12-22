#!/usr/bin/env python3
import sys, os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from benchmarks.medMNIST.utils.train_models_load_datasets import load_models, load_datasets
from FailCatcher.core.utils import evaluate_models_on_loader
from sklearn.metrics import confusion_matrix


def RepeatGrayToRGB(x):
    return x.repeat(3,1,1)


def main():
    flag = 'organamnist'
    model_backbone = 'vit_b_16'
    setup = 'DO'
    batch_size = 4000
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Transforms (grayscale -> repeat to 3 channels)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
        transforms.Lambda(lambda x: RepeatGrayToRGB(x)),
    ])

    print('Loading models...')
    models = load_models(flag, device, size=224, model_backbone=model_backbone, setup=setup)
    print('Loading datasets...')
    datasets, dataloaders, info = load_datasets(flag, color=False, im_size=224, transform=transform, batch_size=batch_size)
    _, calib_loader, _ = dataloaders
    print('Calibration loader size approx:', len(calib_loader.dataset))

    print('Evaluating ensemble on calibration loader...')
    y_true, y_scores, y_pred, correct_idx, incorrect_idx, indiv_scores = evaluate_models_on_loader(models, calib_loader, device)
    print(f'Accuracy on calibration set: {len(correct_idx)/len(y_true):.4f} ({len(correct_idx)} / {len(y_true)})')

    cm = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:\n', cm)

    os.makedirs(os.path.join(repo_root, 'uq_benchmark_results'), exist_ok=True)
    fig_path = os.path.join(repo_root, 'uq_benchmark_results', f'{flag}_calib_confusion_matrix.png')
    np_path = os.path.join(repo_root, 'uq_benchmark_results', f'{flag}_calib_confusion_matrix.npy')

    # Plot
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {flag} calibration')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(fig_path)
    np.save(np_path, cm)
    print('Saved confusion matrix image to', fig_path)
    print('Saved confusion matrix numpy to', np_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e)
        raise
