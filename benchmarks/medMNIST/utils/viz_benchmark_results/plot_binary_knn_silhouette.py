"""Compare train-class silhouette in fold-specific KNN PCA space with saved AUROC-F.

Run from any directory with the FailCatcher environment:
    python plot_binary_knn_silhouette.py --device cuda:0
    python plot_binary_knn_silhouette.py --dry-run
    python plot_binary_knn_silhouette.py --device cuda:0 --resume --allow-cs-mismatch

StandardScaler and PCA(90%) are fitted only on each fold's training set,
exactly as in KNNLatentMethod. Silhouette uses true diagnostic classes,
Euclidean distance, and all training samples of the fold (identical for ID and CS).
The optional CS mismatch override retains historical AUROC-F values and
flags points whose reconstructed predictions do not reproduce those values.
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).resolve().parent / 'results' / 'full_results'
SETUPS = ['standard', 'DA', 'DO', 'DADO']
DATASETS = ['bloodmnist', 'breastmnist', 'dermamnist-e-id', 'octmnist', 'organamnist',
            'pathmnist', 'pneumoniamnist', 'tissuemnist']
COLOR_DATASETS = {'dermamnist-e', 'pathmnist', 'bloodmnist'}


def discover_runs(results_dir, datasets, backbones, setups, shifts=('ID', 'CS')):
    runs = {}
    for directory, shift in [('in_distribution', 'ID'), ('corruption_shifts', 'CS')]:
        if shift not in shifts:
            continue
        for path in sorted((results_dir / directory).glob('uq_benchmark_*.json')):
            data = json.loads(path.read_text())
            setup = data.get('setup') or 'standard'
            if (data.get('flag') not in datasets or data.get('model_backbone') not in backbones
                    or setup not in setups):
                continue
            metrics = data.get('methods', {}).get('KNN_Raw', {})
            if not metrics.get('per_fold_metrics'):
                raise ValueError(f'No KNN_Raw per-fold metrics in {path}')
            corruption = data.get('corruption_info') or ''
            if shift == 'CS' and not re.fullmatch(r'severity[1-5]_test(?:_calib)?', corruption):
                raise ValueError(f'Unsupported corruption specification {corruption!r}: {path}')
            archive = path.with_name(path.name.replace('uq_benchmark_', 'all_metrics_', 1)).with_suffix('.npz')
            if not archive.exists():
                raise FileNotFoundError(archive)
            key = (data['flag'], data['model_backbone'], setup, shift, corruption)
            runs[key] = (path, archive, data)
    return sorted(runs.items())


def knn_projection(train_features, test_features):
    scaler = StandardScaler().fit(train_features)
    pca = PCA(n_components=0.9).fit(scaler.transform(train_features))
    return (pca.transform(scaler.transform(train_features)),
            pca.transform(scaler.transform(test_features)), pca.n_components_)


def save_rows(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        # Union of keys so rows resumed from older CSV schemas still write.
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_completed_rows(runs, output, allow_cs_mismatch):
    if not output.exists():
        return []
    sources = {str(paths[0]) for _, paths in runs}
    with output.open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['result_json'] in sources or row['dataset'] == 'PET/CT']
    keys = set()
    for row in rows:
        key = (row.get('model_name') or row['result_json'], int(row['fold']))
        if key in keys:
            raise ValueError(f'Duplicate completed fold: {key}')
        keys.add(key)
        row.setdefault('reconstructed_auroc_f', row['knn_auroc_f'])
        row.setdefault('auroc_reproduced', 'True')
        if row['auroc_reproduced'] != 'True' and not allow_cs_mismatch:
            raise ValueError('Resuming mismatched CS folds requires --allow-cs-mismatch')
    return rows


def compute_petct_rows(args):
    import torch
    from argparse import Namespace
    sys.path.insert(0, '/workspace')
    from external_petct import EXTERNAL_CSV, prepare_external_run
    from external_test import find_checkpoint
    from external_uq_msr_ensemble_zscore import align_predictions
    from ToolBox.methods.latent import extract_latent_space_and_compute_shap_importance
    from ToolBox.methods.latent import get_layer_from_model

    if args.k != 5:
        raise ValueError('The PET/CT overlay is defined for k=5 to match the train_k5 figure.')

    stem = 'external_DualStream_archDenseNet_do0.1_wd0.0001_bs8_bm1_les1'
    prediction_path = Path('/workspace/train_PET_CT/Inference') / f'{stem}_predictions.csv'
    feature_cache_dir = Path('/workspace/train_PET_CT/UQ/knn_shap_cache')
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)

    prepare_args = Namespace(
        group_name='DualStream_archDenseNet_do0.1_wd0.0001_bs8',
        log_dir='/workspace/train_PET_CT/Logger',
        external_csv=str(EXTERNAL_CSV),
        use_bm_mask=True,
        use_lesion_mask=True,
        output_dir='/workspace/train_PET_CT/Inference',
        num_workers=args.petct_workers,
        inference_batch_size=args.petct_batch_size,
        cache_dir='/workspace/Inference/cache_external_petct',
    )
    config, external_data, patient_ids, labels, _, load_model = prepare_external_run(
        prepare_args, find_checkpoint
    )
    fold_probabilities = align_predictions(prediction_path, patient_ids, labels, range(5))
    ensemble_predictions = (fold_probabilities.mean(axis=0) >= 0.5).astype(int)
    checkpoint_path = prediction_path.with_name(f'{stem}_checkpoints.csv')
    with checkpoint_path.open(newline='') as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    checkpoints = {int(row['fold']): row['checkpoint'] for row in checkpoint_rows}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    external_loader = None

    class PetctFeatureModel(torch.nn.Module):
        def __init__(self, lightning_model):
            super().__init__()
            self.classifier = lightning_model.model
            self.avgpool = self.classifier.class_layers.pool
            self.fc = self.classifier.class_layers.out

        def forward(self, images):
            return self.classifier(images)

    for fold in range(5):
        feature_cache = feature_cache_dir / f'shap_cache_{stem}_fold{fold}_fold0_bg1000.npz'
        if not feature_cache.is_file():
            raise FileNotFoundError(feature_cache)
        with np.load(feature_cache, allow_pickle=False) as cached:
            train_features = cached['features_train']
            train_labels = cached['labels_train'].reshape(-1)

        external_feature_cache = feature_cache_dir / f'{stem}_external_features_fold{fold}.npz'
        model = None
        if external_feature_cache.is_file():
            with np.load(external_feature_cache, allow_pickle=True) as cached:
                if not np.array_equal(cached['patient_ids'], patient_ids):
                    raise ValueError(f'External PET/CT cached feature order mismatch in fold {fold}')
                external_features = cached['features']
                cached_checkpoint = str(cached['checkpoint'].item()) if 'checkpoint' in cached else ''
            if cached_checkpoint and cached_checkpoint != checkpoints[fold]:
                raise ValueError(f'Cached PET/CT features use a different fold {fold} checkpoint')
            checkpoint = checkpoints[fold]
            print(f'PET/CT fold {fold}: reusing cached external embeddings', flush=True)
        else:
            if external_loader is None:
                from external_uq_knn_shap_petct import InMemoryVolumes, build_loader
                from Modules.dataloader_multichannel import get_transforms_multichannel
                from monai.data import PersistentDataset

                external_cache = PersistentDataset(
                    data=external_data,
                    transform=get_transforms_multichannel(step='baseline'),
                    cache_dir=prepare_args.cache_dir,
                )
                external_volumes = InMemoryVolumes(external_cache, args.petct_workers, 'external')
                external_loader = build_loader(
                    external_volumes, list(range(len(external_data))), config, args.petct_batch_size
                )
            model, checkpoint = load_model(None, fold)
            if checkpoint != checkpoints[fold]:
                raise ValueError(f'Loaded PET/CT fold {fold} checkpoint differs from saved predictions')
            backbone = PetctFeatureModel(model).to(device).eval()
            layer = get_layer_from_model(backbone, 'avgpool')
            extracted_features, external_labels, _, extracted_probabilities = \
                extract_latent_space_and_compute_shap_importance(
                    backbone, external_loader, device, layer, importance=False
                )
            if not np.array_equal(external_labels, labels):
                raise ValueError(f'External PET/CT label/order mismatch in fold {fold}')
            extracted_probabilities = np.asarray(extracted_probabilities).reshape(-1)
            extracted_predictions = extracted_probabilities >= 0.5
            saved_predictions = fold_probabilities[fold] >= 0.5
            if not np.array_equal(extracted_predictions, saved_predictions):
                mismatch_count = int(np.sum(extracted_predictions != saved_predictions))
                raise ValueError(f'Checkpoint predictions disagree with saved PET/CT fold {fold} '
                                 f'classes for {mismatch_count} patients')
            if not np.allclose(extracted_probabilities, fold_probabilities[fold], atol=2e-5, rtol=2e-5):
                max_difference = float(np.max(np.abs(extracted_probabilities - fold_probabilities[fold])))
                print(f'  Note: fold {fold} probabilities differ slightly from cache '
                      f'(max {max_difference:.3g}); class decisions match.', flush=True)
            external_features = extracted_features.numpy()
            np.savez_compressed(
                external_feature_cache, features=external_features, patient_ids=patient_ids,
                checkpoint=checkpoint,
            )

        train_projected, external_projected, dimensions = knn_projection(
            train_features, external_features
        )
        train_silhouette = float(silhouette_score(
            train_projected, train_labels, metric='euclidean'
        ))
        neighbor_model = NearestNeighbors(n_neighbors=args.k, metric='euclidean').fit(train_projected)
        distances = neighbor_model.kneighbors(external_projected)[0].mean(axis=1)
        auroc = float(roc_auc_score(ensemble_predictions != labels, distances))
        rows.append(dict(
            dataset='PET/CT', backbone='DualStream DenseNet', setup='BM + lesion',
            shift='external', corruption='', fold=fold, silhouette=train_silhouette,
            knn_auroc_f=auroc, k_selected=args.k, pca_dimensions=int(dimensions),
            n_test=len(labels), n_train=len(train_labels), n_classes=len(np.unique(train_labels)),
            result_json='', score_archive=str(feature_cache),
            reconstructed_auroc_f=auroc, auroc_reproduced=True,
            model_name=stem, prediction_csv=str(prediction_path), checkpoint=str(checkpoint),
        ))
        print(f'PET/CT fold {fold}: silhouette={train_silhouette:.4f}, '
              f'KNN AUROC-F={auroc:.4f}, k={args.k}', flush=True)
        if model is not None:
            model.cpu()

    return rows


def plot_rows(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr

    petct_rows = [row for row in rows if row['dataset'] == 'PET/CT']

    fig, axis = plt.subplots(figsize=(8, 6))
    datasets = [name for name in DATASETS if any(row['dataset'] == name for row in rows)]
    task_colors = dict(zip(datasets, plt.cm.tab10(range(len(datasets)))))
    backbone_markers = {'resnet18': 'o', 'vit_b_16': '^'}
    correlations = []
    for dataset, color in task_colors.items():
        for backbone, marker in backbone_markers.items():
            subset = [row for row in rows if row['dataset'] == dataset and row['backbone'] == backbone]
            for shift in ['ID', 'CS']:
                group = [row for row in subset if row['shift'] == shift]
                scores = [float(row['silhouette']) for row in group]
                aurocs = [float(row['knn_auroc_f']) for row in group]
                if shift == 'ID':
                    axis.scatter(scores, aurocs, color=color, marker=marker, s=45, alpha=0.8,
                                 edgecolors='white', linewidths=0.5)
                else:
                    axis.scatter(scores, aurocs, facecolors='none', edgecolors=color, marker=marker,
                                 s=45, alpha=0.9, linewidths=1.3)
                if len(group) >= 3 and np.ptp(scores) > 0 and np.ptp(aurocs) > 0:
                    statistic = spearmanr(scores, aurocs)
                    correlations.append(dict(dataset=dataset, backbone=backbone, shift=shift,
                                             n=len(group), spearman_rho=float(statistic.statistic)))
    if petct_rows:
        axis.scatter([float(row['silhouette']) for row in petct_rows],
                     [float(row['knn_auroc_f']) for row in petct_rows],
                     color='#171717', marker='X', s=100, linewidths=0.7)
    axis.axhline(0.5, color='0.65', linestyle='--', linewidth=0.8)
    axis.grid(alpha=0.15)
    axis.set_axisbelow(True)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel('Train class silhouette in KNN space (Euclidean)')
    axis.set_ylabel('Fold KNN AUROC-F')
    handles = [Line2D([], [], color=color, marker='s', linestyle='', label=dataset)
               for dataset, color in task_colors.items()]
    handles += [Line2D([], [], color='0.35', marker=marker, linestyle='', label=backbone)
                for backbone, marker in backbone_markers.items()]
    handles += [Line2D([], [], color='0.35', marker='o', linestyle='', label='ID'),
                Line2D([], [], color='0.35', marker='o', markerfacecolor='none', linestyle='',
                       label='CS')]
    if petct_rows:
        handles.append(Line2D([], [], color='#171717', marker='X', linestyle='', markersize=8,
                              label='PET + seg BMI prediction'))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output.with_suffix('.png'), dpi=200)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    output.with_suffix('.correlations.json').write_text(json.dumps(correlations, indent=2))


def compute_rows(runs, args):
    import torch
    from torch.utils.data import DataLoader
    from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
    from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation import dataset_utils
    from ToolBox.methods.latent import extract_latent_space_and_compute_shap_importance, get_layer_from_model

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    current = None
    output_csv = args.output / 'binary_knn_silhouette.csv'
    rows = load_completed_rows(runs, output_csv, args.allow_cs_mismatch) if args.resume else []
    completed = {(row['result_json'], int(row['fold'])) for row in rows}
    print(f'Reusing {len(rows)} completed folds', flush=True)
    for (dataset, backbone, setup, shift, corruption), (json_path, archive, data) in runs:
        if all((str(json_path), fold) in completed
               for fold in range(len(data['methods']['KNN_Raw']['per_fold_metrics']))):
            continue
        group = (dataset, backbone, setup)
        if group != current:
            base = 'dermamnist-e' if dataset == 'dermamnist-e-id' else dataset
            color = base in COLOR_DATASETS
            test_subset = 'id' if dataset == 'dermamnist-e-id' else 'all'
            corruption_flag = 'dermamnist' if 'dermamnist' in base else base
            models = tr.load_models(base, device=torch.device('cpu'), size=224,
                                    model_backbone=backbone, setup='' if setup == 'standard' else setup)
            transform, _ = dataset_utils.get_transforms(color, 224)
            (study, _, clean_test), _, _ = tr.load_datasets(
                base, color, 224, transform, args.batch_size, test_subset=test_subset)
            train_loaders = dataset_utils.create_cv_generator(
                seed=42, batch_size=args.batch_size, num_workers=args.workers)(study, models)
            train_features = {}
            train_labels_by_fold = {}
            train_silhouette = {}
            current = group
        test = clean_test
        if shift == 'CS':
            if not dataset_utils.MEDMNISTC_AVAILABLE:
                raise RuntimeError('CS requires the medmnistc corruption registry and the '
                                   'ImageMagick/MagickWand shared library. Refusing to use clean images.')
            severity = int(re.search(r'severity([1-5])', corruption).group(1))
            test = dataset_utils.apply_random_corruptions(
                clean_test, corruption_flag, severity, cache=True, seed=42)
        loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')
        metrics = data['methods']['KNN_Raw']
        with np.load(archive, allow_pickle=False) as saved:
            fold_scores = saved['KNN_Raw_per_fold']
        if len(models) != len(metrics['per_fold_metrics']) or fold_scores.shape != (len(models), len(test)):
            raise ValueError(f'Fold/sample mismatch: {json_path}')
        for fold, (model, fold_metrics) in enumerate(zip(models, metrics['per_fold_metrics'])):
            if (str(json_path), fold) in completed:
                continue
            print(f'{dataset} {backbone} {setup} {shift} fold {fold}', flush=True)
            model.to(device).eval()
            layer = get_layer_from_model(model, 'avgpool')
            if fold not in train_features:
                features, train_labels, _, _ = extract_latent_space_and_compute_shap_importance(
                    model, train_loaders[fold], device, layer, importance=False)
                train_features[fold] = features.numpy()
                train_labels_by_fold[fold] = train_labels
            features, labels, success, _ = extract_latent_space_and_compute_shap_importance(
                model, loader, device, layer, importance=False)
            train_projected, projected, dimensions = knn_projection(train_features[fold], features.numpy())
            if fold not in train_silhouette:
                n_train = len(train_labels_by_fold[fold])
                # Random subsample bounds the O(n^2) cost on large train sets (e.g. tissuemnist).
                train_silhouette[fold] = float(silhouette_score(
                    train_projected, train_labels_by_fold[fold], metric='euclidean',
                    sample_size=args.silhouette_samples if n_train > args.silhouette_samples else None,
                    random_state=42))
            score = train_silhouette[fold]
            if args.k is not None:
                # Fixed-k recomputation: nothing saved to reproduce.
                distances = NearestNeighbors(n_neighbors=args.k).fit(train_projected)\
                    .kneighbors(projected)[0].mean(axis=1)
                observed_auroc = expected_auroc = float(roc_auc_score(1 - success, distances))
            else:
                observed_auroc = roc_auc_score(1 - success, fold_scores[fold])
                expected_auroc = float(fold_metrics['auroc_f'])
            reproduced = bool(np.isclose(observed_auroc, expected_auroc, atol=1e-6, rtol=0))
            if not reproduced:
                message = (f'Cannot reproduce saved fold AUROC-F: {observed_auroc} != '
                           f'{expected_auroc} ({json_path}, fold {fold}). '
                           'Check checkpoint, corruption cache, and sample ordering.')
                if shift != 'CS' or not args.allow_cs_mismatch:
                    raise ValueError(message)
                print(f'  WARNING: {message} Retaining historical AUROC-F.', flush=True)
            selected_k = args.k
            if selected_k is None:
                selected_k = metrics.get('k_selected', fold_metrics.get('k_selected'))
                if selected_k is None:
                    raise ValueError(f'Missing selected k in {json_path}')
                if isinstance(selected_k, list):
                    selected_k = selected_k[fold]
            rows.append(dict(dataset=dataset, backbone=backbone, setup=setup, shift=shift,
                             corruption=corruption, fold=fold, silhouette=float(score),
                             knn_auroc_f=expected_auroc, k_selected=selected_k,
                             pca_dimensions=int(dimensions), n_test=len(labels),
                             n_train=len(train_labels_by_fold[fold]),
                             n_classes=len(np.unique(train_labels_by_fold[fold])),
                             result_json=str(json_path), score_archive=str(archive),
                             reconstructed_auroc_f=float(observed_auroc), auroc_reproduced=reproduced))
            save_rows(rows, output_csv)
            print(f'  silhouette={score:.4f}, AUROC-F={expected_auroc:.4f}, k={selected_k}', flush=True)
            model.cpu()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--k', type=int, default=None,
                        help='Recompute AUROC-F with this fixed k instead of the saved selected k')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--backbones', nargs='+', choices=['resnet18', 'vit_b_16'],
                        default=['resnet18', 'vit_b_16'])
    parser.add_argument('--setups', nargs='+', choices=SETUPS, default=SETUPS)
    parser.add_argument('--shifts', nargs='+', choices=['ID', 'CS'], default=['ID', 'CS'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--silhouette-samples', type=int, default=10000,
                        help='Subsample training points above this size for the silhouette')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse completed folds from the output CSV')
    parser.add_argument('--allow-cs-mismatch', action='store_true',
                        help='Retain and mark CS points despite historical AUROC-F reproduction mismatch')
    parser.add_argument('--petct-only', action='store_true',
                        help='Recompute and overlay the named five-fold PET/CT experiment')
    parser.add_argument('--petct-batch-size', type=int, default=4,
                        help='Inference batch size for PET/CT feature extraction')
    parser.add_argument('--petct-workers', type=int, default=48,
                        help='DataLoader workers for PET/CT cached-volume loading')
    args = parser.parse_args()
    if args.output is None:
        args.output = RESULTS.parent / 'figures' / (
            'binary_knn_silhouette_train' + (f'_k{args.k}' if args.k is not None else ''))
    if args.plot_only:
        with (args.output / 'binary_knn_silhouette.csv').open() as handle:
            rows = list(csv.DictReader(handle))
    elif args.petct_only:
        csv_path = args.output / 'binary_knn_silhouette.csv'
        if not csv_path.is_file():
            parser.error(f'Cannot find existing MedMNIST score table: {csv_path}')
        with csv_path.open() as handle:
            rows = [row for row in csv.DictReader(handle) if row['dataset'] != 'PET/CT']
        rows.extend(compute_petct_rows(args))
        save_rows(rows, csv_path)
    else:
        runs = discover_runs(args.results_dir, args.datasets, args.backbones, args.setups, args.shifts)
        if not runs:
            parser.error('No matching KNN result JSONs found')
        print(f'{len(runs)} runs, {sum(len(data[2]["methods"]["KNN_Raw"]["per_fold_metrics"]) for _, data in runs)} folds')
        if args.dry_run:
            for key, (_, _, data) in runs:
                print(key, 'k_selected=', data['methods']['KNN_Raw'].get('k_selected'))
            return
        rows = compute_rows(runs, args)
    plot_rows(rows, args.output / 'binary_knn_silhouette_vs_auroc')
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()