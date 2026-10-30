"""
ID+CS AUROC_f radar figure (same dataset layout/orientation as
generate_radar_plots.create_radar_plot_on_axis), but showing only a handful of
series instead of all 10 confidence-function methods:
  - KNN             : raw KNN-distance AUROC_f
  - Best_of_7       : best of {KNN, MSR-S, MLS, MSR, GPS, TTA, DE} per dataset/setup/fold-mean
    - KNN_SHAP        : KNN-SHAP AUROC_f_mean from the KNN-SHAP benchmark JSON files

Layout: 3 rows (ID, CS, PS+NCS) x 2 columns (ResNet18 radar, ViT-B/16 radar). No histogram.

Usage:
    python plot_id_cs_radar_knn_best_shap.py
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

VIZ_UTILS_DIR = Path(__file__).resolve().parent / 'utils' / 'viz_benchmark_results'
sys.path.insert(0, str(VIZ_UTILS_DIR))
from generate_radar_plots import parse_results_directory  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
JSON_DIRS = {
    'id': RESULTS_DIR / 'jsons_results' / 'in_distribution',
    'corruption': RESULTS_DIR / 'jsons_results' / 'corruption_shifts',
    'population': RESULTS_DIR / 'jsons_results' / 'population_shifts',
    'new_class': RESULTS_DIR / 'jsons_results' / 'new_class_shifts',
}
KNN_SHAP_JSON_DIRS = {
    'id': RESULTS_DIR / 'knn_shap_results' / 'jsons_results' / 'in_distribution',
    'corruption': RESULTS_DIR / 'knn_shap_results' / 'jsons_results' / 'corruption_shifts',
    'population': RESULTS_DIR / 'knn_shap_results' / 'jsons_results' / 'population_shifts',
    'new_class': RESULTS_DIR / 'knn_shap_results' / 'jsons_results' / 'new_class_shifts',
}
OUTPUT_DIR = RESULTS_DIR / 'figures' / 'unified'
OCT_CALIBRATION_CSV = RESULTS_DIR / 'knn_shap_results' / 'raw_results_calib_centered_oct_stratified' / 'aurocoffset_auroc_stratified.csv'
CALIBRATED_CSV = Path('/workspace/uq_benchmark_results/raw_results_calib_centered/calibration_centered_auroc.csv')
DEFAULT_CENTERED_OUTPUT = CALIBRATED_CSV.parent / 'calibration_centered_id_cs_auroc_f_radar.png'

BEST_OF_7_CANDIDATES = ['KNN_Raw', 'MSR_calibrated', 'MLS', 'MSR', 'GPS', 'TTA', 'Ensembling']
METHOD_NAMES = ['KNN', 'Best_of_7', 'KNN_SHAP']
METHOD_NAMES.append('KNN_SHAP_CALIBRATED')
METHOD_DISPLAY = {
    'KNN': 'KNN', 'Best_of_7': 'Top1-CSF', 'KNN_SHAP': 'KNN_SHAP',
    'KNN_SHAP_CALIBRATED': 'KNN-SHAP (Test-matched calibration)',
}
METHOD_COLORS = {
    'KNN': 'tab:blue', 'Best_of_7': 'tab:orange', 'KNN_SHAP': 'tab:green',
    'KNN_SHAP_CALIBRATED': 'tab:red',
}
CENTERED_METHOD_COLORS = {'Raw': 'tab:blue', 'Centered': 'tab:green'}
CENTERED_METHOD_DISPLAY = {'Raw': 'Raw KNN', 'Centered': 'Centered KNN'}

# Same manual family ordering used by generate_radar_plots.py, kept for visual consistency
# with the other unified figures. With theta_offset=pi/2 and theta_direction=-1 (clockwise
# from top), this places: tissue=N, breast=NE, pneumonia=E, oct=SE, path=S, organa=SW,
# blood=W, derma-e-id=NW.
PREFERRED_DATASET_ORDER = [
    'tissuemnist', 'breastmnist', 'pneumoniamnist', 'octmnist',
    'pathmnist', 'organamnist', 'bloodmnist', 'dermamnist-e-id',
]
PREFERRED_PS_NCS_DATASET_ORDER = [
    'dermamnist-e-external', 'amos2022', 'hmu-crc', 'midog', 'new_class_amos2022',
]
SETUP_ORDER = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}

# Per-family angular nudge and label radius overrides, copied from
# generate_radar_plots.create_radar_plot_on_axis (metric='auroc_f' branch) so both figures
# orient/label datasets identically.
FAMILY_ANGLE_NUDGE = {
    'bloodmnist': 0.35, 'dermamnist-e-id': 0.18, 'pneumoniamnist': 0.35,
    'organamnist': 0.1, 'breastmnist': 0.07, 'octmnist': -0.05,
    'new_class_midog': 0.2, 'new_class_amos2022': 0.1, 'hmu-crc': -0.1
}
FAMILY_LABEL_RADIUS = {
    'tissuemnist': 1.12, 'octmnist': 1.12, 'pneumoniamnist': 1.1,
    'bloodmnist': 1.1, 'pathmnist': 1.12, 'organamnist': 1.17, 'new_class_midog': 1.2
}
DEFAULT_LABEL_RADIUS = 1.16


def load_knn_shap_json_means(json_dir):
    """Return (flag, backbone, setup) -> KNN-SHAP ``auroc_f_mean``."""
    means = {}
    for json_path in Path(json_dir).glob('*.json'):
        try:
            with json_path.open() as handle:
                data = json.load(handle)
            method = data.get('methods', {}).get('KNN_SHAP', {}).get('auroc_f_mean')
            if method is None:
                continue
            setup = data.get('setup') or 'standard'
            key = (data.get('flag'), data.get('model_backbone'), setup)
            means[key] = float(method)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return means


def load_oct_calibration_means(csv_path):
    """Return (backbone, setup, shift) -> mean auroc_after over folds 0-3."""
    df = pd.read_csv(csv_path)
    df = df[(df['flag'] == 'octmnist') & (df['fold'] < 4)]
    return df.groupby(['backbone', 'setup', 'shift'])['auroc_after'].mean().to_dict()


def build_reduced_results(shift_key):
    """model -> dataset_key -> {'KNN', 'Best_of_7', 'KNN_SHAP'}."""
    raw_results = parse_results_directory(JSON_DIRS[shift_key], metric='auroc_f')
    knn_shap_json_means = load_knn_shap_json_means(KNN_SHAP_JSON_DIRS[shift_key])
    calibration_means = {}
    if shift_key in ('id', 'corruption') and OCT_CALIBRATION_CSV.exists():
        calibration_means = load_oct_calibration_means(OCT_CALIBRATION_CSV)
    calibration_shift = 'ID' if shift_key == 'id' else 'CS'

    reduced = defaultdict(dict)
    for model, datasets in raw_results.items():
        for dataset_key, methods in datasets.items():
            flag, setup = dataset_key.rsplit('_', 1)
            knn = methods.get('KNN_Raw', np.nan)
            candidates = [methods[m] for m in BEST_OF_7_CANDIDATES if m in methods]
            best_of_7 = max(candidates) if candidates else np.nan
            entry = {
                'KNN': knn,
                'Best_of_7': best_of_7,
                'KNN_SHAP': knn_shap_json_means.get((flag, model, setup), np.nan),
                'KNN_SHAP_CALIBRATED': calibration_means.get(
                    (model, setup, calibration_shift), np.nan
                ) if flag == 'octmnist' else np.nan,
            }
            result_dataset_key = f'new_class_{dataset_key}' if shift_key == 'new_class' else dataset_key
            reduced[model][result_dataset_key] = entry
    return dict(reduced)


def ordered_dataset_keys(model_results, preferred_order=PREFERRED_DATASET_ORDER):
    families = defaultdict(list)
    for dataset_key in model_results:
        base = dataset_key.rsplit('_', 1)[0]
        families[base].append(dataset_key)
    for base in families:
        families[base].sort(key=lambda k: SETUP_ORDER.get(k.rsplit('_', 1)[-1], 99))

    keys = []
    for base in preferred_order:
        keys.extend(families.get(base, []))
    for base in sorted(families):
        if base not in preferred_order:
            keys.extend(families[base])
    return keys, families


def compute_family_angles(families, dataset_keys):
    """One angular slot per dataset family, subdivided evenly across its setups."""
    num_families = len(families)
    angle_per_family = 2 * np.pi / num_families
    angles, labels, family_angles, family_names = [], [], [], []

    family_idx = 0
    i = 0
    while i < len(dataset_keys):
        base = dataset_keys[i].rsplit('_', 1)[0]
        family_datasets = families[base]
        family_center = family_idx * angle_per_family
        family_angles.append(family_center)
        family_names.append(base)

        span = angle_per_family * 0.85
        size = len(family_datasets)
        for j, ds_key in enumerate(family_datasets):
            offset = (j - (size - 1) / 2) * (span / size) if size > 1 else 0.0
            angles.append((family_center + offset) % (2 * np.pi))
            setup = ds_key.rsplit('_', 1)[-1]
            labels.append('S' if setup == 'standard' else setup)

        family_idx += 1
        i += size
    return angles, labels, family_angles, family_names


def plot_radar(ax, model_results, preferred_order=PREFERRED_DATASET_ORDER):
    """Draw the reduced-method AUROC_f radar for one model; returns (handles, labels)."""
    dataset_keys, families = ordered_dataset_keys(model_results, preferred_order)
    if not dataset_keys:
        return [], []

    angles, tick_labels, family_angles, family_names = compute_family_angles(families, dataset_keys)
    angles_closed = angles + angles[:1]

    for method in METHOD_NAMES:
        values = [model_results[k].get(method, np.nan) for k in dataset_keys]
        values_closed = values + values[:1]
        if method != 'KNN_SHAP_CALIBRATED':
            ax.plot(angles_closed, values_closed, '-', linewidth=1.5, color=METHOD_COLORS[method],
                    label=METHOD_DISPLAY[method], alpha=0.85)
        else:
            ax.plot(angles_closed, values_closed, '-', linewidth=1.5, color=METHOD_COLORS[method],
                label=METHOD_DISPLAY[method], alpha=0.85)
            ax.plot([], [], 'o', color=METHOD_COLORS[method],
                    markersize=6, markeredgewidth=1, markeredgecolor='white', alpha=0.85)
        ax.plot(angles, values, 'o', color=METHOD_COLORS[method], markersize=6,
                markeredgewidth=1, markeredgecolor='white', alpha=0.85)

    ax.set_xticks(angles)
    ax.set_xticklabels(tick_labels, size=8, fontweight='medium')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, 2 * np.pi)

    y_min, y_max = 0.4, 1.0
    ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.1))
    ax.set_yticklabels([f'{y:g}' for y in np.arange(y_min, y_max + 0.05, 0.1)], size=10)
    ax.set_ylim(y_min, y_max)
    ax.set_rlabel_position(230)

    for angle, name in zip(family_angles, family_names):
        custom_angle = angle + FAMILY_ANGLE_NUDGE.get(name, 0.0)
        radius = FAMILY_LABEL_RADIUS.get(name, DEFAULT_LABEL_RADIUS)
        display_name = name.replace('mnist', '')
        if name == 'new_class_amos2022':
            display_name = 'new class\namos2022'
        elif name == 'new_class_midog':
            display_name = 'new class\nmidog'
        ax.text(custom_angle, radius, display_name, ha='center', va='center',
                size=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.grid(True, linewidth=1.0, alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def load_centered_summary(csv_path, mass_threshold='all'):
    df = pd.read_csv(csv_path)
    if mass_threshold != 'all':
        mass_threshold = float(mass_threshold)
        df = df[np.isclose(df['mass_threshold'], mass_threshold)]
    grouped = df.groupby(['flag', 'backbone', 'setup', 'shift'], as_index=False).agg(
        auroc_before=('auroc_before', 'mean'),
        auroc_after=('auroc_after', 'mean'),
    )

    results = {'id': defaultdict(dict), 'corruption': defaultdict(dict)}
    for _, row in grouped.iterrows():
        shift_key = 'id' if row['shift'] == 'ID' else 'corruption'
        dataset_key = f"{row['flag']}_{row['setup']}"
        results[shift_key][row['backbone']][dataset_key] = {
            'Raw': float(row['auroc_before']),
            'Centered': float(row['auroc_after']),
        }
    return {k: dict(v) for k, v in results.items()}


def plot_centered_radar(ax, model_results):
    """Draw a centered-vs-raw AUROC_f radar for one model."""
    dataset_keys, families = ordered_dataset_keys(model_results)
    if not dataset_keys:
        return [], []

    angles, tick_labels, family_angles, family_names = compute_family_angles(families, dataset_keys)
    angles_closed = angles + angles[:1]

    for method in ['Raw', 'Centered']:
        values = [model_results[k].get(method, np.nan) for k in dataset_keys]
        values_closed = values + values[:1]
        ax.plot(angles_closed, values_closed, '-', linewidth=1.8,
                color=CENTERED_METHOD_COLORS[method], label=CENTERED_METHOD_DISPLAY[method], alpha=0.9)
        ax.plot(angles, values, 'o', color=CENTERED_METHOD_COLORS[method], markersize=6,
                markeredgewidth=1, markeredgecolor='white', alpha=0.9)

    ax.set_xticks(angles)
    ax.set_xticklabels(tick_labels, size=8, fontweight='medium')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, 2 * np.pi)

    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.set_yticklabels([f'{y:g}' for y in np.arange(0.4, 1.01, 0.1)], size=10)
    ax.set_rlabel_position(230)

    for angle, name in zip(family_angles, family_names):
        custom_angle = angle + FAMILY_ANGLE_NUDGE.get(name, 0.0)
        radius = FAMILY_LABEL_RADIUS.get(name, DEFAULT_LABEL_RADIUS)
        ax.text(custom_angle, radius, name.replace('mnist', ''), ha='center', va='center',
                size=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.grid(True, linewidth=1.0, alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def main():
    parser = argparse.ArgumentParser(description='ID/CS AUROC_f radar plots for raw vs calibration-centered KNN results.')
    parser.add_argument('--csv', type=Path, default=None,
                        help='Optional centered-summary CSV. If omitted, plot the JSON benchmark results.')
    parser.add_argument('--mass-threshold', default='all',
                        help="Mass threshold to plot (e.g. 'all', '0.2', '0.8'). Default: 'all'.")
    parser.add_argument('--output', type=Path, default=None,
                        help='Output PNG path. Default: calibration_centered_id_cs_auroc_f_radar.png in the CSV directory.')
    args = parser.parse_args()

    if args.csv is not None and args.csv.exists():
        results = load_centered_summary(args.csv, args.mass_threshold)
        model_names = ['resnet18', 'vit_b_16']
        row_labels = {'id': 'ID', 'corruption': 'CS'}

        fig = plt.figure(figsize=(11, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.15, left=0.08, right=0.96, top=0.88, bottom=0.10)
        all_handles, all_labels = [], []

        for row, shift_key in enumerate(['id', 'corruption']):
            fig.text(0.03, 0.68 if row == 0 else 0.28, row_labels[shift_key],
                     ha='center', va='center', fontsize=16, fontweight='bold', rotation=90)
            for col, model_name in enumerate(model_names):
                ax = fig.add_subplot(gs[row, col], projection='polar')
                handles, labels = plot_centered_radar(ax, results[shift_key].get(model_name, {}))
                model_display = 'ResNet18' if model_name == 'resnet18' else 'ViT-B/16'
                title = f'{model_display}\nAUROC_f' if row == 0 else model_display
                ax.set_title(title, fontsize=13, fontweight='bold', y=1.08, x=-0.15, ha='left')
                if not all_handles and handles:
                    all_handles, all_labels = handles, labels

        if all_handles:
            fig.legend(all_handles, all_labels, loc='lower center', ncol=2,
                       fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

        out_path = args.output if args.output is not None else (args.csv.parent / 'calibration_centered_id_cs_auroc_f_radar.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {shift: build_reduced_results(shift) for shift in ['id', 'corruption']}
    ps_ncs_results = defaultdict(dict)
    for shift in ['population', 'new_class']:
        for model, datasets in build_reduced_results(shift).items():
            ps_ncs_results[model].update(datasets)
    results['ps_ncs'] = dict(ps_ncs_results)
    model_names = ['resnet18', 'vit_b_16']
    row_specs = [
        ('id', 'ID', PREFERRED_DATASET_ORDER),
        ('corruption', 'CS', PREFERRED_DATASET_ORDER),
        ('ps_ncs', 'PS + NCS', PREFERRED_PS_NCS_DATASET_ORDER),
    ]

    fig = plt.figure(figsize=(11, 17))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.15, left=0.08, right=0.96, top=0.90, bottom=0.10)

    all_handles, all_labels = [], []
    for row, (shift_key, row_label, preferred_order) in enumerate(row_specs):
        fig.text(0.03, [0.79, 0.49, 0.19][row], row_label,
                  ha='center', va='center', fontsize=16, fontweight='bold', rotation=90)

        shift_results = results[shift_key]
        for col, model_name in enumerate(model_names):
            ax = fig.add_subplot(gs[row, col], projection='polar')
            handles, labels = plot_radar(ax, shift_results.get(model_name, {}), preferred_order)
            model_display = 'ResNet18' if model_name == 'resnet18' else 'ViT-B/16'
            title = f'{model_display}\nAUROC_f' if row == 0 else ''
            ax.set_title(title, fontsize=13, fontweight='bold', y=1.08, x=-0.15, ha='left')
            if not all_handles and handles:
                all_handles, all_labels = handles, labels

    if all_handles:
        fig.legend(all_handles, all_labels, loc='center', ncol=1,
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.36))

    out_path = OUTPUT_DIR / 'id_cs_auroc_f_radar_knn_best_shap.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
