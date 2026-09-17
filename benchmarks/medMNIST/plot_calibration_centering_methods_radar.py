"""
ID+CS AUROC_f radar figure comparing raw KNN-SHAP distances against both calibration
centering methods (midpoint and auroc-offset), reusing the dataset ordering/family-angle
helpers from plot_id_cs_radar_knn_best_shap.py, with three series instead of two:
  - Raw           : auroc_before (uncentered KNN-SHAP distance AUROC_f)
  - Midpoint      : auroc_after from the midpoint-centering full sweep
  - AUROC-offset  : auroc_after from the auroc-offset-centering full sweep

Usage:
    python plot_calibration_centering_methods_radar.py
    python plot_calibration_centering_methods_radar.py --mass-threshold 0.8
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_id_cs_radar_knn_best_shap import (  # noqa: E402
    ordered_dataset_keys, compute_family_angles, FAMILY_ANGLE_NUDGE,
    FAMILY_LABEL_RADIUS, DEFAULT_LABEL_RADIUS,
)

SWEEP_DIR = Path('/workspace/uq_benchmark_results/raw_results_calib_centered_full_sweep')
DEFAULT_MIDPOINT_CSV = SWEEP_DIR / 'midpoint_auroc.csv'
DEFAULT_AUROCOFFSET_CSV = SWEEP_DIR / 'aurocoffset_auroc.csv'
DEFAULT_STRATIFIED_AUROCOFFSET_CSV = Path('/workspace/uq_benchmark_results/raw_results_calib_centered_oct_stratified/aurocoffset_auroc_stratified.csv')
DEFAULT_OUTPUT = SWEEP_DIR / 'calibration_centering_methods_id_cs_auroc_f_radar.png'

METHODS = ['Raw', 'Midpoint', 'AUROC-offset']
METHOD_COLORS = {'Raw': 'tab:blue', 'Midpoint': 'tab:green', 'AUROC-offset': 'tab:red'}


def load_sweep_csv(csv_path, mass_threshold):
    df = pd.read_csv(csv_path, dtype={'mass_threshold': str})
    mass_threshold = str(mass_threshold)
    df = df[df['mass_threshold'] == mass_threshold]
    return df.groupby(['flag', 'backbone', 'setup', 'shift'], as_index=False).agg(
        auroc_before=('auroc_before', 'mean'),
        auroc_after=('auroc_after', 'mean'),
    )


def build_method_results(midpoint_csv, aurocoffset_csv, mass_threshold, stratified_aurocoffset_csv=None):
    """shift -> backbone -> dataset_key -> {'Raw', 'Midpoint', 'AUROC-offset'}."""
    midpoint_df = load_sweep_csv(midpoint_csv, mass_threshold)
    aurocoffset_df = load_sweep_csv(aurocoffset_csv, mass_threshold)

    results = {'id': defaultdict(dict), 'corruption': defaultdict(dict)}
    for _, row in midpoint_df.iterrows():
        shift_key = 'id' if row['shift'] == 'ID' else 'corruption'
        dataset_key = f"{row['flag']}_{row['setup']}"
        entry = results[shift_key][row['backbone']].setdefault(dataset_key, {})
        entry['Raw'] = float(row['auroc_before'])
        entry['Midpoint'] = float(row['auroc_after'])

    for _, row in aurocoffset_df.iterrows():
        shift_key = 'id' if row['shift'] == 'ID' else 'corruption'
        dataset_key = f"{row['flag']}_{row['setup']}"
        entry = results[shift_key][row['backbone']].setdefault(dataset_key, {})
        entry.setdefault('Raw', float(row['auroc_before']))
        entry['AUROC-offset'] = float(row['auroc_after'])

    if stratified_aurocoffset_csv is not None and stratified_aurocoffset_csv.exists():
        stratified_df = load_sweep_csv(stratified_aurocoffset_csv, mass_threshold)
        for _, row in stratified_df.iterrows():
            if row['flag'] != 'octmnist':
                continue
            shift_key = 'id' if row['shift'] == 'ID' else 'corruption'
            dataset_key = f"{row['flag']}_{row['setup']}"
            entry = results[shift_key][row['backbone']].setdefault(dataset_key, {})
            entry['AUROC-offset'] = float(row['auroc_after'])

    return {k: dict(v) for k, v in results.items()}


def plot_methods_radar(ax, model_results):
    """Draw the Raw/Midpoint/AUROC-offset AUROC_f radar for one model; returns (handles, labels)."""
    dataset_keys, families = ordered_dataset_keys(model_results)
    if not dataset_keys:
        return [], []

    angles, tick_labels, family_angles, family_names = compute_family_angles(families, dataset_keys)
    angles_closed = angles + angles[:1]

    for method in METHODS:
        values = [model_results[k].get(method, np.nan) for k in dataset_keys]
        values_closed = values + values[:1]
        ax.plot(angles_closed, values_closed, '-', linewidth=1.8, color=METHOD_COLORS[method],
                label=method, alpha=0.9)
        ax.plot(angles, values, 'o', color=METHOD_COLORS[method], markersize=6,
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--midpoint-csv', type=Path, default=DEFAULT_MIDPOINT_CSV)
    parser.add_argument('--aurocoffset-csv', type=Path, default=DEFAULT_AUROCOFFSET_CSV)
    parser.add_argument('--stratified-aurocoffset-csv', type=Path,
                        default=DEFAULT_STRATIFIED_AUROCOFFSET_CSV,
                        help='Optional stratified AUROC-offset CSV used to override matching rows.')
    parser.add_argument('--mass-threshold', default='all',
                        help="Mass threshold to plot (e.g. 'all', '0.2', '0.8'). Default: 'all'.")
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = build_method_results(args.midpoint_csv, args.aurocoffset_csv, args.mass_threshold,
                                   args.stratified_aurocoffset_csv)
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
            handles, labels = plot_methods_radar(ax, results[shift_key].get(model_name, {}))
            model_display = 'ResNet18' if model_name == 'resnet18' else 'ViT-B/16'
            title = f'{model_display}\nAUROC_f' if row == 0 else model_display
            ax.set_title(title, fontsize=13, fontweight='bold', y=1.08, x=-0.15, ha='left')
            if not all_handles and handles:
                all_handles, all_labels = handles, labels

    if all_handles:
        fig.legend(all_handles, all_labels, loc='lower center', ncol=3,
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
