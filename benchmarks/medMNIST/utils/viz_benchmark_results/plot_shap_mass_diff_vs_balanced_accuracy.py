"""
For each (backbone, dataset, setup, fold), plot auroc_massX - auroc_all against the fold's
balanced accuracy, overlaying in-distribution (ID) and covariate-shift (CS, severity 3) results
in the same figure (marker = shift, color = dataset).

One row per backbone, one column per SHAP-mass threshold (20/50/70/80/90).

Usage:
    python plot_shap_mass_diff_vs_balanced_accuracy.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
CLASSIFICATION_DIR = RESULTS_DIR / 'classification_results'
OUTPUT_DIR = RESULTS_DIR / 'figures' / 'shap_mass_diff_vs_balanced_accuracy'
MASS_THRESHOLDS = [20, 50, 70, 80, 90]
SEVERITY = 3
SHIFT_MARKERS = {'ID': 'o', 'CS': '^'}


def id_json_path(flag, backbone, setup):
    return CLASSIFICATION_DIR / 'in_distribution' / f'comprehensive_metrics_{flag}_{backbone}_{setup}.json'


def cs_json_path(flag, backbone, setup):
    suffix = '' if setup == 'standard' else f'_{setup}'
    return CLASSIFICATION_DIR / 'corruption_shifts' / f'{flag}_{backbone}{suffix}_severity{SEVERITY}.json'


def id_balanced_accuracy_by_fold(path):
    with open(path) as f:
        data = json.load(f)
    # ID files list per-fold metrics in order with no explicit fold number -> index is the fold.
    return {fold: entry['balanced_accuracy'] for fold, entry in enumerate(data['per_fold_metrics'])}


def cs_balanced_accuracy_by_fold(path):
    with open(path) as f:
        data = json.load(f)
    return {entry['fold']: entry['balanced_accuracy'] for entry in data['per_fold']}


SHIFTS = {
    'ID': (RESULTS_DIR / 'knn_shap_normalization_comparison.csv', id_json_path, id_balanced_accuracy_by_fold),
    'CS': (RESULTS_DIR / 'knn_shap_normalization_comparison_corrupt3_test.csv', cs_json_path, cs_balanced_accuracy_by_fold),
}


def build_plot_dataframe():
    rows = []
    missing_files = set()
    for shift, (csv_path, json_path_fn, load_bacc) in SHIFTS.items():
        df = pd.read_csv(csv_path)
        for (flag, backbone, setup), group in df.groupby(['flag', 'backbone', 'setup']):
            json_path = json_path_fn(flag, backbone, setup)
            if not json_path.exists():
                missing_files.add(f'{shift}: {json_path.name}')
                continue
            bacc_by_fold = load_bacc(json_path)
            for _, row in group.iterrows():
                bacc = bacc_by_fold.get(row['fold'])
                if bacc is None:
                    continue
                for mass in MASS_THRESHOLDS:
                    rows.append({
                        'shift': shift,
                        'flag': flag,
                        'backbone': backbone,
                        'setup': setup,
                        'fold': row['fold'],
                        'mass': mass,
                        'diff': row[f'auroc_mass{mass}'] - row['auroc_all'],
                        'balanced_accuracy': bacc,
                    })
    if missing_files:
        print(f' [WARN] Missing classification result files, skipped: {sorted(missing_files)}')
    return pd.DataFrame(rows)


def plot_all_backbones(plot_df, output_dir):
    backbones = sorted(plot_df['backbone'].unique())
    datasets = sorted(plot_df['flag'].unique())
    cmap = plt.get_cmap('tab10')
    colors = {flag: cmap(i % 10) for i, flag in enumerate(datasets)}

    fig, axes = plt.subplots(len(backbones), len(MASS_THRESHOLDS),
                             figsize=(4 * len(MASS_THRESHOLDS), 5.5 * len(backbones)),
                             sharey='row', sharex=True, squeeze=False)
    for row, backbone in enumerate(backbones):
        backbone_df = plot_df[plot_df['backbone'] == backbone]
        for col, mass in enumerate(MASS_THRESHOLDS):
            ax = axes[row, col]
            mass_df = backbone_df[backbone_df['mass'] == mass]
            for shift, marker in SHIFT_MARKERS.items():
                for flag in datasets:
                    sub = mass_df[(mass_df['flag'] == flag) & (mass_df['shift'] == shift)]
                    ax.scatter(sub['balanced_accuracy'], sub['diff'], color=colors[flag],
                               marker=marker, alpha=0.7, s=25)
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            if row == 0:
                ax.set_title(f'mass{mass}')
            if row == len(backbones) - 1:
                ax.set_xlabel('balanced_accuracy')
        axes[row, 0].set_ylabel(f'{backbone}\nAUROC-F SHAP selected - AUROC-F all features')

    handles = [Line2D([], [], marker='o', linestyle='', color=colors[flag], label=flag) for flag in datasets]
    handles += [Line2D([], [], marker=marker, linestyle='', color='black', label=shift)
                for shift, marker in SHIFT_MARKERS.items()]
    axes[0, -1].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')
    fig.suptitle('SHAP feature selection impact on AUROC-F')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'mass_diff_vs_balanced_accuracy_id_cs.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    plot_df = build_plot_dataframe()
    assert not plot_df.empty, 'No data to plot -- check CSV/classification_results paths'
    plot_all_backbones(plot_df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
