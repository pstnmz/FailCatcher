"""
For each (backbone, dataset, setup, fold) pair, compute the difference between the
SHAP-mass-thresholded KNN AUROC and the full-feature AUROC (auroc_massX - auroc_all,
from knn_shap_normalization_comparison_corrupt3_test.csv) and plot it against the
model's balanced_accuracy on that fold (from the matching classification_results/
corruption_shifts/{flag}_{backbone}[_{setup}]_severity3.json file).

One figure per backbone, with one subplot per SHAP-mass threshold (20/50/70/80/90).

Usage:
    python plot_shap_mass_diff_vs_balanced_accuracy.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
CSV_PATH = RESULTS_DIR / 'knn_shap_normalization_comparison_corrupt3_test.csv'
CLASSIFICATION_DIR = RESULTS_DIR / 'classification_results' / 'corruption_shifts'
OUTPUT_DIR = RESULTS_DIR / 'figures' / 'shap_mass_diff_vs_balanced_accuracy'
MASS_THRESHOLDS = [20, 50, 70, 80, 90]
SEVERITY = 3


def classification_json_path(flag, backbone, setup):
    """standard setup has no suffix in the classification_results filenames."""
    suffix = '' if setup == 'standard' else f'_{setup}'
    return CLASSIFICATION_DIR / f'{flag}_{backbone}{suffix}_severity{SEVERITY}.json'


def load_balanced_accuracy_by_fold(path):
    with open(path) as f:
        data = json.load(f)
    return {entry['fold']: entry['balanced_accuracy'] for entry in data['per_fold']}


def build_plot_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    rows = []
    missing_files = set()
    for (flag, backbone, setup), group in df.groupby(['flag', 'backbone', 'setup']):
        json_path = classification_json_path(flag, backbone, setup)
        if not json_path.exists():
            missing_files.add(json_path.name)
            continue
        bacc_by_fold = load_balanced_accuracy_by_fold(json_path)
        for _, row in group.iterrows():
            bacc = bacc_by_fold.get(row['fold'])
            if bacc is None:
                continue
            for mass in MASS_THRESHOLDS:
                rows.append({
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
                              figsize=(4 * len(MASS_THRESHOLDS), 4 * len(backbones)),
                              sharey='row', sharex=True)
    for row, backbone in enumerate(backbones):
        backbone_df = plot_df[plot_df['backbone'] == backbone]
        for col, mass in enumerate(MASS_THRESHOLDS):
            ax = axes[row, col]
            mass_df = backbone_df[backbone_df['mass'] == mass]
            for flag in datasets:
                flag_df = mass_df[mass_df['flag'] == flag]
                ax.scatter(flag_df['balanced_accuracy'], flag_df['diff'], label=flag,
                           color=colors[flag], alpha=0.7, s=25)
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            if row == 0:
                ax.set_title(f'mass{mass}')
            if row == len(backbones) - 1:
                ax.set_xlabel('balanced_accuracy')
        axes[row, 0].set_ylabel(f'{backbone}\nauroc_massX - auroc_all')
    axes[0, -1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')
    fig.suptitle('SHAP-mass AUROC-F gain vs balanced accuracy (CS)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'mass_diff_vs_balanced_accuracy_cs.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    plot_df = build_plot_dataframe(CSV_PATH)
    assert not plot_df.empty, 'No data to plot -- check CSV/classification_results paths'
    plot_all_backbones(plot_df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
