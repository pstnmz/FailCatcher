"""
Same analysis as plot_shap_mass_diff_vs_balanced_accuracy.py but for in-distribution (ID) data:
for each (backbone, dataset, setup, fold), compute auroc_massX - auroc_all (from
knn_shap_normalization_comparison.csv) and plot it against balanced_accuracy on that fold, read
from classification_results/in_distribution/comprehensive_metrics_{flag}_{backbone}_{setup}.json.

One figure per backbone, with one subplot per SHAP-mass threshold (20/50/70/80/90).

Usage:
    python plot_shap_mass_diff_vs_balanced_accuracy_id.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
CSV_PATH = RESULTS_DIR / 'knn_shap_normalization_comparison.csv'
CLASSIFICATION_DIR = RESULTS_DIR / 'classification_results' / 'in_distribution'
OUTPUT_DIR = RESULTS_DIR / 'figures' / 'shap_mass_diff_vs_balanced_accuracy_id'
MASS_THRESHOLDS = [20, 50, 70, 80, 90]


def classification_json_path(flag, backbone, setup):
    return CLASSIFICATION_DIR / f'comprehensive_metrics_{flag}_{backbone}_{setup}.json'


def load_balanced_accuracy_by_fold(path):
    with open(path) as f:
        data = json.load(f)
    # ID files list per-fold metrics in order with no explicit fold number -> index is the fold.
    return {fold: entry['balanced_accuracy'] for fold, entry in enumerate(data['per_fold_metrics'])}


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


def plot_all_backbones(plot_df, output_dir, x_col='balanced_accuracy', x_label='balanced_accuracy',
                       log_x=False, title='SHAP-mass AUROC-F gain vs balanced accuracy (ID, all setups/folds)',
                       filename='mass_diff_vs_balanced_accuracy_id.png'):
    backbones = sorted(plot_df['backbone'].unique())
    datasets = sorted(plot_df['flag'].unique())
    cmap = plt.get_cmap('tab10')
    colors = {flag: cmap(i % 10) for i, flag in enumerate(datasets)}

    fig, axes = plt.subplots(len(backbones), len(MASS_THRESHOLDS),
                              figsize=(4 * len(MASS_THRESHOLDS), 5.5 * len(backbones)),
                              sharey='row', sharex=True)
    for row, backbone in enumerate(backbones):
        backbone_df = plot_df[plot_df['backbone'] == backbone]
        for col, mass in enumerate(MASS_THRESHOLDS):
            ax = axes[row, col]
            mass_df = backbone_df[backbone_df['mass'] == mass]
            for flag in datasets:
                flag_df = mass_df[mass_df['flag'] == flag]
                ax.scatter(flag_df[x_col], flag_df['diff'], label=flag,
                           color=colors[flag], alpha=0.7, s=25)
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            if log_x:
                ax.set_xscale('log')
            if row == 0:
                ax.set_title(f'mass{mass}', fontsize=14)
            if row == len(backbones) - 1:
                ax.set_xlabel(x_label, fontsize=14)
        axes[row, 0].set_ylabel(f'{backbone}\nAUROC-F SHAP selected - AUROC-F all features', fontsize=14)
    axes[-1, -1].legend(loc='best', fontsize=10, ncol=2, columnspacing=0.6, handletextpad=0.2, borderpad=0.3)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    plot_df = build_plot_dataframe(CSV_PATH)
    assert not plot_df.empty, 'No data to plot -- check CSV/classification_results paths'
    plot_all_backbones(plot_df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
