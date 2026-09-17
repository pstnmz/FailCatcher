"""
Boxplot comparison of auroc_penultimate vs auroc_early vs auroc_earlylayer3,
one subplot for ID results and one for CS (corruption shift, severity 3) results.

auroc_penultimate and auroc_early come from results/knn_early_layer_combined*.csv
(early = layer2). auroc_earlylayer3 comes from the layer3 KNN run in
uq_benchmark_results/knn_early_layer_combined_layer3*.csv, whose computation is
still ongoing, so only rows present in both files (matched on
flag/backbone/setup/fold) are used. auroc_penultimate is asserted identical
across both files as a sanity check.

Usage:
    python plot_early_layer_boxplot.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
UQ_RESULTS_DIR = Path(__file__).resolve().parents[2] / 'uq_benchmark_results'
OUTPUT_DIR = RESULTS_DIR / 'figures' / 'early_layer_boxplots'

ID_CSV_LAYER2 = RESULTS_DIR / 'knn_early_layer_combined.csv'
ID_CSV_LAYER3 = UQ_RESULTS_DIR / 'knn_early_layer_combined_layer3.csv'
CS_CSV_LAYER2 = RESULTS_DIR / 'knn_early_layer_combined_corrupt3_test.csv'
CS_CSV_LAYER3 = UQ_RESULTS_DIR / 'knn_early_layer_combined_layer3_corrupt3_test.csv'

# Same flag -> tab10 index mapping as plot_shap_mass_diff_vs_balanced_accuracy_cs.py,
# so colors stay consistent across plots even though this dataset is a subset.
CANONICAL_FLAGS = ['bloodmnist', 'breastmnist', 'dermamnist-e-id', 'octmnist',
                    'organamnist', 'pathmnist', 'pneumoniamnist', 'tissuemnist']
CATEGORIES = ['auroc_penultimate', 'auroc_early', 'auroc_earlylayer3']


def build_merged_dataframe(csv_layer2, csv_layer3):
    df_layer2 = pd.read_csv(csv_layer2)
    df_layer3 = pd.read_csv(csv_layer3)
    merged = df_layer2.merge(
        df_layer3, on=['flag', 'backbone', 'setup', 'fold', 'n_classes'],
        suffixes=('', '_layer3'))
    assert (merged['auroc_penultimate'] - merged['auroc_penultimate_layer3']).abs().max() < 1e-9, \
        'auroc_penultimate mismatch between layer2 and layer3 CSVs'
    merged = merged.rename(columns={'auroc_early_layer3': 'auroc_earlylayer3'})
    return merged[['flag', 'backbone', 'setup', 'fold'] + CATEGORIES]


def plot_boxplot(ax, df, colors, title):
    datasets = sorted(df['flag'].unique())
    rng = np.random.default_rng(0)

    box_data = [df[cat].values for cat in CATEGORIES]
    ax.boxplot(box_data, tick_labels=['SHAP_penultimate', 'RADIOM_layer1', 'RADIOM_layer3'], showfliers=False, widths=0.5)

    for i, cat in enumerate(CATEGORIES, start=1):
        for flag in datasets:
            values = df.loc[df['flag'] == flag, cat].values
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values,
                       color=colors[flag], label=flag, alpha=0.7, s=20)

    ax.set_ylabel('AUROC-F')
    ax.set_title(title)


def plot_all(id_df, cs_df, output_dir):
    cmap = plt.get_cmap('tab10')
    colors = {flag: cmap(i % 10) for i, flag in enumerate(CANONICAL_FLAGS)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    plot_boxplot(axes[0], id_df, colors, 'ID')
    plot_boxplot(axes[1], cs_df, colors, 'CS')

    # de-duplicate legend entries (one per flag, not per category/subplot)
    handles, labels = axes[1].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axes[1].legend(seen.values(), seen.keys(), loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')

    fig.suptitle('KNN AUROC-F: SHAP penultimate vs RADIOM layer1 vs RADIOM layer3')
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'early_layer_boxplot.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    id_df = build_merged_dataframe(ID_CSV_LAYER2, ID_CSV_LAYER3)
    cs_df = build_merged_dataframe(CS_CSV_LAYER2, CS_CSV_LAYER3)
    assert not id_df.empty, 'No shared rows between ID layer2 and layer3 CSVs'
    assert not cs_df.empty, 'No shared rows between CS layer2 and layer3 CSVs'
    print(f'ID: {len(id_df)} shared rows across flags: {sorted(id_df["flag"].unique())}')
    print(f'CS: {len(cs_df)} shared rows across flags: {sorted(cs_df["flag"].unique())}')
    plot_all(id_df, cs_df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
