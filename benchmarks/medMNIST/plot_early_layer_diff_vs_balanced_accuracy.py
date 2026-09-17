"""
Diff between auroc_combined and auroc_penultimate (KNN AUROC) as a function of
the fold's balanced accuracy, for the early-layer1 and early-layer3 KNN runs.

Row 1 = ID (results/classification_results/in_distribution), row 2 = CS
(results/classification_results/corruption_shifts, severity 3).
Column 1 = combined using the layer1 early features (CSVs with no suffix),
column 2 = combined using the layer3 early features (layer3 KNN computation is
still ongoing, so only rows shared between the layer1/layer3 CSVs are used).

Usage:
    python plot_early_layer_diff_vs_balanced_accuracy.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
UQ_RESULTS_DIR = Path(__file__).resolve().parents[2] / 'uq_benchmark_results'
CLASSIFICATION_DIR = RESULTS_DIR / 'classification_results'
ID_CLASS_DIR = CLASSIFICATION_DIR / 'in_distribution'
CS_CLASS_DIR = CLASSIFICATION_DIR / 'corruption_shifts'
OUTPUT_DIR = RESULTS_DIR / 'figures' / 'early_layer_boxplots'

ID_CSV_LAYER1 = RESULTS_DIR / 'knn_early_layer_combined.csv'
ID_CSV_LAYER3 = UQ_RESULTS_DIR / 'knn_early_layer_combined_layer3.csv'
CS_CSV_LAYER1 = RESULTS_DIR / 'knn_early_layer_combined_corrupt3_test.csv'
CS_CSV_LAYER3 = UQ_RESULTS_DIR / 'knn_early_layer_combined_layer3_corrupt3_test.csv'

# Same flag -> tab10 index mapping as plot_shap_mass_diff_vs_balanced_accuracy_cs.py
# and plot_early_layer_boxplot.py, so colors stay consistent across all figures.
CANONICAL_FLAGS = ['bloodmnist', 'breastmnist', 'dermamnist-e-id', 'octmnist',
                    'organamnist', 'pathmnist', 'pneumoniamnist', 'tissuemnist']


def build_merged_dataframe(csv_layer1, csv_layer3):
    """Combined KNN AUROC (layer1 & layer3) - penultimate AUROC, per flag/setup/fold."""
    df_layer1 = pd.read_csv(csv_layer1)
    df_layer3 = pd.read_csv(csv_layer3)
    merged = df_layer1.merge(
        df_layer3, on=['flag', 'backbone', 'setup', 'fold', 'n_classes'],
        suffixes=('', '_layer3'))
    assert (merged['auroc_penultimate'] - merged['auroc_penultimate_layer3']).abs().max() < 1e-9, \
        'auroc_penultimate mismatch between layer1 and layer3 CSVs'
    merged['diff_layer1'] = merged['auroc_combined'] - merged['auroc_penultimate']
    merged['diff_layer3'] = merged['auroc_combined_layer3'] - merged['auroc_penultimate']
    return merged[['flag', 'backbone', 'setup', 'fold', 'diff_layer1', 'diff_layer3']]


def id_json_path(flag, backbone, setup):
    return ID_CLASS_DIR / f'comprehensive_metrics_{flag}_{backbone}_{setup}.json'


def cs_json_path(flag, backbone, setup):
    """standard setup has no suffix in the corruption_shifts filenames."""
    suffix = '' if setup == 'standard' else f'_{setup}'
    return CS_CLASS_DIR / f'{flag}_{backbone}{suffix}_severity3.json'


def load_id_balanced_accuracy_by_fold(path):
    with open(path) as f:
        data = json.load(f)
    return {fold: entry['balanced_accuracy'] for fold, entry in enumerate(data['per_fold_metrics'])}


def load_cs_balanced_accuracy_by_fold(path):
    with open(path) as f:
        data = json.load(f)
    return {entry['fold']: entry['balanced_accuracy'] for entry in data['per_fold']}


def attach_balanced_accuracy(df, json_path_fn, loader_fn):
    rows = []
    missing_files = set()
    for (flag, backbone, setup), group in df.groupby(['flag', 'backbone', 'setup']):
        json_path = json_path_fn(flag, backbone, setup)
        if not json_path.exists():
            missing_files.add(json_path.name)
            continue
        bacc_by_fold = loader_fn(json_path)
        for _, row in group.iterrows():
            bacc = bacc_by_fold.get(row['fold'])
            if bacc is None:
                continue
            rows.append({**row.to_dict(), 'balanced_accuracy': bacc})
    if missing_files:
        print(f' [WARN] Missing classification result files, skipped: {sorted(missing_files)}')
    return pd.DataFrame(rows)


def plot_scatter(ax, df, colors, diff_col, title):
    datasets = sorted(df['flag'].unique())
    for flag in datasets:
        flag_df = df[df['flag'] == flag]
        ax.scatter(flag_df['balanced_accuracy'], flag_df[diff_col],
                   color=colors[flag], label=flag, alpha=0.7, s=25)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title(title)


def plot_all(id_df, cs_df, output_dir):
    cmap = plt.get_cmap('tab10')
    colors = {flag: cmap(i % 10) for i, flag in enumerate(CANONICAL_FLAGS)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey='row')
    plot_scatter(axes[0, 0], id_df, colors, 'diff_layer1', 'ID - layer1')
    plot_scatter(axes[0, 1], id_df, colors, 'diff_layer3', 'ID - layer3')
    plot_scatter(axes[1, 0], cs_df, colors, 'diff_layer1', 'CS - layer1')
    plot_scatter(axes[1, 1], cs_df, colors, 'diff_layer3', 'CS - layer3')

    for ax in axes[1, :]:
        ax.set_xlabel('balanced_accuracy')
    for ax in axes[:, 0]:
        ax.set_ylabel('AUROC-F gain')

    handles, labels = axes[0, 1].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axes[0, 1].legend(seen.values(), seen.keys(), loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')

    fig.suptitle('KNN SHAP penultimate + RADIOM gain over KNN_SHAP penultimate vs classification balanced accuracy')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'early_layer_diff_vs_balanced_accuracy.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    id_df = attach_balanced_accuracy(
        build_merged_dataframe(ID_CSV_LAYER1, ID_CSV_LAYER3), id_json_path, load_id_balanced_accuracy_by_fold)
    cs_df = attach_balanced_accuracy(
        build_merged_dataframe(CS_CSV_LAYER1, CS_CSV_LAYER3), cs_json_path, load_cs_balanced_accuracy_by_fold)
    assert not id_df.empty, 'No ID data to plot -- check CSV/classification_results paths'
    assert not cs_df.empty, 'No CS data to plot -- check CSV/classification_results paths'
    plot_all(id_df, cs_df, OUTPUT_DIR)


if __name__ == '__main__':
    main()
