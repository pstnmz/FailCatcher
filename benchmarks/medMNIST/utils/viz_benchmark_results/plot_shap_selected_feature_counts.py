"""Plot SHAP-selected feature counts across mass thresholds for ID datasets."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
DEFAULT_CSV = RESULTS_DIR / 'knn_shap_results' / 'shap_selected_feature_counts_by_class_id.csv'
DEFAULT_OUTPUT = RESULTS_DIR / 'figures' / 'unified' / 'shap_selected_feature_counts.png'
MASS_THRESHOLDS = [20, 50, 70, 80, 90]
DATASET_ORDER = [
    'tissuemnist', 'breastmnist', 'pneumoniamnist', 'octmnist',
    'pathmnist', 'organamnist', 'bloodmnist', 'dermamnist-e-id',
]
DATASET_LABELS = {
    'tissuemnist': 'Tissue',
    'breastmnist': 'Breast',
    'pneumoniamnist': 'Pneumonia',
    'octmnist': 'OCT',
    'pathmnist': 'Path',
    'organamnist': 'OrganA',
    'bloodmnist': 'Blood',
    'dermamnist-e-id': 'Derma-E ID',
}
BACKBONES = [('resnet18', 'ResNet-18'), ('vit_b_16', 'ViT-B/16')]


def summarize_feature_counts(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['flag'].isin(DATASET_ORDER)]
    required = ['flag', 'backbone', 'mass_threshold_pct', 'n_selected_features']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    summary = (
        df.groupby(['flag', 'backbone', 'mass_threshold_pct'], as_index=False)['n_selected_features']
        .agg(mean='mean', std='std', n='count')
        .rename(columns={'mass_threshold_pct': 'mass'})
    )
    return summary


def plot_feature_counts(summary, output_path):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(MASS_THRESHOLDS)))
    x = np.arange(len(DATASET_ORDER))
    width = 0.15
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)

    for ax, (backbone, title) in zip(axes, BACKBONES):
        backbone_df = summary[summary['backbone'] == backbone]
        expected = len(DATASET_ORDER) * len(MASS_THRESHOLDS)
        if len(backbone_df) != expected:
            raise ValueError(f'{backbone}: expected {expected} dataset/threshold rows, found {len(backbone_df)}')

        for index, (mass, color) in enumerate(zip(MASS_THRESHOLDS, colors)):
            mass_df = backbone_df[backbone_df['mass'] == mass].set_index('flag').reindex(DATASET_ORDER)
            offset = (index - (len(MASS_THRESHOLDS) - 1) / 2) * width
            ax.bar(
                x + offset,
                mass_df['mean'],
                width,
                yerr=mass_df['std'],
                capsize=2.5,
                label=f'{mass}% mass',
                color=color,
                edgecolor='white',
                linewidth=0.5,
                error_kw={'elinewidth': 0.8, 'alpha': 0.8},
            )

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Selected features')
        ax.yaxis.grid(True, linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend(ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.08),
                   frameon=False, borderaxespad=0)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([DATASET_LABELS[flag] for flag in DATASET_ORDER])
    axes[-1].set_xlabel('Dataset')
    fig.suptitle('SHAP-selected latent features by cumulative importance threshold',
                 fontsize=20, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')
    print(f'Saved {pdf_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_feature_counts(summarize_feature_counts(args.csv), args.output)


if __name__ == '__main__':
    main()