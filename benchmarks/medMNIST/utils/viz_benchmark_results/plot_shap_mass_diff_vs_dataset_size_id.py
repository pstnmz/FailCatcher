"""
Test variant of plot_shap_mass_diff_vs_balanced_accuracy_id.py: same ID figure, but the x-axis is
each fold's KNN training-set size (read from the SHAP cache array headers) instead of balanced accuracy.

Usage:
    python plot_shap_mass_diff_vs_dataset_size_id.py
"""
import zipfile
from pathlib import Path

import numpy as np

from plot_shap_mass_diff_vs_balanced_accuracy_id import CSV_PATH, OUTPUT_DIR, build_plot_dataframe, plot_all_backbones

SHAP_CACHE_DIR = Path(__file__).resolve().parents[2] / 'uq_benchmark_results' / 'shap_cache'


def cached_train_size(flag, backbone, setup, fold):
    """Row count of the cached train features, read from the .npy header without loading the array."""
    setup_suffix = '' if setup == 'standard' else f'_{setup}'
    path = SHAP_CACHE_DIR / f'shap_cache_{flag}_{backbone}{setup_suffix}_fold{fold}_bg1000.npz'
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as z, z.open('features_train.npy') as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        return read_header(f)[0][0]


def main():
    plot_df = build_plot_dataframe(CSV_PATH)
    keys = plot_df[['flag', 'backbone', 'setup', 'fold']].drop_duplicates()
    sizes = {tuple(k): cached_train_size(*k) for k in keys.itertuples(index=False)}
    plot_df['train_size'] = [sizes[k] for k in zip(plot_df['flag'], plot_df['backbone'], plot_df['setup'], plot_df['fold'])]
    missing = plot_df['train_size'].isna().sum()
    if missing:
        print(f' [WARN] {missing} rows without a SHAP cache, skipped')
    plot_df = plot_df.dropna(subset=['train_size'])
    assert not plot_df.empty, 'No data to plot -- check CSV/SHAP cache paths'
    plot_all_backbones(plot_df, OUTPUT_DIR, x_col='train_size', x_label='training set size (per fold, log scale)',
                       log_x=True, title='SHAP-mass AUROC-F gain vs training set size (ID, all setups/folds)',
                       filename='mass_diff_vs_dataset_size_id.png')


if __name__ == '__main__':
    main()
