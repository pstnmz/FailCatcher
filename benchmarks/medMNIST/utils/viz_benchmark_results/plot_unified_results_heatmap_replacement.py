def create_heatmap_figure(heatmap_data, results_dir, aggregation='mean'):
    """
    Create unified heatmap figure.
    
    Layout: 6 rows stacked vertically
    - Top 3: AUROC_f (ID, CS, PS/NCS)
    - Bottom 3: AUGRC (ID, CS, PS/NCS)
    Methods as rows, setups as columns
    
    Args:
        heatmap_data: Dict with keys 'id', 'corruption', 'population'
        results_dir: Main results directory
        aggregation: Aggregation strategy
        
    Returns:
        fig: matplotlib figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nGenerating heatmaps...")
    
    # Define shift configurations
    shift_configs = [
        ('id', 'IN DISTRIBUTION', 'in_distribution'),
        ('corruption', 'CORRUPTION SHIFTS', 'corruption_shifts'),
        ('population', 'POPULATION / NEW CLASS SHIFTS', 'population_shift')
    ]
    
    # Create figure with 6 rows (3 for AUROC_f, 3 for AUGRC)
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(6, 1, hspace=0.15, 
                         left=0.08, right=0.92, top=0.96, bottom=0.04)
    
    # Color scale ranges
    vmin_auroc, vmax_auroc = -0.2, 0.2
    vmin_augrc, vmax_augrc = -0.1, 0.1
    
    all_axes = []
    
    # Generate AUROC_f heatmaps (top 3 rows)
    for row_idx, (shift_key, shift_label, shift_name) in enumerate(shift_configs):
        data = heatmap_data.get(shift_key)
        
        if data is None:
            print(f"  ⚠ No heatmap data for {shift_key}")
            continue
        
        auroc_matrix = data['auroc_matrix']
        methods = data['methods']
        display_names = data['display_names']
        
        # Add Mean_Aggregation row
        auroc_agg_row = compute_mean_agg_row(
            results_dir, display_names, shift_name, 'auroc_f', aggregation
        )
        
        auroc_matrix_with_agg = np.vstack([auroc_matrix, auroc_agg_row])
        methods_with_agg = methods + ['⚡ Mean Agg']
        
        # Create heatmap
        ax = fig.add_subplot(gs[row_idx, 0])
        all_axes.append(ax)
        
        sns.heatmap(auroc_matrix_with_agg,
                    xticklabels=display_names,
                    yticklabels=methods_with_agg,
                    cmap='RdBu_r',
                    center=0,
                    vmin=vmin_auroc,
                    vmax=vmax_auroc,
                    annot=False,
                    cbar=False,
                    ax=ax)
        
        ax.set_title(f'{shift_label} - AUROC_f', fontsize=12, fontweight='bold', pad=5)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # Only show x-tick labels on bottom AUROC_f heatmap
        if row_idx < 2:
            ax.set_xticklabels([])
    
    # Generate AUGRC heatmaps (bottom 3 rows)
    for row_idx, (shift_key, shift_label, shift_name) in enumerate(shift_configs):
        data = heatmap_data.get(shift_key)
        
        if data is None:
            continue
        
        augrc_matrix = data['augrc_matrix']
        methods = data['methods']
        display_names = data['display_names']
        
        # Add Mean_Aggregation row
        augrc_agg_row = compute_mean_agg_row(
            results_dir, display_names, shift_name, 'augrc', aggregation
        )
        
        augrc_matrix_with_agg = np.vstack([augrc_matrix, augrc_agg_row])
        methods_with_agg = methods + ['⚡ Mean Agg']
        
        # Create heatmap
        ax = fig.add_subplot(gs[row_idx + 3, 0])
        all_axes.append(ax)
        
        sns.heatmap(augrc_matrix_with_agg,
                    xticklabels=display_names,
                    yticklabels=methods_with_agg,
                    cmap='RdBu_r',
                    center=0,
                    vmin=vmin_augrc,
                    vmax=vmax_augrc,
                    annot=False,
                    cbar=False,
                    ax=ax)
        
        ax.set_title(f'{shift_label} - AUGRC', fontsize=12, fontweight='bold', pad=5)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # Only show x-tick labels on bottom AUGRC heatmap
        if row_idx < 2:
            ax.set_xticklabels([])
    
    # Add colorbars
    from matplotlib.cm import RdBu_r
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    
    # AUROC_f colorbar (shared for top 3 heatmaps)
    cbar_ax1 = fig.add_axes([0.93, 0.66, 0.015, 0.28])
    norm_auroc = Normalize(vmin=vmin_auroc, vmax=vmax_auroc)
    cbar1 = mpl.colorbar.ColorbarBase(cbar_ax1, cmap=RdBu_r, norm=norm_auroc, orientation='vertical')
    cbar1.set_label('ΔAUROC_f', fontsize=12, fontweight='bold')
    
    # AUGRC colorbar (shared for bottom 3 heatmaps)
    cbar_ax2 = fig.add_axes([0.93, 0.08, 0.015, 0.28])
    norm_augrc = Normalize(vmin=vmin_augrc, vmax=vmax_augrc)
    cbar2 = mpl.colorbar.ColorbarBase(cbar_ax2, cmap=RdBu_r, norm=norm_augrc, orientation='vertical')
    cbar2.set_label('ΔAUGRC', fontsize=12, fontweight='bold')
    
    # Add main title
    fig.suptitle(f'Ensemble vs Per-Fold Differences ({aggregation.capitalize()})', 
                fontsize=18, fontweight='bold', y=0.99)
    
    return fig
