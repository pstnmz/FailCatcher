"""
Greedy policy search for uncertainty quantification.
Select augmentation policies that maximize ROC AUC between correct/incorrect predictions.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from ..visualization.plots import roc_curve_UQ_method_computation


# ============================================================================
# MAIN API
# ============================================================================

def perform_greedy_policy_search(
    npz_dir, good_idx, bad_idx, max_iterations=50, num_workers=1, 
    num_searches=10, top_k=5, plot=True, method='top_k_policies', seed=None
):
    """
    Perform greedy search to find best augmentation policies for UQ.
    
    Args:
        npz_dir: Directory containing .npz files with predictions
        good_idx: Indices of correctly predicted samples
        bad_idx: Indices of incorrectly predicted samples
        max_iterations: Max policies to select per search
        num_workers: Parallel workers (>1 for multiprocessing)
        num_searches: Number of parallel greedy searches with random starts
        top_k: Number of top policy groups to return
        plot: Whether to plot AUC curves
        method: 'top_k_policies' returns top-k groups, 'top_policies' returns single best
        seed: Random seed for reproducibility
    
    Returns:
        list: Selected policy filenames
            - If method='top_k_policies': [[policy1, policy2, ...], [policy3, ...], ...]
            - If method='top_policies': [policy1, policy2, ...]
    
    Example:
        >>> best_policies = perform_greedy_policy_search(
        ...     'gps_augment/224x224/breastmnist_calibration_set',
        ...     correct_idx, incorrect_idx,
        ...     max_iterations=50, num_workers=90, top_k=3, seed=42
        ... )
        >>> # Returns: [['N2_M45___7_...npz', 'N2_M45___13_...npz'], [...], [...]]
    """
    print('Loading predictions...')
    all_preds, all_keys = load_npz_files_for_greedy_search(npz_dir)
    # shapes: [num_policies, num_samples, num_classes]
    num_samples = all_preds.shape[1]
    num_classes = all_preds.shape[2]
    
    print(f"Loaded predictions shape: {all_preds.shape} (policies, samples, classes={num_classes})")

    # Validate indices
    if len(good_idx) or len(bad_idx):
        max_idx = max(
            int(max(good_idx)) if len(good_idx) else -1,
            int(max(bad_idx)) if len(bad_idx) else -1
        )
        if max_idx >= num_samples:
            print(f"Warning: indices exceed predictions length ({max_idx} >= {num_samples}). Clipping.")
            good_idx = [i for i in good_idx if i < num_samples]
            bad_idx = [i for i in bad_idx if i < num_samples]
            if not good_idx or not bad_idx:
                raise ValueError(
                    "After clipping, good_idx or bad_idx is empty. "
                    "Ensure .npz predictions match dataset ordering."
                )

    # Run greedy search
    selected_policies, results = select_greedily_on_ens(
        all_preds, good_idx, bad_idx, all_keys,
        search_set_len=num_samples,
        select_only=max_iterations,
        num_workers=num_workers,
        num_searches=num_searches,
        top_k=top_k,
        method=method,
        seed=seed
    )

    # Convert policy indices to filenames
    if isinstance(selected_policies, list) and all(isinstance(p, list) for p in selected_policies):
        selected_policy_names = [
            [all_keys[i] for i in policy_group]
            for policy_group in selected_policies
        ]
    else:
        selected_policy_names = [all_keys[i] for i in selected_policies]

    if plot:
        plot_auc_curves(results)

    print(f"Best GPS augmentations found: {selected_policy_names}")
    return selected_policy_names


# ============================================================================
# CORE GREEDY SEARCH LOGIC
# ============================================================================

def select_greedily_on_ens(
    all_preds, good_idx, bad_idx, keys, search_set_len, select_only=50,
    num_workers=1, num_searches=10, top_k=5, method='top_k_policies', seed=None
):
    """
    Perform multiple parallel greedy searches with different random starts.
    
    Args:
        all_preds: Predictions array (num_policies, num_samples, num_classes)
        good_idx: Indices of correct predictions
        bad_idx: Indices of incorrect predictions
        keys: Policy filenames corresponding to predictions
        search_set_len: Number of samples to use for search
        select_only: Max policies per search
        num_workers: Parallel workers
        num_searches: Number of independent searches
        top_k: Top groups to return
        method: 'top_k_policies' or 'top_policies'
        seed: Random seed
    
    Returns:
        tuple: (selected_policies, all_search_results)
            - selected_policies: List of policy indices or list of lists
            - all_search_results: List of (best_metric, policy_indices, auc_history)
    """
    val_preds = np.copy(all_preds[:, :search_set_len, :])

    # Prepare random starts (deterministic if seed provided)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    initial_augmentations = [
        int(rng.choice(range(val_preds.shape[0])))
        for _ in range(num_searches)
    ]

    def _run_sequential():
        """Fallback sequential execution."""
        out = []
        for init_aug in initial_augmentations:
            out.append(greedy_search(init_aug, val_preds, good_idx, bad_idx, select_only))
        return out

    results = []
    if num_workers and num_workers > 1:
        try:
            with mp.Pool(processes=min(num_workers, mp.cpu_count() or num_workers)) as pool:
                results = pool.starmap(
                    greedy_search,
                    [(init_aug, val_preds, good_idx, bad_idx, select_only) 
                     for init_aug in initial_augmentations]
                )
        except Exception as e:
            print(f"Parallel greedy_search failed: {repr(e)}; falling back to sequential.")
            results = _run_sequential()
    else:
        results = _run_sequential()

    if not results:
        raise RuntimeError("No greedy_search results produced. Check input shapes and indices.")

    # Select based on method
    if method == 'top_k_policies':
        # Return top-k best searches as separate groups
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        policies = [
            sorted_results[i][1]
            for i in range(min(top_k, len(sorted_results)))
        ]
    else:
        # Return single best search
        best_result = max(results, key=lambda x: x[0])
        policies = np.array(best_result[1])

    best_metric = max(r[0] for r in results)
    print(f"\nGreedy search complete. Best metric: {best_metric:.4f}")

    return policies, results


def greedy_search(initial_aug_idx, val_preds, good_idx, bad_idx, select_only, 
                  min_improvement=0.005, patience=5):
    """
    Single greedy search instance starting from a random initial augmentation.
    
    Iteratively adds policies that maximize ROC AUC for correct vs incorrect prediction separation.
    
    Args:
        initial_aug_idx: Starting policy index
        val_preds: Predictions array (num_policies, num_samples, num_classes)
        good_idx: Indices of correct predictions
        bad_idx: Indices of incorrect predictions
        select_only: Max policies to select
        min_improvement: Min AUC improvement to continue
        patience: Stop if no improvement for this many iterations
    
    Returns:
        tuple: (best_metric, best_group_indices, all_roc_aucs)
            - best_metric: Best ROC AUC achieved
            - best_group_indices: Policy indices in best group
            - all_roc_aucs: AUC history over iterations
    """
    group_indices = [initial_aug_idx]
    best_metric = -np.inf
    best_group_indices = list(group_indices)
    all_roc_aucs = []
    no_improvement_count = 0

    for iteration in range(select_only):
        print(f"Evaluating policy {iteration+1}/{select_only}...", flush=True)
        best_iteration_metric = -np.inf
        best_s = None

        # Try adding each remaining policy
        for new_i in range(val_preds.shape[0]):
            if new_i in group_indices:
                continue

            current_augmentations = group_indices + [int(new_i)]
            
            # Compute std across selected policies
            preds_subset = val_preds[current_augmentations, :, :]
            
            if preds_subset.shape[2] == 1:
                # Binary classification: (num_policies, num_samples, 1)
                preds_std = np.std(preds_subset.squeeze(-1), axis=0)
            else:
                # Multiclass: (num_policies, num_samples, num_classes)
                stds_per_class = np.std(preds_subset, axis=0)  # (num_samples, num_classes)
                preds_std = np.mean(stds_per_class, axis=1)  # (num_samples,)

            # Compute ROC AUC
            roc_auc = roc_curve_UQ_method_computation(
                [preds_std[k] for k in good_idx],
                [preds_std[j] for j in bad_idx]
            )[2]

            if roc_auc > 0.5 and roc_auc > best_iteration_metric:
                best_s = new_i
                best_iteration_metric = roc_auc

        if best_s is None:
            print(f"No valid policy found for iteration {iteration + 1}. Stopping search.")
            break

        # Check early stopping
        if len(all_roc_aucs) > 0:
            improvement = best_iteration_metric - all_roc_aucs[-1]
            if improvement > min_improvement:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        if no_improvement_count >= patience:
            print(
                f"Early stopping at iteration {iteration + 1} "
                f"due to no improvement > {min_improvement} in last {patience} iterations."
            )
            break

        # Update best
        if best_iteration_metric > best_metric:
            best_metric = best_iteration_metric
            best_group_indices = list(group_indices)

        group_indices.append(best_s)
        all_roc_aucs.append(best_iteration_metric)
        print(f"Selected Policy {best_s}: roc_auc={best_iteration_metric:.4f}")

    # Fallback: if only one policy selected, add next best
    if len(best_group_indices) == 1:
        print("Only one policy selected, searching for next best to add...")
        best_second_metric = -np.inf
        best_second = None
        
        for new_i in range(val_preds.shape[0]):
            if new_i == best_group_indices[0]:
                continue
            
            current_augmentations = best_group_indices + [new_i]
            preds_subset = val_preds[current_augmentations, :, :]
            
            if preds_subset.shape[2] == 1:
                preds_std = np.std(preds_subset.squeeze(-1), axis=0)
            else:
                stds_per_class = np.std(preds_subset, axis=0)
                preds_std = np.mean(stds_per_class, axis=1)
            
            roc_auc = roc_curve_UQ_method_computation(
                [preds_std[k] for k in good_idx],
                [preds_std[j] for j in bad_idx]
            )[2]
            
            if roc_auc > 0.5 and roc_auc > best_second_metric:
                best_second = new_i
                best_second_metric = roc_auc
        
        if best_second is not None:
            best_group_indices.append(best_second)
            print(f"Added next best policy {best_second} with roc_auc={best_second_metric:.4f}")

    return best_metric, best_group_indices, all_roc_aucs


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_npz_files_for_greedy_search(npz_dir):
    """
    Load all .npz prediction files from directory.
    
    Args:
        npz_dir: Directory containing .npz files with 'predictions' key
    
    Returns:
        tuple: (all_preds, all_keys)
            - all_preds: np.ndarray (num_policies, num_samples, num_classes)
            - all_keys: list of filenames
    
    Raises:
        RuntimeError: If no valid .npz files found
    
    Example:
        >>> preds, filenames = load_npz_files_for_greedy_search('gps_augment/...')
        >>> preds.shape  # (500, 125, 1) for 500 policies, 125 samples, binary
    """
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
    all_preds_list, all_keys = [], []
    
    for npz_file in npz_files:
        file_path = os.path.join(npz_dir, npz_file)
        try:
            data = np.load(file_path)
            preds = np.asarray(data['predictions'])
            
            # Normalize to (N, C)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            if preds.ndim != 2:
                print(f"Skipping {npz_file}: expected 2D array, got {preds.shape}")
                continue
            
            all_preds_list.append(preds.astype(np.float32, copy=False))
            all_keys.append(npz_file)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
    
    if not all_preds_list:
        raise RuntimeError(f"No valid .npz predictions found in {npz_dir}")

    # Ensure equal sample count; trim to min if needed
    min_len = min(arr.shape[0] for arr in all_preds_list)
    if any(arr.shape[0] != min_len for arr in all_preds_list):
        print(f"Warning: differing sample counts; trimming all to {min_len}")
    
    all_preds = np.stack([arr[:min_len] for arr in all_preds_list], axis=0)  # [P, N, C]
    return all_preds, all_keys


def plot_auc_curves(results):
    """
    Plot ROC AUC progression for all greedy searches.
    
    Args:
        results: List of (best_metric, policy_indices, auc_history) tuples
    
    Example:
        >>> results = [(0.85, [1,5,7], [0.6, 0.7, 0.85]), ...]
        >>> plot_auc_curves(results)
    """
    plt.figure(figsize=(10, 6))

    for idx, (_, _, auc_history) in enumerate(results):
        plt.plot(range(1, len(auc_history) + 1), auc_history, label=f"Search {idx + 1}")

    plt.xlabel("Iteration")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC Progress Over Iterations for Each Greedy Search")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()