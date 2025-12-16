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
    npz_dir, good_idx, bad_idx, max_iterations=50, num_workers=1, num_searches=10, top_k=5, plot=True, method='top_k_policies', seed=None
):
    print('Loading predictions...')
    all_preds, all_keys, format_version = load_npz_files_for_greedy_search(npz_dir)
    # shapes: [num_policies, num_samples, num_classes] (v1) or [num_policies, num_models, num_samples, num_classes] (v2)
    if format_version == 1:
        num_samples = all_preds.shape[1]
        num_classes = all_preds.shape[2]
    else:
        num_samples = all_preds.shape[2]
        num_classes = all_preds.shape[3]

    # Clip indices if they exceed available samples
    if len(good_idx) or len(bad_idx):
        max_idx = max(int(max(good_idx)) if len(good_idx) else -1,
                      int(max(bad_idx)) if len(bad_idx) else -1)
        if max_idx >= num_samples:
            print(f"Warning: indices exceed predictions length ({max_idx} >= {num_samples}). Clipping.")
            good_idx = [i for i in good_idx if i < num_samples]
            bad_idx = [i for i in bad_idx if i < num_samples]
            if not good_idx or not bad_idx:
                raise ValueError("After clipping, good_idx or bad_idx is empty. Ensure .npz match the dataset/ordering.")

    selected_policies, results = select_greedily_on_ens(
        all_preds, good_idx, bad_idx, all_keys,
        search_set_len=num_samples,          # FIX: use sample count, not .size
        select_only=max_iterations,
        num_workers=num_workers,
        num_searches=num_searches,
        top_k=top_k,
        method=method,
        seed=seed,
        format_version=format_version
    )

    if isinstance(selected_policies, list) and all(isinstance(policy, list) for policy in selected_policies):
        selected_policy_names = [[all_keys[i] for i in selected_policy] for selected_policy in selected_policies]
    else:
        selected_policy_names = [all_keys[i] for i in selected_policies]

    if plot:
        plot_auc_curves(results)

    if format_version == 1:
        print(f"Loaded predictions shape: {all_preds.shape} (policies, samples, classes={num_classes})")
    else:
        print(f"Loaded predictions shape: {all_preds.shape} (policies, models, samples, classes={num_classes})")
    return selected_policy_names


# ============================================================================
# CORE GREEDY SEARCH LOGIC
# ============================================================================

def select_greedily_on_ens(
    all_preds, good_idx, bad_idx, keys, search_set_len, select_only=50,
    num_workers=1, num_searches=10, top_k=5, method='top_policies', seed=None, format_version=1
):
    # Extract calibration set: keep all dimensions, slice samples at appropriate axis
    if format_version == 1:
        val_preds = np.copy(all_preds[:, :search_set_len, :])  # [P, N, C]
    else:
        val_preds = np.copy(all_preds[:, :, :search_set_len, :])  # [P, M, N, C]

    # Prepare random starts
    # deterministic initial starts when `seed` is provided
    rng = np.random.RandomState(seed) if seed is not None else np.random
    initial_augmentations = [int(rng.choice(range(val_preds.shape[0]))) for _ in range(num_searches)]
 
    def _run_sequential():
        out = []
        for init_aug in initial_augmentations:
            out.append(greedy_search(init_aug, val_preds, good_idx, bad_idx, select_only, format_version))
        return out

    results = []
    if num_workers and num_workers > 1:
        try:
            with mp.Pool(processes=min(num_workers, mp.cpu_count() or num_workers)) as pool:
                results = pool.starmap(
                    greedy_search,
                    [(init_aug, val_preds, good_idx, bad_idx, select_only, format_version) for init_aug in initial_augmentations]
                )
        except Exception as e:
            print(f"Parallel greedy_search failed: {repr(e)}; falling back to sequential.")
            results = _run_sequential()
    else:
        results = _run_sequential()

    if not results:
        raise RuntimeError("No greedy_search results produced. Check input shapes and indices.")

    # Select the best result based on the ROC AUC metric
    best_result = max(results, key=lambda x: x[0])
    best_metric, best_group_indices, _ = best_result

    print("\nGreedy search complete. Best metric:", best_metric)

    if method == 'top_k_policies':
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        policies = [sorted_results[i][1] for i in range(min(top_k, len(sorted_results)))]
    else:
        policies = np.array(best_group_indices)

    return policies, results


def greedy_search(initial_aug_idx, val_preds, good_idx, bad_idx, select_only, format_version=1, min_improvement=0.005, patience=5):
    """
    Single greedy search instance starting from a random initial augmentation.
    
    Args:
        val_preds: [P, N, C] (v1) or [P, M, N, C] (v2)
        format_version: 1 = std(ensemble), 2 = avg(std(models))
    """
    group_indices = [initial_aug_idx]
    best_metric = -np.inf
    best_group_indices = list(group_indices)
    all_roc_aucs = []
    no_improvement_count = 0
    
    for new_member_i in range(select_only):
        print(f"Evaluating policy {new_member_i+1}/{select_only}...", flush=True)
        best_iteration_metric = -np.inf
        best_s = None

        for new_i in range(val_preds.shape[0]):
            if new_i in group_indices:
                continue

            current_augmentations = group_indices + [np.int64(new_i)]
            
            # Compute std based on format
            if format_version == 1:
                # Old format: [P, N, C] → std across policies
                preds_subset = val_preds[current_augmentations, :, :]  # [len(current_augmentations), N, C]
                if preds_subset.shape[2] == 1:
                    # Binary: (num_policies, num_samples, 1)
                    preds_std = np.std(preds_subset, axis=0)  # (num_samples, 1)
                    preds_std = preds_std.flatten()  # ← FLATTEN to (num_samples,) for indexing
                else:
                    # Multiclass: (num_policies, num_samples, num_classes)
                    stds_per_class = np.std(preds_subset, axis=0)  # (num_samples, num_classes)
                    preds_std = np.mean(stds_per_class, axis=1)  # (num_samples,)
            else:
                # New format: [P, M, N, C] → std per model, then average across models
                preds_subset = val_preds[current_augmentations, :, :, :]  # [len(current_augmentations), M, N, C]
                # Compute std across policies per model: [M, N, C]
                std_per_model = np.std(preds_subset, axis=0)  # [M, N, C]
                
                if std_per_model.shape[2] == 1:
                    # Binary: average across models → [N, 1]
                    preds_std = np.mean(std_per_model, axis=0).flatten()  # [N]
                else:
                    # Multiclass: average std per class across models → [N, C], then mean across classes → [N]
                    avg_std_per_model = np.mean(std_per_model, axis=0)  # [N, C]
                    preds_std = np.mean(avg_std_per_model, axis=1)  # [N]
            
            # Compute ROC AUC
            roc_auc = roc_curve_UQ_method_computation(
                [preds_std[k] for k in good_idx],
                [preds_std[j] for j in bad_idx]
            )[2]
            
            if roc_auc > 0.5 and roc_auc > best_iteration_metric:
                best_s = new_i
                best_iteration_metric = roc_auc
        
        if best_s is None:
            print(f"No valid policy found for iteration {new_member_i + 1}. Stopping search.")
            break
        
        if len(all_roc_aucs) > 0:
            improvement = best_iteration_metric - all_roc_aucs[-1]
            if improvement > min_improvement:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {new_member_i + 1} due to no improvement > {min_improvement} in last {patience} iterations.")
            break
    
        if best_iteration_metric > best_metric:
            best_metric = best_iteration_metric
            best_group_indices = list(group_indices)
        
        group_indices.append(best_s)
        all_roc_aucs.append(best_iteration_metric)
        print(f"Selected Policy {best_s}: roc_auc={best_iteration_metric:.4f}")

    # Fallback if only one policy selected
    if len(best_group_indices) == 1:
        print("Only one policy selected, searching for the next best policy to add...")
        best_second_metric = -np.inf
        best_second = None
        
        for new_i in range(val_preds.shape[0]):
            if new_i == best_group_indices[0]:
                continue
            
            current_augmentations = best_group_indices + [new_i]
            
            # Compute std based on format
            if format_version == 1:
                preds_subset = val_preds[current_augmentations, :, :]
                if preds_subset.shape[2] == 1:
                    preds_std = np.std(preds_subset, axis=0).flatten()
                else:
                    stds_per_class = np.std(preds_subset, axis=0)
                    preds_std = np.mean(stds_per_class, axis=1)
            else:
                preds_subset = val_preds[current_augmentations, :, :, :]
                std_per_model = np.std(preds_subset, axis=0)  # [M, N, C]
                if std_per_model.shape[2] == 1:
                    preds_std = np.mean(std_per_model, axis=0).flatten()
                else:
                    avg_std_per_model = np.mean(std_per_model, axis=0)
                    preds_std = np.mean(avg_std_per_model, axis=1)
            
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
    Load all .npz files from the specified directory.
    Return stacked predictions and filenames.
    
    Supports two formats:
    - v1 (old): predictions = [num_samples, num_classes] → returns [num_policies, num_samples, num_classes]
    - v2 (new): predictions = [num_models, num_samples, num_classes] → returns [num_policies, num_models, num_samples, num_classes]
    
    Returns:
        all_preds: [P, N, C] or [P, M, N, C] depending on format
        all_keys: List of filenames
        format_version: 1 or 2
    """
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
    all_preds_list, all_keys = [], []
    format_version = None
    
    for npz_file in npz_files:
        file_path = os.path.join(npz_dir, npz_file)
        try:
            data = np.load(file_path)
            preds = np.asarray(data['predictions'])
            
            # Detect format version
            version = data.get('version', 1)
            if format_version is None:
                format_version = version
            elif format_version != version:
                print(f"Warning: {npz_file} has version {version}, expected {format_version}. Skipping.")
                continue
            
            # Handle old format (v1): [N, C]
            if format_version == 1:
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                if preds.ndim != 2:
                    print(f"Skipping {npz_file}: expected 2D array for v1, got {preds.shape}")
                    continue
            # Handle new format (v2): [M, N, C]
            else:
                if preds.ndim != 3:
                    print(f"Skipping {npz_file}: expected 3D array for v2, got {preds.shape}")
                    continue
            
            all_preds_list.append(preds.astype(np.float32, copy=False))
            all_keys.append(npz_file)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
    
    if not all_preds_list:
        raise RuntimeError(f"No valid .npz predictions found in {npz_dir}")
    
    # Ensure equal sample count; trim to min if needed
    if format_version == 1:
        min_len = min(arr.shape[0] for arr in all_preds_list)
        if any(arr.shape[0] != min_len for arr in all_preds_list):
            print(f"Warning: differing sample counts; trimming to {min_len}")
        all_preds = np.stack([arr[:min_len] for arr in all_preds_list], axis=0)  # [P, N, C]
    else:
        # v2: [M, N, C] per file
        min_len = min(arr.shape[1] for arr in all_preds_list)  # N is at axis 1
        if any(arr.shape[1] != min_len for arr in all_preds_list):
            print(f"Warning: differing sample counts; trimming to {min_len}")
        all_preds = np.stack([arr[:, :min_len, :] for arr in all_preds_list], axis=0)  # [P, M, N, C]
    
    print(f"Loaded format version {format_version}: predictions shape = {all_preds.shape}")
    return all_preds, all_keys, format_version


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
    plt.close()  # Close instead of show to avoid blocking execution