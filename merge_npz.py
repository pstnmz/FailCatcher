#!/usr/bin/env python3
"""
Merge two .npz files containing UQ benchmark results.
"""
import numpy as np
import sys

# File paths
file1 = "uq_benchmark_results/all_metrics_tissuemnist_vit_b_16_DADO_20251221_131707.npz"
file2 = "uq_benchmark_results/all_metrics_tissuemnist_vit_b_16_DADO_20251222_132523.npz"
output = "uq_benchmark_results/all_metrics_tissuemnist_vit_b_16_DADO_merged.npz"

print("Loading files...")
data1 = np.load(file1, allow_pickle=True)
data2 = np.load(file2, allow_pickle=True)

print(f"\nFile 1 ({file1}):")
print(f"  Keys: {list(data1.keys())}")

print(f"\nFile 2 ({file2}):")
print(f"  Keys: {list(data2.keys())}")

# Merge: combine all keys from both files
merged = {}

# Add all from file1
for key in data1.keys():
    merged[key] = data1[key]
    print(f"  Added from file1: {key}")

# Add from file2 (will overwrite if key exists)
for key in data2.keys():
    if key in merged:
        print(f"  WARNING: Overwriting {key} with data from file2")
    merged[key] = data2[key]
    print(f"  Added from file2: {key}")

print(f"\nMerged keys: {list(merged.keys())}")

# Save merged file
print(f"\nSaving to {output}...")
np.savez_compressed(output, **merged)

print("✓ Done!")
print(f"\nMerged file contains {len(merged)} entries")
