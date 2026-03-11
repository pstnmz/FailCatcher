#!/usr/bin/env python3
"""
Calculate mean AUGRC and AUROC_F for each UQ method across all shifts and datasets.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Directory containing all JSON results
RESULTS_DIR = "/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/uq_benchmark_results/jsons_results"

# Methods to analyze
METHODS_TO_ANALYZE = [
    "TTA",
    "GPS",
    "KNN_Raw",
    "MSR",
    "MSR_calibrated",
    "MLS",
    "Ensembling",
    "MCDropout",
    "Mean_Aggregation",
    "Mean_Aggregation_Ensemble"
]

# Shift type mapping based on directory names
SHIFT_TYPES = {
    "in_distribution": "ID",
    "corruption_shifts": "CS",
    "population_shifts": "PS",
    "new_class_shifts": "NCS"
}

def find_all_json_files(root_dir):
    """Find all JSON files recursively in the directory."""
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def get_shift_type(json_path, root_dir):
    """Determine the shift type from the file path."""
    rel_path = os.path.relpath(json_path, root_dir)
    if 'in_distribution' in rel_path:
        return 'ID'
    elif 'corruption_shifts' in rel_path:
        return 'CS'
    elif 'population_shifts' in rel_path:
        return 'PS'
    elif 'new_class_shifts' in rel_path:
        return 'NCS'
    else:
        return 'Unknown'

def extract_augrc_values(json_files, root_dir):
    """Extract AUGRC and AUROC_F values for each method from all JSON files."""
    method_augrc_values = defaultdict(list)
    method_augrc_by_shift = defaultdict(lambda: defaultdict(list))
    method_auroc_values = defaultdict(list)
    method_auroc_by_shift = defaultdict(lambda: defaultdict(list))
    error_rates_by_shift = defaultdict(list)
    
    files_with_errors = []
    files_processed = 0
    shift_counts = defaultdict(int)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if 'methods' key exists
            if 'methods' not in data:
                continue
            
            shift_type = get_shift_type(json_file, root_dir)
            shift_counts[shift_type] += 1
            
            # Extract classification error rate
            if 'error_rate' in data and data['error_rate'] is not None:
                error_rates_by_shift[shift_type].append(data['error_rate'])
            elif 'test_accuracy' in data and data['test_accuracy'] is not None:
                error_rates_by_shift[shift_type].append(1 - data['test_accuracy'])
            
            # Extract AUGRC and AUROC_F for each method
            methods = data['methods']
            for method_name in METHODS_TO_ANALYZE:
                if method_name in methods:
                    method_data = methods[method_name]
                    if 'Mean_Aggregation' in method_name or 'Ensembling' in method_name:
                        # Extract AUGRC
                        if 'augrc' in method_data and method_data['augrc'] is not None:
                            method_augrc_values[method_name].append(method_data['augrc'])
                            method_augrc_by_shift[method_name][shift_type].append(method_data['augrc'])
                        
                        # Extract AUROC_F
                        if 'auroc_f' in method_data and method_data['auroc_f'] is not None:
                            method_auroc_values[method_name].append(method_data['auroc_f'])
                            method_auroc_by_shift[method_name][shift_type].append(method_data['auroc_f'])
                    else:
                        # Extract AUGRC
                        if 'augrc_mean' in method_data and method_data['augrc_mean'] is not None:
                            method_augrc_values[method_name].append(method_data['augrc_mean'])
                            method_augrc_by_shift[method_name][shift_type].append(method_data['augrc_mean'])
                        
                        # Extract AUROC_F
                        if 'auroc_f_mean' in method_data and method_data['auroc_f_mean'] is not None:
                            method_auroc_values[method_name].append(method_data['auroc_f_mean'])
                            method_auroc_by_shift[method_name][shift_type].append(method_data['auroc_f_mean'])
            
            files_processed += 1
            
        except Exception as e:
            files_with_errors.append((json_file, str(e)))
    
    print(f"\nProcessed {files_processed} JSON files successfully")
    print(f"Shift distribution: {dict(shift_counts)}")
    if files_with_errors:
        print(f"Encountered errors in {len(files_with_errors)} files")
    
    return method_augrc_values, method_augrc_by_shift, method_auroc_values, method_auroc_by_shift, error_rates_by_shift

def calculate_statistics(method_augrc_values):
    """Calculate mean and std for each method."""
    statistics = {}
    
    for method_name in METHODS_TO_ANALYZE:
        if method_name in method_augrc_values and len(method_augrc_values[method_name]) > 0:
            values = np.array(method_augrc_values[method_name])
            statistics[method_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        else:
            statistics[method_name] = {
                'mean': None,
                'std': None,
                'median': None,
                'min': None,
                'max': None,
                'count': 0
            }
    
    return statistics

def print_combined_results(augrc_statistics, auroc_statistics):
    """Print combined AUGRC and AUROC_F results in a formatted table."""
    print("\n" + "="*120)
    print("MEAN AUGRC AND AUROC_F FOR EACH UQ METHOD ACROSS ALL SHIFTS")
    print("="*120)
    print(f"\n{'Method':<30} {'AUGRC Mean':<15} {'AUGRC Std':<15} {'AUROC_F Mean':<15} {'AUROC_F Std':<15} {'Count':<10}")
    print("-"*120)
    
    # Sort by mean AUGRC (descending, ignoring None values)
    sorted_methods = sorted(
        augrc_statistics.items(),
        key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else -1,
        reverse=True
    )
    
    for method_name, augrc_stats in sorted_methods:
        auroc_stats = auroc_statistics.get(method_name, {})
        
        if augrc_stats['mean'] is not None or auroc_stats.get('mean') is not None:
            augrc_mean_str = f"{augrc_stats['mean']:.6f}" if augrc_stats['mean'] is not None else "N/A"
            augrc_std_str = f"{augrc_stats['std']:.6f}" if augrc_stats['std'] is not None else "N/A"
            auroc_mean_str = f"{auroc_stats.get('mean', 0):.6f}" if auroc_stats.get('mean') is not None else "N/A"
            auroc_std_str = f"{auroc_stats.get('std', 0):.6f}" if auroc_stats.get('std') is not None else "N/A"
            count = augrc_stats['count'] if augrc_stats['count'] > 0 else auroc_stats.get('count', 0)
            
            print(f"{method_name:<30} {augrc_mean_str:<15} {augrc_std_str:<15} {auroc_mean_str:<15} {auroc_std_str:<15} {count:<10}")
        else:
            print(f"{method_name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {augrc_stats['count']:<10}")

def print_combined_results_by_shift(augrc_by_shift, auroc_by_shift):
    """Print combined AUGRC and AUROC_F results broken down by shift type."""
    shift_order = ['ID', 'CS', 'PS', 'NCS']
    shift_names = {
        'ID': 'In-Distribution',
        'CS': 'Corruption Shifts',
        'PS': 'Population Shifts',
        'NCS': 'New Class Shifts'
    }
    
    print("\n" + "="*120)
    print("BREAKDOWN BY SHIFT TYPE")
    print("="*120)
    
    for shift_type in shift_order:
        if shift_type not in augrc_by_shift and shift_type not in auroc_by_shift:
            continue
        
        shift_name = shift_names.get(shift_type, shift_type)
        augrc_shift_stats = augrc_by_shift.get(shift_type, {})
        auroc_shift_stats = auroc_by_shift.get(shift_type, {})
        
        print(f"\n{'='*120}")
        print(f"{shift_name} ({shift_type})")
        print(f"{'='*120}")
        print(f"{'Method':<30} {'AUGRC Mean':<15} {'AUGRC Std':<15} {'AUROC_F Mean':<15} {'AUROC_F Std':<15} {'Count':<10}")
        print("-"*120)
        
        # Get all methods that have data in either metric
        all_methods = set(augrc_shift_stats.keys()) | set(auroc_shift_stats.keys())
        sorted_methods = sorted(
            [(m, augrc_shift_stats.get(m, {'mean': None})) for m in all_methods],
            key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else -1,
            reverse=True
        )
        
        for method_name, augrc_stats in sorted_methods:
            auroc_stats = auroc_shift_stats.get(method_name, {})
            
            if augrc_stats.get('mean') is not None or auroc_stats.get('mean') is not None:
                augrc_mean_str = f"{augrc_stats.get('mean', 0):.6f}" if augrc_stats.get('mean') is not None else "N/A"
                augrc_std_str = f"{augrc_stats.get('std', 0):.6f}" if augrc_stats.get('std') is not None else "N/A"
                auroc_mean_str = f"{auroc_stats.get('mean', 0):.6f}" if auroc_stats.get('mean') is not None else "N/A"
                auroc_std_str = f"{auroc_stats.get('std', 0):.6f}" if auroc_stats.get('std') is not None else "N/A"
                count = augrc_stats.get('count', 0) if augrc_stats.get('count', 0) > 0 else auroc_stats.get('count', 0)
                
                print(f"{method_name:<30} {augrc_mean_str:<15} {augrc_std_str:<15} {auroc_mean_str:<15} {auroc_std_str:<15} {count:<10}")
            else:
                count = augrc_stats.get('count', 0) if augrc_stats else auroc_stats.get('count', 0)
                print(f"{method_name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {count:<10}")

def print_results(statistics):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("MEAN AUGRC FOR EACH UQ METHOD ACROSS ALL SHIFTS")
    print("="*80)
    print(f"\n{'Method':<35} {'Mean AUGRC':<15} {'Std':<15} {'Count':<10}")
    print("-"*80)
    
    # Sort by mean AUGRC (descending, ignoring None values)
    sorted_methods = sorted(
        statistics.items(),
        key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else -1,
        reverse=True
    )
    
    for method_name, stats in sorted_methods:
        if stats['mean'] is not None:
            print(f"{method_name:<35} {stats['mean']:<15.6f} {stats['std']:<15.6f} {stats['count']:<10}")
        else:
            print(f"{method_name:<35} {'N/A':<15} {'N/A':<15} {stats['count']:<10}")
    
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    for method_name, stats in sorted_methods:
        if stats['mean'] is not None:
            print(f"\n{method_name}:")
            print(f"  Mean:   {stats['mean']:.6f}")
            print(f"  Std:    {stats['std']:.6f}")
            print(f"  Median: {stats['median']:.6f}")
            print(f"  Min:    {stats['min']:.6f}")
            print(f"  Max:    {stats['max']:.6f}")
            print(f"  Count:  {stats['count']}")
        else:
            print(f"\n{method_name}: No data available")

def print_results_by_shift(statistics_by_shift, shift_order=['ID', 'CS', 'PS', 'NCS']):
    """Print results broken down by shift type."""
    print("\n" + "="*100)
    print("MEAN AUGRC BREAKDOWN BY SHIFT TYPE")
    print("="*100)
    
    for shift_type in shift_order:
        if shift_type not in statistics_by_shift:
            continue
            
        shift_name = {
            'ID': 'In-Distribution',
            'CS': 'Corruption Shifts',
            'PS': 'Population Shifts',
            'NCS': 'New Class Shifts'
        }.get(shift_type, shift_type)
        
        print(f"\n{'='*100}")
        print(f"{shift_name} ({shift_type})")
        print(f"{'='*100}")
        print(f"{'Method':<35} {'Mean AUGRC':<15} {'Std':<15} {'Count':<10}")
        print("-"*100)
        
        shift_stats = statistics_by_shift[shift_type]
        sorted_methods = sorted(
            shift_stats.items(),
            key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else -1,
            reverse=True
        )
        
        for method_name, stats in sorted_methods:
            if stats['mean'] is not None:
                print(f"{method_name:<35} {stats['mean']:<15.6f} {stats['std']:<15.6f} {stats['count']:<10}")
            else:
                print(f"{method_name:<35} {'N/A':<15} {'N/A':<15} {stats['count']:<10}")

def print_error_rates_by_shift(error_rates_by_shift):
    """Print mean classification error rates by shift type."""
    shift_order = ['ID', 'CS', 'PS', 'NCS']
    shift_names = {
        'ID': 'In-Distribution',
        'CS': 'Corruption Shifts',
        'PS': 'Population Shifts',
        'NCS': 'New Class Shifts'
    }
    
    print("\n" + "="*100)
    print("MEAN CLASSIFICATION ERROR RATE BY DISTRIBUTION SHIFT")
    print("="*100)
    print(f"\n{'Shift Type':<30} {'Mean Error Rate':<20} {'Std':<15} {'Count':<10}")
    print("-"*100)
    
    for shift_type in shift_order:
        if shift_type in error_rates_by_shift and len(error_rates_by_shift[shift_type]) > 0:
            values = np.array(error_rates_by_shift[shift_type])
            mean_err = np.mean(values)
            std_err = np.std(values)
            count = len(values)
            shift_name = shift_names.get(shift_type, shift_type)
            
            print(f"{shift_name:<30} {mean_err:<20.6f} {std_err:<15.6f} {count:<10}")
        else:
            shift_name = shift_names.get(shift_type, shift_type)
            print(f"{shift_name:<30} {'N/A':<20} {'N/A':<15} {0:<10}")
    
    # Calculate overall mean (weighted by sample count)
    all_errors = []
    for errors in error_rates_by_shift.values():
        all_errors.extend(errors)
    
    # Calculate unweighted mean (simple average of shift-type means)
    shift_means = []
    for shift_type in shift_order:
        if shift_type in error_rates_by_shift and len(error_rates_by_shift[shift_type]) > 0:
            shift_means.append(np.mean(error_rates_by_shift[shift_type]))
    
    if all_errors:
        overall_mean = np.mean(all_errors)
        overall_std = np.std(all_errors)
        overall_count = len(all_errors)
        print("-"*100)
        print(f"{'Overall (Weighted Mean)':<30} {overall_mean:<20.6f} {overall_std:<15.6f} {overall_count:<10}")
        
        if shift_means:
            unweighted_mean = np.mean(shift_means)
            print(f"{'Overall (Unweighted Mean)':<30} {unweighted_mean:<20.6f} {'N/A':<15} {len(shift_means):<10}")

def main():
    print(f"Searching for JSON files in: {RESULTS_DIR}")
    
    # Find all JSON files
    json_files = find_all_json_files(RESULTS_DIR)
    print(f"Found {len(json_files)} JSON files")
    
    # Extract AUGRC and AUROC_F values
    print("Extracting AUGRC and AUROC_F values...")
    method_augrc_values, method_augrc_by_shift, method_auroc_values, method_auroc_by_shift, error_rates_by_shift = extract_augrc_values(json_files, RESULTS_DIR)
    
    # Calculate overall statistics for AUGRC
    augrc_statistics = calculate_statistics(method_augrc_values)
    
    # Calculate overall statistics for AUROC_F
    auroc_statistics = calculate_statistics(method_auroc_values)
    
    # Calculate statistics by shift type for AUGRC
    augrc_by_shift = {}
    for shift_type in ['ID', 'CS', 'PS', 'NCS']:
        shift_values = {}
        for method_name in METHODS_TO_ANALYZE:
            if method_name in method_augrc_by_shift and shift_type in method_augrc_by_shift[method_name]:
                shift_values[method_name] = method_augrc_by_shift[method_name][shift_type]
            else:
                shift_values[method_name] = []
        augrc_by_shift[shift_type] = calculate_statistics(shift_values)
    
    # Calculate statistics by shift type for AUROC_F
    auroc_by_shift = {}
    for shift_type in ['ID', 'CS', 'PS', 'NCS']:
        shift_values = {}
        for method_name in METHODS_TO_ANALYZE:
            if method_name in method_auroc_by_shift and shift_type in method_auroc_by_shift[method_name]:
                shift_values[method_name] = method_auroc_by_shift[method_name][shift_type]
            else:
                shift_values[method_name] = []
        auroc_by_shift[shift_type] = calculate_statistics(shift_values)
    
    # Print results
    print_combined_results(augrc_statistics, auroc_statistics)
    print_combined_results_by_shift(augrc_by_shift, auroc_by_shift)
    print_error_rates_by_shift(error_rates_by_shift)
    
    # Save results to file
    output_file = "/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/mean_augrc_auroc_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("MEAN AUGRC AND AUROC_F FOR EACH UQ METHOD ACROSS ALL SHIFTS\n")
        f.write("="*120 + "\n\n")
        f.write(f"{'Method':<30} {'AUGRC Mean':<15} {'AUGRC Std':<15} {'AUROC_F Mean':<15} {'AUROC_F Std':<15} {'Count':<10}\n")
        f.write("-"*120 + "\n")
        
        sorted_methods = sorted(
            augrc_statistics.items(),
            key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else -1,
            reverse=True
        )
        
        for method_name, augrc_stats in sorted_methods:
            auroc_stats = auroc_statistics.get(method_name, {})
            if augrc_stats['mean'] is not None or auroc_stats.get('mean') is not None:
                augrc_mean_str = f"{augrc_stats['mean']:<15.6f}" if augrc_stats['mean'] is not None else f"{'N/A':<15}"
                augrc_std_str = f"{augrc_stats['std']:<15.6f}" if augrc_stats['std'] is not None else f"{'N/A':<15}"
                auroc_mean_str = f"{auroc_stats.get('mean', 0):<15.6f}" if auroc_stats.get('mean') is not None else f"{'N/A':<15}"
                auroc_std_str = f"{auroc_stats.get('std', 0):<15.6f}" if auroc_stats.get('std') is not None else f"{'N/A':<15}"
                count = augrc_stats['count'] if augrc_stats['count'] > 0 else auroc_stats.get('count', 0)
                f.write(f"{method_name:<30} {augrc_mean_str} {augrc_std_str} {auroc_mean_str} {auroc_std_str} {count:<10}\n")
            else:
                f.write(f"{method_name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {augrc_stats['count']:<10}\n")
        
        # Add shift breakdown to file
        f.write("\n\n" + "="*120 + "\n")
        f.write("BREAKDOWN BY SHIFT TYPE\n")
        f.write("="*120 + "\n")
        
        shift_names = {
            'ID': 'In-Distribution',
            'CS': 'Corruption Shifts',
            'PS': 'Population Shifts',
            'NCS': 'New Class Shifts'
        }
        
        for shift_type in ['ID', 'CS', 'PS', 'NCS']:
            if shift_type not in augrc_by_shift and shift_type not in auroc_by_shift:
                continue
            
            shift_name = shift_names.get(shift_type, shift_type)
            augrc_shift_stats = augrc_by_shift.get(shift_type, {})
            auroc_shift_stats = auroc_by_shift.get(shift_type, {})
            
            f.write(f"\n{shift_name} ({shift_type})\n")
            f.write("-"*120 + "\n")
            f.write(f"{'Method':<30} {'AUGRC Mean':<15} {'AUGRC Std':<15} {'AUROC_F Mean':<15} {'AUROC_F Std':<15} {'Count':<10}\n")
            f.write("-"*120 + "\n")
            
            all_methods = set(augrc_shift_stats.keys()) | set(auroc_shift_stats.keys())
            sorted_shift_methods = sorted(
                [(m, augrc_shift_stats.get(m, {'mean': None})) for m in all_methods],
                key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else -1,
                reverse=True
            )
            
            for method_name, augrc_stats in sorted_shift_methods:
                auroc_stats = auroc_shift_stats.get(method_name, {})
                if augrc_stats.get('mean') is not None or auroc_stats.get('mean') is not None:
                    augrc_mean_str = f"{augrc_stats.get('mean', 0):<15.6f}" if augrc_stats.get('mean') is not None else f"{'N/A':<15}"
                    augrc_std_str = f"{augrc_stats.get('std', 0):<15.6f}" if augrc_stats.get('std') is not None else f"{'N/A':<15}"
                    auroc_mean_str = f"{auroc_stats.get('mean', 0):<15.6f}" if auroc_stats.get('mean') is not None else f"{'N/A':<15}"
                    auroc_std_str = f"{auroc_stats.get('std', 0):<15.6f}" if auroc_stats.get('std') is not None else f"{'N/A':<15}"
                    count = augrc_stats.get('count', 0) if augrc_stats.get('count', 0) > 0 else auroc_stats.get('count', 0)
                    f.write(f"{method_name:<30} {augrc_mean_str} {augrc_std_str} {auroc_mean_str} {auroc_std_str} {count:<10}\n")
                else:
                    count = augrc_stats.get('count', 0) if augrc_stats else auroc_stats.get('count', 0)
                    f.write(f"{method_name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {count:<10}\n")
        
        # Add error rates by shift
        f.write("\n\n" + "="*100 + "\n")
        f.write("MEAN CLASSIFICATION ERROR RATE BY DISTRIBUTION SHIFT\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Shift Type':<30} {'Mean Error Rate':<20} {'Std':<15} {'Count':<10}\n")
        f.write("-"*100 + "\n")
        
        for shift_type in ['ID', 'CS', 'PS', 'NCS']:
            if shift_type in error_rates_by_shift and len(error_rates_by_shift[shift_type]) > 0:
                values = np.array(error_rates_by_shift[shift_type])
                mean_err = np.mean(values)
                std_err = np.std(values)
                count = len(values)
                shift_name = shift_names.get(shift_type, shift_type)
                f.write(f"{shift_name:<30} {mean_err:<20.6f} {std_err:<15.6f} {count:<10}\n")
            else:
                shift_name = shift_names.get(shift_type, shift_type)
                f.write(f"{shift_name:<30} {'N/A':<20} {'N/A':<15} {0:<10}\n")
        
        # Overall mean error rate
        all_errors = []
        for errors in error_rates_by_shift.values():
            all_errors.extend(errors)
        
        if all_errors:
            overall_mean = np.mean(all_errors)
            overall_std = np.std(all_errors)
            overall_count = len(all_errors)
            f.write("-"*100 + "\n")
            f.write(f"{'Overall (Weighted Mean)':<30} {overall_mean:<20.6f} {overall_std:<15.6f} {overall_count:<10}\n")
            
            # Calculate unweighted mean
            shift_means = []
            for shift_type in ['ID', 'CS', 'PS', 'NCS']:
                if shift_type in error_rates_by_shift and len(error_rates_by_shift[shift_type]) > 0:
                    shift_means.append(np.mean(error_rates_by_shift[shift_type]))
            
            if shift_means:
                unweighted_mean = np.mean(shift_means)
                f.write(f"{'Overall (Unweighted Mean)':<30} {unweighted_mean:<20.6f} {'N/A':<15} {len(shift_means):<10}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
