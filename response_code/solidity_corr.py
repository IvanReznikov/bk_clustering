import sys
import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

# Increase recursion limit
sys.setrecursionlimit(100000)

# Add parent directory to path to import bk_clustering
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bk_clustering.main import BurjKhalifaClustering
from bk_clustering.utilities import load_save, metrics

SKIP_COLUMNS_IN_DATASET = {
    ("real", "wdbc"): ["idnumber"],
    ("real", "spectrometer"): ["LRS-name"],
}

import threading

def _process_dataset_impl(dataset_name):
    # Ensure recursion limit is set in child processes
    sys.setrecursionlimit(100000)
    folder, filename = dataset_name
    try:
        skip_columns = (
            []
            if dataset_name not in SKIP_COLUMNS_IN_DATASET
            else SKIP_COLUMNS_IN_DATASET[dataset_name]
        )
        df = load_save.read_arff(folder, filename, skip_columns)
        if df is None:
            return {
                'dataset_name': str(dataset_name),
                'status': 'file_not_found'
            }
        
        X, true_labels = df.loc[:, df.columns != "class"], df["class"]
        
        # Fit BurjKhalifaClustering with linkage="ward"
        bk_model = BurjKhalifaClustering(
            depth=2,
            chain_ratio=5,
            parent_split_ratio=10,
            min_leaves=0,
            n_clusters=None,
            linkage="ward",
        )
        bk_model.fit(X)
        
        # Calculate metrics
        eval_metrics = metrics.calculate_metrics(true_labels, bk_model.labels_)
        
        return {
            'dataset_name': str(dataset_name),
            'status': 'success',
            'average_solidity': float(bk_model.average_solidity_),
            'chosen_linkage': bk_model.linkage,
            'ARI': float(eval_metrics['rand_index_1']),
            'NMI': float(eval_metrics['mutual_similarity_1']),
            'AMI': float(eval_metrics['mutual_similarity_2']),
            'V-measure': float(eval_metrics['v_measure_2']),
            'FMI': float(eval_metrics['fm_score']),
            'num_datapoints': len(true_labels),
            'num_clusters_detected': int(bk_model.n_clusters),
            'num_clusters_true': int(eval_metrics['clusters_true'])
        }
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        return {
            'dataset_name': str(dataset_name),
            'status': 'error',
            'error_msg': str(e),
            'error_detail': err_detail
        }

def process_dataset(dataset_name):
    res = {}
    def run_thread():
        nonlocal res
        res = _process_dataset_impl(dataset_name)
    
    # Run in thread with 64MB stack size
    old_size = threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=run_thread)
    t.start()
    t.join()
    threading.stack_size(old_size)
    return res

def main():
    skip_datasets = [
        "water-treatment",  # no class
        "autos",
        "credit.a",  # duplicate dataset
        "credit.g",  # duplicate dataset
        "sick",  # duplicate dataset
        "golfball",  # as 1 cluster, incorrect metric definition for clustering methods
        "Colon",  # multiple duplicated column names
        "jm1",  # gmm throws error
        "KDDTest+",  # gmm throws error
        "Rice_MSC_Dataset",  # run separately
        "click_data",  # takes forever long for kmeans?
    ]
    folders = ["real", "artificial"]

    dataset_names = []
    for folder in folders:
        dataset_names += [
            (folder, x[:-5])
            for x in os.listdir(f"../data/{folder}")
            if x[:-5] not in skip_datasets
        ]
        
    print(f"Analyzing {len(dataset_names)} datasets...")
    
    small_datasets = []
    large_datasets = []
    
    # Classify datasets by size before processing
    for name in dataset_names:
        folder, filename = name
        try:
            skip_columns = (
                []
                if name not in SKIP_COLUMNS_IN_DATASET
                else SKIP_COLUMNS_IN_DATASET[name]
            )
            # Just inspect shape
            df_path = f"../data/{folder}/{filename}.arff"
            if not os.path.exists(df_path):
                small_datasets.append((name, 0))
                continue
            
            # Read shape only to avoid full preprocessing memory overhead here
            with open(df_path, "r", encoding="utf-8") as f:
                # Approximate number of rows by reading the file
                num_rows = sum(1 for line in f if not line.strip().startswith('@') and len(line.strip()) > 0)
            
            if num_rows > 5000:
                large_datasets.append((name, num_rows))
            else:
                small_datasets.append((name, num_rows))
        except Exception:
            # Fallback to small if any reading issues
            small_datasets.append((name, 0))
            
    print(f"Classified: {len(small_datasets)} small/medium datasets (<=5000 rows) and {len(large_datasets)} large datasets (>5000 rows).")
    for name, rows in large_datasets:
        print(f"  Large dataset: {name} ({rows} rows)")
        
    results = []
    
    # Phase 1: Process small datasets in parallel
    if small_datasets:
        max_workers = 4
        print(f"\n--- Phase 1: Processing {len(small_datasets)} small/medium datasets in parallel (workers={max_workers}) ---")
        small_names = [item[0] for item in small_datasets]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_dataset, name): name for name in small_names}
            completed = 0
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                completed += 1
                if completed % 10 == 0 or completed == len(small_names):
                    print(f"Progress (Small): {completed}/{len(small_names)} completed.")
                    
    # Phase 2: Process large datasets sequentially
    if large_datasets:
        print(f"\n--- Phase 2: Processing {len(large_datasets)} large datasets sequentially ---")
        for idx, (name, rows) in enumerate(large_datasets):
            print(f"[{idx+1}/{len(large_datasets)}] Starting {name} ({rows} rows)...", flush=True)
            start_t = time.time()
            res = process_dataset(name)
            results.append(res)
            elapsed = time.time() - start_t
            status_str = "SUCCESS" if res.get('status') == 'success' else f"FAILED ({res.get('error_msg')})"
            print(f"[{idx+1}/{len(large_datasets)}] Finished {name} - {status_str} in {elapsed:.2f}s", flush=True)
            
    # Filter successful runs
    success_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'error']
    
    print(f"\nSuccessfully processed {len(success_results)}/{len(dataset_names)} datasets.")
    print(f"Failed to process {len(failed_results)}/{len(dataset_names)} datasets.")
    
    # Save raw results
    output_path = '../results/solidity_correlation_results.json'
    with open(output_path, 'w') as f:
        json.dump(success_results, f, indent=4)
    print(f"Raw results saved to {output_path}")
    
    failed_path = '../results/solidity_correlation_failures.json'
    if failed_results:
        with open(failed_path, 'w') as f:
            json.dump(failed_results, f, indent=4)
        print(f"Failure logs saved to {failed_path}")
    elif os.path.exists(failed_path):
        import shutil
        os.makedirs('../legacy', exist_ok=True)
        shutil.move(failed_path, '../legacy/solidity_correlation_failures.json')
    
    if not success_results:
        print("No successful runs. Exiting.")
        return
        
    # Convert to DataFrame for correlation analysis
    df = pd.DataFrame(success_results)
    
    metrics_to_correlate = ['ARI', 'NMI', 'AMI', 'V-measure', 'FMI']
    
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS: Solidity vs. External Validity Metrics")
    print("="*50)
    
    summary_results = []
    for metric in metrics_to_correlate:
        # Spearman rank correlation
        spearman_rho, spearman_p = stats.spearmanr(df['average_solidity'], df[metric])
        # Pearson linear correlation
        pearson_r, pearson_p = stats.pearsonr(df['average_solidity'], df[metric])
        
        print(f"\nMetric: {metric}")
        print(f"  Spearman rho: {spearman_rho:8.4f} (p-value: {spearman_p:.4e})")
        print(f"  Pearson r:    {pearson_r:8.4f} (p-value: {pearson_p:.4e})")
        
        summary_results.append({
            'Metric': metric,
            'Spearman_rho': spearman_rho,
            'Spearman_p': spearman_p,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p
        })
        
    # Save summary results
    summary_path = '../results/solidity_correlation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
        
if __name__ == '__main__':
    main()
