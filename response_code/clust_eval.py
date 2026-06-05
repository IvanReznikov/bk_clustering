import json
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load all results
methods = {
    'bk_clustering': 'A0: MultiFusion',
    'affinity': 'A1: Affinity Propagation',
    'dbscan': 'A2: DBSCAN',
    'hdbscan': 'A3: HDBSCAN',
    'density_peak': 'A4: Fast Search',
    'mean_shift': 'A5: Mean Shift',
    'optics': 'A6: OPTICS',
    'agglomerative': 'R1: Agglomerative',
    'birch': 'R2: BIRCH',
    'gaussian_mixture': 'R3: Gaussian Mixture',
    'k_means': 'R4: k-Means',
    'mini_batch_kmeans': 'R5: Mini-batch k-Means',
}

metric_map = {
    'MI': 'mutual_similarity_0',
    'NMI': 'mutual_similarity_1',
    'AMI': 'mutual_similarity_2',
    'RI': 'rand_index_0',
    'ARI': 'rand_index_1',
    'Homogeneity': 'v_measure_0',
    'Completeness': 'v_measure_1',
    'V-measure': 'v_measure_2',
    'FMI': 'fm_score',
}

raw_data = {}
for fname, label in methods.items():
    with open(f'../results/{fname}_results.json') as f:
        raw_data[label] = json.load(f)

# Get all dataset keys common across methods
all_keys = set(raw_data['A0: MultiFusion'].keys())
for label, data in raw_data.items():
    all_keys = all_keys.intersection(set(data.keys()))
all_keys = sorted(list(all_keys))
print(f"Datasets common to all methods: {len(all_keys)}")

# Build per-dataset per-metric DataFrames
metric_dfs = {}
for metric_name, metric_key in metric_map.items():
    rows = {}
    for label, data in raw_data.items():
        rows[label] = {k: data[k].get(metric_key, np.nan) for k in all_keys}
    metric_dfs[metric_name] = pd.DataFrame(rows)

print("Metrics loaded:", list(metric_dfs.keys()))
print("Shape:", metric_dfs['ARI'].shape)

from scipy.stats import wilcoxon, friedmanchisquare
import itertools

multifusion_label = 'A0: MultiFusion'
competitors = [m for m in methods.values() if m != multifusion_label]

# ---- 1. Wilcoxon signed-rank tests: MultiFusion vs each competitor ----
wilcoxon_results = []
key_metrics = ['ARI', 'NMI', 'AMI', 'FMI', 'V-measure']

for metric in key_metrics:
    df = metric_dfs[metric]
    mf_scores = df[multifusion_label].values
    for comp in competitors:
        comp_scores = df[comp].values
        # Remove NaN pairs
        mask = ~(np.isnan(mf_scores) | np.isnan(comp_scores))
        mf_clean = mf_scores[mask]
        comp_clean = comp_scores[mask]
        diff = mf_clean - comp_clean
        # Skip if all differences are zero
        if np.all(diff == 0):
            continue
        try:
            stat, p = wilcoxon(mf_clean, comp_clean, alternative='greater')
            # Effect size: rank-biserial correlation
            n = len(diff)
            r = 1 - (2 * stat) / (n * (n + 1) / 2)
            wilcoxon_results.append({
                'Metric': metric,
                'Competitor': comp,
                'W-stat': round(stat, 1),
                'p-value': p,
                'r (effect)': round(r, 3),
                'n': int(mask.sum()),
                'MF_median': round(np.median(mf_clean), 4),
                'Comp_median': round(np.median(comp_clean), 4),
            })
        except Exception as e:
            pass

wilcoxon_df = pd.DataFrame(wilcoxon_results)
# Bonferroni correction
wilcoxon_df['p_bonferroni'] = np.minimum(wilcoxon_df['p-value'] * len(wilcoxon_df), 1.0)
wilcoxon_df['sig'] = wilcoxon_df['p_bonferroni'].apply(lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns')))
print(wilcoxon_df[['Metric','Competitor','MF_median','Comp_median','p-value','p_bonferroni','sig','r (effect)']].to_string())

# ---- 2. Friedman test + Nemenyi post-hoc for ARI ----
try:
    import scikit_posthocs as sp
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'scikit-posthocs', '-q'])
    import scikit_posthocs as sp

friedman_results = {}
for metric in key_metrics:
    df = metric_dfs[metric]
    # Drop rows with any NaN
    df_clean = df.dropna()
    groups = [df_clean[col].values for col in df_clean.columns]
    stat, p = friedmanchisquare(*groups)
    friedman_results[metric] = {'stat': round(stat, 2), 'p': p, 'n_datasets': len(df_clean)}
    
friedman_df = pd.DataFrame(friedman_results).T
print("Friedman test results:")
print(friedman_df)

# Nemenyi post-hoc for ARI
metric = 'ARI'
df_ari = metric_dfs[metric].dropna()
nemenyi = sp.posthoc_nemenyi_friedman(df_ari.values)
nemenyi.index = df_ari.columns
nemenyi.columns = df_ari.columns

# Extract MultiFusion row
mf_row = nemenyi.loc[multifusion_label]
print("Nemenyi p-values: MultiFusion vs. others (ARI):")
print(mf_row.sort_values())

# ---- 3. Solidity vs External Metric correlation ----
try:
    with open('../results/solidity_correlation_results.json') as f:
        corr_data = json.load(f)
    
    corr_df = pd.DataFrame(corr_data)
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS: Solidity vs. External Validity Metrics")
    print("="*50)
    
    for metric in ['ARI', 'NMI', 'AMI', 'V-measure', 'FMI']:
        spearman_rho, spearman_p = stats.spearmanr(corr_df['average_solidity'], corr_df[metric])
        pearson_r, pearson_p = stats.pearsonr(corr_df['average_solidity'], corr_df[metric])
        print(f"\nMetric: {metric}")
        print(f"  Spearman rho: {spearman_rho:8.4f} (p-value: {spearman_p:.4e})")
        print(f"  Pearson r:    {pearson_r:8.4f} (p-value: {pearson_p:.4e})")
except Exception as e:
    print("Failed to run correlation analysis in clust_eval.py:", e)