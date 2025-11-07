import numpy as np
import os
import json
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
num_tasks = 198
shift_len = 988
bins = np.linspace(-1e-3, 1e3, 2001)  # adjust bin range if needed
folder = "encode_dataset/final_3k/all_npz"
output_folder = "encode_dataset/final_3k/task_stats"
os.makedirs(output_folder, exist_ok=True)

prefixes = ['aln', 'pv', 'fc']

with open('encode_dataset/final_3k/task_names.json', 'r') as f:
    task_names = json.load(f)

# Map prefix -> list of task indices
prefix_task_idx = {p: [i for i, name in enumerate(task_names) if name.startswith(p)] for p in prefixes}

# ---------------- PROCESS FILE ----------------
def process_file(file_path):
    local_hist = {p: np.zeros(len(bins)-1, dtype=np.int64) for p in prefixes}
    local_count = {p: 0 for p in prefixes}
    local_sum = {p: 0.0 for p in prefixes}
    local_sum_sq = {p: 0.0 for p in prefixes}
    local_max = {p: -np.inf for p in prefixes}
    local_min = {p: np.inf for p in prefixes}
    
    data = np.load(file_path, allow_pickle=True)['arr_0'][()]
    
    for y in data.values():
        for prefix, indices in prefix_task_idx.items():
            if not indices:
                continue
            vals = np.concatenate([y[task_names[i]][shift_len:-shift_len] for i in indices])
            # set values greater than global_max to nan
            vals = np.where(vals > 1e+4, np.nan, vals)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue

            # Histogram
            local_hist[prefix] += np.histogram(vals, bins=bins)[0]

            # Aggregate stats
            local_count[prefix] += len(vals)
            local_sum[prefix] += vals.sum()
            local_sum_sq[prefix] += (vals**2).sum()
            
            # Min/Max
            local_max[prefix] = max(local_max[prefix], np.max(vals))
            local_min[prefix] = min(local_min[prefix], np.min(vals))
    
    return local_hist, local_count, local_sum, local_sum_sq, local_max, local_min

# ---------------- MULTIPROCESS ----------------
all_files = [os.path.join(folder, f) for f in os.listdir(folder)]

num_workers = 16

print(f"Processing {len(all_files)} files using {num_workers} cores...")
with mp.Pool(processes=num_workers) as pool:
    results = list(tqdm(pool.imap(process_file, all_files), total=len(all_files)))

# ---------------- MERGE RESULTS ----------------
hist_counts = {p: np.zeros(len(bins)-1, dtype=np.int64) for p in prefixes}
count = {p: 0 for p in prefixes}
sum_vals = {p: 0.0 for p in prefixes}
sum_sq_vals = {p: 0.0 for p in prefixes}
max_vals = {p: -np.inf for p in prefixes}
min_vals = {p: np.inf for p in prefixes}

for local_hist, local_count, local_sum, local_sum_sq, local_max, local_min in results:
    for p in prefixes:
        hist_counts[p] += local_hist[p]
        count[p] += local_count[p]
        sum_vals[p] += local_sum[p]
        sum_sq_vals[p] += local_sum_sq[p]
        max_vals[p] = max(max_vals[p], local_max[p])
        min_vals[p] = min(min_vals[p], local_min[p])

# Compute stats
stats = {}
for p in prefixes:
    c = count[p]
    mean = sum_vals[p] / c if c > 0 else np.nan
    std = np.sqrt(sum_sq_vals[p]/c - mean**2) if c > 0 else np.nan
    stats[p] = {"mean": mean, "std": std, "count": c}

# ---------------- SAVE HISTOGRAMS ----------------
bin_centers = 0.5*(bins[:-1] + bins[1:])
bin_width = 1

for p in prefixes:
    plt.figure()
    plt.bar(bin_centers, hist_counts[p], width=bin_width)
    plt.xlim([min_vals[p], max_vals[p]])
    plt.title(f"Histogram of all tasks with prefix '{p}'")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_folder, f"group/reg/{p}_hist.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Log-scale version
    plt.figure()
    plt.bar(bin_centers, hist_counts[p]+1, width=bin_width)  # add 1 to avoid log(0)
    plt.yscale('log')
    plt.xlim([min_vals[p], max_vals[p]])
    plt.title(f"Histogram of all tasks with prefix '{p}' (Log Scale)")
    plt.xlabel("Value")
    plt.ylabel("Frequency (log scale)")
    plt.savefig(os.path.join(output_folder, f"group/log/{p}_hist.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------- PRINT STATS ----------------
for p in prefixes:
    print(f"{p}: mean={stats[p]['mean']:.4f}, std={stats[p]['std']:.4f}, count={stats[p]['count']}")

'''
aln: mean=1.6112, std=90.6698, count=526054477
pv: mean=2.9956, std=48.2432, count=22951717258
fc: mean=1.1778, std=3.3032, count=21790697976
'''