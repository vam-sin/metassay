import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

num_tasks = 198
shift_len = 988

bins = np.linspace(-1e-3, 1e+3, 2001)  # histogram bins
folder = "encode_dataset/final_3k/all_npz"
output_folder = "encode_dataset/final_3k/task_stats"

with open('encode_dataset/final_3k/task_names.json', 'r') as f:
    task_names = json.load(f)

def process_file(file_path):
    local_hist = np.zeros((num_tasks, len(bins) - 1), dtype=np.int64)
    local_count = np.zeros(num_tasks, dtype=np.int64)
    local_mean = np.zeros(num_tasks, dtype=np.float64)
    local_M2 = np.zeros(num_tasks, dtype=np.float64)
    local_max = np.full(num_tasks, -np.inf, dtype=np.float64)
    local_min = np.full(num_tasks, np.inf, dtype=np.float64)

    data = np.load(file_path, allow_pickle=True)['arr_0'][()]
    
    for y in data.values():
        for t in range(num_tasks):
            vals = y[task_names[t]][shift_len:-shift_len]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue

            # Histogram
            local_hist[t] += np.histogram(vals, bins=bins)[0]

            # Vectorized Welford update
            n = len(vals)
            mean_vals = np.mean(vals)
            delta = mean_vals - local_mean[t]
            total_count = local_count[t] + n

            # Update mean & M2
            local_mean[t] = local_mean[t] + delta * n / total_count
            local_M2[t] = (
                local_M2[t] + np.sum((vals - mean_vals) * (vals - local_mean[t])) 
                + delta ** 2 * local_count[t] * n / total_count
            )
            local_count[t] = total_count

            # Max
            local_max[t] = max(local_max[t], np.max(vals))
            local_min[t] = min(local_min[t], np.min(vals))

    return local_hist, local_count, local_mean, local_M2, local_max, local_min

all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]

num_workers = 16
print(f"Processing {len(all_files)} files using {num_workers} cores...")

with mp.Pool(processes=num_workers) as pool:
    results = list(tqdm(pool.imap(process_file, all_files), total=len(all_files)))

hist_counts = np.zeros((num_tasks, len(bins) - 1), dtype=np.int64)
count = np.zeros(num_tasks, dtype=np.int64)
mean = np.zeros(num_tasks, dtype=np.float64)
M2 = np.zeros(num_tasks, dtype=np.float64)
max_vals = np.full(num_tasks, -np.inf, dtype=np.float64)
min_vals = np.full(num_tasks, np.inf, dtype=np.float64)

for local_hist, local_count, local_mean, local_M2, local_max, local_min in results:
    # Merge histograms
    hist_counts += local_hist

    # Merge Welford stats
    for t in range(num_tasks):
        if local_count[t] == 0:
            continue
        total = count[t] + local_count[t]
        delta = local_mean[t] - mean[t]

        M2[t] = M2[t] + local_M2[t] + delta**2 * count[t] * local_count[t] / total
        mean[t] = (mean[t] * count[t] + local_mean[t] * local_count[t]) / total
        count[t] = total

        max_vals[t] = max(max_vals[t], local_max[t])
        min_vals[t] = min(min_vals[t], local_min[t])

std = np.sqrt(M2 / np.maximum(count - 1, 1))

percentiles_25 = np.zeros(num_tasks)
percentiles_50 = np.zeros(num_tasks)
percentiles_75 = np.zeros(num_tasks)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

for t in range(num_tasks):
    cumulative = np.cumsum(hist_counts[t])
    total = cumulative[-1]
    if total == 0:
        percentiles_25[t] = percentiles_50[t] = percentiles_75[t] = np.nan
        continue
    percentiles_25[t] = bin_centers[np.searchsorted(cumulative, 0.25 * total)]
    percentiles_50[t] = bin_centers[np.searchsorted(cumulative, 0.50 * total)]
    percentiles_75[t] = bin_centers[np.searchsorted(cumulative, 0.75 * total)]

stats_df = pd.DataFrame({
    "Task": task_names,
    "Mean": mean,
    "Std Dev": std,
    "Max": max_vals,
    "Min": min_vals,
    "25th Percentile": percentiles_25,
    "Median (50th)": percentiles_50,
    "75th Percentile": percentiles_75
})
stats_df.to_csv(os.path.join(output_folder, "stats.csv"), index=False)

print("Saving histograms...")

for t in range(num_tasks):
    plt.figure()
    plt.bar(bin_centers, hist_counts[t], width=1)
    plt.xlim([min_vals[t], max_vals[t]])
    plt.title(f"{task_names[t]} Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_folder, f"reg/{task_names[t]}_hist.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.bar(bin_centers, hist_counts[t], width=1)
    plt.yscale('log')
    plt.xlim([min_vals[t], max_vals[t]])
    plt.title(f"{task_names[t]} Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_folder, f"log/{task_names[t]}_hist.png"), dpi=300, bbox_inches='tight')
    plt.close()
