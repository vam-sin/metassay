import pyBigWig
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from intervaltree import IntervalTree
import pyranges as pr
from joblib import Parallel, delayed
import time

chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']

# data stats
file_path = 'encode_dataset/dataset/'
folder_list = os.listdir(file_path)
# remove .chrom.sizes file
folder_list = [f for f in folder_list if not f.endswith('.chrom.sizes') and not f.endswith('regions.bed')]
# get all files in the folders
file_list = []
for folder in folder_list:
    files = os.listdir(os.path.join(file_path, folder))
    for f in files:
        file_list.append(os.path.join(file_path, folder, f))

print(f'Number of Tasks: {len(file_list)}')  # should be 198

# Load blacklist once
blr_df = pr.read_bed('encode_dataset/dataset/blacklist_regions.bed').df

# Build one interval tree per chromosome
blacklist_trees = {}
for chrom, group in blr_df.groupby("Chromosome"):
    blacklist_trees[chrom] = IntervalTree.from_tuples(zip(group.Start, group.End))

def checkBLR_fast(chrom, start, end):
    """Return True if region overlaps a blacklist interval."""
    tree = blacklist_trees.get(chrom)
    if tree is None:
        return False
    return bool(tree.overlap(start, end))

in_size = 2048
out_size = 1024
num_task = 198

matrix_dict = {}

# get all chromosome lengths and make the chromosome length dict
bw_file = pyBigWig.open(file_list[0])

chromosome_lengths = {}
for chrom in chromosomes:
    chromosome_lengths[chrom] = bw_file.chroms()[chrom]

bw_file.close()

def load_signal_in_chunks(bw_file, chrom, length, chunk_size=10_000_000):
    signal = np.empty(length, dtype=np.float32)
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        signal[start:end] = np.array(bw_file.values(chrom, start, end))
    return signal

def process_file(key, chromosomes, chromosome_lengths, out_size):
    print(f'Processing file: {key}')
    bw_file = pyBigWig.open(key)

    for ch in chromosomes:
        per_chrom_res = []
        length = chromosome_lengths[ch]
        signal = load_signal_in_chunks(bw_file, ch, length)
        if np.all(np.isnan(signal)):
            continue

        starts = np.arange(out_size // 2, length - int(out_size * 1.5), out_size, dtype=int)
        cout_starts = starts
        cout_ends = starts + out_size
        cin_starts = starts - out_size // 2
        cin_ends = starts + out_size + out_size // 2

        for s, e, cs, ce in zip(cin_starts, cin_ends, cout_starts, cout_ends):
            # skip blacklist
            if checkBLR_fast(ch, s, e):
                continue

            values = signal[cs:ce]
            if np.all(np.isnan(values)):
                continue

            per_chrom_res.append({
                "task": os.path.basename(key),
                "chrom": ch,
                "in_start": s,
                "max": np.nanmax(values),
                "median": np.nanmedian(values),
                "mean": np.nanmean(values)
            })
        
        # get stats for the file
        df = pd.DataFrame(per_chrom_res)
        df = df.astype({
            "in_start": "int64",
            "max": "float32",
            "median": "float32",
            "mean": "float32"
        })

        df_file = f'encode_dataset/proc/bin_stats/{os.path.basename(key)}_{ch}.csv'
        df.to_csv(df_file, index=False)

    bw_file.close()

# Parallelize across files
n_jobs = 64  # adjust to CPU count
Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_file)(key, chromosomes, chromosome_lengths, out_size)
    for key in tqdm(file_list)
)

# # flatten results
# print("Flattening results...")
# flat_results = [r for sublist in all_results for r in sublist]

# print(len(flat_results))

# # save the data as a npz
# print("Saving results...")
# np.savez('encode_dataset/proc/data_stats.npz', flat_results)

# 509184788 total segments