# libraries
'''
goes through all of the 198 tasks and does summary stats 
at a chromosome level and saves them in individual csv files
'''
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
folder_list = [f for f in folder_list if not f.endswith('.chrom.sizes') and not f.endswith('regions.bed') and not f.endswith('satellite_repeats.out')]
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

# Load satellite repeats from RepeatMasker output file
satellite_repeats_file = 'encode_dataset/dataset/satellite_repeats.out'
satellite_trees = {}

if os.path.exists(satellite_repeats_file):
    print(f"Loading satellite repeats from {satellite_repeats_file}...")
    # Collect intervals per chromosome (similar to blacklist pattern)
    satellite_data_by_chrom = {}
    with open(satellite_repeats_file, 'r') as f:
        for line in f:
            # Skip header lines (start with SW, score, or are blank)
            if line.startswith('SW') or line.startswith('score') or line.startswith('   ') or not line.strip():
                continue
            # Skip lines that don't have enough columns
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            try:
                chrom_name = parts[4]  # Column 5 (0-indexed: 4)
                start_pos = int(parts[5]) - 1  # Column 6, convert to 0-based
                end_pos = int(parts[6])  # Column 7, already 1-based
                repeat_class = parts[10]  # Column 11 (0-indexed: 10)
                
                # Only include satellite repeats
                if 'Satellite' in repeat_class:
                    if chrom_name not in satellite_data_by_chrom:
                        satellite_data_by_chrom[chrom_name] = []
                    satellite_data_by_chrom[chrom_name].append((start_pos, end_pos))
            except (ValueError, IndexError):
                continue
    
    # Build interval trees per chromosome using from_tuples (same as blacklist, fast)
    for chrom, intervals in satellite_data_by_chrom.items():
        satellite_trees[chrom] = IntervalTree.from_tuples(intervals)
    
    total_intervals = sum(len(intervals) for intervals in satellite_data_by_chrom.values())
    print(f"Loaded {total_intervals} satellite repeat intervals across {len(satellite_trees)} chromosomes")
else:
    print(f"Warning: {satellite_repeats_file} not found. Satellite repeat checking will be disabled.")

def checkSatelliteRepeats(chrom, start, end):
    """
    Return True if region overlaps a satellite repeat interval.
    
    Args:
        chrom: Chromosome name (e.g., 'chr1')
        start: Start position (0-based)
        end: End position (1-based)
    
    Returns:
        bool: True if region overlaps any satellite repeat, False otherwise
    """
    tree = satellite_trees.get(chrom)
    if tree is None:
        return False
    return bool(tree.overlap(start, end))

bin_size = 3000
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

def process_file(key, chromosomes, chromosome_lengths, bin_size):
    print(f'Processing file: {key}')
    bw_file = pyBigWig.open(key)

    for ch in chromosomes:
        per_chrom_res = []
        length = chromosome_lengths[ch]
        signal = load_signal_in_chunks(bw_file, ch, length)
        if np.all(np.isnan(signal)):
            continue

        # make a list of starts (which are spaced 3k bp apart)
        bin_starts = np.arange(0, length - bin_size, bin_size, dtype=int)
        bin_ends = bin_starts + bin_size

        for s, e in zip(bin_starts, bin_ends):
            # skip blacklist
            if checkBLR_fast(ch, s, e) or checkSatelliteRepeats(ch, s, e):
                continue

            values = signal[s:e]
            if np.all(np.isnan(values)):
                continue

            per_chrom_res.append({
                "task": os.path.basename(key),
                "chrom": ch,
                "bin_start": s,
                "max": np.nanmax(values),
                "median": np.nanmedian(values),
                "mean": np.nanmean(values)
            })
        
        # get stats for the file
        df = pd.DataFrame(per_chrom_res)
        df = df.astype({
            "bin_start": "int64",
            "max": "float32",
            "median": "float32",
            "mean": "float32"
        })

        df_file = f'encode_dataset/proc_3k/bin_stats/{os.path.basename(key)}_{ch}.csv'
        df.to_csv(df_file, index=False)

    bw_file.close()

# Parallelize across files
n_jobs = 32  # adjust to CPU count
Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_file)(key, chromosomes, chromosome_lengths, bin_size)
    for key in tqdm(file_list)
)