import pyBigWig
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import time
from utils import seq_loader

chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
in_size = 2048
out_size = 1024
num_task = 198

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
file_list = sorted(file_list)

# get all chromosome lengths and make the chromosome length dict
bw_file = pyBigWig.open(file_list[0])

chromosome_lengths = {}
for chrom in chromosomes:
    chromosome_lengths[chrom] = bw_file.chroms()[chrom]

bw_file.close()

bin_regions = np.load('encode_dataset/proc/uniqueBins.npz', allow_pickle=True)['bins'].astype(str)

print(f'Number of Unique Bin Regions: {len(bin_regions)}')

def load_signal_in_chunks(bw_file, chrom, length, chunk_size=10_000_000):
    signal = np.empty(length, dtype=np.float32)
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        signal[start:end] = np.array(bw_file.values(chrom, start, end))
    return signal

# id_str:{dna_seq: DNA, task1: values, task2: values, ...}
def process_file_values(key, chromosomes, chromosome_lengths):
    bw_file = pyBigWig.open(key)
    task = os.path.basename(key)
    seqloader_instance = seq_loader('hg38', 2048)
    local_data = {}
    print(f'Processing file: {task}')

    for ch in chromosomes:
        length = chromosome_lengths[ch]
        signal = load_signal_in_chunks(bw_file, ch, length)

        # get those in bin_regions for this chromosome
        chr_bins = [br for br in bin_regions if f'{ch}_' in br]
        cin_starts = [int(br.split('_')[1]) for br in chr_bins]
        cin_ends = [x + in_size for x in cin_starts]
        cout_starts = [x + out_size // 2 for x in cin_starts]
        cout_ends = [x + out_size for x in cout_starts]

        for s, e, cs, ce in zip(cin_starts, cin_ends, cout_starts, cout_ends):
            id_str = f"{ch}_{s}"

            values = signal[cs:ce]
            dna_seq = seqloader_instance.get_seq_start(ch, s, '+', ohe=False)  # one-hot encoded DNA sequence

            local_data[id_str] = {
                task: values,
                "dna_seq": dna_seq
            }

    bw_file.close()
    
    # save intermediate result
    print(f'Finished processing file: {task}, collected {len(local_data)} bins.')
    np.savez(f'encode_dataset/proc/binTasks/{os.path.basename(task)}.npz', local_data)

# Parallelize across files
n_jobs = 16  # adjust to CPU count
Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_file_values)(key, chromosomes, chromosome_lengths)
    for key in tqdm(file_list)
)
