import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tangermeme.utils import one_hot_encode
import json
from joblib import Parallel, delayed

chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']

# load tasks json
with open('encode_dataset/final/task_names.json', 'r') as f:
    task_names = json.load(f)

bin_regions = np.load('encode_dataset/proc/uniqueBins.npz', allow_pickle=True)['bins'].astype(str)
bin_regions = sorted(bin_regions)
bin_regions = [str(br) for br in bin_regions]

# split into chromosomes
bin_regions_by_chr = {chrom: [] for chrom in chromosomes}
for bin_region in bin_regions:
    chrom = bin_region.split('_')[0]
    bin_regions_by_chr[chrom].append(bin_region)

print(f'Number of Unique Bin Regions: {len(bin_regions)}')

assert len(bin_regions) == sum([len(bin_regions_by_chr[chrom]) for chrom in chromosomes]), "Mismatch in bin region counts!"

# sample output file
proc_folder = 'encode_dataset/proc/binTasks'

output_folder = 'encode_dataset/final/all'
chunk_size = 1000

def process_chrom(ch):
    chrom_data = {}
    for task in tqdm(task_names, desc='Loading Task Data'):
        chrom_data[task] = np.load(f'encode_dataset/proc/binTasksSep/{task}_{ch}.npz', allow_pickle=True)['arr_0'][()]
    
    for i in tqdm(range(0, len(bin_regions_by_chr[ch]), chunk_size), desc=f'Processing {task} {ch}'):
        chunk_bins = bin_regions_by_chr[ch][i:i+chunk_size]
        chunk_out_file = os.path.join(output_folder, f'bins_{ch}_{i}_{i+len(chunk_bins)}.npz')

        chunk_data = {}
        for bin_region in chunk_bins:
            chunk_data[bin_region] = {}
            chunk_data[bin_region]['dna_seq'] = one_hot_encode(chrom_data[task_names[0]][bin_region]['dna_seq'], seq_len=2048) 
        
            for task in task_names:
                chunk_data[bin_region][task] = chrom_data[task][bin_region][task]

        assert len(chunk_data) == len(chunk_bins)
        for j in range(len(chunk_bins)):
            assert len(chunk_data[chunk_bins[j]]) == 199 # 198 tasks + dna_seq
        
        np.savez_compressed(chunk_out_file, chunk_data)

n_jobs = 12  # adjust to CPU count
Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_chrom)(ch)
    for ch in tqdm(chromosomes)
)

print('Final data preparation complete.')