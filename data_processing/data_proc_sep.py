import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=8)
parser.add_argument('--section', type=int, default=0)
args = parser.parse_args()

chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']

# load tasks json
with open('encode_dataset/final_3k/task_names.json', 'r') as f:
    task_names = json.load(f)

bin_regions = np.load('encode_dataset/proc_3k/uniqueBins.npz', allow_pickle=True)['bins'].astype(str)
bin_regions = sorted(bin_regions)
bin_regions = [str(br) for br in bin_regions]

# split into chromosomes
bin_regions_by_chr = {chrom: [] for chrom in chromosomes}
for bin_region in bin_regions:
    chrom = bin_region.split('_')[0]
    bin_regions_by_chr[chrom].append(bin_region)

print(f'Number of Unique Bin Regions: {len(bin_regions)}')

assert len(bin_regions) == sum([len(bin_regions_by_chr[chrom]) for chrom in chromosomes]), "Mismatch in bin region counts!"

file_path = 'encode_dataset/proc_3k/binTasks'
folder_list = os.listdir(file_path)
# remove .chrom.sizes file
file_list = [f for f in folder_list]
file_list = [file_path + '/' + f for f in file_list]

if args.section == 0:
    file_list = file_list[:50]
elif args.section == 1:
    file_list = file_list[50:100]
elif args.section == 2:
    file_list = file_list[100:150]
elif args.section == 3:
    file_list = file_list[150:200]
else:
    raise ValueError(f'Invalid section: {args.section}')

print(f'Num files: {len(file_list)}')

# go through each of the files and split by chromosome and save in 'binTasksSep'
def process_file(file_loc):
    key = os.path.basename(file_loc).replace('.npz', '')
    print(f'Processing file: {key}')
    data = np.load(file_loc, allow_pickle=True)['arr_0'][()]

    for ch in chromosomes:
        data_chrom = {}
        for x in bin_regions_by_chr[ch]:
            data_chrom[x] = data[x]
        np.savez_compressed(f'encode_dataset/proc_3k/binTasksSep/{key}_{ch}.npz', data_chrom)

n_jobs = 32  # adjust to CPU count
Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_file)(f)
    for f in tqdm(file_list)
)