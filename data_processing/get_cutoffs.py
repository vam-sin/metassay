# libraries
'''
get_cutoffs.py
script to analyse the csv files obtained from bin_stats.py to set cutoffs that would
obtain a similar number of bins from all 3 output types. 
downsampling is conducted so that no specific chromosome has a super high number of bins from that task. (done after setting the thresh for fc, and then pv/aln follow that)
output: uniqueBins.npz (file that consists of all the bins that were chosen across all the 22 chromosomes)
dont have to check blacklist regions here as we are only checking those bins that are not in the blacklist regions. (removed in the bin_stats.py script)
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import random 

# set random seed for reproducibility
random.seed(42)
np.random.seed(42)

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

tasks = [os.path.basename(f) for f in file_list]
fc_tasks = [f for f in tasks if 'fc' in f]
pv_tasks = [f for f in tasks if 'pv' in f]
aln_tasks = [f for f in tasks if 'aln' in f]

print(f'Number of Tasks (Total): {len(tasks)}')  # should be 198
print(f'Number of Tasks (FC): {len(fc_tasks)}')  # should be 63
print(f'Number of Tasks (PV): {len(pv_tasks)}')  # should be 68
print(f'Number of Tasks (ALN): {len(aln_tasks)}')  # should be 67

assert len(fc_tasks) + len(pv_tasks) + len(aln_tasks) == len(tasks), "Task categorization error."

# get mean and std of the mean of all the bins for each output type (fc, pc, aln)
# fc_means_list = []
# pv_means_list = []
# aln_means_list = []
# for task in tqdm(tasks):
#     for ch in chromosomes:
#         df_file = f'encode_dataset/proc_3k/bin_stats/{task}_{ch}.csv'
#         df = pd.read_csv(df_file)
#         means = df['mean'].tolist()
#         if 'fc' in task:
#             fc_means_list.extend(means)
#         elif 'pv' in task:
#             pv_means_list.extend(means)
#         elif 'aln' in task:
#             aln_means_list.extend(means)

# fc_means = np.mean(fc_means_list)
# pv_means = np.mean(pv_means_list)
# aln_means = np.mean(aln_means_list)

# fc_std = np.std(fc_means_list)
# pv_std = np.std(pv_means_list)
# aln_std = np.std(aln_means_list)

# print(f'FC Means: {fc_means}')  
# print(f'PV Means: {pv_means}')
# print(f'ALN Means: {aln_means}')
# print(f'FC Std: {fc_std}')
# print(f'PV Std: {pv_std}')
# print(f'ALN Std: {aln_std}')

# cutoffs = {
#     'fc': {'mean': fc_means, 'std': fc_std},
#     'pv': {'mean': pv_means, 'std': pv_std},
#     'aln': {'mean': aln_means, 'std': aln_std}
# }

cutoffs = {
    'fc': {'mean': 0.6241491606827647, 'std': 0.9414642527728895},
    'pv': {'mean': 0.7038336184982527, 'std': 7.333628925645824},
    'aln': {'mean': 1.022139049185703, 'std': 6.443636625764475}
}

# check that each chromosome in each task has a reasonable number of bins above cutoff
def get_bins_above_cutoff(task):
    if 'fc' in task:
        cutoff = 2.0
    elif 'pv' in task:
        cutoff = cutoffs['pv']['mean'] + 0.00 * cutoffs['pv']['std']
    elif 'aln' in task:
        cutoff = cutoffs['aln']['mean'] + 0.00 * cutoffs['aln']['std']
    
    # print(f'--- Task: {task}, Cutoff: {cutoff} ---')
    total_bins_above_cutoff = 0
    chroms_start_bins = []
    for ch in chromosomes:
        df_file = f'encode_dataset/proc_3k/bin_stats/{task}_{ch}.csv'
        if os.path.exists(df_file):
            df_ch = pd.read_csv(df_file)
            bins_above_cutoff = np.sum(df_ch['mean'] > cutoff)
            total_bins_above_cutoff += bins_above_cutoff
            # merge chrom and in_start to get chrom_start
            df_ch['chrom_start'] = df_ch['chrom'] + '_' + df_ch['bin_start'].astype(str)
            chroms_start_bins.extend(df_ch[df_ch['mean'] > cutoff]['chrom_start'].tolist())
    
    # print(f'Total Bins Above Cutoff for Task {task}: {total_bins_above_cutoff}')

    return {task: chroms_start_bins}

chrom_start_bins_lists = Parallel(n_jobs=128)(delayed(get_bins_above_cutoff)(task) for task in tqdm(tasks))

print('--- Summary of Chromosome Start Bins Across All Tasks ---')

chrom_start_bins = []
fc_chrom_start_bins = []
pv_chrom_start_bins = []
aln_chrom_start_bins = []
# merge all lists
for task_dict in chrom_start_bins_lists:
    for task, bins in task_dict.items():
        if 'fc' in task:
            fc_chrom_start_bins.extend(bins)
        elif 'pv' in task:
            pv_chrom_start_bins.extend(bins)
        elif 'aln' in task:
            aln_chrom_start_bins.extend(bins)

        chrom_start_bins.extend(bins)

print(f'Total Unique Chromosome Start Bins Across All Tasks: {len(set(chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins Across All FC Tasks: {len(set(fc_chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins Across All PV Tasks: {len(set(pv_chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins Across All ALN Tasks: {len(set(aln_chrom_start_bins))}')

# get min for fc, pv, and aln tasks
min_fc = float('inf')
min_pv = float('inf')
min_aln = float('inf')
for task_dict in chrom_start_bins_lists:
    for task, bins in task_dict.items():
        if 'fc' in task:
            min_fc = min(min_fc, len(bins))
        elif 'pv' in task:
            min_pv = min(min_pv, len(bins))
        elif 'aln' in task:
            min_aln = min(min_aln, len(bins))

# # downsample each of the tasks to their respective mins
downsampled_chrom_start_bins = []
downsamples_fc_chrom_start_bins = []
downsamples_pv_chrom_start_bins = []
downsamples_aln_chrom_start_bins = []

for task_dict in chrom_start_bins_lists:
    for task, bins in task_dict.items():
        if 'fc' in task:
            downsampled_bins = np.random.choice(bins, min_fc, replace=False).tolist()
        elif 'pv' in task:
            downsampled_bins = np.random.choice(bins, min_pv, replace=False).tolist()
        elif 'aln' in task:
            downsampled_bins = np.random.choice(bins, min_aln, replace=False).tolist()
        downsampled_chrom_start_bins.extend(downsampled_bins)

        if 'fc' in task:
            downsamples_fc_chrom_start_bins.extend(downsampled_bins)
        elif 'pv' in task:
            downsamples_pv_chrom_start_bins.extend(downsampled_bins)
        elif 'aln' in task:
            downsamples_aln_chrom_start_bins.extend(downsampled_bins)

print('--- Summary of Downsampled Chromosome Start Bins Across All Tasks ---')
print(f'Total Unique Chromosome Start Bins After Downsampling: {len(set(downsampled_chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins After Downsampling (FC): {len(set(downsamples_fc_chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins After Downsampling (PV): {len(set(downsamples_pv_chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins After Downsampling (ALN): {len(set(downsamples_aln_chrom_start_bins))}')

# save all downsampled chrom start bins to a file into npz (250,476 bins)
np.savez_compressed('encode_dataset/proc_3k/uniqueBins.npz', bins=list(set(downsampled_chrom_start_bins)))