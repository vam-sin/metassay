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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

# # get mean and std of the mean of all the bins for each output type (fc, pc, aln)
# fc_means_list = []
# pv_means_list = []
# aln_means_list = []
# for task in tqdm(tasks):
#     for ch in chromosomes:
#         if 'aln' in task:
#             df_file = f'encode_dataset/proc_3k/bin_stats/{task}_{ch}.csv'
#             df = pd.read_csv(df_file)
#             means = df['mean'].tolist()
#             if 'fc' in task:
#                 fc_means_list.extend(means)
#             elif 'pv' in task:
#                 pv_means_list.extend(means)
#             elif 'aln' in task:
#                 aln_means_list.extend(means)

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
#     'fc': {'mean': 0.623366859516083, 'std': 0.9404246963283813},
#     'pv': {'mean': 0.7048715054568124, 'std': 7.292673462668909},
#     'aln': {'mean': 1.0364701598686537, 'std': 2.3632760703633244}
# }

# get task wise means
task_wise_means = {}
for task in tqdm(tasks):
    task_wise_means[task] = {}
    task_list = []
    for ch in chromosomes:
        df_file = f'encode_dataset/proc_3k/bin_stats/{task}_{ch}.csv'
        if os.path.exists(df_file):
            df_ch = pd.read_csv(df_file)
            means = df_ch['mean'].tolist()
            task_list.extend(means)
    task_wise_means[task] = np.mean(task_list)

# check that each chromosome in each task has a reasonable number of bins above cutoff
def get_bins_above_cutoff(task):
    cutoff = task_wise_means[task]
    
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

print(len(chrom_start_bins_lists))

print('--- Summary of Chromosome Start Bins Across All Tasks ---')

chrom_start_bins = []
fc_chrom_start_bins = []
pv_chrom_start_bins = []
aln_chrom_start_bins = []
fc_lists_size = []
pv_lists_size = []
aln_lists_size = []
# merge all lists
for task_dict in chrom_start_bins_lists:
    for task, bins in task_dict.items():
        if 'fc' in task:
            fc_chrom_start_bins.extend(bins)
            fc_lists_size.append(len(bins))
        elif 'pv' in task:
            pv_chrom_start_bins.extend(bins)
            pv_lists_size.append(len(bins))
        elif 'aln' in task:
            aln_chrom_start_bins.extend(bins)
            aln_lists_size.append(len(bins))
        chrom_start_bins.extend(bins)

# combine all lists with their types, then sort together
all_sizes = []
all_types = []
for size in fc_lists_size:
    all_sizes.append(size)
    all_types.append('FC')
for size in pv_lists_size:
    all_sizes.append(size)
    all_types.append('PV')
for size in aln_lists_size:
    all_sizes.append(size)
    all_types.append('ALN')

# sort together by size
sorted_indices = np.argsort(all_sizes)
sorted_sizes = [all_sizes[i] for i in sorted_indices]
sorted_types = [all_types[i] for i in sorted_indices]

# create color map
colors = {'FC': 'red', 'PV': 'blue', 'ALN': 'green'}
bar_colors = [colors[t] for t in sorted_types]

# calculate total bins
total_bins = sum(sorted_sizes)

max_bins_per_task = 2500

# plot the value of each of values in lists_sizes, and color by the output type, sort it, all together so the x axis range is 198, and make it a bar plot
plt.figure()
plt.bar(range(len(sorted_sizes)), sorted_sizes, color=bar_colors)
# create custom legend
legend_elements = [Patch(facecolor='red', label='FC'),
                   Patch(facecolor='blue', label='PV'),
                   Patch(facecolor='green', label='ALN')]
plt.legend(handles=legend_elements)
plt.xlabel('Task Index (sorted by size)')
plt.ylabel('Number of Bins')
# plt.ylim(0, max_bins_per_task)
plt.savefig('encode_dataset/proc_3k/lists_size_distribution.png', dpi=300, bbox_inches='tight')

print(f'Total Unique Chromosome Start Bins Across All Tasks: {len(set(chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins Across All FC Tasks: {len(set(fc_chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins Across All PV Tasks: {len(set(pv_chrom_start_bins))}')
print(f'Total Unique Chromosome Start Bins Across All ALN Tasks: {len(set(aln_chrom_start_bins))}')

# print the min number per output type
print('Min bins per task (global): ', min(sorted_sizes))

# downsample each task to max 100k bins
downsampled_chrom_start_bins = []
downsampled_fc_chrom_start_bins = []
downsampled_pv_chrom_start_bins = []
downsampled_aln_chrom_start_bins = []

print(f'\n--- Downsampling each task to max {max_bins_per_task:,} bins ---')
for task_dict in tqdm(chrom_start_bins_lists):
    for task, bins in task_dict.items():
        if len(bins) > max_bins_per_task:
            # randomly downsample to max_bins_per_task
            downsampled_bins = np.random.choice(bins, max_bins_per_task, replace=False).tolist()
        else:
            # keep all bins if under the limit
            downsampled_bins = bins
        
        downsampled_chrom_start_bins.extend(downsampled_bins)
        
        if 'fc' in task:
            downsampled_fc_chrom_start_bins.extend(downsampled_bins)
        elif 'pv' in task:
            downsampled_pv_chrom_start_bins.extend(downsampled_bins)
        elif 'aln' in task:
            downsampled_aln_chrom_start_bins.extend(downsampled_bins)

print(f'\n--- Summary After Downsampling Each Task to Max {max_bins_per_task:,} Bins ---')
print(f'Total Unique Chromosome Start Bins: {len(set(downsampled_chrom_start_bins)):,}')
print(f'Total Unique Chromosome Start Bins (FC): {len(set(downsampled_fc_chrom_start_bins)):,}')
print(f'Total Unique Chromosome Start Bins (PV): {len(set(downsampled_pv_chrom_start_bins)):,}')
print(f'Total Unique Chromosome Start Bins (ALN): {len(set(downsampled_aln_chrom_start_bins)):,}')

# # save all downsampled chrom start bins to a file into npz (250,476 bins)
np.savez_compressed('encode_dataset/proc_3k/uniqueBins.npz', bins=list(set(downsampled_chrom_start_bins)))