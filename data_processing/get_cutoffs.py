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
folder_list = [f for f in folder_list if not f.endswith('.chrom.sizes') and not f.endswith('regions.bed')]
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

# means_pv = []
# # PV processing
# for task in tqdm(pv_tasks, desc='Processing PV tasks'):
#     # load in all the chromosome bin stats files for this task, and get the mean +- stds for cutoff calculation
#     for ch in chromosomes:
#         df_file = f'encode_dataset/proc/bin_stats/{task}_{ch}.csv'
#         if os.path.exists(df_file):
#             df_ch = pd.read_csv(df_file)
#             means_pv.extend(list(df_ch['mean']))

# means_pv = np.array(means_pv)

# print(f'PV Tasks - Overall Mean: {means_pv.mean()}, Overall Std: {means_pv.std()}. (Bins Perc Above Mean: {100 * np.sum(means_pv > means_pv.mean()) / len(means_pv):.2f}%)')
# print(f'PV Tasks (Mean + 1*Std): {means_pv.mean() + 1 * means_pv.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_pv > (means_pv.mean() + 1 * means_pv.std())) / len(means_pv):.2f}%)')
# print(f'PV Tasks (Mean + 2*Std): {means_pv.mean() + 2 * means_pv.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_pv > (means_pv.mean() + 2 * means_pv.std())) / len(means_pv):.2f}%)')
# print(f'PV Tasks (Mean + 3*Std): {means_pv.mean() + 3 * means_pv.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_pv > (means_pv.mean() + 3 * means_pv.std())) / len(means_pv):.2f}%)')

# means_aln = []
# # ALN processing
# for task in tqdm(aln_tasks, desc='Processing ALN tasks'):
#     # load in all the chromosome bin stats files for this task, and get the mean +- stds for cutoff calculation
#     for ch in chromosomes:
#         df_file = f'encode_dataset/proc/bin_stats/{task}_{ch}.csv'
#         if os.path.exists(df_file):
#             df_ch = pd.read_csv(df_file)
#             means_aln.extend(list(df_ch['mean']))

# means_aln = np.array(means_aln)

# print(f'ALN Tasks - Overall Mean: {means_aln.mean()}, Overall Std: {means_aln.std()}. (Bins Perc Above Mean: {100 * np.sum(means_aln > means_aln.mean()) / len(means_aln):.2f}%)')
# print(f'ALN Tasks (Mean + 1*Std): {means_aln.mean() + 1 * means_aln.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_aln > (means_aln.mean() + 1 * means_aln.std())) / len(means_aln):.2f}%)')
# print(f'ALN Tasks (Mean + 2*Std): {means_aln.mean() + 2 * means_aln.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_aln > (means_aln.mean() + 2 * means_aln.std())) / len(means_aln):.2f}%)')
# print(f'ALN Tasks (Mean + 3*Std): {means_aln.mean() + 3 * means_aln.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_aln > (means_aln.mean() + 3 * means_aln.std())) / len(means_aln):.2f}%)')

# means_fc = []
# # FC processing
# for task in tqdm(fc_tasks, desc='Processing FC tasks'):
#     # load in all the chromosome bin stats files for this task, and get the mean +- stds for cutoff calculation
#     for ch in chromosomes:
#         df_file = f'encode_dataset/proc/bin_stats/{task}_{ch}.csv'
#         if os.path.exists(df_file):
#             df_ch = pd.read_csv(df_file)
#             means_fc.extend(list(df_ch['mean']))

# means_fc = np.array(means_fc)

# print(f'FC Tasks - Overall Mean: {means_fc.mean()}, Overall Std: {means_fc.std()}. (Bins Perc Above Mean: {100 * np.sum(means_fc > means_fc.mean()) / len(means_fc):.2f}%)')
# print(f'FC Tasks (Mean + 1*Std): {means_fc.mean() + 1 * means_fc.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_fc > (means_fc.mean() + 1 * means_fc.std())) / len(means_fc):.2f}%)')
# print(f'FC Tasks (Mean + 2*Std): {means_fc.mean() + 2 * means_fc.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_fc > (means_fc.mean() + 2 * means_fc.std())) / len(means_fc):.2f}%)')
# print(f'FC Tasks (Mean + 3*Std): {means_fc.mean() + 3 * means_fc.std()} (Bins Perc Above Cutoff: {100 * np.sum(means_fc > (means_fc.mean() + 3 * means_fc.std())) / len(means_fc):.2f}%)')

'''
FC Tasks - Overall Mean: 0.6233088767015336, Overall Std: 1.1291127701216044
FC Tasks (Mean + 1*Std): 1.752421646823138 (Bins Perc Above Cutoff: 3.51%)
FC Tasks (Mean + 2*Std): 2.881534416944742 (Bins Perc Above Cutoff: 1.54%)
FC Tasks (Mean + 3*Std): 4.010647187066347 (Bins Perc Above Cutoff: 0.90%)

PV Tasks - Overall Mean: 0.7049074885691617, Overall Std: 10.938644432346093. (Bins Perc Above Mean: 7.46%)
PV Tasks (Mean + 1*Std): 11.643551920915256 (Bins Perc Above Cutoff: 0.69%)
PV Tasks (Mean + 2*Std): 22.58219635326135 (Bins Perc Above Cutoff: 0.38%)
PV Tasks (Mean + 3*Std): 33.52084078560744 (Bins Perc Above Cutoff: 0.26%)

ALN Tasks - Overall Mean: 1.0188139424530114, Overall Std: 5.983112741975139. (Bins Perc Above Mean: 10.20%)
ALN Tasks (Mean + 1*Std): 7.001926684428151 (Bins Perc Above Cutoff: 0.01%)
ALN Tasks (Mean + 2*Std): 12.98503942640329 (Bins Perc Above Cutoff: 0.00%)
ALN Tasks (Mean + 3*Std): 18.968152168378428 (Bins Perc Above Cutoff: 0.00%)
'''

cutoffs = {
    'fc': {'mean': 0.6233088767015336, 'std': 1.1291127701216044},
    'pv': {'mean': 0.7049074885691617, 'std': 10.938644432346093},
    'aln': {'mean': 1.0188139424530114, 'std': 5.983112741975139}
}

# check that each chromosome in each task has a reasonable number of bins above cutoff
def get_bins_above_cutoff(task):
    if 'fc' in task:
        cutoff = 2.0
    elif 'pv' in task:
        cutoff = cutoffs['pv']['mean'] + 0.05 * cutoffs['pv']['std']
    elif 'aln' in task:
        cutoff = cutoffs['aln']['mean'] + 0.015 * cutoffs['aln']['std']
    
    # print(f'--- Task: {task}, Cutoff: {cutoff} ---')
    total_bins_above_cutoff = 0
    chroms_start_bins = []
    for ch in chromosomes:
        df_file = f'encode_dataset/proc/bin_stats/{task}_{ch}.csv'
        if os.path.exists(df_file):
            df_ch = pd.read_csv(df_file)
            bins_above_cutoff = np.sum(df_ch['mean'] > cutoff)
            total_bins_above_cutoff += bins_above_cutoff
            # merge chrom and in_start to get chrom_start
            df_ch['chrom_start'] = df_ch['chrom'] + '_' + df_ch['in_start'].astype(str)
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

# save all downsampled chrom start bins to a file into npz
np.savez_compressed('encode_dataset/proc/uniqueBins.npz', bins=list(set(downsampled_chrom_start_bins)))