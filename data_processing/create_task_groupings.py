"""
Script to create task_groupings.json from encode_experiments.csv and task_names.json
Groups tasks by: cell type, output type, assay type, and individual tasks
"""
import pandas as pd
import json
import os
from collections import defaultdict

# Load task names
with open('encode_dataset/final_3k/task_names.json', 'r') as f:
    task_names = json.load(f)

print(f"Loaded {len(task_names)} task names")

# Load experiments CSV
df = pd.read_csv('encode_experiments.csv', sep='\t')

# Create mappings from experiment IDs to metadata
# Map: ENCFF ID -> (cell_type, assay, target, output_type)
experiment_to_metadata = {}

for _, row in df.iterrows():
    cell_type = row['Cell Type']
    assay = row['Assay']
    target = row['Target'] if pd.notna(row['Target']) else ''
    
    # ALN tasks are mapped via Experiments column (ENCSR IDs), not via ALN columns
    
    # Map PV columns (there are two PV columns in the CSV - check by index)
    # Column indices: 0=Cell Type, 1=Assay, 2=Target, 3=ALN1, 4=ALN2, 5=ALN3, 6=PV, 7=PV, 8=FC, 9=Experiments
    pv_indices = [6, 7]  # Two PV columns
    for idx in pv_indices:
        if idx < len(row):
            encff_id = row.iloc[idx] if hasattr(row, 'iloc') else row[idx]
            if pd.notna(encff_id) and encff_id != '-':
                experiment_to_metadata[encff_id] = (cell_type, assay, target, 'pv')
    
    # Map FC column
    if 'FC' in row:
        encff_id = row['FC']
        if pd.notna(encff_id) and encff_id != '-':
            experiment_to_metadata[encff_id] = (cell_type, assay, target, 'fc')

print(f"Mapped {len(experiment_to_metadata)} experiments")

# Build groupings
groupings = {
    'by_cell_type': defaultdict(lambda: {'task_names': [], 'task_indices': []}),
    'by_output_type': defaultdict(lambda: {'task_names': [], 'task_indices': []}),
    'by_assay_type': defaultdict(lambda: {'task_names': [], 'task_indices': []}),
    'by_task': {}
}

# Create ENCSR -> row mapping for ALN tasks
encsr_to_row = {}
for _, row in df.iterrows():
    encsr_id = row['Experiments']
    if pd.notna(encsr_id):
        encsr_to_row[encsr_id] = row

# Process each task name
for task_idx, task_name in enumerate(task_names):
    # Extract ID from task name
    # Format: "aln_ENCSR000AKE.bw" or "fc_ENCFF102ARJ.bigWig" or "pv_ENCFF012DMX.bigWig"
    parts = task_name.replace('.bw', '').replace('.bigWig', '').split('_')
    if len(parts) < 2:
        print(f"Warning: Could not parse task name: {task_name}")
        continue
    
    output_type = parts[0]  # 'aln', 'fc', or 'pv'
    encsr_or_encff = parts[1]  # ENCSR or ENCFF ID
    
    # Find metadata based on task type
    metadata = None
    if output_type == 'aln' and encsr_or_encff.startswith('ENCSR'):
        # For ALN tasks with ENCSR IDs, look up via Experiments column
        if encsr_or_encff in encsr_to_row:
            row = encsr_to_row[encsr_or_encff]
            cell_type = row['Cell Type']
            assay = row['Assay']
            target = row['Target'] if pd.notna(row['Target']) else ''
            metadata = (cell_type, assay, target, 'aln')
    elif encsr_or_encff.startswith('ENCFF'):
        # For FC/PV tasks with ENCFF IDs, direct lookup
        metadata = experiment_to_metadata.get(encsr_or_encff)
    
    # Add to by_task
    groupings['by_task'][task_name] = {
        'task_names': [task_name],
        'task_indices': [task_idx]
    }
    
    # Add to by_output_type
    groupings['by_output_type'][output_type]['task_names'].append(task_name)
    groupings['by_output_type'][output_type]['task_indices'].append(task_idx)
    
    # Add to by_cell_type and by_assay_type if we have metadata
    if metadata:
        cell_type, assay, target, _ = metadata
        groupings['by_cell_type'][cell_type]['task_names'].append(task_name)
        groupings['by_cell_type'][cell_type]['task_indices'].append(task_idx)
        
        # For assay type, use the assay name
        groupings['by_assay_type'][assay]['task_names'].append(task_name)
        groupings['by_assay_type'][assay]['task_indices'].append(task_idx)
    else:
        print(f"Warning: No metadata found for task: {task_name}")

# Convert defaultdicts to regular dicts
groupings['by_cell_type'] = dict(groupings['by_cell_type'])
groupings['by_output_type'] = dict(groupings['by_output_type'])
groupings['by_assay_type'] = dict(groupings['by_assay_type'])

# Print summary
print("\n=== Grouping Summary ===")
print(f"Cell types: {list(groupings['by_cell_type'].keys())}")
print(f"Output types: {list(groupings['by_output_type'].keys())}")
print(f"Assay types: {list(groupings['by_assay_type'].keys())}")
print(f"Total tasks: {len(groupings['by_task'])}")

for cell_type, info in groupings['by_cell_type'].items():
    print(f"  {cell_type}: {len(info['task_indices'])} tasks")
for output_type, info in groupings['by_output_type'].items():
    print(f"  {output_type}: {len(info['task_indices'])} tasks")
for assay_type, info in groupings['by_assay_type'].items():
    print(f"  {assay_type}: {len(info['task_indices'])} tasks")

# Save to file
output_path = 'encode_dataset/final_3k/task_groupings.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(groupings, f, indent=2)

print(f"\nSaved task_groupings.json to {output_path}")

