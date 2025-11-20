# libraries
import pandas as pd 
import numpy as np
import torch
import json
import os 
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import itertools
import time
import lightning as L
from joblib import Parallel, delayed
import hashlib
from collections import defaultdict
import random

# Import architectures
try:
    from architectures import (
        build_unet_architecture, build_chrombpnet_architecture, build_resnet_architecture,
        forward_unet, forward_chrombpnet, forward_resnet
    )
except ImportError:
    # If relative import fails, try absolute import from models package
    from models.architectures import (
        build_unet_architecture, build_chrombpnet_architecture, build_resnet_architecture,
        forward_unet, forward_chrombpnet, forward_resnet
    )

# Load task groupings for task-wise metrics
_task_groupings = None
_task_names = None
_excluded_aln_tasks = None

def _load_task_groupings():
    """Load task groupings once and cache them."""
    global _task_groupings, _task_names
    if _task_groupings is None:
        with open('../data_processing/encode_dataset/final_3k/task_groupings.json', 'r') as f:
            _task_groupings = json.load(f)
        with open('../data_processing/encode_dataset/final_3k/task_names.json', 'r') as f:
            _task_names = json.load(f)
    return _task_groupings, _task_names

def _get_excluded_aln_tasks():
    """Get set of ALN tasks to exclude (Histone CHIP-Seq and PRO-cap ALN tasks)."""
    global _excluded_aln_tasks
    if _excluded_aln_tasks is None:
        groupings, _ = _load_task_groupings()
        excluded = set()
        
        # Get ALN tasks from Histone CHIP-Seq
        if 'by_assay_type' in groupings and 'Histone CHIP-Seq' in groupings['by_assay_type']:
            histone_tasks = groupings['by_assay_type']['Histone CHIP-Seq']['task_names']
            excluded.update([t for t in histone_tasks if t.startswith('aln_')])
        
        # Get ALN tasks from PRO-cap
        if 'by_assay_type' in groupings and 'PRO-cap' in groupings['by_assay_type']:
            procap_tasks = groupings['by_assay_type']['PRO-cap']['task_names']
            excluded.update([t for t in procap_tasks if t.startswith('aln_')])
        
        _excluded_aln_tasks = excluded
    
    return _excluded_aln_tasks

def compute_task_wise_metrics(y_pred, y_true, mask, include_individual_tasks=True):
    """
    Compute task-wise metrics grouped by different criteria.
    Returns dict with metrics for each grouping.
    
    Args:
        y_pred: Predictions tensor [B, seq_len, num_tasks]
        y_true: Targets tensor [B, seq_len, num_tasks]
        mask: Valid values mask [B, seq_len, num_tasks]
        include_individual_tasks: If True, include by_task metrics (default: True)
    """
    groupings, task_names = _load_task_groupings()
    metrics = {}
    
    # Convert to numpy for easier indexing
    y_pred_np = y_pred.detach().cpu().numpy()  # [B, seq_len, num_tasks]
    y_true_np = y_true.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    
    # Flatten batch and sequence dimensions
    y_pred_flat = y_pred_np.reshape(-1, y_pred_np.shape[-1])  # [B*seq_len, num_tasks]
    y_true_flat = y_true_np.reshape(-1, y_true_np.shape[-1])
    mask_flat = mask_np.reshape(-1, mask_np.shape[-1])
    
    # Determine which groupings to compute
    groupings_to_compute = ['by_cell_type', 'by_output_type', 'by_assay_type']
    if include_individual_tasks:
        groupings_to_compute.append('by_task')
    
    # Compute metrics for each grouping
    for group_name, groups in groupings.items():
        if group_name not in groupings_to_compute:
            continue  # Only log specific groupings (separate, not combined)
        
        group_metrics = {}
        for group_key, group_info in groups.items():
            task_indices = group_info['task_indices']
            if not task_indices:
                continue
            
            # Get predictions and targets for tasks in this group
            group_preds = []
            group_trues = []
            
            for task_idx in task_indices:
                if task_idx >= y_pred_flat.shape[1]:
                    continue
                task_pred = y_pred_flat[:, task_idx]
                task_true = y_true_flat[:, task_idx]
                task_mask = mask_flat[:, task_idx]
                
                # Only use valid (non-NaN) values
                valid = task_mask.astype(bool)
                if valid.sum() > 0:
                    group_preds.append(task_pred[valid])
                    group_trues.append(task_true[valid])
            
            if not group_preds:
                continue
            
            # Concatenate all valid predictions and targets for this group
            all_preds = np.concatenate(group_preds)
            all_trues = np.concatenate(group_trues)
            
            if len(all_preds) == 0:
                continue
            
            # Compute Pearson correlation
            try:
                pred_mean = all_preds.mean()
                true_mean = all_trues.mean()
                pred_centered = all_preds - pred_mean
                true_centered = all_trues - true_mean
                
                numerator = (pred_centered * true_centered).sum()
                pred_std = np.sqrt((pred_centered ** 2).sum())
                true_std = np.sqrt((true_centered ** 2).sum())
                
                if pred_std > 1e-8 and true_std > 1e-8:
                    pcc = numerator / (pred_std * true_std)
                else:
                    pcc = 0.0
                
                # Compute MSE
                mse = np.mean((all_preds - all_trues) ** 2)
                
                # Store metrics with sanitized group key (for wandb logging)
                # For individual tasks, use a shorter key format
                if group_name == 'by_task':
                    # For task names like "aln_ENCFF008LEY.bw", use just "aln_ENCFF008LEY"
                    safe_key = group_key.replace('.bw', '').replace('.bigWig', '').replace(' ', '_').replace('/', '_').replace('-', '_').replace('+', '_').replace('(', '').replace(')', '')
                else:
                    safe_key = group_key.replace(' ', '_').replace('/', '_').replace('-', '_')
                
                group_metrics[f'{safe_key}_pcc'] = float(pcc)
                group_metrics[f'{safe_key}_mse'] = float(mse)
            except Exception as e:
                # Skip if computation fails
                continue
        
        if group_metrics:
            metrics[group_name] = group_metrics
    
    return metrics

def _process_sample(sample, task_names, i_s, o_s, tc, hc):
    """Helper function to process a single sample (for parallel processing)."""
    try:
        X = sample['dna_seq'].float()[:, i_s:-i_s]  # should already be ohe
        
        # Get excluded ALN tasks (Histone CHIP-Seq and PRO-cap)
        excluded_tasks = _get_excluded_aln_tasks()
        
        # Filter out excluded tasks - only stack the tasks we want to predict
        filtered_task_names = [t for t in task_names if t not in excluded_tasks]
        y = torch.from_numpy(np.stack([sample[t] for t in filtered_task_names])).permute(1, 0)[o_s:-o_s, :]

        # hard-clipping the targets: f(x) = min(x, hc)
        y = torch.clamp(y, max=hc)

        # set negative values to 0
        y = torch.clamp(y, min=0.0)

        # soft-clipping the targets: f(x) = min(x, tc + sqrt(max(0, x - tc))) where tc = 32
        y_clipped = torch.clamp(y, max=tc + torch.sqrt(torch.clamp(y - tc, min=0.0)))
        # Preserve NaN values (don't clip NaNs)
        y = torch.where(torch.isnan(y), y, y_clipped)

        return X, y
    except Exception as e:
        print(f"Warning: Could not process sample: {e}")
        return None

def _load_and_process_file(file_path, folder, task_names, i_s, o_s, tc, hc):
    """Helper function to load a file and process all samples (for parallel processing)."""
    try:
        full_path = os.path.join(folder, file_path)
        data = np.load(full_path, allow_pickle=True)['arr_0'][()]
        
        processed_samples = []
        for key, sample in data.items():
            processed = _process_sample(sample, task_names, i_s, o_s, tc, hc)
            if processed is not None:
                processed_samples.append(processed)
        
        return processed_samples
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return []

class ChromDS(Dataset):
    def __init__(self, chroms_list, input_size=2048, output_size=1024, reverse_compl=False, random_shift=False, n_jobs_load=16, data_cache_dir=None, dataset_downsampling=1.0, section=None):
        """
        In-memory dataset that loads and pre-processes all data during initialization.
        
        Args:
            chroms_list: List of chromosome patterns to match (e.g., ['_chr1_', '_chr2_'])
            input_size: Input sequence size
            output_size: Output sequence size
            reverse_compl: Whether to use reverse complement augmentation (not implemented yet)
            random_shift: Whether to use random shifting (not implemented yet)
            n_jobs_load: Number of parallel jobs for loading and processing (default: 16)
            data_cache_dir: Directory to cache processed data (default: None, no caching)
            dataset_downsampling: Fraction of dataset to use (default: 1.0, use 0.5 for half the dataset)
        """
        self.folder = '../data_processing/encode_dataset/final_3k/all_npz'
        
        # Find all relevant files (exclude cache directory and non-npz files)
        all_files = os.listdir(self.folder)
        # Filter out directories and ensure we only get .npz files
        all_files = [f for f in all_files 
                    if os.path.isfile(os.path.join(self.folder, f)) 
                    and f.endswith('.npz')]
        
        self.files_ds = []
        for ch in chroms_list:
            ch_list = [f for f in all_files if ch in f]
            self.files_ds.extend(sorted(ch_list))  # Sort for reproducibility
        
        # load tasks json
        with open('../data_processing/encode_dataset/final_3k/task_names.json', 'r') as f:
            all_task_names = json.load(f)
        
        # Get excluded ALN tasks (Histone CHIP-Seq and PRO-cap)
        excluded_tasks = _get_excluded_aln_tasks()
        
        # Filter out excluded tasks - dataset will only have 167 tasks
        self.task_names = [t for t in all_task_names if t not in excluded_tasks]
        print(f"Filtered out {len(excluded_tasks)} ALN tasks (Histone CHIP-Seq and PRO-cap)")
        print(f"Using {len(self.task_names)} tasks (down from {len(all_task_names)})")

        self.i_s = int((3000-input_size)/2)
        self.o_s = int((3000-output_size)/2)

        # transforms
        self.reverse_compl = reverse_compl
        self.random_shift = random_shift
        self.dataset_downsampling = dataset_downsampling

        # soft-clip
        self.tc = 32.0

        # hard-clip
        self.hc = 1e+5
        
        # Check for cached processed data
        cache_file = None
        if data_cache_dir is not None:
            if section is not None:
                cache_file = os.path.join(data_cache_dir, f'data_{section}.npz')
            else:
                os.makedirs(data_cache_dir, exist_ok=True)
                cache_key = hashlib.md5(
                    (str(sorted(self.files_ds)) + str(input_size) + str(output_size) + 
                    str(self.reverse_compl) + str(self.random_shift) + str(dataset_downsampling)).encode()
                ).hexdigest()
                cache_file = os.path.join(data_cache_dir, f'data_{cache_key}.npz')
            
            if os.path.exists(cache_file):
                print(f"Loading cached processed data from {cache_file}...")
                try:
                    cache_data = np.load(cache_file, allow_pickle=True)
                    # Load X and y arrays
                    X_list = cache_data['X_list']
                    y_list = cache_data['y_list']
                    # Convert back to list of tuples (X, y) as tensors
                    self.data = [(torch.from_numpy(X), torch.from_numpy(y)) 
                                for X, y in zip(X_list, y_list)]
                    print(f"Loaded {len(self.data)} pre-processed samples from cache")
                    # Note: Cache already contains downsampled data if dataset_downsampling < 1.0 was used
                    return
                except Exception as e:
                    print(f"Failed to load cache: {e}. Processing data...")
        
        # Load and process all data
        print(f"Loading and processing {len(self.files_ds)} files (using {n_jobs_load} workers)...")
        print("This may take several minutes. All data will be loaded into memory.")
        
        # Parallel loading and processing
        # Need to pass all_task_names to _process_sample so it can filter correctly
        with open('../data_processing/encode_dataset/final_3k/task_names.json', 'r') as f:
            all_task_names = json.load(f)
        
        processed_chunks = Parallel(n_jobs=n_jobs_load)(
            delayed(_load_and_process_file)(
                file_path, self.folder, all_task_names, 
                self.i_s, self.o_s, self.tc, self.hc
            )
            for file_path in tqdm(self.files_ds, desc="Loading and processing files")
        )
        
        # Flatten the list of lists
        self.data = []
        for chunk in processed_chunks:
            self.data.extend(chunk)
        
        print(f"Loaded and processed {len(self.data)} samples into memory")
        
        # Apply dataset downsampling if requested (BEFORE saving to cache)
        if dataset_downsampling < 1.0:
            original_size = len(self.data)
            target_size = int(len(self.data) * dataset_downsampling)
            # Randomly sample without replacement
            random.seed(42)  # For reproducibility
            indices = random.sample(range(len(self.data)), target_size)
            self.data = [self.data[i] for i in sorted(indices)]  # Sort to maintain some order
            print(f"Downsampled dataset from {original_size} to {len(self.data)} samples ({dataset_downsampling*100:.1f}%)")
        
        # Save processed data to cache if requested (as npz) - cache already contains downsampled data
        if cache_file is not None:
            print(f"Saving processed data to cache: {cache_file}")
            try:
                # Convert tensors to numpy arrays for saving
                X_list = [X.numpy() for X, y in self.data]
                y_list = [y.numpy() for X, y in self.data]
                # Save as npz (already downsampled if requested)
                np.savez(cache_file, X_list=X_list, y_list=y_list)
                print("Processed data cached successfully!")
            except Exception as e:
                print(f"Warning: Failed to cache processed data: {e}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a pre-processed sample by index.
        All processing is done during initialization, so this just returns the stored data.
        """
        X, y = self.data[idx]
        
        if self.reverse_compl:
            # Reverse complement augmentation: reverse sequence and complement bases
            # For one-hot encoding: [A, C, G, T] -> reverse and swap A<->T, C<->G
            # Randomly apply reverse complement
            if torch.rand(1).item() > 0.5:
                # Reverse along sequence dimension (dim=1 for X, dim=0 for y)
                X_rc = torch.flip(X, dims=[1])
                # Complement: swap channels 0<->3 (A<->T) and 1<->2 (C<->G)
                # Create index tensor for channel reordering
                channel_idx = torch.tensor([3, 2, 1, 0], device=X.device, dtype=torch.long)
                X = X_rc[channel_idx, :]
                
                # Also reverse the output along sequence dimension to maintain correspondence
                # y is [seq_len, num_tasks], so flip along dim=0
                y = torch.flip(y, dims=[0])
        
        return X, y

class MaskedPoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.poisson_nll_loss(y_pred_mask, y_true_mask, reduction="none")
        return loss.mean()

class MaskedMSE():
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        mse = nn.functional.mse_loss(y_pred_mask, y_true_mask, reduction="mean")
        
        return mse

class MaskedPearsonCorr(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)

        return cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

class MaskedJSDiv(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _kl_divergence(self, p, q, eps=1e-10):
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        return torch.sum(p * torch.log(p / q))
    
    def __call__(self, y_pred, y_true, mask, eps=1e-10):
        # Apply mask and flatten
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()
        
        # Normalize to probability distributions (ensure non-negative and sum to 1)
        y_pred_norm = torch.clamp(y_pred_mask, min=0.0)
        y_true_norm = torch.clamp(y_true_mask, min=0.0)
        
        # Normalize to sum to 1
        pred_sum = torch.sum(y_pred_norm)
        true_sum = torch.sum(y_true_norm)
        
        if pred_sum < eps or true_sum < eps:
            # If either distribution is all zeros, return a large divergence
            return torch.tensor(float('inf'), device=y_pred.device)
        
        p = y_pred_norm / pred_sum
        q = y_true_norm / true_sum
        
        # Compute mixture distribution M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        
        # Compute Jensen-Shannon divergence
        kl_pm = self._kl_divergence(p, m, eps)
        kl_qm = self._kl_divergence(q, m, eps)
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js_div

class CombinedMaskedLoss(nn.Module):
    """Combined loss function: Poisson NLL + MSE for better training stability."""
    def __init__(self, poisson_weight=0.7, mse_weight=0.3):
        super().__init__()
        self.poisson_weight = poisson_weight
        self.mse_weight = mse_weight
        self.poisson_loss = MaskedPoissonLoss()
        self.mse_loss = MaskedMSE()
    
    def __call__(self, y_pred, y_true, mask):
        poisson = self.poisson_loss(y_pred, y_true, mask)
        mse = self.mse_loss(y_pred, y_true, mask)
        return self.poisson_weight * poisson + self.mse_weight * mse

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Linearly increases learning rate from warmup_start_lr to base_lr during warmup_epochs,
    then follows cosine annealing from base_lr to min_lr over the remaining epochs.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, warmup_start_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * lr_scale 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay 
                    for base_lr in self.base_lrs]

class CosineAnnealingWarmRestartsScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with warmup and warm restarts.
    
    Starts with linear warmup, then follows cosine annealing with periodic restarts.
    Each cycle consists of warmup followed by cosine decay, with cycles getting longer.
    """
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=None, 
                 min_lr=0, warmup_steps=0, warmup_start_lr=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr if max_lr is not None else max([group['lr'] for group in optimizer.param_groups])
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * lr_scale 
                    for _ in self.base_lrs]
        else:
            # Cosine annealing with restarts
            # Calculate which cycle we're in
            steps_since_warmup = self.last_epoch - self.warmup_steps
            cycle_steps = 0
            cycle = 0
            while cycle_steps + int(self.first_cycle_steps * (self.cycle_mult ** cycle)) <= steps_since_warmup:
                cycle_steps += int(self.first_cycle_steps * (self.cycle_mult ** cycle))
                cycle += 1
            
            self.cycle = cycle
            self.cur_cycle_steps = int(self.first_cycle_steps * (self.cycle_mult ** cycle))
            self.step_in_cycle = steps_since_warmup - cycle_steps
            
            # Cosine annealing within current cycle
            progress = self.step_in_cycle / self.cur_cycle_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            
            # Apply gamma decay to max_lr for each cycle
            cycle_max_lr = self.max_lr * (self.gamma ** cycle)
            
            return [self.min_lr + (cycle_max_lr - self.min_lr) * cosine_decay 
                    for _ in self.base_lrs]

class Model(L.LightningModule):
    """
    Unified model class supporting multiple architectures:
    - 'unet': UNet with 4 encoder/decoder blocks, residual connections
    - 'chrombpnet': ChromBPNet-style with dilated convolutions
    - 'resnet': ResNet-style with residual blocks and global pooling
    
    All models use combined loss (Poisson + MSE) and support various schedulers.
    """
    def __init__(self, model_type='unet', dropout_val=0.1, num_epochs=50, bs=64, lr=5e-4, 
                 scheduler_type='cosine_warmup', warmup_epochs=3, min_lr=1e-6, first_cycle_steps=None, 
                 cycle_mult=1.0, warmup_steps=0, gamma=1.0, weight_decay=1e-4,
                 poisson_weight=0.7, mse_weight=0.3,
                 # UNet-specific args
                 num_blocks=4, base_channels=64, conv_kernel_size=3, pool_kernel_size=2,
                 input_conv_kernel_size=21, task_specific_conv_kernel_size=5,
                 task_specific_output_binary=True,
                 # ChromBPNet-specific args
                 filters=64, n_dil_layers=8, conv1_kernel_size=21, profile_kernel_size=75):
        super().__init__()
        
        self.model_type = model_type
        self.loss = CombinedMaskedLoss(poisson_weight=poisson_weight, mse_weight=mse_weight)
        self.pcc_metric = MaskedPearsonCorr()
        self.mse_metric = MaskedMSE()
        self.jsd_metric = MaskedJSDiv()
        
        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        
        # Scheduler parameters
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.first_cycle_steps = first_cycle_steps if first_cycle_steps is not None else num_epochs // 4
        self.cycle_mult = cycle_mult
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        # Build architecture based on model_type
        if model_type == 'unet':
            build_unet_architecture(self, dropout_val, 
                                   num_blocks=num_blocks,
                                   base_channels=base_channels,
                                   conv_kernel_size=conv_kernel_size,
                                   pool_kernel_size=pool_kernel_size,
                                   input_conv_kernel_size=input_conv_kernel_size,
                                   task_specific_conv_kernel_size=task_specific_conv_kernel_size,
                                   task_specific_output_binary=task_specific_output_binary)
        elif model_type == 'chrombpnet':
            build_chrombpnet_architecture(self, dropout_val, filters, n_dil_layers, conv1_kernel_size, profile_kernel_size)
        elif model_type == 'resnet':
            build_resnet_architecture(self, dropout_val, num_blocks, base_channels)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'unet', 'chrombpnet', or 'resnet'")

    def forward(self, x):
        if self.model_type == 'unet':
            return forward_unet(self, x)
        elif self.model_type == 'chrombpnet':
            return forward_chrombpnet(self, x)
        elif self.model_type == 'resnet':
            return forward_resnet(self, x)
    
    def _get_loss(self, batch):
        # get features and labels
        x, y = batch

        # pass through model
        y_pred = self.forward(x)

        mask = ~torch.isnan(y)

        loss = self.loss(y_pred, y, mask)
        pcc_perf = self.pcc_metric(y_pred, y, mask)
        mse_metric = self.mse_metric(y_pred, y, mask)
        jsd_metric = self.jsd_metric(y_pred, y, mask)

        return loss, pcc_perf, mse_metric, jsd_metric

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=self.min_lr
            )
        elif self.scheduler_type == 'cosine_warmup':
            scheduler = CosineWarmupScheduler(
                optimizer, 
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.num_epochs,
                min_lr=self.min_lr,
                warmup_start_lr=self.min_lr
            )
        elif self.scheduler_type == 'warm_restarts':
            scheduler = CosineAnnealingWarmRestartsScheduler(
                optimizer,
                first_cycle_steps=self.first_cycle_steps,
                cycle_mult=self.cycle_mult,
                max_lr=self.lr,
                min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
                warmup_start_lr=self.min_lr,
                gamma=self.gamma
            )
        elif self.scheduler_type == 'lronplateau':
            scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 'monitor': 'eval/loss'}
        else:
            scheduler = None
        
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def training_step(self, batch):
        loss, pcc_perf, mse_metric, jsd_metric = self._get_loss(batch)

        self.log('train/loss', loss, batch_size=self.bs)
        self.log('train/pcc', pcc_perf, batch_size=self.bs)
        self.log('train/mse', mse_metric, batch_size=self.bs)
        self.log('train/jsdiv', jsd_metric, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, pcc_perf, mse_metric, jsd_metric = self._get_loss(batch)

        self.log('eval/loss', loss, on_epoch=True)
        self.log('eval/pcc', pcc_perf, on_epoch=True)
        self.log('eval/mse', mse_metric, on_epoch=True)
        self.log('eval/jsdiv', jsd_metric, on_epoch=True)

        # Store predictions and targets for task-wise metrics
        x, y = batch
        y_pred = self.forward(x)
        mask = ~torch.isnan(y)
        
        # Store in a list that accumulates across batches
        if not hasattr(self, '_val_preds'):
            self._val_preds = []
            self._val_targets = []
            self._val_masks = []
        
        # Move to CPU immediately to save GPU memory
        self._val_preds.append(y_pred.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        self._val_masks.append(mask.detach().cpu())

        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log task-wise metrics at the end of validation epoch (cell types and output types only)."""
        if hasattr(self, '_val_preds'):
            # Concatenate all batches
            y_pred_all = torch.cat(self._val_preds, dim=0)
            y_true_all = torch.cat(self._val_targets, dim=0)
            mask_all = torch.cat(self._val_masks, dim=0)
            
            # Compute average loss and PCC for printing
            avg_loss = self.loss(y_pred_all, y_true_all, mask_all).item()
            avg_pcc = self.pcc_metric(y_pred_all, y_true_all, mask_all).item()
            
            # Print average metrics
            print(f"\n[Validation Epoch {self.current_epoch}] Avg Loss: {avg_loss:.4f}, Avg PCC: {avg_pcc:.4f}")
            
            # Compute task-wise metrics (exclude individual tasks for validation)
            task_metrics = compute_task_wise_metrics(y_pred_all, y_true_all, mask_all, include_individual_tasks=False)
            
            # Log metrics for each grouping
            for group_name, group_metrics in task_metrics.items():
                for metric_name, metric_value in group_metrics.items():
                    self.log(f'eval/{group_name}/{metric_name}', metric_value, on_epoch=True)
            
            # Clear stored predictions and free GPU memory
            del self._val_preds
            del self._val_targets
            del self._val_masks
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def test_step(self, batch):
        loss, pcc_perf, mse_metric, jsd_metric = self._get_loss(batch)  

        self.log('test/loss', loss, on_epoch=True)
        self.log('test/pcc_perf', pcc_perf, on_epoch=True)
        self.log('test/mse_metric', mse_metric, on_epoch=True)
        self.log('test/jsdiv', jsd_metric, on_epoch=True)

        # Store predictions and targets for task-wise metrics
        x, y = batch
        y_pred = self.forward(x)
        mask = ~torch.isnan(y)
        
        # Store in a list that accumulates across batches
        if not hasattr(self, '_test_preds'):
            self._test_preds = []
            self._test_targets = []
            self._test_masks = []
        
        # Move to CPU immediately to save GPU memory
        self._test_preds.append(y_pred.detach().cpu())
        self._test_targets.append(y.detach().cpu())
        self._test_masks.append(mask.detach().cpu())

        return loss
    
    def on_test_epoch_end(self):
        """Compute and log task-wise metrics at the end of test epoch."""
        if hasattr(self, '_test_preds'):
            # Concatenate all batches
            y_pred_all = torch.cat(self._test_preds, dim=0)
            y_true_all = torch.cat(self._test_targets, dim=0)
            mask_all = torch.cat(self._test_masks, dim=0)
            
            # Compute average loss and PCC for printing
            avg_loss = self.loss(y_pred_all, y_true_all, mask_all).item()
            avg_pcc = self.pcc_metric(y_pred_all, y_true_all, mask_all).item()
            
            # Print average metrics
            print(f"\n[Test] Avg Loss: {avg_loss:.4f}, Avg PCC: {avg_pcc:.4f}")
            
            # Compute task-wise metrics
            task_metrics = compute_task_wise_metrics(y_pred_all, y_true_all, mask_all)
            
            # Log metrics for each grouping
            for group_name, group_metrics in task_metrics.items():
                for metric_name, metric_value in group_metrics.items():
                    self.log(f'test/{group_name}/{metric_name}', metric_value, on_epoch=True)
            
            # Clear stored predictions and free GPU memory
            del self._test_preds
            del self._test_targets
            del self._test_masks
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
def train_model(model_type='unet', num_epochs=50, bs=64, lr=5e-4, save_loc=None, 
                train_loader=None, test_loader=None, val_loader=None, dropout_val=0.1, 
                logger=None, scheduler_type='cosine_warmup', warmup_epochs=3, 
                min_lr=1e-6, first_cycle_steps=None, cycle_mult=1.0, warmup_steps=0, gamma=1.0, 
                weight_decay=1e-4, poisson_weight=0.7, mse_weight=0.3, gradient_clip_val=1.0,
                # Architecture-specific parameters
                # UNet-specific
                num_blocks=4, base_channels=64, conv_kernel_size=3, pool_kernel_size=2,
                input_conv_kernel_size=21, task_specific_conv_kernel_size=5,
                task_specific_output_binary=True,
                # ChromBPNet-specific
                filters=64, n_dil_layers=8, conv1_kernel_size=21, profile_kernel_size=75):
    """
    Unified training function for all model architectures.
    
    Args:
        model_type: 'unet', 'chrombpnet', or 'resnet'
        All other args are standard training parameters or architecture-specific.
    """
    # Create a PyTorch Lightning trainer with gradient clipping
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=1,
        logger=logger,
        max_epochs=num_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='eval/loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="eval/loss", patience=30),
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    model = Model(
        model_type=model_type,
        dropout_val=dropout_val,
        num_epochs=num_epochs,
        bs=bs,
        lr=lr,
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=cycle_mult,
        warmup_steps=warmup_steps,
        gamma=gamma,
        weight_decay=weight_decay,
        poisson_weight=poisson_weight,
        mse_weight=mse_weight,
        filters=filters,
        n_dil_layers=n_dil_layers,
        conv1_kernel_size=conv1_kernel_size,
        profile_kernel_size=profile_kernel_size,
        num_blocks=num_blocks,
        base_channels=base_channels,
        conv_kernel_size=conv_kernel_size,
        pool_kernel_size=pool_kernel_size,
        input_conv_kernel_size=input_conv_kernel_size,
        task_specific_conv_kernel_size=task_specific_conv_kernel_size,
        task_specific_output_binary=task_specific_output_binary
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total {model_type.upper()} parameters: {pytorch_total_params:,}")

    # fit trainer
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result} 

    return model, result
