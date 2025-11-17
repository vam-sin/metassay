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
import pickle
from collections import defaultdict

# Load task groupings for task-wise metrics
_task_groupings = None
_task_names = None

def _load_task_groupings():
    """Load task groupings once and cache them."""
    global _task_groupings, _task_names
    if _task_groupings is None:
        with open('../data_processing/encode_dataset/final_3k/task_groupings.json', 'r') as f:
            _task_groupings = json.load(f)
        with open('../data_processing/encode_dataset/final_3k/task_names.json', 'r') as f:
            _task_names = json.load(f)
    return _task_groupings, _task_names

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
        y = torch.from_numpy(np.stack([sample[t] for t in task_names])).permute(1, 0)[o_s:-o_s, :]

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
    def __init__(self, chroms_list, input_size=2048, output_size=1024, reverse_compl=False, random_shift=False, n_jobs_load=16, data_cache_dir=None):
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
            self.task_names = json.load(f)

        self.i_s = int((3000-input_size)/2)
        self.o_s = int((3000-output_size)/2)

        # transforms
        self.reverse_compl = reverse_compl
        self.random_shift = random_shift

        # soft-clip
        self.tc = 32.0

        # hard-clip
        self.hc = 1e+5
        
        # Check for cached processed data
        cache_file = None
        if data_cache_dir is not None:
            os.makedirs(data_cache_dir, exist_ok=True)
            cache_key = hashlib.md5(
                (str(sorted(self.files_ds)) + str(input_size) + str(output_size) + 
                 str(self.reverse_compl) + str(self.random_shift)).encode()
            ).hexdigest()
            cache_file = os.path.join(data_cache_dir, f'data_{cache_key}.pkl')
            
            if os.path.exists(cache_file):
                print(f"Loading cached processed data from {cache_file}...")
                try:
                    with open(cache_file, 'rb') as f:
                        self.data = pickle.load(f)
                    print(f"Loaded {len(self.data)} pre-processed samples from cache")
                    return
                except Exception as e:
                    print(f"Failed to load cache: {e}. Processing data...")
        
        # Load and process all data
        print(f"Loading and processing {len(self.files_ds)} files (using {n_jobs_load} workers)...")
        print("This may take several minutes. All data will be loaded into memory.")
        
        # Parallel loading and processing
        processed_chunks = Parallel(n_jobs=n_jobs_load)(
            delayed(_load_and_process_file)(
                file_path, self.folder, self.task_names, 
                self.i_s, self.o_s, self.tc, self.hc
            )
            for file_path in tqdm(self.files_ds, desc="Loading and processing files")
        )
        
        # Flatten the list of lists
        self.data = []
        for chunk in processed_chunks:
            self.data.extend(chunk)
        
        print(f"Loaded and processed {len(self.data)} samples into memory")
        
        # Save processed data to cache if requested
        if cache_file is not None:
            print(f"Saving processed data to cache: {cache_file}")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.data, f)
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

class ResNetBlock1D(nn.Module):
    """Residual block for 1D convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        out = self.relu(out)
        return out

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
                 # ChromBPNet-specific args
                 filters=64, n_dil_layers=8, conv1_kernel_size=21, profile_kernel_size=75,
                 # ResNet-specific args
                 num_blocks=4, base_channels=64):
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
            self._build_unet(dropout_val)
        elif model_type == 'chrombpnet':
            self._build_chrombpnet(dropout_val, filters, n_dil_layers, conv1_kernel_size, profile_kernel_size)
        elif model_type == 'resnet':
            self._build_resnet(dropout_val, num_blocks, base_channels)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'unet', 'chrombpnet', or 'resnet'")
    
    def _build_unet(self, dropout_val):
        """Build UNet architecture."""
        # Encoder (Downsampling path) - 4 blocks
        self.enc1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.enc1_residual = nn.Conv1d(4, 64, kernel_size=1)  # Residual connection
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.enc2_residual = nn.Conv1d(64, 128, kernel_size=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.enc3_residual = nn.Conv1d(128, 256, kernel_size=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Additional encoder block
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.enc4_residual = nn.Conv1d(256, 512, kernel_size=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Decoder (Upsampling path) - 4 blocks
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Improved final output layer with intermediate layer
        # Note: We skip the last upconv to maintain 1024 output size (4 pools, 3 upconvs)
        self.final_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_val * 0.5),
            nn.Conv1d(256, 198, kernel_size=1)  # 198 output tracks
        )
    
    def _build_chrombpnet(self, dropout_val, filters, n_dil_layers, conv1_kernel_size, profile_kernel_size):
        """Build ChromBPNet architecture."""
        self.filters = filters
        self.n_dil_layers = n_dil_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.profile_kernel_size = profile_kernel_size
        
        # First convolution without dilation
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, filters, kernel_size=conv1_kernel_size, padding='valid'),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        
        # Dilated convolutions with residual connections
        self.dilated_convs = nn.ModuleList()
        for i in range(1, n_dil_layers + 1):
            dilation = 2 ** i
            conv = nn.Sequential(
                nn.Conv1d(filters, filters, kernel_size=3, padding='valid', dilation=dilation),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout_val)
            )
            self.dilated_convs.append(conv)
        
        # Profile prediction branch
        self.prof_conv = nn.Conv1d(filters, 198, kernel_size=profile_kernel_size, padding='valid')
    
    def _build_resnet(self, dropout_val, num_blocks, base_channels):
        """Build ResNet architecture."""
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(4, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        
        # Residual blocks with increasing channels
        self.blocks = nn.ModuleList()
        channels = base_channels
        for i in range(num_blocks):
            next_channels = channels * 2 if i < num_blocks - 1 else channels
            stride = 2 if i < num_blocks - 1 else 1
            self.blocks.append(ResNetBlock1D(channels, next_channels, kernel_size=3, 
                                            stride=stride, dropout=dropout_val))
            channels = next_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final output layers
        self.final_layers = nn.Sequential(
            nn.Linear(channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_val * 0.5),
            nn.Linear(256, 198)  # 198 output tracks
        )
    
    def _crop_center(self, x, target_length):
        """Crop or interpolate tensor to target length from center (for ChromBPNet)."""
        current_length = x.shape[-1]
        if current_length == target_length:
            return x
        elif current_length > target_length:
            crop_size = (current_length - target_length) // 2
            return x[:, :, crop_size:current_length - crop_size]
        else:
            x_interp = torch.nn.functional.interpolate(
                x, size=target_length, mode='linear', align_corners=False
            )
            return x_interp

    def forward(self, x):
        if self.model_type == 'unet':
            return self._forward_unet(x)
        elif self.model_type == 'chrombpnet':
            return self._forward_chrombpnet(x)
        elif self.model_type == 'resnet':
            return self._forward_resnet(x)
    
    def _forward_unet(self, x):
        """Forward pass for UNet."""
        # Encoder Path with residual connections
        enc1_out = self.enc1(x)
        enc1_res = self.enc1_residual(x)
        enc1 = enc1_out + enc1_res  # Residual connection
        p1 = self.pool1(enc1)
        
        enc2_out = self.enc2(p1)
        enc2_res = self.enc2_residual(p1)
        enc2 = enc2_out + enc2_res
        p2 = self.pool2(enc2)
        
        enc3_out = self.enc3(p2)
        enc3_res = self.enc3_residual(p2)
        enc3 = enc3_out + enc3_res
        p3 = self.pool3(enc3)
        
        enc4_out = self.enc4(p3)
        enc4_res = self.enc4_residual(p3)
        enc4 = enc4_out + enc4_res
        p4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        
        # Decoder Path with Skip Connections
        up4 = self.upconv4(bottleneck)
        concat4 = torch.cat([up4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(concat4)
        
        up3 = self.upconv3(dec4)
        concat3 = torch.cat([up3, enc3], dim=1)  # Skip connection
        dec3 = self.dec3(concat3)
        
        up2 = self.upconv2(dec3)
        concat2 = torch.cat([up2, enc2], dim=1)  # Skip connection
        dec2 = self.dec2(concat2)
        
        # Final convolution (skip last upconv to maintain 1024 output size)
        out = self.final_conv(dec2).permute(0, 2, 1)
        return out
    
    def _forward_chrombpnet(self, x):
        """Forward pass for ChromBPNet."""
        # First convolution
        x = self.conv1(x)  # [B, filters, seq_len']
        
        # Dilated convolutions with residual connections
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_x = dilated_conv(x)
            # Crop x to match conv_x size (symmetric cropping)
            x_len = x.shape[-1]
            conv_x_len = conv_x.shape[-1]
            crop_size = (x_len - conv_x_len) // 2
            x_cropped = x[:, :, crop_size:x_len - crop_size]
            # Residual connection
            x = conv_x + x_cropped
        
        # Profile prediction branch
        prof_out = self.prof_conv(x)  # [B, 198, seq_len'']
        
        # Crop to match output size (1024)
        target_length = 1024
        prof_out = self._crop_center(prof_out, target_length)
        
        # Permute to [B, seq_len, 198]
        out = prof_out.permute(0, 2, 1)
        return out
    
    def _forward_resnet(self, x):
        """Forward pass for ResNet."""
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)  # [B, C, 1]
        x = x.squeeze(-1)  # [B, C]
        
        # Final layers
        x = self.final_layers(x)  # [B, 198]
        
        # Expand to [B, seq_len, 198] by repeating across sequence dimension
        seq_len = 1024
        x = x.unsqueeze(1).repeat(1, seq_len, 1)  # [B, 1024, 198]
        return x
    
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
        else:
            scheduler = None
        
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def training_step(self, batch):
        loss, pcc_perf, mse_metric, jsd_metric = self._get_loss(batch)

        self.log('train/loss', loss, batch_size=self.bs, sync_dist=True)
        self.log('train/pcc', pcc_perf, batch_size=self.bs, sync_dist=True)
        self.log('train/mse', mse_metric, batch_size=self.bs, sync_dist=True)
        self.log('train/jsdiv', jsd_metric, batch_size=self.bs, sync_dist=True)

        return loss
    
    def validation_step(self, batch):
        loss, pcc_perf, mse_metric, jsd_metric = self._get_loss(batch)

        self.log('eval/loss', loss, on_epoch=True, sync_dist=True)
        self.log('eval/pcc', pcc_perf, on_epoch=True, sync_dist=True)
        self.log('eval/mse', mse_metric, on_epoch=True, sync_dist=True)
        self.log('eval/jsdiv', jsd_metric, on_epoch=True, sync_dist=True)

        # Store predictions and targets for task-wise metrics (only on rank 0 to avoid duplicates)
        if self.trainer.global_rank == 0:
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
        if self.trainer.global_rank == 0 and hasattr(self, '_val_preds'):
            # Concatenate all batches
            y_pred_all = torch.cat(self._val_preds, dim=0)
            y_true_all = torch.cat(self._val_targets, dim=0)
            mask_all = torch.cat(self._val_masks, dim=0)
            
            # Compute task-wise metrics (exclude individual tasks for validation)
            task_metrics = compute_task_wise_metrics(y_pred_all, y_true_all, mask_all, include_individual_tasks=False)
            
            # Log metrics for each grouping
            for group_name, group_metrics in task_metrics.items():
                for metric_name, metric_value in group_metrics.items():
                    self.log(f'eval/{group_name}/{metric_name}', metric_value, on_epoch=True, sync_dist=False)
            
            # Clear stored predictions and free GPU memory
            del self._val_preds
            del self._val_targets
            del self._val_masks
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def test_step(self, batch):
        loss, pcc_perf, mse_metric, jsd_metric = self._get_loss(batch)  

        self.log('test/loss', loss, on_epoch=True, sync_dist=True)
        self.log('test/pcc_perf', pcc_perf, on_epoch=True, sync_dist=True)
        self.log('test/mse_metric', mse_metric, on_epoch=True, sync_dist=True)
        self.log('test/jsdiv', jsd_metric, on_epoch=True, sync_dist=True)

        # Store predictions and targets for task-wise metrics (only on rank 0 to avoid duplicates)
        if self.trainer.global_rank == 0:
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
        if self.trainer.global_rank == 0 and hasattr(self, '_test_preds'):
            # Concatenate all batches
            y_pred_all = torch.cat(self._test_preds, dim=0)
            y_true_all = torch.cat(self._test_targets, dim=0)
            mask_all = torch.cat(self._test_masks, dim=0)
            
            # Compute task-wise metrics
            task_metrics = compute_task_wise_metrics(y_pred_all, y_true_all, mask_all)
            
            # Log metrics for each grouping
            for group_name, group_metrics in task_metrics.items():
                for metric_name, metric_value in group_metrics.items():
                    self.log(f'test/{group_name}/{metric_name}', metric_value, on_epoch=True, sync_dist=False)
            
            # Clear stored predictions and free GPU memory
            del self._test_preds
            del self._test_targets
            del self._test_masks
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
def train_model(model_type='unet', num_epochs=50, bs=64, lr=5e-4, save_loc=None, 
                train_loader=None, test_loader=None, val_loader=None, dropout_val=0.1, 
                logger=None, num_gpus=1, scheduler_type='cosine_warmup', warmup_epochs=3, 
                min_lr=1e-6, first_cycle_steps=None, cycle_mult=1.0, warmup_steps=0, gamma=1.0, 
                weight_decay=1e-4, poisson_weight=0.7, mse_weight=0.3, gradient_clip_val=1.0,
                # Architecture-specific parameters
                filters=64, n_dil_layers=8, conv1_kernel_size=21, profile_kernel_size=75,
                num_blocks=4, base_channels=64):
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
        devices=num_gpus,
        strategy="ddp" if num_gpus > 1 else "auto",
        accumulate_grad_batches=1,
        logger=logger,
        max_epochs=num_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='eval/loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="eval/loss", patience=15),
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
        base_channels=base_channels
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total {model_type.upper()} parameters: {pytorch_total_params:,}")

    # fit trainer
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result} 

    return model, result
