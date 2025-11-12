# libraries
import pandas as pd 
import numpy as np
import torch
import json
import lightning as L
import argparse
import wandb
import os
import json
import getpass

# Set Tensor Core optimization for A100
torch.set_float32_matmul_precision('medium')  # Use Tensor Cores for better performance

with open('api_keys.json', 'r') as f:
    api_keys = json.load(f)
os.environ['WANDB_API_KEY'] = api_keys['wandb_api_key']

# Set fallback username if getuser() fails (common in containers)
try:
    username = getpass.getuser()
except (KeyError, OSError):
    # Set environment variables that wandb will use
    os.environ['USER'] = os.environ.get('USER', 'wandb_user')
    os.environ['LOGNAME'] = os.environ.get('LOGNAME', 'wandb_user')
    # Also set wandb-specific env var
    os.environ['WANDB_USERNAME'] = 'wandb_user'

# For multi-GPU: Prevent WandB from making API calls during initialization
# WandbLogger will handle rank 0 detection after Lightning sets up DDP
# This reduces the chance of API conflicts during process spawning
os.environ['WANDB_SILENT'] = 'true'  # Reduce verbose output
os.environ['WANDB_INIT_TIMEOUT'] = '60'  # Increase timeout for API calls

from unet_utils import ChromDS, trainUNet 

# %%
if __name__ == '__main__':
    # Note: wandb.login() is not needed when using WandbLogger with WANDB_API_KEY set
    # WandbLogger will handle authentication and multi-GPU setup automatically
    # It only logs on rank 0 by default when using DDP
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_jobs_load', type=int, default=32, help='Number of parallel jobs for loading and processing data')
    parser.add_argument('--dropout_val', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (default: 1, use 2 for dual A100)')
    args = parser.parse_args()

    # reproducibility
    L.seed_everything(42)

    # load dataset
    train_ds = ChromDS(['_chr1_', '_chr2_', '_chr3_', '_chr4_', '_chr5_', '_chr6_', '_chr7_', '_chr8_', '_chr9_', '_chr10_', '_chr11_', '_chr12_', '_chr13_', '_chr14_', '_chr15_', '_chr16_'], n_jobs_load=args.n_jobs_load, data_cache_dir='../data_processing/encode_dataset/final_3k/all_npz/cache_dir')
    val_ds = ChromDS(['_chr17_', '_chr18_', '_chr19_'], n_jobs_load=args.n_jobs_load, data_cache_dir='../data_processing/encode_dataset/final_3k/all_npz/cache_dir')
    test_ds = ChromDS(['_chr20_', '_chr21_', '_chr22_'], n_jobs_load=args.n_jobs_load, data_cache_dir='../data_processing/encode_dataset/final_3k/all_npz/cache_dir')

    print(f'Train: {len(train_ds)}, Test: {len(test_ds)}, Val: {len(val_ds)}')

    # Set multiprocessing start method to avoid conflicts
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # convert to dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.bs, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=args.bs, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=args.bs, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # Configure WandbLogger for multi-GPU
    # WandbLogger automatically handles DDP and only logs on rank 0
    # However, to prevent API initialization errors, we configure it with settings
    # that delay initialization until after Lightning sets up the distributed environment
    # Create config dictionary with all hyperparameters for wandb web interface
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.bs,
        'learning_rate': args.lr,
        'dropout_val': args.dropout_val
    }
    logger = L.pytorch.loggers.WandbLogger(
        project='metassay',
        log_model=False,  # Don't log model checkpoints to save space
        config=config,  # Log hyperparameters to wandb web interface
    )

    save_loc = f'saved_models/UNet_{args.num_epochs}_{args.bs}_{args.lr}_{args.dropout_val}'

    model, result = trainUNet(args.num_epochs, args.bs, args.lr, save_loc, train_dataloader, test_dataloader, val_dataloader, args.dropout_val, logger, args.num_gpus)