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

# Configure WandB environment
os.environ['WANDB_SILENT'] = 'true'  # Reduce verbose output
os.environ['WANDB_INIT_TIMEOUT'] = '60'  # Increase timeout for API calls

from utils import ChromDS, train_model 

# %%
if __name__ == '__main__':
    # Note: wandb.login() is not needed when using WandbLogger with WANDB_API_KEY set
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_jobs_load', type=int, default=32, help='Number of parallel jobs for loading and processing data')
    parser.add_argument('--dropout_val', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--scheduler_type', type=str, default='warm_restarts', choices=['cosine', 'cosine_warmup', 'warm_restarts'], help='Scheduler type to use')
    parser.add_argument('--dataset_downsampling', type=float, default=1.0, help='Fraction of dataset to use (default: 1.0, use 0.5 for half the dataset)')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'chrombpnet', 'resnet'], 
                        help='Model type to use: unet, chrombpnet, or resnet')
    parser.add_argument('--reverse_compl', action='store_true', help='Enable reverse complement augmentation')
    # UNet-specific arguments
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of encoder/decoder blocks (default: 4)')
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels (default: 64)')
    parser.add_argument('--conv_kernel_size', type=int, default=3, help='Kernel size for encoder/decoder convs (default: 3)')
    parser.add_argument('--pool_kernel_size', type=int, default=4, help='Kernel size for pooling (default: 4)')
    parser.add_argument('--input_conv_kernel_size', type=int, default=21, help='Kernel size for input conv block (default: 21)')
    parser.add_argument('--task_specific_conv_kernel_size', type=int, default=5, help='Kernel size for task-specific convs (default: 5)')
    args = parser.parse_args()

    # reproducibility
    L.seed_everything(42)

    # load dataset
    train_ds = ChromDS(['_chr1_', '_chr2_', '_chr3_', '_chr4_', '_chr5_', '_chr6_', '_chr7_', '_chr8_', '_chr9_', '_chr10_', '_chr11_', '_chr12_', '_chr13_', '_chr14_', '_chr15_', '_chr16_'], 
                       n_jobs_load=args.n_jobs_load, data_cache_dir='../data_processing/encode_dataset/final_3k/cache_dir',
                       reverse_compl=args.reverse_compl, dataset_downsampling=args.dataset_downsampling)
    val_ds = ChromDS(['_chr17_', '_chr18_', '_chr19_'], n_jobs_load=args.n_jobs_load, data_cache_dir='../data_processing/encode_dataset/final_3k/cache_dir',
                     reverse_compl=False, dataset_downsampling=args.dataset_downsampling)
    test_ds = ChromDS(['_chr20_', '_chr21_', '_chr22_'], n_jobs_load=args.n_jobs_load, data_cache_dir='../data_processing/encode_dataset/final_3k/cache_dir',
                       reverse_compl=False, dataset_downsampling=args.dataset_downsampling)

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

    # Configure WandbLogger and create config dictionary with all hyperparameters
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.bs,
        'learning_rate': args.lr,
        'dropout_val': args.dropout_val,
        'model_type': args.model_type,
        'reverse_compl': args.reverse_compl,
        'dataset_downsampling': args.dataset_downsampling,
        'scheduler_type': args.scheduler_type
    }
    # Add UNet-specific parameters to config if using UNet
    if args.model_type == 'unet':
        config.update({
            'num_blocks': args.num_blocks,
            'base_channels': args.base_channels,
            'conv_kernel_size': args.conv_kernel_size,
            'pool_kernel_size': args.pool_kernel_size,
            'input_conv_kernel_size': args.input_conv_kernel_size,
            'task_specific_conv_kernel_size': args.task_specific_conv_kernel_size
        })
    logger = L.pytorch.loggers.WandbLogger(
        project='metassay',
        log_model=False,  # Don't log model checkpoints to save space
        config=config,  # Log hyperparameters to wandb web interface
    )

    model_name_map = {'unet': 'UNet', 'resnet': 'ResNet', 'chrombpnet': 'ChromBPNet'}
    model_name = model_name_map.get(args.model_type, 'UNet')
    save_loc = f'saved_models/{model_name}_{args.num_epochs}_{args.bs}_{args.lr}_{args.dropout_val}'

    # Train the selected model using unified train_model function
    print(f"Training {model_name}...")
    train_kwargs = {
        'model_type': args.model_type,
        'num_epochs': args.num_epochs,
        'bs': args.bs,
        'lr': args.lr,
        'save_loc': save_loc,
        'train_loader': train_dataloader,
        'test_loader': test_dataloader,
        'val_loader': val_dataloader,
        'dropout_val': args.dropout_val,
        'logger': logger,
        'scheduler_type': args.scheduler_type
    }
    # Add UNet-specific parameters if using UNet
    if args.model_type == 'unet':
        train_kwargs.update({
            'num_blocks': args.num_blocks,
            'base_channels': args.base_channels,
            'conv_kernel_size': args.conv_kernel_size,
            'pool_kernel_size': args.pool_kernel_size,
            'input_conv_kernel_size': args.input_conv_kernel_size,
            'task_specific_conv_kernel_size': args.task_specific_conv_kernel_size
        })
    model, result = train_model(**train_kwargs)