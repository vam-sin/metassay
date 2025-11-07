# libraries
import pandas as pd 
import numpy as np
import torch
import json
import os 
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
# from torchmetrics.regression import JensenShannonDivergence
import itertools
import time
import lightning as L
from joblib import Parallel, delayed
import hashlib
import pickle

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
            pass  # TODO: implement reverse complement augmentation
        
        return X, y

class MaskedPoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.poisson_nll_loss(y_pred_mask, y_true_mask, reduction="none")
        return loss.mean()

# class MaskedJSDiv(nn.Module):
#     def __init__(self):
#         super.__init__()
#         jsd_func = JensenShannonDivergence()
#     def __call__(self, y_pred, y_true, mask):
#         y_pred_mask = torch.masked_select(y_pred, mask).float()
#         y_true_mask = torch.masked_select(y_true, mask).float()

#         jsd = jsd_func(y_pred_mask, y_true_mask)

#         return jsd

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

class UNet(L.LightningModule):
    def __init__(self, dropout_val, num_epochs, bs, lr):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        
        # Decoder (Upsampling path)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        
        self.final_conv = nn.Conv1d(128, 198, kernel_size=1) # 198 output tracks
        
        # one less upconv than pool as the output is 1024 and the input is 2048

        self.loss = MaskedPoissonLoss()
        self.pcc_metric = MaskedPearsonCorr()
        self.mse_metric = MaskedMSE()
        # self.jsd_metric = MaskedJSDiv()
        
        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

    def forward(self, x):
        # Encoder Path
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)
        
        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)
        
        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(p3)
        
        # Decoder Path with Skip Connections
        up3 = self.upconv3(bottleneck)
        concat3 = torch.cat([up3, enc3], dim=1)  # Skip connection
        dec3 = self.dec3(concat3)
        
        up2 = self.upconv2(dec3)
        concat2 = torch.cat([up2, enc2], dim=1)  # Skip connection
        dec2 = self.dec2(concat2)
        
        # Final convolution
        out = self.final_conv(dec2).permute(0, 2, 1)
        
        return out
    
    def _get_loss(self, batch):
        # get features and labels
        x, y = batch

        # pass through model
        y_pred = self.forward(x)

        mask = ~torch.isnan(y)

        loss = self.loss(y_pred, y, mask)
        pcc_perf = self.pcc_metric(y_pred, y, mask)
        mse_metric = self.mse_metric(y_pred, y, mask)
        # jsd_metric = self.jsd_metric(y_pred, y, mask)

        return loss, pcc_perf, mse_metric

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):

        loss, pcc_perf, mse_metric = self._get_loss(batch)

        self.log('train/loss', loss, batch_size=self.bs, sync_dist=True)
        self.log('train/pcc', pcc_perf, batch_size=self.bs, sync_dist=True)
        self.log('train/mse', mse_metric, batch_size=self.bs, sync_dist=True)

        return loss
    
    def validation_step(self, batch):
        loss, pcc_perf, mse_metric = self._get_loss(batch)

        self.log('eval/loss', loss, on_epoch=True, sync_dist=True)
        self.log('eval/pcc', pcc_perf, on_epoch=True, sync_dist=True)
        self.log('eval/mse', mse_metric, on_epoch=True, sync_dist=True)

        return loss
    
    def test_step(self, batch):
        loss, pcc_perf, mse_metric = self._get_loss(batch)  

        self.log('test/loss', loss, on_epoch=True, sync_dist=True)
        self.log('test/pcc_perf', pcc_perf, on_epoch=True, sync_dist=True)
        self.log('test/mse_metric', mse_metric, on_epoch=True, sync_dist=True)

        return loss
    
def trainUNet(num_epochs, bs, lr, save_loc, train_loader, test_loader, val_loader, dropout_val, logger, num_gpus=1):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=num_gpus,  # Use multiple GPUs if available
        strategy="ddp" if num_gpus > 1 else "auto",  # Use DDP for multi-GPU
        accumulate_grad_batches=1,
        logger=logger,
        max_epochs=num_epochs,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='eval/loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="eval/loss", patience=10),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    model = UNet(dropout_val, num_epochs, bs, lr)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total UNet parameters: ", pytorch_total_params)

    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result} 

    return model, result
    

        

