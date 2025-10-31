# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import itertools
import time
import lightning as L

class ChromDS(Dataset):
    def __init__(self, files, tasks):
        # load in all the files and merge them
        self.merged_dict = {}

        # load tasks json
        with open('../data_processing/encode_dataset/final/task_names.json', 'r') as f:
            self.task_names = json.load(f)

    def __len__(self):
        return len(self.merged_dict)
    
    def __getitem__(self, idx):
        X = self.merged_dict[idx]['dna_seq'] # should already be ohe
        y = [self.merged_dict[idx][t] for t in self.task_names]

        # flatten y

        return X, y

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())

class UNet(L.LightningModule):
    def __init__(self, dropout_val=0.1, num_epochs=100, bs=32, lr=0.001):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
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
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Decoder (Upsampling path)
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
        
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.final_conv = nn.Conv1d(64, 4, kernel_size=1)
        
        self.loss = MaskedL1Loss()
        
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
        
        up1 = self.upconv1(dec2)
        concat1 = torch.cat([up1, enc1], dim=1)  # Skip connection
        dec1 = self.dec1(concat1)
        
        # Final convolution
        out = self.final_conv(dec1)
        
        return out
    
    def _get_loss(self, batch):
        # get features and labels
        x, y, g, t, c = batch

        y = y.squeeze(dim=0)

        # pass through model
        y_pred = self.forward(x)

        # remove the first dim of y_pred
        y_pred = y_pred[1:, :]

        # condition
        condition_ = x[0].item()

        # calculate masks
        lengths_full = torch.tensor([y.shape[0]]).to(y_pred)
        mask_full = torch.arange(y_pred.shape[0])[None, :].to(lengths_full) < lengths_full[:, None]
        mask_full = torch.logical_and(mask_full, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask_full = mask_full.squeeze(dim=0)

        y_pred = torch.squeeze(y_pred, dim=1)

        print(y_pred.shape, y.shape, mask_full.shape)

        loss, perf, mae = self.loss(y_pred, y, mask_full, condition_)

        return loss, perf, mae, c

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):

        loss, perf, mae, c = self._get_loss(batch)

        self.log('train/loss', loss, batch_size=self.bs)
        self.log('train/r', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae, c = self._get_loss(batch)

        self.log('eval/loss', loss)
        self.log('eval/r', perf)

        return loss
    
    def test_step(self, batch):
        loss, perf, mae, c = self._get_loss(batch)

        self.log('test/loss', loss)
        self.log('test/r', perf)

        self.perf_list.append(perf.item())
        self.conds_list.append(c)

        return loss
    
def trainUNet(num_epochs, bs, lr, save_loc, train_loader, test_loader, val_loader, dropout_val):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
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
    model = UNet(dropout_val, num_epochs, bs, lr, num_layers, num_nodes)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total UNet parameters: ", pytorch_total_params)

    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result} 

    return model, result
    

        

