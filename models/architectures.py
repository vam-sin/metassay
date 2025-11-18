"""
Model architectures for genomic sequence prediction.

This module contains the architecture definitions for:
- UNet: Encoder-decoder with skip connections
- ChromBPNet: Dilated convolutions with residual connections
- ResNet: Residual blocks with global pooling
"""
import torch
import torch.nn as nn


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


def crop_center(x, target_length):
    """Crop or interpolate tensor to target length from center."""
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


def build_unet_architecture(model, dropout_val, num_blocks=4, base_channels=64, 
                            conv_kernel_size=3, pool_kernel_size=2, input_conv_kernel_size=21,
                            task_specific_conv_kernel_size=5):
    """
    Build UNet architecture components on the model instance.
    
    Args:
        dropout_val: Dropout value
        num_blocks: Number of encoder/decoder blocks (default: 4)
        base_channels: Base number of channels (default: 64)
        conv_kernel_size: Kernel size for encoder/decoder convs (default: 3)
        pool_kernel_size: Kernel size for pooling (default: 2)
        input_conv_kernel_size: Kernel size for input conv block (default: 21)
        task_specific_conv_kernel_size: Kernel size for task-specific convs (default: 5)
    """
    model.num_blocks = num_blocks
    model.base_channels = base_channels
    model.conv_kernel_size = conv_kernel_size
    model.pool_kernel_size = pool_kernel_size
    
    # Input convolution block with large kernel
    input_padding = (input_conv_kernel_size - 1) // 2
    model.input_conv = nn.Sequential(
        nn.Conv1d(4, base_channels, kernel_size=input_conv_kernel_size, padding=input_padding),
        nn.BatchNorm1d(base_channels),
        nn.ReLU(),
        nn.Dropout(dropout_val)
    )
    
    # Encoder blocks (downsampling path)
    model.encoders = nn.ModuleList()
    model.enc_residuals = nn.ModuleList()
    model.pools = nn.ModuleList()
    
    in_channels = base_channels
    for i in range(num_blocks):
        out_channels = base_channels * (2 ** i)
        padding = (conv_kernel_size - 1) // 2
        
        # Encoder block: two conv layers with residual
        enc_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        model.encoders.append(enc_block)
        
        # Residual connection
        if in_channels != out_channels:
            model.enc_residuals.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        else:
            model.enc_residuals.append(nn.Identity())
        
        # Pooling layer
        model.pools.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size))
        
        in_channels = out_channels
    
    # Bottleneck
    bottleneck_channels = base_channels * (2 ** num_blocks)
    bottleneck_padding = (conv_kernel_size - 1) // 2
    model.bottleneck = nn.Sequential(
        nn.Conv1d(in_channels, bottleneck_channels, kernel_size=conv_kernel_size, padding=bottleneck_padding),
        nn.BatchNorm1d(bottleneck_channels),
        nn.ReLU(),
        nn.Dropout(dropout_val),
        nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=conv_kernel_size, padding=bottleneck_padding),
        nn.BatchNorm1d(bottleneck_channels),
        nn.ReLU()
    )
    
    # Decoder blocks (upsampling path)
    model.upconvs = nn.ModuleList()
    model.decoders = nn.ModuleList()
    
    in_channels = bottleneck_channels
    for i in range(num_blocks - 1, -1, -1):  # Reverse order
        out_channels = base_channels * (2 ** i)
        padding = (conv_kernel_size - 1) // 2
        
        # Upsampling
        model.upconvs.append(nn.ConvTranspose1d(in_channels, out_channels, 
                                                kernel_size=pool_kernel_size, stride=pool_kernel_size))
        
        # Decoder block: two conv layers
        # Input will be concatenated with skip connection, so in_channels doubles
        dec_block = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv1d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        model.decoders.append(dec_block)
        
        in_channels = out_channels
    
    # Task-specific 1D convolutions - one conv per task, each outputs 1 channel
    model.task_specific_convs = nn.ModuleList()
    task_padding = (task_specific_conv_kernel_size - 1) // 2
    for _ in range(167):  # One conv per task
        task_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=task_specific_conv_kernel_size, padding=task_padding),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_val * 0.5),
            nn.Conv1d(in_channels, 1, kernel_size=1)  # Output 1 channel per task
        )
        model.task_specific_convs.append(task_conv)


def build_chrombpnet_architecture(model, dropout_val, filters, n_dil_layers, conv1_kernel_size, profile_kernel_size):
    """Build ChromBPNet architecture components on the model instance."""
    model.filters = filters
    model.n_dil_layers = n_dil_layers
    model.conv1_kernel_size = conv1_kernel_size
    model.profile_kernel_size = profile_kernel_size
    
    # First convolution without dilation
    model.conv1 = nn.Sequential(
        nn.Conv1d(4, filters, kernel_size=conv1_kernel_size, padding='valid'),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Dropout(dropout_val)
    )
    
    # Dilated convolutions with residual connections
    model.dilated_convs = nn.ModuleList()
    for i in range(1, n_dil_layers + 1):
        dilation = 2 ** i
        conv = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size=3, padding='valid', dilation=dilation),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        model.dilated_convs.append(conv)
    
    # Profile prediction branch
    model.prof_conv = nn.Conv1d(filters, 167, kernel_size=profile_kernel_size, padding='valid')


def build_resnet_architecture(model, dropout_val, num_blocks, base_channels):
    """Build ResNet architecture components on the model instance."""
    # Initial convolution
    model.initial_conv = nn.Sequential(
        nn.Conv1d(4, base_channels, kernel_size=7, padding=3),
        nn.BatchNorm1d(base_channels),
        nn.ReLU(),
        nn.Dropout(dropout_val)
    )
    
    # Residual blocks with increasing channels
    model.blocks = nn.ModuleList()
    channels = base_channels
    for i in range(num_blocks):
        next_channels = channels * 2 if i < num_blocks - 1 else channels
        stride = 2 if i < num_blocks - 1 else 1
        model.blocks.append(ResNetBlock1D(channels, next_channels, kernel_size=3, 
                                        stride=stride, dropout=dropout_val))
        channels = next_channels
    
    # Global average pooling
    model.global_pool = nn.AdaptiveAvgPool1d(1)
    
    # Final output layers
    model.final_layers = nn.Sequential(
        nn.Linear(channels, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_val),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout_val * 0.5),
        nn.Linear(256, 167)  # 167 output tracks
    )


def forward_unet(model, x):
    """Forward pass for UNet."""
    # Input convolution
    x = model.input_conv(x)
    
    # Encoder path with residual connections
    encoder_outputs = []
    for i, (enc, enc_res, pool) in enumerate(zip(model.encoders, model.enc_residuals, model.pools)):
        enc_out = enc(x)
        enc_res_out = enc_res(x)
        x = enc_out + enc_res_out  # Residual connection
        encoder_outputs.append(x)  # Store for skip connections
        x = pool(x)
    
    # Bottleneck
    x = model.bottleneck(x)
    
    # Decoder path with skip connections
    for i, (upconv, dec) in enumerate(zip(model.upconvs, model.decoders)):
        x = upconv(x)
        # Get corresponding encoder output for skip connection (reverse order)
        skip_idx = model.num_blocks - 1 - i
        skip_connection = encoder_outputs[skip_idx]
        
        # Crop skip connection to match upsampled size if needed
        if skip_connection.shape[-1] != x.shape[-1]:
            skip_connection = crop_center(skip_connection, x.shape[-1])
        
        x = torch.cat([x, skip_connection], dim=1)  # Skip connection
        x = dec(x)
    
    # Task-specific convolutions - each outputs 1 channel
    task_outputs = []
    for task_conv in model.task_specific_convs:
        task_out = task_conv(x)  # [B, 1, L]
        task_outputs.append(task_out.squeeze(1))  # [B, L]
    
    # Stack task outputs: [B, L] for each task -> [B, num_tasks, L]
    out = torch.stack(task_outputs, dim=1)  # [B, num_tasks, L]
    
    # Crop to center 1024 to match output shape
    target_length = 1024
    out = crop_center(out, target_length)  # [B, num_tasks, 1024]
    
    # Permute to [B, 1024, num_tasks]
    out = out.permute(0, 2, 1)  # [B, 1024, num_tasks]
    
    return out


def forward_chrombpnet(model, x):
    """Forward pass for ChromBPNet."""
    # First convolution
    x = model.conv1(x)  # [B, filters, seq_len']
    
    # Dilated convolutions with residual connections
    for i, dilated_conv in enumerate(model.dilated_convs):
        conv_x = dilated_conv(x)
        # Crop x to match conv_x size (symmetric cropping)
        x_len = x.shape[-1]
        conv_x_len = conv_x.shape[-1]
        crop_size = (x_len - conv_x_len) // 2
        x_cropped = x[:, :, crop_size:x_len - crop_size]
        # Residual connection
        x = conv_x + x_cropped
    
    # Profile prediction branch
    prof_out = model.prof_conv(x)  # [B, 167, seq_len'']
    
    # Crop to match output size (1024)
    target_length = 1024
    prof_out = crop_center(prof_out, target_length)
    
    # Permute to [B, seq_len, 167]
    out = prof_out.permute(0, 2, 1)
    return out


def forward_resnet(model, x):
    """Forward pass for ResNet."""
    # Initial convolution
    x = model.initial_conv(x)
    
    # Residual blocks
    for block in model.blocks:
        x = block(x)
    
    # Global average pooling
    x = model.global_pool(x)  # [B, C, 1]
    x = x.squeeze(-1)  # [B, C]
    
    # Final layers
    x = model.final_layers(x)  # [B, 167]
    
    # Expand to [B, seq_len, 167] by repeating across sequence dimension
    seq_len = 1024
    x = x.unsqueeze(1).repeat(1, seq_len, 1)  # [B, 1024, 167]
    return x

