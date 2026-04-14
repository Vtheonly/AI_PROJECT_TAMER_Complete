"""
Swin-Base Encoder with Gradient Checkpointing and 2D Positional Encoding.

FIXED: 
- Explicit dimension checking to prevent shape mismatch in self.proj.
- Handles both (B, C, H, W) and (B, H, W, C) outputs from timm backbones.
- Corrected flattened length calculation for linear layers.
"""

import torch
import torch.nn as nn
import timm
import logging
from .attention import PositionalEncoding2D

logger = logging.getLogger("TAMER.Encoder")

class SwinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create the backbone
        # We target Stage 3 (out_indices=2) which typically has 512 channels in Swin-Base.
        self.backbone = timm.create_model(
            config.encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(2,),
            img_size=(config.img_height, config.img_width)
        )

        # Enable gradient checkpointing to save VRAM on Tesla T4/Colab
        self.backbone.set_grad_checkpointing(True)

        # Get the actual channel count from the backbone's feature info
        # This prevents hardcoding errors if the model variant changes.
        feature_info = self.backbone.feature_info.get_dicts()
        self.in_channels = feature_info[0]['num_chs']
        
        logger.info(f"Swin Encoder: Backbone features detected with {self.in_channels} channels.")

        # Projection layer to transform backbone channels to model dimension (d_model=512)
        self.proj = nn.Linear(self.in_channels, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        # 2D Positional Encoding
        # For 256x1024 input at 16x downsample, grid is 16x64. 
        # We set max higher for safety.
        self.pos_enc_2d = PositionalEncoding2D(
            d_model=config.d_model, 
            max_h=64, 
            max_w=128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            features: (B, L, d_model) where L is the number of spatial patches
        """
        # 1. Extract features from Swin
        # Returns a list of tensors; we take the first one since we requested only one index.
        features = self.backbone(x)[0] 
        B = features.shape[0]

        # 2. Robust Dimension Handling
        # Different timm versions/models return (B, C, H, W) or (B, H, W, C).
        # We must ensure the last dimension is the channel dimension for the Linear layer.
        
        if features.dim() == 4:
            # Check if it's (B, C, H, W)
            if features.shape[1] == self.in_channels:
                # Permute to (B, H, W, C)
                features = features.permute(0, 2, 3, 1).contiguous()
            # If it's already (B, H, W, C), do nothing
        elif features.dim() == 3:
            # If it's (B, L, C), we reshape it to (B, H, W, C) to apply 2D Positional Encoding
            H_feat = self.config.img_height // 16
            W_feat = self.config.img_width // 16
            # Ensure L matches H*W
            if features.shape[1] != (H_feat * W_feat):
                # Adaptive calculation if input image size was different
                L = features.shape[1]
                aspect = self.config.img_width / self.config.img_height
                H_feat = int((L / aspect) ** 0.5)
                W_feat = L // H_feat
            features = features.view(B, H_feat, W_feat, self.in_channels)

        # 3. Channel Projection
        # features shape is (B, H, W, C_in) -> (B, H, W, d_model)
        x = self.proj(features)
        x = self.norm(x)

        # 4. Apply 2D Positional Encoding
        # This allows the model to know the spatial location of each math symbol
        x = self.pos_enc_2d(x)

        # 5. Flatten spatial patches into a sequence
        # (B, H, W, D) -> (B, H*W, D)
        x = x.flatten(1, 2)

        return x