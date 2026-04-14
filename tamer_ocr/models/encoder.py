"""
Swin-Base Encoder with Gradient Checkpointing and 2D Positional Encoding.
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
        # timm automatically handles the interpolation of positional embeddings
        # for the 256x1024 rectangular input size.
        self.backbone = timm.create_model(
            config.encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(2,),
            img_size=(config.img_height, config.img_width)
        )

        # CRITICAL: Enable gradient checkpointing to fit the model in VRAM
        self.backbone.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing ENABLED on Swin backbone")

        # Dynamically get the output feature dimension from timm
        feature_dim = self.backbone.feature_info.channels()[-1]
        logger.info(f"Swin backbone feature_dim: {feature_dim}")
        
        # Projection layer to match the Decoder's d_model (512)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        # FIX: Add the missing 2D Positional Encoding!
        # For a 256x1024 input with 16x downsampling, the grid is 16x64.
        # We set max_h and max_w slightly larger to be safe.
        self.pos_enc_2d = PositionalEncoding2D(
            d_model=config.d_model, 
            max_h=32, 
            max_w=128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image tensor
        Returns:
            (B, H_feat * W_feat, d_model) flattened spatial features with 2D PE
        """
        # 1. Extract features from the Swin backbone
        # Stage 2 has a total downsampling factor of 16x.
        features = self.backbone(x)[0]
        B = features.shape[0]

        # 2. Reshape into consistent (B, H_feat, W_feat, C) format
        # timm with features_only=True outputs (B, C, H, W)
        if features.dim() == 4:
            # Convert (B, C, H, W) -> (B, H, W, C)
            features = features.permute(0, 2, 3, 1)
        elif features.dim() == 3:
            # Fallback if it outputs (B, L, C)
            H_feat = self.config.img_height // 16
            W_feat = self.config.img_width // 16
            features = features.view(B, H_feat, W_feat, -1)

        # 3. Project to model dimension (d_model)
        features = self.proj(features)

        # 4. FIX: Apply 2D Positional Encoding BEFORE flattening
        # This gives the decoder the spatial awareness it needs to read left-to-right
        features = self.pos_enc_2d(features)

        # 5. Flatten spatial dimensions into a sequence
        # (B, H_feat, W_feat, d_model) -> (B, L_feat, d_model)
        features = features.reshape(B, -1, self.config.d_model)

        return features