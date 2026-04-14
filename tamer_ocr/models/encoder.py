"""
Swin-Base Encoder with Gradient Checkpointing.

Key changes from the old version:
- Uses swin_base_patch4_window7_224 (not tiny)
- Enables gradient checkpointing for VRAM savings
- NO square resizing — input can be non-square (256x1024)
- Projects 1024 -> 512 (d_model)
- Adds 2D positional encoding to spatial features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .attention import PositionalEncoding2D

import logging
logger = logging.getLogger("TAMER.Encoder")


class SwinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create Swin-Base backbone
        self.backbone = timm.create_model(
            config.encoder_name,  # "swin_base_patch4_window7_224"
            pretrained=True,
            features_only=True,
            out_indices=(2,)  # Extract deep features from stage 2
        )

        # CRITICAL: Enable gradient checkpointing to fit on T4 (16GB VRAM)
        self.backbone.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing ENABLED on Swin backbone")

        # Detect output format dynamically
        dummy_input = torch.randn(1, 3, 256, 1024)
        with torch.no_grad():
            dummy_out = self.backbone(dummy_input)[0]

        if dummy_out.dim() == 3:
            if dummy_out.shape[1] > dummy_out.shape[2]:
                self.format = "BLC"
                feature_dim = dummy_out.shape[2]
            else:
                self.format = "BCL"
                feature_dim = dummy_out.shape[1]
        else:
            if dummy_out.shape[1] > dummy_out.shape[-1]:
                self.format = "BCHW"
                feature_dim = dummy_out.shape[1]
            else:
                self.format = "BHWC"
                feature_dim = dummy_out.shape[-1]

        logger.info(f"Swin output format: {self.format}, feature_dim: {feature_dim}")
        logger.info(f"Swin output shape: {dummy_out.shape}")

        # Project from Swin output dim (1024) to d_model (512)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        # 2D Positional Encoding for spatial features
        self.pos2d = PositionalEncoding2D(config.d_model, max_h=32, max_w=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) grayscale images
        Returns:
            (B, L, D) flattened spatial features with 2D positional encoding
        """
        # Expand grayscale to 3 channels for Swin
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)

        # Run through Swin backbone
        features = self.backbone(x)[0]

        B = features.shape[0]

        # Convert to BHWC format
        if self.format == "BCHW":
            features = features.permute(0, 2, 3, 1)  # NCHW -> NHWC
        elif self.format == "BLC":
            L = features.shape[1]
            H = W = int(L ** 0.5)
            features = features.view(B, H, W, -1)
        elif self.format == "BCL":
            features = features.permute(0, 2, 1)  # BCL -> BLC
            L = features.shape[1]
            H = W = int(L ** 0.5)
            features = features.view(B, H, W, -1)

        # Project to d_model
        features = self.proj(features)

        # Add 2D positional encoding
        features = self.pos2d(features)

        # Flatten spatial dims: (B, H, W, D) -> (B, H*W, D)
        B, H, W, D = features.shape
        features = features.reshape(B, H * W, D)

        return features
