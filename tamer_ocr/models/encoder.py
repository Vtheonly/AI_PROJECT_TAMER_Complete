"""
Swin-Base Encoder with Gradient Checkpointing.
Fixed for 256x1024 rectangular input and dynamic feature reshaping.
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger("TAMER.Encoder")

class SwinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # FIX: We pass the specific rectangular img_size to the timm constructor.
        # This prevents the AssertionError and tells the model to interpolate 
        # its internal position embeddings to fit the 256x1024 grid.
        self.backbone = timm.create_model(
            config.encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(2,),
            img_size=(config.img_height, config.img_width)
        )

        # CRITICAL: Enable gradient checkpointing to fit the model in T4/P100 VRAM
        self.backbone.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing ENABLED on Swin backbone")

        # Perform a dummy forward pass to detect the output dimension and format
        # Stage 2 of Swin-Base typically results in a 512-dimension feature map.
        dummy_input = torch.randn(1, 3, config.img_height, config.img_width)
        with torch.no_grad():
            dummy_out = self.backbone(dummy_input)[0]

        # Determine if the backbone outputs BLC (sequence) or BCHW (image tensor)
        if dummy_out.dim() == 3:
            feature_dim = dummy_out.shape[-1]
            self.format = "BLC"
        else:
            feature_dim = dummy_out.shape[1]
            self.format = "BCHW"

        logger.info(f"Swin backbone detected output format: {self.format}, feature_dim: {feature_dim}")
        
        # Projection layer to match the Decoder's d_model (512)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 1024) input image tensor
        Returns:
            (B, 1024, 512) flattened spatial features for the Transformer Decoder
        """
        # 1. Extract features from the Swin backbone
        # Stage 2 has a total downsampling factor of 16x.
        features = self.backbone(x)[0]
        B = features.shape[0]

        # 2. Reshape into consistent (B, H_feat, W_feat, C) format
        if self.format == "BCHW":
            # Convert (B, C, H, W) -> (B, H, W, C)
            features = features.permute(0, 2, 3, 1)
        
        elif self.format == "BLC":
            # For 256x1024 input:
            # H_feat = 256 // 16 = 16
            # W_feat = 1024 // 16 = 64
            # Total sequence length L = 1024
            H_feat = self.config.img_height // 16
            W_feat = self.config.img_width // 16
            features = features.view(B, H_feat, W_feat, -1)

        # 3. Project to model dimension (d_model)
        features = self.proj(features)

        # 4. Flatten spatial dimensions into a sequence
        # (B, 16, 64, 512) -> (B, 1024, 512)
        features = features.reshape(B, -1, self.config.d_model)

        return features