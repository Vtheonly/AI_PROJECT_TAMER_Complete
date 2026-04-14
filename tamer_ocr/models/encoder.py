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

        # We pass img_size to allow timm to interpolate position embeddings
        self.backbone = timm.create_model(
            config.encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(2,),
            img_size=(config.img_height, config.img_width)
        )

        self.backbone.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing ENABLED on Swin backbone")

        # Detect the output dimension and format of the backbone
        dummy_input = torch.randn(1, 3, config.img_height, config.img_width)
        with torch.no_grad():
            dummy_out = self.backbone(dummy_input)[0]

        if dummy_out.dim() == 3:
            feature_dim = dummy_out.shape[-1]
            self.format = "BLC"
        else:
            feature_dim = dummy_out.shape[1]
            self.format = "BCHW"

        logger.info(f"Swin output format: {self.format}, feature_dim: {feature_dim}")
        
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 1024) images
        Returns:
            (B, 1024, 512) flattened spatial features
        """
        features = self.backbone(x)[0]
        B = features.shape[0]

        # Convert everything to B, H, W, C format first
        if self.format == "BCHW":
            # (B, C, H, W) -> (B, H, W, C)
            features = features.permute(0, 2, 3, 1)
        
        elif self.format == "BLC":
            # At Stage 2, Swin-Base downsamples by 16x.
            # 256 // 16 = 16 | 1024 // 16 = 64
            H_feat = self.config.img_height // 16
            W_feat = self.config.img_width // 16
            features = features.view(B, H_feat, W_feat, -1)

        # Apply the projection and LayerNorm
        features = self.proj(features)

        # Flatten spatial dims into a sequence for the Transformer Decoder
        # (B, 16, 64, 512) -> (B, 1024, 512)
        features = features.reshape(B, -1, self.config.d_model)

        return features