"""
Swin-Base Encoder with Gradient Checkpointing.
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

        # FIX: explicitly pass img_size to bypass strict shape assertions in timm
        self.backbone = timm.create_model(
            config.encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(2,),
            img_size=(config.img_height, config.img_width)
        )

        self.backbone.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing ENABLED on Swin backbone")

        dummy_input = torch.randn(1, 3, config.img_height, config.img_width)
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
        
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[0]
        B = features.shape[0]

        # Convert everything to B, H, W, C
        if self.format == "BCHW":
            features = features.permute(0, 2, 3, 1)
        elif self.format == "BLC":
            # FIX: Dynamically calculate Height and Width of feature map based on aspect ratio
            L = features.shape[1]
            ratio = self.config.img_width / self.config.img_height
            H_feat = int((L / ratio) ** 0.5)
            W_feat = int(H_feat * ratio)
            features = features.view(B, H_feat, W_feat, -1)
        elif self.format == "BCL":
            features = features.permute(0, 2, 1)
            L = features.shape[1]
            ratio = self.config.img_width / self.config.img_height
            H_feat = int((L / ratio) ** 0.5)
            W_feat = int(H_feat * ratio)
            features = features.view(B, H_feat, W_feat, -1)

        features = self.proj(features)
        B, H, W, D = features.shape
        features = features.reshape(B, H * W, D)

        return features