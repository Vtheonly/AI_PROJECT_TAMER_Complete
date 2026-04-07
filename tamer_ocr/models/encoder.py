import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(max_h, d_model // 2) * 0.1)
        self.col_embed = nn.Parameter(torch.randn(max_w, d_model // 2) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, H, W, D]
        h, w = x.shape[1], x.shape[2]
        pos = torch.cat([
            self.row_embed[:h].unsqueeze(1).expand(-1, w, -1),
            self.col_embed[:w].unsqueeze(0).expand(h, -1, -1)
        ], dim=-1)
        return x + pos.unsqueeze(0)

class SwinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(
            config.encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(2,)  # Extract deep features
        )
        
        # SwinV2 tiny features out channel is usually 384 at index 2 (varies by model, assuming standard)
        dummy_input = torch.randn(1, 3, 256, 256)
        dummy_out = self.backbone(dummy_input)[0]
        feature_dim = dummy_out.shape[1]
        
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        self.pos2d = PositionalEncoding2D(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
            
        # Standardize size for Swin
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        features = self.backbone(x)[0]
        
        if features.dim() == 4:
            features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        features = self.proj(features)
        features = self.pos2d(features)
        
        B, H, W, D = features.shape
        features = features.reshape(B, H * W, D)
        return features