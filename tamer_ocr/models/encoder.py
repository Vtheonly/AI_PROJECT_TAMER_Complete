import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(max_h, d_model // 2) * 0.1)
        self.col_embed = nn.Parameter(torch.randn(max_w, d_model // 2) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, H, W, D]
        h, w = x.shape[1], x.shape[2]
        
        # Safe slicing to prevent dimension mismatch crashes
        r_h = min(h, self.row_embed.size(0))
        c_w = min(w, self.col_embed.size(0))
        
        row = self.row_embed[:r_h]
        col = self.col_embed[:c_w]
        
        # If input is unexpectedly large, pad embeddings so it doesn't crash
        if h > r_h:
            row = F.pad(row, (0, 0, 0, h - r_h))
        if w > c_w:
            col = F.pad(col, (0, 0, 0, w - c_w))

        pos = torch.cat([
            row.unsqueeze(1).expand(-1, w, -1),
            col.unsqueeze(0).expand(h, -1, -1)
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
        
        # Dynamically detect output format to prevent dimension crashes
        dummy_input = torch.randn(1, 3, 256, 256)
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
        
        B = features.shape[0]
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
        
        features = self.proj(features)
        features = self.pos2d(features)
        
        B, H, W, D = features.shape
        features = features.reshape(B, H * W, D)
        return features