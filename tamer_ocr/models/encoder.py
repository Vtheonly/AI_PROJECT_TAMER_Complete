"""
Swin-Base Encoder with Strong 2D Positional Encoding and Row Boundary Markers.

v2.3 Changes:
  - Replaced weak PositionalEncoding2D (attention.py) with STRONG learned
    row/column embeddings that survive the flatten operation.
    The decoder can now distinguish "row 1 col 3" from "row 3 col 1"
    after flatten, which is critical for matrix recognition.

  - Added row boundary markers: a learned "newline" embedding is inserted
    between each row of encoder patches. This gives the decoder an explicit
    signal for row transitions, directly analogous to \\\\ in LaTeX.
    The sequence length grows from H*W to H*W + (H-1).

  - Gradient checkpointing enabled for VRAM savings on T4.
  - img_size=(H, W) passed to timm for rectangular input support.
  - Robust (B, C, H, W) vs (B, H, W, C) dimension handling.
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

        img_size = (config.img_height, config.img_width)  # e.g. (256, 1024)

        # Try the configured model name, fall back to known-good alternatives.
        _BACKBONE_FALLBACKS = [
            config.encoder_name,                              # primary: swinv2_base_window8_256.ms_in1k
            "swinv2_base_window8_256.ms_in1k",               # explicit safe fallback (same model)
            "swinv2_small_window8_256.ms_in1k",              # smaller variant
            "swin_base_patch4_window7_224.ms_in22k_ft_in1k", # v1 last-resort
        ]

        self.backbone = None
        for _name in _BACKBONE_FALLBACKS:
            try:
                self.backbone = timm.create_model(
                    _name,
                    pretrained=True,
                    features_only=True,
                    out_indices=(3,),
                    img_size=img_size,  # override patch embed assertion for 256x1024
                )
                logger.info(f"Swin backbone loaded: {_name} | img_size={img_size}")
                break
            except (RuntimeError, KeyError, Exception) as e:
                logger.warning(f"timm model not available: {_name} ({e}), trying next...")

        if self.backbone is None:
            raise RuntimeError(f"No Swin backbone could be loaded. Tried: {_BACKBONE_FALLBACKS}")

        # Enable gradient checkpointing to save VRAM on Tesla T4/Colab
        self.backbone.set_grad_checkpointing(True)

        # Get the actual channel count from the backbone's feature info.
        feature_info = self.backbone.feature_info.get_dicts()
        self.in_channels = feature_info[0]['num_chs']

        logger.info(f"Swin Encoder: Backbone features detected with {self.in_channels} channels.")

        # Projection layer to transform backbone channels to model dimension (d_model)
        self.proj = nn.Linear(self.in_channels, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        # ----------------------------------------------------------------
        # STRONG 2D Positional Encoding (replaces PositionalEncoding2D)
        #
        # Uses nn.Embedding for row and column positions. Each position
        # gets d_model//2 dimensions for row + d_model//2 for column,
        # concatenated to d_model. These embeddings SURVIVE the flatten
        # operation because they're ADDED to the features, not applied
        # as a separate spatial structure.
        #
        # Why this is better than the old approach:
        # - nn.Embedding indices are discrete and learned per-position
        # - After flatten, token at position [r, c] still carries
        #   distinct row+col information through its embedding
        # ----------------------------------------------------------------
        max_h = 64   # supports up to 64 patch rows (1024px / 16 downsample)
        max_w = 128  # supports up to 128 patch cols (2048px / 16 downsample)
        half_d = config.d_model // 2

        self.row_embed = nn.Embedding(max_h, half_d)
        self.col_embed = nn.Embedding(max_w, half_d)

        # Initialize with small values for stable training start
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

        # ----------------------------------------------------------------
        # Row Boundary Markers
        #
        # A learned "newline" embedding inserted between each row of
        # patches in the flattened sequence. This gives the decoder an
        # explicit signal for row transitions.
        #
        # For an 8×32 feature grid (256×1024 input):
        #   Without markers: 256 tokens
        #   With markers:    256 + 7 = 263 tokens (7 row boundaries)
        #
        # The row boundary is computed as a projection of the mean
        # features in that row, giving it content-awareness.
        # ----------------------------------------------------------------
        self.row_boundary_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

        # Learnable base embedding for row boundaries (added to projection)
        self.row_boundary_base = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            features: (B, L, d_model) where L = H_feat * W_feat + (H_feat - 1)
                      The extra (H_feat - 1) tokens are row boundary markers.
        """
        # 1. Extract features from Swin
        features = self.backbone(x)[0]
        B = features.shape[0]

        # 2. Robust Dimension Handling
        if features.dim() == 4:
            if features.shape[1] == self.in_channels:
                # (B, C, H, W) → (B, H, W, C)
                features = features.permute(0, 2, 3, 1).contiguous()
            # else: already (B, H, W, C)
        elif features.dim() == 3:
            # (B, L, C) → need to figure out H, W
            H_feat = self.config.img_height // 32  # 4 stages of 2x downsample
            W_feat = self.config.img_width // 32
            L = features.shape[1]
            if L != H_feat * W_feat:
                # Adaptive fallback
                aspect = self.config.img_width / self.config.img_height
                H_feat = int((L / aspect) ** 0.5)
                W_feat = L // H_feat
            features = features.view(B, H_feat, W_feat, self.in_channels)

        H, W = features.shape[1], features.shape[2]

        # 3. Channel Projection: (B, H, W, C_in) → (B, H, W, d_model)
        x = self.proj(features)
        x = self.norm(x)

        # 4. Strong 2D Positional Encoding
        # Clamp indices to embedding size for safety
        row_ids = torch.arange(H, device=x.device).clamp(max=self.row_embed.num_embeddings - 1)
        col_ids = torch.arange(W, device=x.device).clamp(max=self.col_embed.num_embeddings - 1)

        row_emb = self.row_embed(row_ids)  # (H, d_model//2)
        col_emb = self.col_embed(col_ids)  # (W, d_model//2)

        # Broadcast to (H, W, d_model)
        row_emb = row_emb.unsqueeze(1).expand(-1, W, -1)  # (H, W, d_model//2)
        col_emb = col_emb.unsqueeze(0).expand(H, -1, -1)  # (H, W, d_model//2)
        pos_2d = torch.cat([row_emb, col_emb], dim=-1)     # (H, W, d_model)

        x = x + pos_2d.unsqueeze(0)  # (B, H, W, d_model)

        # 5. Build flattened sequence WITH row boundary markers
        # Instead of simple flatten, interleave rows with boundary tokens.
        # This gives the decoder explicit row-transition signals.
        rows = []
        # Compute per-row mean for content-aware boundary embeddings
        row_means = x.mean(dim=2)  # (B, H, d_model)
        boundary_embeds = self.row_boundary_proj(row_means) + self.row_boundary_base  # (B, H, d_model)

        for h in range(H):
            rows.append(x[:, h, :, :])  # (B, W, d_model) — patches for row h
            if h < H - 1:
                # Insert boundary marker between rows
                rows.append(boundary_embeds[:, h:h + 1, :])  # (B, 1, d_model)

        # Concatenate: (B, H*W + (H-1), d_model)
        x = torch.cat(rows, dim=1)

        return x