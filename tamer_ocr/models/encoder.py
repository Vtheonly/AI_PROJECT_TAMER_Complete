"""
Swin-Base Encoder with Strong 2D Positional Encoding and Row Boundary Markers.

v2.5 Changes (Offline + Dynamic Positional Fix):
  - [FIX] Dynamic positional embedding dimensions based on actual image resolution
    instead of hardcoded 64x128, preventing clamp-induced position collapse
  - Offline pre-trained weight loading fully supported via config.local_backbone_path
  - No internet required when local weights are provided
"""

import os
import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger("TAMER.Encoder")


class SwinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        img_size = (config.img_height, config.img_width)

        # ----------------------------------------------------------------
        # Offline-first: Check for local pretrained weights
        # ----------------------------------------------------------------
        local_weights_path = getattr(config, "local_backbone_path", None)
        use_local_weights = (
            local_weights_path is not None
            and os.path.exists(local_weights_path)
        )

        want_pretrained = not use_local_weights

        _BACKBONE_FALLBACKS = [
            getattr(config, "encoder_name", "swinv2_base_window8_256.ms_in1k"),
            "swinv2_base_window8_256.ms_in1k",
            "swinv2_small_window8_256.ms_in1k",
            "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        ]
        
        seen = set()
        FALLBACKS = []
        for n in _BACKBONE_FALLBACKS:
            if n not in seen:
                seen.add(n)
                FALLBACKS.append(n)

        self.backbone = None
        for _name in FALLBACKS:
            try:
                self.backbone = timm.create_model(
                    _name,
                    pretrained=want_pretrained,  # False when using local weights
                    features_only=True,
                    out_indices=(3,),
                    img_size=img_size,
                )
                logger.info(
                    f"Swin backbone created: {_name} | "
                    f"pretrained={want_pretrained} | img_size={img_size}"
                )
                break
            except Exception as e:
                logger.warning(f"timm model not available: {_name} ({e}), trying next...")

        if self.backbone is None:
            raise RuntimeError(f"No Swin backbone could be loaded. Tried: {FALLBACKS}")

        # ----------------------------------------------------------------
        # Load offline weights from local safetensors/pt file
        # ----------------------------------------------------------------
        if use_local_weights:
            logger.info(f"Loading offline pre-trained weights from: {local_weights_path}")
            try:
                from safetensors.torch import load_file
                state_dict = load_file(local_weights_path)
                logger.info("✅ Loaded weights using safetensors")
            except ImportError:
                logger.warning("safetensors not installed, falling back to torch.load")
                state_dict = torch.load(local_weights_path, map_location="cpu")
            except Exception as e:
                logger.error(f"Failed to load with safetensors: {e}, trying torch.load")
                state_dict = torch.load(local_weights_path, map_location="cpu")

            # Remove classification head keys (incompatible with features_only=True)
            keys_to_delete = [k for k in state_dict if "head" in k]
            for k in keys_to_delete:
                del state_dict[k]
                logger.debug(f"  Removed key from checkpoint: {k}")

            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(
                    f"  Missing keys ({len(missing)}): "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            if unexpected:
                logger.warning(
                    f"  Unexpected keys ({len(unexpected)}): "
                    f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
                )

            logger.info("✅ Pre-trained weights loaded successfully (offline mode)")
        else:
            if want_pretrained:
                logger.info("✅ Pre-trained ImageNet weights loaded via timm (online mode)")
            else:
                logger.warning(
                    f"⚠️ local_backbone_path set but file not found: {local_weights_path} "
                    "— training encoder from SCRATCH"
                )

        self.backbone.set_grad_checkpointing(True)

        feature_info = self.backbone.feature_info.get_dicts()
        self.in_channels = feature_info[0]["num_chs"]
        logger.info(f"Swin Encoder: backbone output channels = {self.in_channels}")

        self.proj = nn.Linear(self.in_channels, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        # ----------------------------------------------------------------
        # [FIX: Dynamic Positional Embedding Dimensions]
        # Swin Transformer reduces spatial dims by ~4-32x depending on stage.
        # For stage 3 (out_indices=(3,)), typical reduction is ~8x.
        # We calculate max embeddings dynamically to prevent clamp-induced
        # positional collapse at high resolutions (e.g., 384x1280).
        # ----------------------------------------------------------------
        reduction_factor = 4  # Conservative estimate for safety
        max_h = max(64, (config.img_height // reduction_factor) + 16)
        max_w = max(128, (config.img_width // reduction_factor) + 16)
        half_d = config.d_model // 2

        logger.info(
            f"Positional embeddings initialized: "
            f"max_h={max_h}, max_w={max_w} (from img_size={img_size})"
        )

        self.row_embed = nn.Embedding(max_h, half_d)
        self.col_embed = nn.Embedding(max_w, half_d)

        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

        self.row_boundary_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.row_boundary_base = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[0]
        B = features.shape[0]

        # Handle both (B, C, H, W) and (B, L, C) feature formats
        if features.dim() == 4:
            if features.shape[1] == self.in_channels:
                features = features.permute(0, 2, 3, 1).contiguous()
        elif features.dim() == 3:
            aspect = max(self.config.img_width / self.config.img_height, 1.0)
            L = features.shape[1]
            best_h, best_w = 1, L
            min_diff = float("inf")
            for h in range(1, int(L ** 0.5) + 1):
                if L % h == 0:
                    w = L // h
                    diff = abs((w / h) - aspect)
                    if diff < min_diff:
                        min_diff = diff
                        best_h, best_w = h, w
            features = features.view(B, best_h, best_w, self.in_channels)

        H, W = features.shape[1], features.shape[2]

        # Project and normalize
        x = self.proj(features)
        x = self.norm(x)

        # Apply 2D positional encoding
        # With dynamic sizing, clamp is now a safety check rather than a bug
        row_ids = torch.arange(H, device=x.device).clamp(max=self.row_embed.num_embeddings - 1)
        col_ids = torch.arange(W, device=x.device).clamp(max=self.col_embed.num_embeddings - 1)

        row_emb = self.row_embed(row_ids).unsqueeze(1).expand(-1, W, -1)
        col_emb = self.col_embed(col_ids).unsqueeze(0).expand(H, -1, -1)
        pos_2d = torch.cat([row_emb, col_emb], dim=-1)

        x = x + pos_2d.unsqueeze(0)

        # Add row boundary markers
        row_means = x.mean(dim=2)
        boundary_embeds = self.row_boundary_proj(row_means) + self.row_boundary_base

        rows = []
        for h in range(H):
            rows.append(x[:, h, :, :])
            if h < H - 1:
                rows.append(boundary_embeds[:, h:h+1, :])

        x = torch.cat(rows, dim=1)
        return x