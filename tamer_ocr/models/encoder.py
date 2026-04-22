
"""
Swin-Base Encoder with Strong 2D Positional Encoding.

v3.0 Changes (Article-Guided Fixes):
  - [FIX] Model name corrected to swinv2_base_window12_192.ms_in22k
  - [FIX] Removed row boundary markers: they created a 1D snake that destroyed
    vertical spatial awareness. Each token now carries baked-in 2D GPS coords
    before flattening, so (row, col) position survives sequence conversion.
  - [FIX] Positional embeddings changed from nn.Embedding to nn.Parameter
    with trunc_normal_ init, matching article specification.
  - [FIX] Forward now adds 2D position while still in grid form (H, W),
    then flattens to (B, H*W, d_model) in one clean step.
  - [FIX] FATAL SPATIAL SCRAMBLING FIX: Replaced the broken aspect-ratio
    guessing loop with exact mathematical grid reconstruction using the
    actual input tensor dimensions and derived backbone stride. This
    prevents silent corruption of spatial token ordering.
  - [ADD] Three-layer weight verification (polygraph test):
      Layer 1: Equality check - did weights actually change after loading?
      Layer 2: Statistical signature - are these real pretrained weights or noise?
      Layer 3: Zero tolerance - crash if any transformer blocks are missing.
  - [KEEP] All original offline loading logic, fallback list, safetensors
    handling, key cleanup, grad checkpointing, feature_info detection, logging.
"""

import os
import math
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

        # Article specifies swinv2_base_window12_192.ms_in22k as the correct
        # model name. It must match exactly the model used to export weights.
        _BACKBONE_FALLBACKS = [
            getattr(config, "encoder_name", "swinv2_base_window12_192.ms_in22k"),
            "swinv2_base_window12_192.ms_in22k",
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
                    pretrained=want_pretrained,
                    features_only=True,
                    out_indices=(3,),
                    img_size=img_size,
                    strict_img_size=False,
                )
                logger.info(
                    f"Swin backbone created: {_name} | "
                    f"pretrained={want_pretrained} | img_size={img_size}"
                )
                break
            except Exception as e:
                logger.warning(
                    f"timm model not available: {_name} ({e}), trying next..."
                )

        if self.backbone is None:
            raise RuntimeError(
                f"No Swin backbone could be loaded. Tried: {FALLBACKS}"
            )

        # ----------------------------------------------------------------
        # Load offline weights from local safetensors/pt file
        # ----------------------------------------------------------------
        if use_local_weights:
            logger.info(
                f"Loading offline pre-trained weights from: {local_weights_path}"
            )
            try:
                from safetensors.torch import load_file
                state_dict = load_file(local_weights_path)
                logger.info("Loaded weights using safetensors")
            except ImportError:
                logger.warning(
                    "safetensors not installed, falling back to torch.load"
                )
                state_dict = torch.load(local_weights_path, map_location="cpu")
            except Exception as e:
                logger.error(
                    f"Failed to load with safetensors: {e}, trying torch.load"
                )
                state_dict = torch.load(local_weights_path, map_location="cpu")

            # Remove classification head keys
            keys_to_delete = [k for k in state_dict if "head" in k]
            for k in keys_to_delete:
                del state_dict[k]
                logger.debug(f"  Removed key from checkpoint: {k}")

            # ----------------------------------------------------------------
            # Key name fix: timm features_only uses 'layers_' with underscore.
            # A full model export uses 'layers.' with a dot.
            # Article identifies this as necessary for compatibility.
            # ----------------------------------------------------------------
            fixed_weights = {}
            for k, v in state_dict.items():
                new_key = k.replace("layers.", "layers_")
                fixed_weights[new_key] = v

            # ----------------------------------------------------------------
            # Three-layer polygraph verification (article: polygraph test)
            # ----------------------------------------------------------------

            # Layer 1: Snapshot before loading
            pre_weight = (
                self.backbone.layers_0.blocks[0].mlp.fc1.weight.clone()
            )

            missing, unexpected = self.backbone.load_state_dict(
                fixed_weights, strict=False
            )

            # Layer 1 continued: Did weights actually change?
            post_weight = self.backbone.layers_0.blocks[0].mlp.fc1.weight
            if torch.equal(pre_weight, post_weight):
                logger.error(
                    "Weight probe failed: model weights are identical to "
                    "random initialization after loading. The file may be "
                    "empty or completely mismatched."
                )
                raise RuntimeError(
                    "Failed to load weights. The model is still blind."
                )

            # Layer 2: Statistical signature check
            # Random weights always have mean~0.0 and std~0.02 from init.
            # Real pretrained weights have a complex non-random distribution.
            weight_mean = post_weight.mean().item()
            weight_std = post_weight.std().item()
            logger.info(
                f"Weight probe [layers_0.blocks[0].mlp.fc1.weight]: "
                f"mean={weight_mean:.6f}, std={weight_std:.6f}"
            )
            if abs(weight_mean) < 1e-5 and abs(weight_std - 0.02) < 1e-3:
                logger.error(
                    "Loaded weights have the statistical signature of random "
                    "initialization (mean~0, std~0.02). These are not real "
                    "pre-trained weights."
                )
                raise RuntimeError(
                    "Weights loaded but they are not real pre-trained weights."
                )

            # Layer 3: Zero tolerance for missing transformer blocks
            # relative_coords_table and attn_mask are regenerated at runtime
            # for the target resolution, so they are expected to be missing.
            # Actual transformer block weights missing means the brain is gone.
            missing_blocks = [m for m in missing if "blocks" in m]
            if len(missing_blocks) > 0:
                logger.error(
                    f"Surgery failed: {len(missing_blocks)} transformer "
                    f"blocks are missing from the loaded weights."
                )
                logger.error(f"Sample missing key: {missing_blocks[0]}")
                raise RuntimeError(
                    "Encoder is blind. Check your weight file or model_name."
                )

            ignored_missing = [m for m in missing if "blocks" not in m]
            if ignored_missing:
                logger.info(
                    f"Ignored {len(ignored_missing)} expected missing keys "
                    f"(dynamic attention masks, relative coords, etc.)"
                )
            if unexpected:
                logger.warning(
                    f"Unexpected keys ({len(unexpected)}): "
                    f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
                )

            logger.info(
                "Verification passed: pre-trained weights loaded and active "
                "(offline mode)"
            )

        else:
            if want_pretrained:
                logger.info(
                    "Pre-trained ImageNet weights loaded via timm (online mode)"
                )
            else:
                logger.warning(
                    f"local_backbone_path set but file not found: "
                    f"{local_weights_path} -- training encoder from scratch"
                )

        self.backbone.set_grad_checkpointing(True)

        feature_info = self.backbone.feature_info.get_dicts()
        self.in_channels = feature_info[0]["num_chs"]
        logger.info(f"Swin Encoder: backbone output channels = {self.in_channels}")

        self.proj = nn.Linear(self.in_channels, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        # ----------------------------------------------------------------
        # 2D Positional Embeddings as nn.Parameter (article fix).
        # Using nn.Embedding caused clamp-induced position collapse at high
        # resolutions. nn.Parameter with trunc_normal_ init gives every
        # token its own learnable (row, col) GPS coordinate.
        # 256x256 grid gives overhead for large images like 384x1280.
        # ----------------------------------------------------------------
        self.row_embed = nn.Parameter(
            torch.zeros(256, config.d_model // 2)
        )
        self.col_embed = nn.Parameter(
            torch.zeros(256, config.d_model // 2)
        )
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------------------------------------------------------------
        # Capture the EXACT input dimensions of this specific batch.
        # These are used later to mathematically reconstruct the spatial
        # grid WITHOUT guessing. This is the foundation of the spatial fix.
        # ----------------------------------------------------------------
        B_in, C_in, H_in, W_in = x.shape

        # Extract features from backbone
        features = self.backbone(x)[0]

        # Handle both (B, C, H, W) and (B, L, C) feature formats
        if features.dim() == 4:
            if features.shape[1] == self.in_channels:
                # Proper format, just move channels to the end: (B, H, W, C)
                features = features.permute(0, 2, 3, 1).contiguous()

        elif features.dim() == 3:
            B_dim, L, C_dim = features.shape

            # ----------------------------------------------------------------
            # THE FIX: Exact Mathematical Spatial Reconstruction.
            #
            # THE OLD BUG: The previous code looped through integer factors
            # of L and picked the one whose (w/h) ratio best matched the
            # config aspect ratio. This is algebraic guessing. If even one
            # pixel of padding shifted the actual grid from e.g. 12x40 to
            # 12x41, the loop would choose a completely wrong factorization
            # such as 10x48, physically scrambling the image. Every token
            # for "top-left" would end up tagged as "middle-right". The
            # 2D positional encodings added on top of this scramble would
            # then teach the model a hallucinated and broken sense of space.
            #
            # THE FIX: We know the sequence L is exactly H_feat * W_feat,
            # and that H_feat = H_in / stride, W_feat = W_in / stride.
            # Therefore: stride^2 = (H_in * W_in) / L.
            # We reverse-engineer the exact backbone stride from the actual
            # tensor shapes, then reconstruct the grid with zero guessing.
            # ----------------------------------------------------------------
            stride_sq = (H_in * W_in) / L
            stride = int(round(math.sqrt(stride_sq)))

            # Calculate the exact feature grid dimensions.
            # ceil safely handles any internal Swin window-alignment padding.
            feat_h = math.ceil(H_in / stride)
            feat_w = math.ceil(W_in / stride)

            # Failsafe pass 1: if Swin's internal padding caused a 1-token
            # width shift, pin the height and strictly derive width from L.
            if feat_h * feat_w != L:
                feat_w = L // feat_h

            # Zero-tolerance check: If the math still does not resolve to L
            # exactly, we must crash rather than silently scramble the grid.
            # A scrambled spatial grid with baked-in 2D positional encodings
            # is worse than no positional encoding at all: it actively teaches
            # the model wrong spatial relationships that it cannot unlearn.
            if feat_h * feat_w != L:
                raise RuntimeError(
                    f"CRITICAL SPATIAL SCRAMBLING PREVENTED!\n"
                    f"Input image shape : {H_in} x {W_in}\n"
                    f"Sequence length   : {L}\n"
                    f"Derived stride    : {stride}\n"
                    f"Calculated grid   : {feat_h} x {feat_w} = "
                    f"{feat_h * feat_w} != {L}\n"
                    f"The backbone's downsampling stride appears non-uniform.\n"
                    f"Check your backbone configuration or input image size."
                )

            # Safely reshape knowing the math is exact and verified
            features = features.view(B_dim, feat_h, feat_w, C_dim)

        # At this point we are guaranteed a pristine (B, H, W, C) grid
        # with correct spatial ordering regardless of whether the backbone
        # returned 4D or 3D tensors.
        B, H, W, C = features.shape

        # Project channels to d_model and normalize
        x_proj = self.proj(features)
        x_proj = self.norm(x_proj)

        # ----------------------------------------------------------------
        # Bake 2D GPS coordinates into every token BEFORE flattening.
        #
        # Article diagnosis: the old boundary-marker approach turned the
        # 2D grid into a 1D snake. At 1280px wide, row N was 1000+ steps
        # away from row N+1 in sequence space. The decoder lost its vertical
        # anchor and started repeating tokens because it could not see that
        # a numerator sits above a denominator.
        #
        # Fix: add (row, col) position while data is still a (H, W) grid.
        # Even after flattening, each token retains its spatial coordinates.
        # Space is meaning in math OCR: x/y is just x above y. Without
        # baked-in 2D coords, that vertical relationship is invisible.
        #
        # Slicing [:H, :] and [:W, :] is safe because our input resolution
        # fits within the 256x256 parameter bounds initialized in __init__.
        # ----------------------------------------------------------------
        row_pos = self.row_embed[:H, :].unsqueeze(1).repeat(1, W, 1)
        col_pos = self.col_embed[:W, :].unsqueeze(0).repeat(H, 1, 1)
        pos = torch.cat([row_pos, col_pos], dim=-1).unsqueeze(0)

        # Add positional encoding while still in 2D grid form, then flatten.
        # Output shape: (B, H*W, d_model)
        return (x_proj + pos).view(B, -1, self.config.d_model)
