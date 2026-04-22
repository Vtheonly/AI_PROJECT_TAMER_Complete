"""
TAMER Model: Swin-Base Encoder + KV-Cached Transformer Decoder.

v2.0 Changes (Encoder Padding Mask Edition):
  - [FIXED] generate_memory_mask(): Computes a precise boolean (B, L_mem) mask
    from the real content dimensions returned by the dataset pipeline.
    Cross-attention now attends ONLY to spatial tokens that correspond to actual
    ink strokes, completely ignoring white-padding feature tokens.
  - [FIXED] encode() now accepts optional real_ws / real_hs and returns a
    (memory, memory_mask) tuple so inference can use the mask too.
  - [FIXED] forward() passes memory_mask all the way into the decoder.
  - [RETAINED] Explicit encode() method for inference compatibility.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .encoder import SwinEncoder
from .decoder import TransformerDecoder


class TAMERModel(nn.Module):
    """
    Swin-Base Encoder + KV-Cached Transformer Decoder for Math OCR.

    The model takes RGB images padded to (img_height x img_width) with a
    top-left anchor and predicts LaTeX token sequences autoregressively.

    Encoder padding mask
    --------------------
    Swin-Base downsamples the input image by 32x (4x patch embed + 3x
    PatchMerging). A 384x1280 canvas produces a (12, 40) = 480-token feature
    grid. For an equation that only occupies the top-left 384x320 pixels,
    only the left (12, 10) = 120 tokens carry real information. Without a
    mask, cross-attention softmax spreads probability mass across all 480
    tokens, wasting 75% of its capacity on blank white patches.

    generate_memory_mask() converts the per-image (real_w, real_h) pixel
    values into a boolean (B, L_mem) mask where True = padding (ignore) and
    False = valid content (attend). The mask is passed directly into
    KVCachedAttention as an additive float mask (-inf on ignored positions).
    """

    # Swin-Base total spatial downsampling factor:
    #   PatchEmbed: 4x  +  PatchMerging x3: 2x each  =  4 * 2 * 2 * 2 = 32x
    DOWNSAMPLE_FACTOR: int = 32

    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.config = config
        self.encoder = SwinEncoder(config)
        self.decoder = TransformerDecoder(vocab_size, config)

        # Pre-compute feature grid shape so we never call division in the hot path
        self._feat_h = config.img_height // self.DOWNSAMPLE_FACTOR
        self._feat_w = config.img_width // self.DOWNSAMPLE_FACTOR

    # ------------------------------------------------------------------
    # Encoder Padding Mask
    # ------------------------------------------------------------------

    def generate_memory_mask(
        self,
        real_ws: torch.Tensor,
        real_hs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a boolean encoder padding mask for cross-attention.

        Args:
            real_ws: (B,) int tensor — pixel width of content before padding
            real_hs: (B,) int tensor — pixel height of content before padding
            device:  target device

        Returns:
            mask: (B, feat_h * feat_w) bool tensor
                  True  = padding token  → cross-attention will ignore it
                  False = content token  → cross-attention will attend to it

        Implementation note:
            We use ceiling division so a partially-covered feature cell is
            treated as valid (conservative: never mask real content).
            The resulting 2D boolean grid is flattened row-major to match
            the (B, L_mem, D) memory layout produced by SwinEncoder.
        """
        B = real_ws.size(0)
        feat_h = self._feat_h
        feat_w = self._feat_w

        # Start with everything masked (True = ignore everything)
        mask = torch.ones((B, feat_h, feat_w), dtype=torch.bool, device=device)

        for i in range(B):
            # Ceiling division: a cell partially covered by content is valid
            valid_w = min(
                feat_w,
                (int(real_ws[i].item()) + self.DOWNSAMPLE_FACTOR - 1) // self.DOWNSAMPLE_FACTOR,
            )
            valid_h = min(
                feat_h,
                (int(real_hs[i].item()) + self.DOWNSAMPLE_FACTOR - 1) // self.DOWNSAMPLE_FACTOR,
            )
            # Mark the content region as False (do NOT ignore these tokens)
            mask[i, :valid_h, :valid_w] = False

        # Flatten spatial dims to match encoder output: (B, feat_h*feat_w)
        return mask.view(B, -1)

    # ------------------------------------------------------------------
    # encode() — called by both forward() and inference loops
    # ------------------------------------------------------------------

    def encode(
        self,
        images: torch.Tensor,
        real_ws: Optional[torch.Tensor] = None,
        real_hs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode images into spatial features and optionally build a padding mask.

        Args:
            images:  (B, 3, H, W) input image tensor
            real_ws: (B,) pixel width of content region (None → no mask)
            real_hs: (B,) pixel height of content region (None → no mask)

        Returns:
            memory:      (B, L_mem, d_model) spatial feature sequence
            memory_mask: (B, L_mem) bool mask, or None if dimensions not provided
        """
        memory = self.encoder(images)   # (B, L_mem, d_model)

        memory_mask = None
        if real_ws is not None and real_hs is not None:
            memory_mask = self.generate_memory_mask(
                real_ws.to(images.device),
                real_hs.to(images.device),
                images.device,
            )

        return memory, memory_mask

    # ------------------------------------------------------------------
    # forward() — training path
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        tgt_ids: torch.Tensor,
        real_ws: Optional[torch.Tensor] = None,
        real_hs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard forward pass for training with teacher forcing.

        Args:
            images:  (B, 3, H, W) input image tensor
            tgt_ids: (B, L) target token indices (already shifted: no EOS at input)
            real_ws: (B,) pixel width of content before padding (from dataset)
            real_hs: (B,) pixel height of content before padding (from dataset)

        Returns:
            logits: (B, L, vocab_size) token predictions
        """
        # Encode image and build the spatial padding mask in one call
        memory, memory_mask = self.encode(images, real_ws, real_hs)

        # Decoder handles causal masking internally
        logits = self.decoder(tgt_ids, memory, memory_mask=memory_mask)

        return logits