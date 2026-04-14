"""
TAMER Model: Swin-Base Encoder + Standard Transformer Decoder.

Simplified from the original:
- NO pointer network
- NO coverage attention
- NO text-only pretraining mode
- NO tree structure
"""

import torch
import torch.nn as nn
from .encoder import SwinEncoder
from .decoder import TransformerDecoder


class TAMERModel(nn.Module):
    """
    Swin-Base Encoder + Standard Transformer Decoder for Math OCR.
    """

    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.config = config
        self.encoder = SwinEncoder(config)
        self.decoder = TransformerDecoder(vocab_size, config)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into spatial features."""
        return self.encoder(images)

    def forward(self, images: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 1, H, W) input images
            tgt_ids: (B, L) target token indices (teacher forcing)

        Returns:
            logits: (B, L, vocab_size) token predictions
        """
        memory = self.encode(images)
        logits = self.decoder(tgt_ids, memory)
        return logits
