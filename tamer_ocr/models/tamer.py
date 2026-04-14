"""
TAMER Model: Swin-Base Encoder + Standard Transformer Decoder.

Simplified from the original:
- NO pointer network
- NO coverage attention
- NO tree structure
- FIXED: Explicit encode() method for batched inference compatibility.
"""

import torch
import torch.nn as nn
from .encoder import SwinEncoder
from .decoder import TransformerDecoder


class TAMERModel(nn.Module):
    """
    Swin-Base Encoder + Standard Transformer Decoder for Math OCR.
    
    This model takes RGB images and predicts LaTeX token sequences.
    """

    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.config = config
        self.encoder = SwinEncoder(config)
        self.decoder = TransformerDecoder(vocab_size, config)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into spatial features. 
        This is called by inference loops to compute memory once.
        
        Args:
            images: (B, 3, H, W) input image tensor
        Returns:
            (B, L_feat, d_model) spatial feature sequence
        """
        return self.encoder(images)

    def forward(self, images: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training with teacher forcing.
        
        Args:
            images: (B, 3, H, W) input image tensor
            tgt_ids: (B, L) target token indices

        Returns:
            logits: (B, L, vocab_size) token predictions
        """
        # 1. Extract memory from encoder
        memory = self.encode(images)
        
        # 2. Pass memory and targets to decoder
        # The decoder handles causal masking internally
        logits = self.decoder(tgt_ids, memory)
        
        return logits