"""
Standard Transformer Decoder for Math OCR.

Key changes from the old version:
- NO pointer network (pointer_q, pointer_k removed)
- NO coverage attention
- NO parent_proj (tree structure removed)
- Standard nn.TransformerDecoder with sinusoidal positional encoding
- 6 layers, 8 heads, d_model=512, dim_feedforward=2048
"""

import torch
import torch.nn as nn
import math
from .attention import PositionalEncoding1D


class TransformerDecoder(nn.Module):
    """
    Standard Transformer Decoder for LaTeX sequence generation.

    Uses PyTorch's built-in TransformerDecoderLayer for reliability.
    Sinusoidal positional encoding is applied to the token embeddings.
    """

    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.d_model = config.d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0)

        # Sinusoidal positional encoding
        self.pos_encoding = PositionalEncoding1D(config.d_model, config.dropout, config.max_seq_len)

        # Standard Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )

        self.layers = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.d_model)
        )

        # Output projection
        self.output_proj = nn.Linear(config.d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generate a causal (upper triangular) mask for autoregressive decoding."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt_ids: (B, L) target token indices
            memory: (B, S, D) encoder output features
            tgt_mask: (L, L) causal mask (optional, generated if None)

        Returns:
            logits: (B, L, vocab_size) token predictions
        """
        B, L = tgt_ids.shape

        # Embed tokens + positional encoding
        tgt_emb = self.embedding(tgt_ids)  # (B, L, D)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(L, tgt_ids.device)

        # Create padding mask (True where padded)
        tgt_key_padding_mask = (tgt_ids == 0)  # pad_id = 0

        # Run through decoder
        output = self.layers(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(output)  # (B, L, vocab_size)

        return logits
