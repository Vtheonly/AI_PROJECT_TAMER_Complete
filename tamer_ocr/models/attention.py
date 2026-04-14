"""
Positional Encoding modules for the TAMER model.

- PositionalEncoding1D: Sinusoidal positional encoding for decoder tokens.
- PositionalEncoding2D: Learned 2D positional encoding for encoder features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for 1D sequences (decoder tokens)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence
        Returns:
            (B, L, D) with positional encoding added
        """
        return self.dropout(x + self.pe[:, :x.size(1)])


class PositionalEncoding2D(nn.Module):
    """
    Learned 2D positional encoding for spatial features.

    Since our images are wide (256x1024), the Swin feature grid
    needs to know spatial positions. This adds learned row/column
    embeddings to the encoder features.
    """

    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 128):
        super().__init__()
        # Split d_model in half for row and column embeddings
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        
        self.row_embed = nn.Parameter(torch.randn(max_h, d_model // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(max_w, d_model // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D) spatial feature map from Swin backbone
        Returns:
            (B, H, W, D) with 2D positional encoding added
        """
        h, w = x.shape[1], x.shape[2]

        # 1. Slice embeddings to match current feature map size
        # This handles cases where the input image might be smaller than max
        r_h = min(h, self.row_embed.size(0))
        c_w = min(w, self.col_embed.size(0))

        row = self.row_embed[:r_h]
        col = self.col_embed[:c_w]

        # 2. If input is larger than max_h/max_w, pad embeddings (safety fallback)
        if h > r_h:
            row = F.pad(row, (0, 0, 0, h - r_h))
        if w > c_w:
            col = F.pad(col, (0, 0, 0, w - c_w))

        # 3. Expand and concatenate to create a (H, W, D) grid
        # row: (H, 1, D/2) -> (H, W, D/2)
        # col: (1, W, D/2) -> (H, W, D/2)
        pos = torch.cat([
            row.unsqueeze(1).expand(-1, w, -1),
            col.unsqueeze(0).expand(h, -1, -1)
        ], dim=-1) # Result: (H, W, D)

        # 4. Add to input features
        # x: (B, H, W, D), pos: (H, W, D)
        return x + pos.unsqueeze(0)