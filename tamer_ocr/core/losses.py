"""
Loss functions for TAMER model.

v2.3 Changes:
  - Added StructureAwareLoss: weights structural tokens (\\\\, &,
    \\begin{env}, \\end{env}) higher in the loss computation.
    Getting row/column separators wrong destroys entire matrix structure,
    so penalizing these errors more heavily significantly improves
    multi-line and matrix accuracy.

  - Original LabelSmoothedCELoss preserved unchanged for backward compat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional

logger = logging.getLogger("TAMER.Losses")


class LabelSmoothedCELoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for autoregressive sequence modeling.

    Args:
        pad_id: Token ID to ignore in loss computation (typically 0).
        label_smoothing: Smoothing factor (0.1 = 10% uniform distribution).
    """

    def __init__(self, pad_id: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing
        )
        self.pad_id = pad_id

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, L, vocab_size) model predictions.
            targets: (B, L) target token indices.

        Returns:
            scalar loss.
        """
        # Autoregressive Shift:
        # The model at index 't' predicts the token at index 't+1'.
        # logits[:, :-1, :] are the predictions for tokens at targets[:, 1:]

        # 1. Slice to align predictions with targets
        # (B, L-1, V)
        pred_logits = logits[:, :-1, :].contiguous()
        # (B, L-1)
        target_tokens = targets[:, 1:].contiguous()

        # 2. Flatten for CrossEntropyLoss
        # (B * (L-1), V)
        pred_logits = pred_logits.view(-1, logits.size(-1))
        # (B * (L-1))
        target_tokens = target_tokens.view(-1)

        # 3. Compute loss (ignore_index handles the padding)
        return self.criterion(pred_logits, target_tokens)


class StructureAwareLoss(nn.Module):
    """
    Structure-aware cross-entropy loss for multi-line and matrix support.

    Weights structural tokens MORE heavily in the loss. Getting \\\\, &,
    \\begin{env}, and \\end{env} wrong doesn't just affect one token —
    it destroys the entire matrix/aligned structure. A wrong row separator
    turns a 3×3 matrix into garbage.

    Default weights:
      - Normal tokens:      1.0
      - Structural tokens:  3.0 (configurable)

    Args:
        tokenizer: LaTeXTokenizer instance (to look up structural token IDs)
        pad_id: Token ID to ignore in loss computation
        label_smoothing: Smoothing factor
        structural_weight: Weight multiplier for structural tokens
    """

    def __init__(
        self,
        tokenizer,
        pad_id: int = 0,
        label_smoothing: float = 0.1,
        structural_weight: float = 3.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.structural_weight = structural_weight

        # Build per-token weight vector
        vocab_size = len(tokenizer)
        weights = torch.ones(vocab_size)

        # Structural tokens that control multi-line layout
        STRUCTURAL_TOKENS = ['\\\\', '&']

        # Environment tokens — both begin and end
        ENV_NAMES = [
            'aligned', 'align', 'cases', 'gathered', 'split',
            'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix',
            'smallmatrix', 'array', 'eqnarray', 'multline',
        ]

        structural_count = 0
        for token in STRUCTURAL_TOKENS:
            if token in tokenizer.vocab:
                weights[tokenizer.vocab[token]] = structural_weight
                structural_count += 1

        for env in ENV_NAMES:
            for prefix in ['\\begin', '\\end']:
                env_token = f'{prefix}{{{env}}}'
                if env_token in tokenizer.vocab:
                    weights[tokenizer.vocab[env_token]] = structural_weight
                    structural_count += 1

        logger.info(
            f"StructureAwareLoss: {structural_count} tokens weighted at {structural_weight}x"
        )

        self.register_buffer('token_weights', weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, L, vocab_size) model predictions.
            targets: (B, L) target token indices.

        Returns:
            scalar loss.
        """
        # Autoregressive shift
        pred_logits = logits[:, :-1, :].contiguous()   # (B, L-1, V)
        target_tokens = targets[:, 1:].contiguous()     # (B, L-1)

        B, L, V = pred_logits.shape

        # Flatten
        pred_flat = pred_logits.view(B * L, V)
        target_flat = target_tokens.view(B * L)

        # Compute per-token cross-entropy (no reduction)
        ce_loss = F.cross_entropy(
            pred_flat,
            target_flat,
            ignore_index=self.pad_id,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )  # (B*L,)

        # Apply structural weights
        weights = self.token_weights[target_flat]  # (B*L,)

        # Mask out padding
        mask = (target_flat != self.pad_id).float()

        # Weighted average
        weighted_loss = (ce_loss * weights * mask).sum() / mask.sum().clamp(min=1)

        return weighted_loss