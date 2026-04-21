"""
Loss functions for TAMER model.

v2.4 Changes:
  - [FIXED] Loss Function Denominator Flaw: StructureAwareLoss now normalizes
    per-sequence before taking the batch mean. This ensures that batches with
    long sequences don't produce massive gradient spikes compared to short batches.

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
            logits: (B, L, vocab_size) model predictions (Pre-aligned).
            targets: (B, L) target token indices (Pre-aligned).
        """
        # Flatten for CrossEntropyLoss
        pred_logits = logits.view(-1, logits.size(-1))  # (B * L, V)
        target_tokens = targets.view(-1)                # (B * L)

        return self.criterion(pred_logits, target_tokens)


class StructureAwareLoss(nn.Module):
    """
    Structure-aware cross-entropy loss for multi-line and matrix support.

    Weights structural tokens MORE heavily in the loss. Getting \\\\, &,
    \\begin{env}, and \\end{env} wrong doesn't just affect one token —
    it destroys the entire matrix/aligned structure. A wrong row separator
    turns a 3x3 matrix into garbage.

    Default weights:
      - Normal tokens:      1.0
      - Structural tokens:  3.0 (configurable)

    v2.4 Fix — Loss Function Denominator Flaw:
      Previously, the loss was computed as:
        sum(weighted_loss_all_tokens) / total_non_pad_tokens_in_batch
      This caused gradient scale to swing wildly: a batch of long equations
      produces a tiny loss (large denominator) while a batch of short equations
      produces a huge loss (small denominator), making training unstable.

      Now we compute:
        mean over batch of [ sum(weighted_loss_per_seq) / non_pad_tokens_in_that_seq ]
      This keeps the gradient scale stable regardless of equation length.

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
            logits:  (B, L, V) model predictions.
            targets: (B, L)    target token indices.

        Returns:
            Scalar loss: mean over batch of per-sequence normalized weighted CE.
        """
        B, L, V = logits.shape

        # Flatten to (B*L, V) and (B*L,) for F.cross_entropy
        pred_flat   = logits.view(B * L, V)
        target_flat = targets.view(B * L)

        # [FIX: Loss Function Denominator Flaw]
        # reduction='none' gives us per-token loss as (B*L,).
        # We immediately reshape back to (B, L) so we can do per-sequence math.
        ce_loss = F.cross_entropy(
            pred_flat,
            target_flat,
            ignore_index=self.pad_id,
            label_smoothing=self.label_smoothing,
            reduction='none'
        ).view(B, L)  # (B, L) — crucial reshape back before any aggregation

        # Build weight and mask tensors in (B, L) space — NOT flattened space
        weights = self.token_weights.to(targets.device)[targets]  # (B, L)
        mask    = (targets != self.pad_id).float()                 # (B, L)

        # Apply structural weights and padding mask
        weighted_loss = ce_loss * weights * mask  # (B, L)

        # [FIX] Sum over sequence dimension, normalize by THIS sequence's token count.
        # Each sequence gets its own denominator — long and short equations
        # contribute equally stable gradients to the batch mean.
        loss_per_sequence = (
            weighted_loss.sum(dim=1)           # (B,) — sum over tokens in each seq
            / mask.sum(dim=1).clamp(min=1)     # (B,) — divide by non-pad count per seq
        )

        # [FIX] Average across the batch dimension only after per-seq normalization.
        return loss_per_sequence.mean()         # scalar