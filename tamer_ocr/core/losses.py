"""
Loss function for TAMER model.

Simplified from the original:
- NO pointer loss
- NO coverage loss
- Label-smoothed CrossEntropyLoss with strict sequence alignment.
"""

import torch
import torch.nn as nn


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