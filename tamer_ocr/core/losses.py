"""
Loss function for TAMER model.

Simplified from the original:
- NO pointer loss
- NO coverage loss
- Just label-smoothed CrossEntropyLoss with ignore_index for padding
"""

import torch
import torch.nn as nn


class LabelSmoothedCELoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Args:
        pad_id: Token ID to ignore in loss computation (typically 0)
        label_smoothing: Smoothing factor (0.1 = 10% uniform distribution)
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
            logits: (B, L, vocab_size) model predictions
            targets: (B, L) target token indices
        
        Returns:
            scalar loss
        """
        # Shift: predict token t+1 from state at t
        # logits[:, :-1, :] predicts targets[:, 1:]
        pred_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        target_tokens = targets[:, 1:].contiguous().view(-1)
        
        return self.criterion(pred_logits, target_tokens)
