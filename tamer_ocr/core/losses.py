import torch
import torch.nn as nn

class TreeGuidedLoss(nn.Module):
    def __init__(self, pad_idx: int, config):
        super().__init__()
        self.pad_idx = pad_idx
        self.config = config
        
        self.token_criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx, 
            label_smoothing=config.label_smoothing
        )
        # Pointer padding target is safely routed to -100
        self.pointer_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, pointer_scores, tgt_ids, tgt_parents, coverage=None):
        """
        Input predictions are unshifted (includes <sos> prediction context).
        Targets are shifted right by 1 for teacher forcing.
        """
        # Shift targets by 1 (We predict token t+1 from state t)
        target_tokens = tgt_ids[:, 1:].contiguous().view(-1)
        pred_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        
        token_loss = self.token_criterion(pred_logits, target_tokens)
        
        # Pointer Loss
        target_pointers = tgt_parents[:, 1:].contiguous().view(-1)
        pred_pointers = pointer_scores[:, :-1, :].contiguous().view(-1, pointer_scores.size(-1))
        
        # Mask out pointers where the target token is padding
        pad_mask = (target_tokens == self.pad_idx)
        target_pointers = target_pointers.masked_fill(pad_mask, -100)
        
        pointer_loss = self.pointer_criterion(pred_pointers, target_pointers)
        
        # Coverage penalty (L1 on un-attended areas)
        cov_loss = torch.tensor(0.0, device=logits.device)
        if coverage is not None:
            cov_loss = torch.mean(torch.clamp(1.0 - coverage, min=0.0))
            
        total_loss = (self.config.seq_loss_weight * token_loss) + \
                     (self.config.pointer_loss_weight * pointer_loss) + \
                     (self.config.coverage_loss_weight * cov_loss)
                     
        return total_loss, token_loss, pointer_loss, cov_loss