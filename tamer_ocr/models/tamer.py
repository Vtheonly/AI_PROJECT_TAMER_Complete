import torch
import torch.nn as nn
from .encoder import SwinEncoder
from .decoder import TreeGuidedDecoder

class TAMERCore(nn.Module):
    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.config = config
        self.encoder = SwinEncoder(config)
        self.decoder = TreeGuidedDecoder(vocab_size, config)
        self._init_weights()

    def _init_weights(self):
        for p in self.decoder.token_out.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.decoder.embedding.weight, std=0.02)

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images, tgt_ids, tgt_parents, coverage=None, text_only=False):
        if text_only:
            # PHASE 0: Bypass Encoder entirely. Send dummy visual memory.
            B = tgt_ids.size(0)
            # 256 is the sequence length output of the tiny swin encoder
            memory = torch.zeros(B, 256, self.config.d_model, device=tgt_ids.device)
        else:
            memory = self.encode(images)
            
        causal_mask = self.generate_causal_mask(tgt_ids.size(1), tgt_ids.device)
        logits, pointer_scores, new_coverage = self.decoder(tgt_ids, tgt_parents, memory, causal_mask, coverage)
        return logits, pointer_scores, new_coverage