import torch
import torch.nn as nn
import math
from .attention import PositionalEncoding1D, CoverageAttention

class TreeGuidedDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = CoverageAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, coverage=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        
        tgt2, new_coverage = self.cross_attn(tgt, memory, coverage)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        
        tgt = self.norm3(tgt + self.dropout3(self.ffn(tgt)))
        return tgt, new_coverage

class TreeGuidedDecoder(nn.Module):
    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding1D(config.d_model, config.dropout, config.max_seq_len)
        
        # Projection to merge sequential and topological embeddings
        self.parent_proj = nn.Linear(config.d_model, config.d_model)
        
        self.layers = nn.ModuleList([
            TreeGuidedDecoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout) 
            for _ in range(config.num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        
        # Next token output
        self.token_out = nn.Linear(config.d_model, vocab_size)
        
        # Pointer Network for Parent prediction
        self.pointer_q = nn.Linear(config.d_model, config.d_model)
        self.pointer_k = nn.Linear(config.d_model, config.d_model)

    def forward(self, tgt_ids, tgt_parents, memory, tgt_mask, coverage=None):
        B, L = tgt_ids.shape
        
        # Embed sequence tokens
        tgt_emb = self.embedding(tgt_ids)
        
        # Embed topological parent tokens (Gathered safely using tgt_parents)
        parent_emb = torch.gather(tgt_emb, 1, tgt_parents.unsqueeze(-1).expand(-1, -1, self.d_model))
        
        # The core structural shift: x_t combines token_t and parent_t
        x = tgt_emb + self.parent_proj(parent_emb)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x, coverage = layer(x, memory, tgt_mask, coverage)
            
        x = self.norm(x)
        
        # 1. Token Prediction Logits
        logits = self.token_out(x)
        
        # 2. Pointer Network Scores (Predicting parent index for the *next* step)
        Q = self.pointer_q(x)
        K = self.pointer_k(x)
        pointer_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
        
        # Prevent pointing to future tokens
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        pointer_scores.masked_fill_(causal_mask, float('-inf'))
        
        return logits, pointer_scores, coverage