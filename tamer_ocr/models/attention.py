import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

class CoverageAttention(nn.Module):
    """Spatial Coverage Memory mappings"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.coverage_proj = nn.Linear(1, nhead, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, coverage: torch.Tensor = None):
        B, Tq, _ = query.shape
        _, Tk, _ = memory.shape

        Q = self.q_proj(query).view(B, Tq, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(memory).view(B, Tk, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(memory).view(B, Tk, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if coverage is not None:
            cov_bias = self.coverage_proj(coverage.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(2)
            scores = scores + cov_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights_drop, V).transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        
        avg_attn = attn_weights.mean(dim=1)
        new_coverage = (coverage if coverage is not None else torch.zeros(B, Tk, device=query.device))
        new_coverage = new_coverage + avg_attn.sum(dim=1)
        
        return self.out_proj(output), new_coverage