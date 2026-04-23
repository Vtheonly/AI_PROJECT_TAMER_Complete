
"""
KV-Cached Transformer Decoder for Math OCR.

v2.1 Changes (Fix KV-Cache for Cross Attention):
  - [FIXED] KVCachedAttention now correctly identifies cross-attention (where K/V
    come from the static encoder memory) and calculates/caches K and V exactly
    once, rather than incorrectly concatenating them every step.
  - [RETAINED] Encoder Padding Mask propagation via memory_mask.
  - [RETAINED] Pre-norm architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .attention import PositionalEncoding1D


class KVCachedAttention(nn.Module):
    """
    Multi-head attention with optional KV caching.
    Uses F.scaled_dot_product_attention for fused CUDA kernels / Flash Attention.

    Mask convention (unified):
      - Additive float mask: 0.0 = attend, float('-inf') = block.
        Used for causal self-attention masks (shape: L_q x L_k).
      - Boolean mask (cross-attention from memory_mask):
        True = attend, False = block.
        Internally converted to additive float before SDPA.
      Both are broadcast-expanded to (B, nhead, L_q, L_k) before SDPA.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict] = None,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            query:     (B, L_q, D)
            key:       (B, L_k, D)
            value:     (B, L_k, D)
            mask:      Additive float mask (L_q, L_k) for self-attention, OR
                       Boolean mask (B, 1, 1, L_k) for cross-attention.
                       None = no masking.
            cache:     KV cache dict (mutated in-place during inference)
            cache_key: Unique key for this layer/type in the cache dict

        Returns:
            output: (B, L_q, D)
        """
        B, L_q, _ = query.shape

        # Linear projections for query
        q = self.q_proj(query)   # (B, L_q, D)

        # Split into heads: (B, nhead, L, head_dim)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, L_, _ = t.shape
            return t.view(B_, L_, self.nhead, self.head_dim).transpose(1, 2)

        q = _split_heads(q)

        # Check if we are doing cross-attention (where K/V are static encoder features)
        is_cross_attn = cache_key is not None and cache_key.startswith("cross")

        if cache is not None and cache_key is not None:
            if is_cross_attn:
                # Cross-attention: K and V represent the entire memory.
                # Compute them exactly once at step 0 and reuse for all future steps.
                if cache_key in cache:
                    k, v = cache[cache_key]
                else:
                    k = _split_heads(self.k_proj(key))
                    v = _split_heads(self.v_proj(value))
                    cache[cache_key] = (k, v)
            else:
                # Self-attention: K and V represent the newest token.
                # Compute them and concatenate with the historical context.
                k = _split_heads(self.k_proj(key))
                v = _split_heads(self.v_proj(value))
                if cache_key in cache:
                    past_k, past_v = cache[cache_key]
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
                cache[cache_key] = (k, v)
        else:
            # No caching (e.g. training path)
            k = _split_heads(self.k_proj(key))
            v = _split_heads(self.v_proj(value))

        # Normalise mask to additive float for SDPA
        # SDPA expects: 0.0 = attend, -inf = block, shape broadcastable to
        # (B, nhead, L_q, L_k).
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                # Boolean mask: True = attend (keep), False = block (ignore).
                # Convert to additive: blocked positions get -inf.
                attn_mask = torch.zeros_like(mask, dtype=query.dtype)
                attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
                # Ensure shape is (B, 1, 1, L_k) for broadcast
                if attn_mask.dim() == 2 and attn_mask.shape[0] == B:
                    # (B, L_k) -> (B, 1, 1, L_k)
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            else:
                # Float additive mask: (L_q, L_k) -> (1, 1, L_q, L_k)
                attn_mask = mask
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # causality enforced by explicit mask
        )  # (B, nhead, L_q, head_dim)

        # Merge heads: (B, L_q, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.out_proj(attn_out)


class DecoderBlock(nn.Module):
    """
    Pre-Norm Transformer Decoder Block with KV-Cached self- and cross-attention.

    Cross-attention now accepts memory_mask to focus exclusively on spatial
    feature tokens that correspond to real ink strokes in the image.
    """

    def __init__(self, config):
        super().__init__()
        self.self_attn = KVCachedAttention(config.d_model, config.nhead, config.dropout)
        self.cross_attn = KVCachedAttention(config.d_model, config.nhead, config.dropout)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, L, D) target sequence embeddings
            memory:       (B, S, D) encoder output
            tgt_mask:     (L, L) additive causal mask for self-attention
            memory_mask:  (B, 1, 1, S) bool mask for cross-attention
                          True = attend, False = ignore (padding)
                          Built by TAMERModel.generate_memory_mask().
            cache:        KV cache dict (mutated in-place)
            layer_idx:    layer index for unique cache keys

        Returns:
            x: (B, L, D)
        """
        # --- Pre-Norm Self Attention ---
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(
            x, x, x,
            mask=tgt_mask,
            cache=cache,
            cache_key=f"self_{layer_idx}",
        )

        # --- Pre-Norm Cross Attention (with encoder padding mask) ---
        # memory_mask forces the softmax to assign exactly 0 probability to
        # white-padding feature tokens, so attention capacity is concentrated
        # on the real ink-stroke tokens only.
        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(
            x, memory, memory,
            mask=memory_mask,   # None during unconstrained inference; bool mask during training
            cache=cache,
            cache_key=f"cross_{layer_idx}",
        )

        # --- Pre-Norm Feed Forward ---
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)

        return x


class TransformerDecoder(nn.Module):
    """
    Full KV-Cached Transformer Decoder with encoder padding mask support.

    Training  (use_cache=False): full parallel forward, memory_mask applied.
    Inference (use_cache=True):  single-token forward with growing KV cache,
                                 memory_mask applied on step 0 (cached after that).

    Sentinel: use_kv_cache_flag = True  →  inference.py selects O(N^2) path.
    """

    use_kv_cache_flag: bool = True

    def __init__(self, vocab_size: int, config):
        super().__init__()
        self.d_model = config.d_model

        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding1D(
            config.d_model,
            config.dropout,
            config.max_seq_len,
        )

        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_decoder_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, vocab_size)

        self._init_output_proj()

    def _init_output_proj(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Additive upper-triangular causal mask: 0=attend, -inf=block."""
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device),
            diagonal=1,
        )

    @staticmethod
    def _bool_memory_mask_to_attn_mask(
        memory_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert dataset-style bool mask to SDPA-compatible additive float mask.

        Dataset convention:  True  = IGNORE (padding)
                             False = ATTEND (content)

        SDPA additive convention:  0.0     = attend
                                   -inf    = block

        Input  shape: (B, L_mem)          — from generate_memory_mask()
        Output shape: (B, 1, 1, L_mem)    — broadcasts over heads and query len
        """
        # Invert: True (ignore) -> -inf, False (attend) -> 0.0
        float_mask = torch.zeros_like(memory_mask, dtype=dtype)
        float_mask = float_mask.masked_fill(memory_mask, float("-inf"))
        # (B, L_mem) -> (B, 1, 1, L_mem)
        return float_mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_cache: Optional[Dict] = None,
        step: int = 0,
    ):
        """
        Args:
            tgt_ids:      (B, L) target token indices
            memory:       (B, S, D) encoder output
            tgt_mask:     (L, L) additive causal mask (auto-generated if None)
            memory_mask:  (B, S) bool mask from TAMERModel.generate_memory_mask()
                          True=padding/ignore, False=content/attend.
                          Pass None to attend to all encoder tokens (e.g. inference
                          without dimension info).
            use_cache:    True for single-token KV-cache inference
            past_cache:   cache dict from previous step (mutated in-place)
            step:         current decoding step index (for positional encoding)

        Returns:
            Training  (use_cache=False): logits (B, L, vocab_size)
            Inference (use_cache=True):  Tuple[logits (B, 1, vocab_size), cache dict]
        """
        # Convert bool memory mask to additive float once, reuse across all layers
        cross_attn_mask: Optional[torch.Tensor] = None
        if memory_mask is not None:
            cross_attn_mask = self._bool_memory_mask_to_attn_mask(
                memory_mask, dtype=memory.dtype
            )
            # cross_attn_mask: (B, 1, 1, L_mem)

        if use_cache:
            # ------------------------------------------------------------------
            # INFERENCE PATH — O(N^2)
            # Embed only the newest (last) token. Positional encoding is applied
            # at absolute position `step` to stay consistent with training.
            # ------------------------------------------------------------------
            cache = past_cache if past_cache is not None else {}

            # (B, 1, D)
            x = self.embedding(tgt_ids[:, -1:])

            # Positional encoding at the correct absolute position
            pe = self.pos_encoding.pe[:, step : step + 1, :]  # (1, 1, D)
            x = self.pos_encoding.dropout(x + pe)

            # No causal self-attention mask needed: query length is 1,
            # there are no future positions to block.
            for i, layer in enumerate(self.layers):
                x = layer(
                    x, memory,
                    tgt_mask=None,
                    memory_mask=cross_attn_mask,
                    cache=cache,
                    layer_idx=i,
                )

            x = self.norm(x)
            logits = self.output_proj(x)   # (B, 1, vocab_size)
            return logits, cache

        else:
            # ------------------------------------------------------------------
            # TRAINING PATH — full sequence, parallel, no cache
            # ------------------------------------------------------------------
            B, L = tgt_ids.shape

            x = self.embedding(tgt_ids)    # (B, L, D)
            x = self.pos_encoding(x)       # (B, L, D)

            if tgt_mask is None:
                tgt_mask = self.generate_causal_mask(L, tgt_ids.device)

            for i, layer in enumerate(self.layers):
                x = layer(
                    x, memory,
                    tgt_mask=tgt_mask,
                    memory_mask=cross_attn_mask,
                    cache=None,
                    layer_idx=i,
                )

            x = self.norm(x)
            logits = self.output_proj(x)   # (B, L, vocab_size)
            return logits