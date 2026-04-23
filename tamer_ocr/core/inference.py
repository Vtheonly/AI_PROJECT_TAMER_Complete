
"""
Greedy and Beam Search inference for the TAMER model.

v2.1 Changes (Fix KV-Cache bugs):
  - [FIXED] beam_search() now correctly defaults to the eager O(N^3) SDPA path
    to avoid complex caching mechanics required for beam management. SDPA is
    extremely fast for short generation lengths anyway.
  - [RETAINED] greedy_decode() KV-Cache for ultra-fast batch decoding.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
import logging

logger = logging.getLogger("TAMER.Inference")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_model(model):
    """Strip DataParallel / DDP / torch.compile wrappers."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _has_kv_cache(model) -> bool:
    """Return True if the decoder supports KV-cache inference."""
    return getattr(model.decoder, "use_kv_cache_flag", False)


def _postprocess(tokens: List[int], sos_id: int, eos_id: int) -> List[int]:
    """Strip leading SOS and everything from EOS onward."""
    if tokens and tokens[0] == sos_id:
        tokens = tokens[1:]
    if eos_id in tokens:
        tokens = tokens[: tokens.index(eos_id)]
    return tokens


# ---------------------------------------------------------------------------
# Greedy Decode
# ---------------------------------------------------------------------------

@torch.no_grad()
@torch.compiler.disable       # Prevents graph-recompile at every sequence length
def greedy_decode(
    model,
    images: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int = 200,
    device: torch.device = None,
    real_ws: Optional[torch.Tensor] = None,
    real_hs: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """
    Greedy (argmax) autoregressive decoding.

    Args:
        model:   TAMER model (may be wrapped).
        images:  (B, C, H, W) input image batch.
        sos_id:  Start-of-sequence token id.
        eos_id:  End-of-sequence token id.
        max_len: Maximum tokens to generate per sequence.
        device:  Target device; inferred from images if None.
        real_ws: (B,) pixel width of content before padding (for encoder mask).
                 Pass None to attend to all encoder tokens.
        real_hs: (B,) pixel height of content before padding (for encoder mask).

    Returns:
        List of B token-id lists, SOS/EOS stripped.
    """
    model.eval()
    model = _unwrap_model(model)

    if device is None:
        device = images.device

    B = images.size(0)

    # Encode image ONCE — memory is reused for every decoding step.
    # memory_mask is None when real_ws/real_hs are not provided.
    memory, memory_mask = model.encode(images, real_ws=real_ws, real_hs=real_hs)

    tokens = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
    unfinished = torch.ones(B, dtype=torch.bool, device=device)

    use_cache = _has_kv_cache(model)
    past_cache: Dict = {} if use_cache else None

    for step in range(max_len):
        if use_cache:
            # O(N^2): process only the newest token, reuse cached K/V
            logits, past_cache = model.decoder(
                tokens,
                memory,
                tgt_mask=None,
                memory_mask=memory_mask,
                use_cache=True,
                past_cache=past_cache,
                step=step,
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1)

        else:
            # O(N^3) eager fallback for legacy decoders
            L = tokens.size(1)
            tgt_mask = model.decoder.generate_causal_mask(L, device)
            logits = model.decoder(
                tokens, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1)

        # Replace finished sequences with EOS to keep them inert
        next_tokens = torch.where(
            unfinished,
            next_tokens,
            torch.tensor(eos_id, dtype=torch.long, device=device),
        )

        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        unfinished = unfinished & (next_tokens != eos_id)

        if not unfinished.any():
            break

    result: List[List[int]] = []
    for seq in tokens.cpu().tolist():
        result.append(_postprocess(seq, sos_id, eos_id))

    return result


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------

@torch.no_grad()
@torch.compiler.disable       # Prevents graph-recompile at every sequence length
def beam_search(
    model,
    image: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    beam_width: int = 5,
    max_len: int = 200,
    length_penalty: float = 0.6,
    device: torch.device = None,
    real_ws: Optional[torch.Tensor] = None,
    real_hs: Optional[torch.Tensor] = None,
) -> List[int]:
    """
    Beam search decoding for a single image.

    Args:
        model:          TAMER model.
        image:          (1, C, H, W) single image tensor.
        sos_id:         Start-of-sequence token id.
        eos_id:         End-of-sequence token id.
        pad_id:         Padding token id (kept for API compatibility).
        beam_width:     Number of parallel beams.
        max_len:        Maximum generation length.
        length_penalty: Score normalisation exponent (higher → longer sequences).
        device:         Target device; inferred from image if None.
        real_ws:        (1,) pixel width of content before padding.
        real_hs:        (1,) pixel height of content before padding.

    Returns:
        Best token-id list with SOS/EOS stripped.
    """
    model.eval()
    model = _unwrap_model(model)

    if device is None:
        device = image.device

    # Encode the single image ONCE — shape (1, S, D)
    memory, memory_mask = model.encode(image, real_ws=real_ws, real_hs=real_hs)

    # Each beam: (token_list, cumulative_log_prob)
    beams: List[tuple] = [([sos_id], 0.0)]
    completed: List[tuple] = []

    # Force eager path for beam search to avoid complex KV cache beam indexing.
    # SDPA is extremely fast for small sequence lengths anyway.
    use_cache = False 

    for step in range(max_len):
        if not beams:
            break

        # Move finished beams to completed
        active_beams = []
        for tokens, score in beams:
            if tokens[-1] == eos_id:
                n = len(tokens) - 1
                completed.append((tokens, score / (max(n, 1) ** length_penalty)))
            else:
                active_beams.append((tokens, score))

        if not active_beams:
            break

        num_active = len(active_beams)

        # Build batch input from all active beams
        tgt_ids = torch.tensor(
            [b[0] for b in active_beams], dtype=torch.long, device=device
        )  # (num_active, L)

        # Expand memory to num_active beams
        batched_memory = memory.expand(num_active, -1, -1)  # (num_active, S, D)

        # Expand memory_mask too if present
        batched_memory_mask = None
        if memory_mask is not None:
            batched_memory_mask = memory_mask.expand(num_active, -1)  # (num_active, S)

        if use_cache:
            # Replaced with use_cache=False above.
            pass
        else:
            # O(N^3) eager path
            L = tgt_ids.size(1)
            tgt_mask = model.decoder.generate_causal_mask(L, device)
            logits = model.decoder(
                tgt_ids, batched_memory,
                tgt_mask=tgt_mask,
                memory_mask=batched_memory_mask,
            )
            next_logits = logits[:, -1, :]   # (num_active, vocab)

        log_probs = F.log_softmax(next_logits, dim=-1)
        topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)

        # Expand each active beam into beam_width candidates
        all_candidates: List[tuple] = []
        seen: set = set()

        for i, (current_tokens, current_score) in enumerate(active_beams):
            for j in range(beam_width):
                new_token = topk_indices[i, j].item()
                new_score = current_score + topk_log_probs[i, j].item()
                candidate = current_tokens + [new_token]
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    all_candidates.append((candidate, new_score))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    # Drain remaining active beams
    for tokens, score in beams:
        n = len(tokens) - 1
        completed.append((tokens, score / (max(n, 1) ** length_penalty)))

    completed.sort(key=lambda x: x[1], reverse=True)
    return _postprocess(completed[0][0], sos_id, eos_id)    