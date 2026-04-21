"""
Standard Beam Search and Greedy Inference for TAMER model.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger("TAMER.Inference")


def _unwrap_model(model):
    """Unwrap DataParallel or torch.compile wrappers to access raw model methods."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


@torch.no_grad()
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
) -> List[int]:
    model.eval()
    model = _unwrap_model(model)
    if device is None:
        device = image.device

    memory = model.encode(image)

    beams = [([sos_id], 0.0)]
    completed = []

    for step in range(max_len):
        if not beams:
            break

        active_beams = []
        for tokens, score in beams:
            if tokens[-1] == eos_id:
                length = len(tokens) - 1
                penalized_score = score / ((length ** length_penalty) if length > 0 else 1.0)
                completed.append((tokens, penalized_score))
            else:
                active_beams.append((tokens, score))

        if not active_beams:
            break

        num_active = len(active_beams)
        tgt_ids = torch.tensor([b[0] for b in active_beams], dtype=torch.long, device=device)
        batched_memory = memory.expand(num_active, -1, -1)

        L = tgt_ids.size(1)
        tgt_mask = model.decoder.generate_causal_mask(L, device)

        logits = model.decoder(tgt_ids, batched_memory, tgt_mask)
        next_logits = logits[:, -1, :]
        log_probs = F.log_softmax(next_logits, dim=-1)

        topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=-1)

        all_candidates = []
        seen_sequences = set()

        for i in range(num_active):
            current_tokens, current_score = active_beams[i]
            for j in range(beam_width):
                new_token = topk_indices[i, j].item()
                new_score = current_score + topk_log_probs[i, j].item()
                candidate_tokens = current_tokens + [new_token]
                seq_tuple = tuple(candidate_tokens)
                if seq_tuple not in seen_sequences:
                    seen_sequences.add(seq_tuple)
                    all_candidates.append((candidate_tokens, new_score))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    if not completed:
        for tokens, score in beams:
            length = len(tokens) - 1
            penalized_score = score / ((length ** length_penalty) if length > 0 else 1.0)
            completed.append((tokens, penalized_score))

    completed.sort(key=lambda x: x[1], reverse=True)
    best_tokens = completed[0][0]

    if best_tokens[0] == sos_id:
        best_tokens = best_tokens[1:]
    if best_tokens and best_tokens[-1] == eos_id:
        best_tokens = best_tokens[:-1]

    return best_tokens


@torch.no_grad()
def greedy_decode(
    model,
    images: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int = 200,
    device: torch.device = None,
) -> List[List[int]]:
    model.eval()
    model = _unwrap_model(model)
    if device is None:
        device = images.device

    B = images.size(0)
    memory = model.encode(images)

    tokens = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
    unfinished = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        L = tokens.size(1)
        tgt_mask = model.decoder.generate_causal_mask(L, device)
        logits = model.decoder(tokens, memory, tgt_mask)
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        next_tokens = torch.where(
            unfinished,
            next_tokens,
            torch.tensor(eos_id, device=device),
        )
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        unfinished = unfinished & (next_tokens != eos_id)
        if not unfinished.any():
            break

    result = []
    tokens_list = tokens.cpu().tolist()

    for seq in tokens_list:
        if seq[0] == sos_id:
            seq = seq[1:]
        if eos_id in seq:
            seq = seq[: seq.index(eos_id)]
        result.append(seq)

    return result