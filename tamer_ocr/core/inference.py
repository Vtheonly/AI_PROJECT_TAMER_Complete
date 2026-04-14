"""
Standard Beam Search Inference for TAMER model.

Key changes from old version:
- NO grammar constraints
- NO pointer scores
- NO coverage tracking
- Added length penalty to prevent short predictions
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
import logging

logger = logging.getLogger("TAMER.Inference")


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
    """
    Standard beam search decoding.
    
    Args:
        model: TAMERModel instance
        image: (1, 1, H, W) input image tensor
        sos_id: Start-of-sequence token ID
        eos_id: End-of-sequence token ID
        pad_id: Padding token ID
        beam_width: Number of beams
        max_len: Maximum output sequence length
        length_penalty: Penalty for short sequences (>0 encourages longer output)
        device: Device for tensors
    
    Returns:
        List of token IDs (excluding SOS and EOS)
    """
    model.eval()
    if device is None:
        device = image.device
    
    # Encode image once
    memory = model.encode(image)  # (1, S, D)
    
    # Initialize beams: list of (tokens, log_prob)
    beams = [([sos_id], 0.0)]
    completed = []
    
    for step in range(max_len):
        if not beams:
            break
        
        all_candidates = []
        
        for tokens, score in beams:
            # If this beam has ended, move to completed
            if tokens[-1] == eos_id:
                # Apply length penalty
                length = len(tokens) - 1  # Exclude SOS
                penalized_score = score / ((length ** length_penalty) if length > 0 else 1.0)
                completed.append((tokens, penalized_score))
                continue
            
            # Prepare input
            tgt_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            # Generate causal mask
            L = len(tokens)
            tgt_mask = model.decoder.generate_causal_mask(L, device)
            
            # Forward pass
            logits = model.decoder(tgt_ids, memory, tgt_mask)  # (1, L, V)
            next_logits = logits[0, -1, :]  # (V,)
            
            # Log probabilities
            log_probs = F.log_softmax(next_logits, dim=-1)
            
            # Get top-k candidates
            topk_log_probs, topk_indices = log_probs.topk(beam_width)
            
            for i in range(beam_width):
                new_token = topk_indices[i].item()
                new_score = score + topk_log_probs[i].item()
                all_candidates.append((tokens + [new_token], new_score))
        
        # Select top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
        
        # Early termination if all beams ended
        if all(b[0][-1] == eos_id for b in beams):
            for tokens, score in beams:
                length = len(tokens) - 1
                penalized_score = score / ((length ** length_penalty) if length > 0 else 1.0)
                completed.append((tokens, penalized_score))
            break
    
    # If no completed beams, take the best incomplete one
    if not completed:
        for tokens, score in beams:
            length = len(tokens) - 1
            penalized_score = score / ((length ** length_penalty) if length > 0 else 1.0)
            completed.append((tokens, penalized_score))
    
    # Select best
    completed.sort(key=lambda x: x[1], reverse=True)
    best_tokens = completed[0][0]
    
    # Remove SOS and EOS
    if best_tokens[0] == sos_id:
        best_tokens = best_tokens[1:]
    if best_tokens and best_tokens[-1] == eos_id:
        best_tokens = best_tokens[:-1]
    
    return best_tokens


@torch.no_grad()
def greedy_decode(
    model,
    image: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int = 200,
    device: torch.device = None,
) -> List[int]:
    """Simple greedy decoding for fast inference."""
    model.eval()
    if device is None:
        device = image.device
    
    memory = model.encode(image)
    tokens = [sos_id]
    
    for _ in range(max_len):
        tgt_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        L = len(tokens)
        tgt_mask = model.decoder.generate_causal_mask(L, device)
        
        logits = model.decoder(tgt_ids, memory, tgt_mask)
        next_token = logits[0, -1, :].argmax().item()
        tokens.append(next_token)
        
        if next_token == eos_id:
            break
    
    # Remove SOS and EOS
    if tokens[0] == sos_id:
        tokens = tokens[1:]
    if tokens and tokens[-1] == eos_id:
        tokens = tokens[:-1]
    
    return tokens
