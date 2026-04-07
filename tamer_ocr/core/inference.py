import torch
import torch.nn.functional as F
from typing import List
from data.tokenizer import LaTeXTokenizer
from core.constraints import LaTeXGrammarConstraints

@torch.no_grad()
def constrained_beam_search(
    model, image: torch.Tensor, tokenizer: LaTeXTokenizer, 
    grammar: LaTeXGrammarConstraints, config
) -> List[int]:
    model.eval()
    device = image.device
    memory = model.encode(image)
    
    # Beam state: (tokens, parents, score, coverage)
    beams = [([tokenizer.sos_id], [0], 0.0, None)]
    completed = []
    
    for step in range(config.max_seq_len):
        if not beams:
            break
            
        all_candidates = []
        for tokens, parents, score, cov in beams:
            if tokens[-1] == tokenizer.eos_id:
                completed.append((tokens, parents, score))
                continue
                
            tgt_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_parents = torch.tensor([parents], dtype=torch.long, device=device)
            causal_mask = model.generate_causal_mask(len(tokens), device)
            
            logits, pointer_scores, new_cov = model.decoder(tgt_ids, tgt_parents, memory, causal_mask, cov)
            
            next_logits = logits[0, -1, :]
            next_pointers = pointer_scores[0, -1, :step+1]
            
            if config.use_grammar_constraints:
                mask = grammar.get_valid_mask(tokens)
                next_logits[~mask.to(device)] = float('-inf')
                
            token_log_probs = F.log_softmax(next_logits, dim=-1)
            pointer_log_probs = F.log_softmax(next_pointers, dim=-1)
            
            # Keep top-K tokens
            topk_tokens = token_log_probs.topk(config.beam_width)
            topk_ptrs = pointer_log_probs.topk(min(config.beam_width, len(pointer_log_probs)))
            
            for tk_idx in range(len(topk_tokens.values)):
                token_idx = topk_tokens.indices[tk_idx].item()
                token_score = topk_tokens.values[tk_idx].item()
                
                # Assume best pointer for structural consistency
                ptr_idx = topk_ptrs.indices[0].item()
                ptr_score = topk_ptrs.values[0].item()
                
                combined_score = score + token_score + (config.pointer_loss_weight * ptr_score)
                all_candidates.append(
                    (tokens + [token_idx], parents + [ptr_idx], combined_score, new_cov)
                )
                
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        beams = all_candidates[:config.beam_width]
        
        if all(b[0][-1] == tokenizer.eos_id for b in beams):
            completed.extend([b[:3] for b in beams])
            break
            
    if not completed:
        completed.extend([b[:3] for b in beams])
        
    completed.sort(key=lambda x: x[2], reverse=True)
    best_tokens = completed[0][0]
    
    return best_tokens[1:-1] if best_tokens[-1] == tokenizer.eos_id else best_tokens[1:]