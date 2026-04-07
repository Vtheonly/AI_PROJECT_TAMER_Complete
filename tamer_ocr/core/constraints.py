import torch
from data.tokenizer import LaTeXTokenizer

class LaTeXGrammarConstraints:
    def __init__(self, tokenizer: LaTeXTokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        
        self.lbrace_id = tokenizer.vocab.get('{', -1)
        self.rbrace_id = tokenizer.vocab.get('}', -1)
        self.eos_id = tokenizer.eos_id
        
        self.requires_brace = {tokenizer.vocab.get(t) for t in ['^', '_', '\\frac', '\\sqrt'] if t in tokenizer.vocab}
        
    def get_valid_mask(self, generated_ids: list) -> torch.Tensor:
        mask = torch.ones(self.vocab_size, dtype=torch.bool)
        
        if not generated_ids:
            return mask
            
        last_id = generated_ids[-1]
        brace_depth = sum(1 if t == self.lbrace_id else -1 if t == self.rbrace_id else 0 for t in generated_ids)
        
        # Constraint 1: Structural requirement (e.g. \frac MUST be followed by {)
        if last_id in self.requires_brace and self.lbrace_id >= 0:
            mask[:] = False
            mask[self.lbrace_id] = True
            return mask
            
        # Constraint 2: Balanced braces
        if brace_depth <= 0 and self.rbrace_id >= 0:
            mask[self.rbrace_id] = False
            
        if brace_depth != 0:
            mask[self.eos_id] = False
            
        return mask