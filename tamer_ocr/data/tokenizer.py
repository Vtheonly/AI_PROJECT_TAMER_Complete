"""
LaTeX Tokenizer for Math OCR Training.

Builds a global vocabulary from all datasets. Special tokens are fixed at indices 0-3.
Crucially splits multi-digit numbers into individual digit tokens to ensure 
robustness to unseen numbers and minimize vocabulary size.
"""

import logging
from collections import Counter
from typing import List, Dict, Optional
import json
import os

logger = logging.getLogger("TAMER.Tokenizer")


class LaTeXTokenizer:
    """Tokenizer that respects LaTeX command structure and digit-level granularity."""

    # Special tokens with FIXED indices
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self):
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        self.reverse_vocab: Dict[int, str] = {i: t for t, i in self.vocab.items()}

    def tokenize(self, latex: str) -> List[str]:
        """
        Tokenize a LaTeX string into a list of tokens.
        
        Fixes implemented:
        - Splits multi-digit numbers (e.g., "123" -> ["1", "2", "3"])
        - Treats decimal points as separate tokens.
        - Preserves LaTeX commands (e.g., "\\frac").
        """
        tokens = []
        i = 0
        latex = latex.strip()
        while i < len(latex):
            if latex[i].isspace():
                i += 1
                continue
            
            # Handle LaTeX commands and escaped characters
            if latex[i] == '\\':
                j = i + 1
                while j < len(latex) and latex[j].isalpha():
                    j += 1
                if j == i + 1:
                    # Escaped special character like \{ \} \% or \#
                    if j < len(latex):
                        tokens.append(latex[i:j+1])
                        j += 1
                    else:
                        tokens.append(latex[i])
                        j += 1
                else:
                    # Regular command like \frac or \sqrt
                    tokens.append(latex[i:j])
                i = j
            
            # Handle structural symbols
            elif latex[i] in '{}()[]+-=_^&|<>':
                tokens.append(latex[i])
                i += 1
            
            # FIX: Treat every digit and decimal point as a single, separate token
            # This prevents <unk> tokens for new numbers seen during inference.
            elif latex[i].isdigit() or latex[i] == '.':
                tokens.append(latex[i])
                i += 1
            
            # Handle everything else (letters, other symbols)
            else:
                tokens.append(latex[i])
                i += 1
                
        return tokens

    def build_from_corpus(self, corpus: List[str]):
        """Build vocabulary from a list of LaTeX strings."""
        logger.info("Building vocabulary from corpus...")
        token_counts = Counter()
        for text in corpus:
            token_counts.update(self.tokenize(text))

        for token, _ in token_counts.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.reverse_vocab = {i: t for t, i in self.vocab.items()}
        logger.info(f"Vocabulary built: {len(self.vocab)} total tokens.")

    def build_from_samples(self, samples: list):
        """Build vocabulary from a list of sample dicts with 'latex' key."""
        corpus = [s.get('latex', '') for s in samples if s.get('latex')]
        self.build_from_corpus(corpus)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to a list of integer indices."""
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Convert a list of integer indices back to a LaTeX string."""
        res = []
        for idx in indices:
            # Ensure index is treated as int
            idx_int = int(idx)
            t = self.reverse_vocab.get(idx_int, self.UNK_TOKEN)
            
            if skip_special and t in self.SPECIAL_TOKENS:
                continue
            if t == self.EOS_TOKEN:
                break
            res.append(t)
        return ' '.join(res)

    def save(self, path: str):
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab}, f, ensure_ascii=False, indent=2)
        logger.info(f"Tokenizer saved to {path} ({len(self.vocab)} tokens)")

    def load(self, path: str):
        """Load vocabulary from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        # Convert string keys from JSON back to integers for reverse lookup
        self.reverse_vocab = {int(i) if str(i).isdigit() else i: t for t, i in self.vocab.items()}
        # Re-map correctly where indices are the keys
        self.reverse_vocab = {idx: tok for tok, idx in self.vocab.items()}
        
        # Integrity check for special tokens
        for token, expected_idx in zip(self.SPECIAL_TOKENS, range(4)):
            if self.vocab.get(token) != expected_idx:
                logger.warning(f"Special token {token} not at expected index {expected_idx}")
        
        logger.info(f"Tokenizer loaded from {path} ({len(self.vocab)} tokens)")

    @property
    def pad_id(self) -> int: return self.vocab[self.PAD_TOKEN]
    
    @property
    def sos_id(self) -> int: return self.vocab[self.SOS_TOKEN]
    
    @property
    def eos_id(self) -> int: return self.vocab[self.EOS_TOKEN]
    
    @property
    def unk_id(self) -> int: return self.vocab[self.UNK_TOKEN]
    
    def __len__(self) -> int: return len(self.vocab)