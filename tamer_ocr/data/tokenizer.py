"""
LaTeX Tokenizer for Math OCR Training.

Builds a global vocabulary from all datasets. Special tokens are fixed at indices 0-3.

v2.4 Changes:
  - FIXED: decode() now uses simple space-joining instead of complex smart-joining rules.
    The previous "intelligent" parsing caused evaluation failures by producing
    inconsistent output. Evaluation metrics (exact_match) handle normalization.

v2.3 Changes:
  - FIXED: \\\\ (double backslash / row separator) is now tokenized as a single
    atomic token '\\\\' instead of being split into two backslash commands.
    This is CRITICAL for multi-line equation and matrix support.

  - FIXED: \\begin{env} and \\end{env} are now atomic tokens instead of being
    split into \\begin, {, e, n, v, }. This preserves environment structure
    and prevents the model from hallucinating invalid environments.

  - Digit-level tokenization preserved for robustness to unseen numbers.
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

    # Structural tokens that MUST be in vocabulary for multi-line support
    STRUCTURAL_TOKENS = [
        '\\\\',              # row separator in matrices/aligned
        '&',                 # column separator
    ]

    # Environment tokens — treated as atomic units
    ENVIRONMENT_NAMES = [
        'aligned', 'align', 'cases', 'gathered', 'split',
        'eqnarray', 'multline',
        'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix',
        'smallmatrix', 'array',
    ]

    def __init__(self):
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        self.reverse_vocab: Dict[int, str] = {i: t for t, i in self.vocab.items()}

    def tokenize(self, latex: str) -> List[str]:
        """
        Tokenize a LaTeX string into a list of tokens.

        Handles:
        - \\\\  (double backslash) as atomic row separator — checked BEFORE single backslash
        - \\begin{env} and \\end{env} as atomic environment tokens
        - LaTeX commands (\\frac, \\sqrt, etc.)
        - Escaped characters (\\{, \\}, etc.)
        - Individual digits and decimal points
        - Structural symbols: { } ( ) [ ] + - = _ ^ & | < >
        """
        tokens = []
        i = 0
        latex = latex.strip()
        n = len(latex)

        while i < n:
            # Skip whitespace
            if latex[i].isspace():
                i += 1
                continue

            # ----------------------------------------------------------------
            # PRIORITY 1: Double backslash \\\\ (row separator)
            # Must be checked BEFORE single backslash handling.
            # Without this fix, \\\\ is parsed as two separate \\ commands,
            # which destroys matrix/aligned row structure.
            # ----------------------------------------------------------------
            if i + 1 < n and latex[i] == '\\' and latex[i + 1] == '\\':
                if i + 2 >= n or not latex[i + 2].isalpha():
                    tokens.append('\\\\')
                    i += 2
                    continue

            # ----------------------------------------------------------------
            # PRIORITY 2: \\begin{env} and \\end{env} as atomic tokens
            # Prevents {matrix} from being split into {, m, a, t, r, i, x, }
            # ----------------------------------------------------------------
            if latex[i] == '\\' and (latex[i:].startswith('\\begin') or latex[i:].startswith('\\end')):
                cmd_len = 6 if latex[i:].startswith('\\begin') else 4
                j = i + cmd_len
                # Skip whitespace between command and {
                while j < n and latex[j].isspace():
                    j += 1
                if j < n and latex[j] == '{':
                    # Find the closing }
                    k = j + 1
                    while k < n and latex[k] != '}':
                        k += 1
                    if k < n:
                        # Full token: e.g., \\begin{matrix} or \\end{cases}
                        full_token = latex[i:k + 1]
                        # Normalize whitespace within: \\begin  {matrix} → \\begin{matrix}
                        full_token = full_token[:cmd_len] + full_token[cmd_len:].lstrip()
                        tokens.append(full_token)
                        i = k + 1
                        continue

            # ----------------------------------------------------------------
            # Handle LaTeX commands and escaped characters: \\frac, \\{, etc.
            # ----------------------------------------------------------------
            if latex[i] == '\\':
                j = i + 1
                while j < n and latex[j].isalpha():
                    j += 1
                if j == i + 1:
                    # Escaped special character like \{ \} \% \# or bare backslash
                    if j < n:
                        tokens.append(latex[i:j + 1])
                        j += 1
                    else:
                        tokens.append(latex[i])
                        j += 1
                else:
                    # Regular command like \\frac or \\sqrt
                    tokens.append(latex[i:j])
                i = j
                continue

            # ----------------------------------------------------------------
            # Structural symbols — each is a single token
            # ----------------------------------------------------------------
            if latex[i] in '{}()[]+-=_^&|<>':
                tokens.append(latex[i])
                i += 1
                continue

            # ----------------------------------------------------------------
            # Digits and decimal points — each separate token
            # This prevents <unk> for unseen numbers during inference.
            # ----------------------------------------------------------------
            if latex[i].isdigit() or latex[i] == '.':
                tokens.append(latex[i])
                i += 1
                continue

            # ----------------------------------------------------------------
            # Everything else (letters, commas, semicolons, etc.)
            # ----------------------------------------------------------------
            tokens.append(latex[i])
            i += 1

        return tokens

    def build_from_corpus(self, corpus: List[str]):
        """Build vocabulary from a list of LaTeX strings."""
        logger.info("Building vocabulary from corpus...")
        token_counts = Counter()
        for text in corpus:
            token_counts.update(self.tokenize(text))

        # Add all structural tokens first (even if never seen in corpus)
        for token in self.STRUCTURAL_TOKENS:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Add environment tokens
        for env in self.ENVIRONMENT_NAMES:
            for prefix in ['\\begin', '\\end']:
                env_token = f'{prefix}{{{env}}}'
                if env_token not in self.vocab:
                    self.vocab[env_token] = len(self.vocab)

        # Add corpus tokens
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
        """
        Convert a list of integer indices back to a LaTeX string.
        Safely joins tokens with a space. The evaluation metrics
        (exact_match) handle stripping/normalizing spaces for final scoring.
        """
        res = []
        for idx in indices:
            idx_int = int(idx)
            t = self.reverse_vocab.get(idx_int, self.UNK_TOKEN)

            if skip_special and t in self.SPECIAL_TOKENS:
                continue
            if t == self.EOS_TOKEN:
                break
            res.append(t)

        if not res:
            return ""

        # Just join with a space. It is mathematically safe and
        # guarantees we don't accidentally merge commands like \alpha b
        out = ' '.join(res)

        # Clean up any accidental double spaces
        out = ' '.join(out.split())
        return out

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
        self.reverse_vocab = {int(idx): tok for tok, idx in self.vocab.items()}

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