import logging
from collections import Counter
from typing import List, Tuple, Dict

logger = logging.getLogger("TAMER.Tokenizer")

class LaTeXTokenizer:
    """Tokenizer that respects LaTeX commands and extracts structural pointers."""
    
    def __init__(self):
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.reverse_vocab: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        
    def tokenize(self, latex: str) -> List[str]:
        tokens = []
        i = 0
        latex = latex.strip()
        while i < len(latex):
            if latex[i].isspace():
                i += 1
                continue
            if latex[i] == '\\':
                j = i + 1
                while j < len(latex) and latex[j].isalpha():
                    j += 1
                tokens.append(latex[i:j])
                i = j
            elif latex[i] in '{}()[]+-=_^&':
                tokens.append(latex[i])
                i += 1
            elif latex[i].isdigit():
                j = i
                while j < len(latex) and (latex[j].isdigit() or latex[j] == '.'):
                    j += 1
                tokens.append(latex[i:j])
                i = j
            else:
                tokens.append(latex[i])
                i += 1
        return tokens

    def build_from_corpus(self, corpus: List[str]):
        logger.info("Building vocabulary from corpus...")
        token_counts = Counter()
        for text in corpus:
            token_counts.update(self.tokenize(text))
            
        for token, _ in token_counts.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                
        self.reverse_vocab = {i: t for t, i in self.vocab.items()}
        logger.info(f"Vocabulary built: {len(self.vocab)} total tokens.")

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        res = []
        for idx in indices:
            t = self.reverse_vocab.get(idx, '<unk>')
            if skip_special and t in self.special_tokens:
                continue
            if t == '<eos>':
                break
            res.append(t)
        return ' '.join(res)
        
    @property
    def pad_id(self) -> int: return self.vocab['<pad>']
    @property
    def sos_id(self) -> int: return self.vocab['<sos>']
    @property
    def eos_id(self) -> int: return self.vocab['<eos>']
    @property
    def unk_id(self) -> int: return self.vocab['<unk>']
    def __len__(self) -> int: return len(self.vocab)


def extract_structural_pointers(tokens: List[str]) -> List[int]:
    """
    Core function for Tree-Guided Decoding.
    Assigns each token an index representing its structural parent.
    -1 means root.
    """
    parents = [-1] * len(tokens)
    stack = []
    
    for i, t in enumerate(tokens):
        if t == '{':
            stack.append(i)
            parents[i] = i - 1 if i > 0 else -1
        elif t == '}':
            if stack:
                start = stack.pop()
                parents[i] = parents[start]
        elif t in ['^', '_', '\\frac', '\\sqrt']:
            parents[i] = i - 1 if i > 0 else -1
        else:
            parents[i] = i - 1 if i > 0 and parents[i] == -1 else parents[i]
            
    return parents