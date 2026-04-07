import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import logging
from PIL import Image
from typing import List, Dict, Any
from .tokenizer import LaTeXTokenizer, extract_structural_pointers

logger = logging.getLogger("TAMER.Dataset")

class TreeMathDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], config, tokenizer: LaTeXTokenizer, transform=None):
        """
        samples: list of dicts with keys 'image_path' and 'latex'
        """
        self.samples = samples
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        self.complexities = [self._compute_complexity(s['latex']) for s in self.samples]

    def _compute_complexity(self, latex: str) -> int:
        return latex.count('{') + latex.count('\\frac') * 2 + latex.count('^') * 2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = sample['image_path']
        latex = sample['latex']
        
        # Image Processing
        try:
            img = Image.open(img_path).convert('L')
            arr = np.array(img)
            h, w = arr.shape
            scale = min(self.config.img_height / h, self.config.img_width / w)
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
            
            canvas = np.full((self.config.img_height, self.config.img_width), 255, dtype=np.uint8)
            canvas[:nh, :nw] = arr
            
            if self.transform:
                canvas = self.transform(image=canvas)['image']
                
            tensor = 1.0 - (torch.from_numpy(canvas).float() / 255.0)
            tensor = tensor.unsqueeze(0)
            
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            tensor = torch.zeros(1, self.config.img_height, self.config.img_width)
            latex = ""

        # Tree Structural Extraction
        tokens = self.tokenizer.tokenize(latex)[:self.config.max_seq_len - 2]
        token_parents = extract_structural_pointers(tokens)
        
        # Add <SOS> and <EOS>
        ids = [self.tokenizer.sos_id] + self.tokenizer.encode(tokens) + [self.tokenizer.eos_id]
        
        # Shift parents by 1 because <SOS> is at index 0. <EOS> points to the last token.
        shifted_parents = [0] + [p + 1 for p in token_parents] + [len(ids) - 2]
        
        return {
            'image': tensor,
            'ids': torch.tensor(ids, dtype=torch.long),
            'parents': torch.tensor(shifted_parents, dtype=torch.long),
            'length': len(ids),
            'latex': latex,
            'complexity': self.complexities[idx]
        }

def get_collate_fn(pad_id: int):
    def collate_fn(batch):
        batch = [x for x in batch if x['length'] > 0]
        if not batch:
            return None
            
        images = torch.stack([x['image'] for x in batch])
        lengths = torch.tensor([x['length'] for x in batch])
        latices = [x['latex'] for x in batch]
        
        ids = pad_sequence([x['ids'] for x in batch], batch_first=True, padding_value=pad_id)
        # Parent pointers for padding tokens point safely to index 0 (<SOS>)
        parents = pad_sequence([x['parents'] for x in batch], batch_first=True, padding_value=0)
        
        return {
            'image': images,
            'ids': ids,
            'parents': parents,
            'length': lengths,
            'latex': latices
        }
    return collate_fn