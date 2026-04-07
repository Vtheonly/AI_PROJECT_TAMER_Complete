import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import logging
from PIL import Image
from typing import List, Dict, Any, Union
from .tokenizer import LaTeXTokenizer, extract_structural_pointers

logger = logging.getLogger("TAMER.Dataset")

class TreeMathDataset(Dataset):
    """
    PyTorch Dataset for math formula recognition with tree-structure guidance.
    
    Handles both:
    - File path samples: {'image': str, 'latex': str}
    - PIL Image samples: {'image': PIL.Image, 'latex': str}
    
    This unified interface supports datasets from different sources:
    - CROHME/HME100K/Im2LaTeX: File paths
    - MathWriting (HF): In-memory PIL Images
    """
    def __init__(self, samples: List[Dict[str, Any]], config, tokenizer: LaTeXTokenizer, transform=None):
        """
        Initialize the dataset.
        
        Args:
            samples: List of dicts with keys:
                - 'image': str (file path) or PIL.Image.Image
                - 'latex': str (LaTeX formula)
            config: Configuration object with img_height, img_width, max_seq_len
            tokenizer: LaTeXTokenizer instance
            transform: Albumentations transform (optional)
        """
        self.samples = samples
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        self.complexities = [self._compute_complexity(s.get('latex', '')) for s in self.samples]

    def _compute_complexity(self, latex: str) -> int:
        """Compute structural complexity score for curriculum learning."""
        return latex.count('{') + latex.count('\\frac') * 2 + latex.count('^') * 2

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, img_source: Union[str, Image.Image]) -> torch.Tensor:
        """
        Process an image from either a file path or PIL Image.
        
        Args:
            img_source: File path (str) or PIL.Image.Image
            
        Returns:
            Normalized tensor of shape (1, height, width)
        """
        # Handle both string paths and PIL images
        if isinstance(img_source, str):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        else:
            raise ValueError(f"Unsupported image type: {type(img_source)}")
        
        # Convert to grayscale
        img = img.convert('L')
        arr = np.array(img)
        
        # Handle edge cases (empty images, etc.)
        if arr.size == 0:
            return torch.zeros(1, self.config.img_height, self.config.img_width)
            
        h, w = arr.shape
        if h == 0 or w == 0:
            return torch.zeros(1, self.config.img_height, self.config.img_width)
        
        # Scale to fit config dimensions while maintaining aspect ratio
        scale = min(self.config.img_height / h, self.config.img_width / w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        
        if nh > 0 and nw > 0:
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        
        # Create canvas and place resized image
        canvas = np.full((self.config.img_height, self.config.img_width), 255, dtype=np.uint8)
        canvas[:nh, :nw] = arr
        
        # Apply augmentation if provided
        if self.transform:
            canvas = self.transform(image=canvas)['image']
            
        # Normalize to [0, 1] and invert (black=1, white=0 for typical math formulas)
        tensor = 1.0 - (torch.from_numpy(canvas).float() / 255.0)
        tensor = tensor.unsqueeze(0)  # Add channel dimension
        
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Get image source (path or PIL Image)
        img_source = sample.get('image') or sample.get('image_path')
        latex = sample.get('latex', '')
        
        # Image Processing
        try:
            tensor = self._process_image(img_source)
        except Exception as e:
            logger.warning(f"Failed to load image (idx={idx}): {e}")
            tensor = torch.zeros(1, self.config.img_height, self.config.img_width)
            latex = ""  # Skip tokenization for invalid images

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