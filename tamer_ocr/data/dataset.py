import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import logging
from PIL import Image
from typing import List, Dict, Any, Union, Optional
from .tokenizer import LaTeXTokenizer

logger = logging.getLogger("TAMER.Dataset")


class MathDataset(Dataset):
    """
    PyTorch Dataset for math formula recognition.
    
    Key design decisions (per the blueprint):
    - Images maintain aspect ratio: height=256, pad width to 1024
    - Images are inverted: 0 for background, 1 for ink
    - No tree structure or parent pointers — just token sequences
    - Each sample carries its dataset_name for temperature-based sampling
    
    Handles both:
    - File path samples: {'image': str, 'latex': str, 'dataset_name': str}
    - PIL Image samples: {'image': PIL.Image, 'latex': str, 'dataset_name': str}
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        config,
        tokenizer: LaTeXTokenizer,
        transform=None,
    ):
        self.samples = samples
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Pre-compute token lengths for filtering
        self._token_lengths = []
        for s in self.samples:
            tokens = self.tokenizer.tokenize(s.get('latex', ''))
            self._token_lengths.append(len(tokens))

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, img_source: Union[str, Image.Image]) -> torch.Tensor:
        """
        Process an image with aspect-ratio-preserving resizing.
        
        Blueprint rules:
        1. Resize so height = 256, maintaining aspect ratio
        2. If resulting width < 1024, pad with white (255) to 1024
        3. If resulting width > 1024, resize width to 1024 (let height shrink), then pad height to 256
        4. Invert: 0 for background, 1 for ink
        
        Returns:
            Tensor of shape (1, 256, 1024)
        """
        target_h = self.config.img_height   # 256
        target_w = self.config.img_width    # 1024

        # Load image
        if isinstance(img_source, str):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        else:
            raise ValueError(f"Unsupported image type: {type(img_source)}")

        # Convert to grayscale
        img = img.convert('L')
        arr = np.array(img)

        # Handle edge cases
        if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
            canvas = np.full((target_h, target_w), 0.0, dtype=np.float32)
            return torch.from_numpy(canvas).unsqueeze(0)

        h, w = arr.shape

        # Check aspect ratio filter
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > self.config.max_aspect_ratio:
            # Return blank — will be filtered downstream
            canvas = np.full((target_h, target_w), 0.0, dtype=np.float32)
            return torch.from_numpy(canvas).unsqueeze(0)

        # Step 1: Resize so height = target_h, maintaining aspect ratio
        scale = target_h / h
        new_w = int(w * scale)

        if new_w > target_w:
            # Step 2: If width exceeds target, resize width to target_w instead
            scale = target_w / w
            new_h = int(h * scale)
            arr = cv2.resize(arr, (target_w, new_h), interpolation=cv2.INTER_AREA)
            # Pad height to target_h
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
            # Center vertically
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, :] = arr
        else:
            # Resize height to target_h
            arr = cv2.resize(arr, (new_w, target_h), interpolation=cv2.INTER_AREA)
            # Pad width to target_w with white pixels
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
            canvas[:, :new_w] = arr

        # Apply augmentation if provided
        if self.transform:
            canvas = self.transform(image=canvas)['image']

        # Invert and normalize: 0 for background (white), 1 for ink (black)
        tensor = 1.0 - (torch.from_numpy(canvas.astype(np.float32)) / 255.0)
        tensor = tensor.unsqueeze(0)  # (1, 256, 1024)

        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Get image source (path or PIL Image)
        img_source = sample.get('image') or sample.get('image_path')
        latex = sample.get('latex', '')
        dataset_name = sample.get('dataset_name', 'unknown')

        # Image Processing
        try:
            tensor = self._process_image(img_source)
        except Exception as e:
            logger.warning(f"Failed to load image (idx={idx}): {e}")
            tensor = torch.zeros(1, self.config.img_height, self.config.img_width)
            latex = ""

        # Tokenize — NO structural pointers
        tokens = self.tokenizer.tokenize(latex)[:self.config.max_seq_len - 2]

        # Add <SOS> and <EOS>
        ids = [self.tokenizer.sos_id] + self.tokenizer.encode(tokens) + [self.tokenizer.eos_id]

        return {
            'image': tensor,
            'ids': torch.tensor(ids, dtype=torch.long),
            'length': len(ids),
            'latex': latex,
            'dataset_name': dataset_name,
        }


def get_collate_fn(pad_id: int):
    """Collate function that pads sequences but does NOT handle parent pointers."""
    def collate_fn(batch):
        # Filter out empty sequences
        batch = [x for x in batch if x['length'] > 0]
        if not batch:
            return None

        images = torch.stack([x['image'] for x in batch])
        lengths = torch.tensor([x['length'] for x in batch])
        latices = [x['latex'] for x in batch]
        dataset_names = [x['dataset_name'] for x in batch]

        ids = pad_sequence([x['ids'] for x in batch], batch_first=True, padding_value=pad_id)

        return {
            'image': images,
            'ids': ids,
            'length': lengths,
            'latex': latices,
            'dataset_name': dataset_names,
        }
    return collate_fn
