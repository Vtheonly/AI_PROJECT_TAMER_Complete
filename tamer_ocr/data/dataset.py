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
    
    Key design decisions:
    - Images maintain aspect ratio: height=256, pad width to 1024.
    - Images use normal ImageNet-style colors (white background, black text).
    - ImageNet normalization applied (3 channels).
    - Handles both file paths and PIL Image objects.
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
        
        # Pre-compute token lengths for filtering if needed
        self._token_lengths = []
        for s in self.samples:
            latex = s.get('latex', '')
            tokens = self.tokenizer.tokenize(latex)
            self._token_lengths.append(len(tokens))

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, img_source: Union[str, Image.Image]) -> torch.Tensor:
        """
        Process an image with aspect-ratio-preserving resizing.
        
        Rules:
        1. Resize so height = 256, maintaining aspect ratio.
        2. If resulting width < 1024, pad with white (255) to 1024.
        3. If resulting width > 1024, resize width to 1024, then pad height to 256.
        4. Normalize using ImageNet mean/std across 3 channels.
        
        Returns:
            Tensor of shape (3, 256, 1024)
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

        # Convert to grayscale to standardize inputs before padding
        img = img.convert('L')
        arr = np.array(img)

        # Handle edge cases (empty or corrupt images)
        if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
        else:
            h, w = arr.shape
            
            # Aspect ratio filter (optional based on config)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.config.max_aspect_ratio:
                canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
            else:
                # Step 1: Scale height to target_h
                scale = target_h / h
                new_w = int(w * scale)

                if new_w > target_w:
                    # Step 2: If width exceeds target, scale width instead
                    scale = target_w / w
                    new_h = int(h * scale)
                    arr = cv2.resize(arr, (target_w, new_h), interpolation=cv2.INTER_AREA)
                    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
                    y_offset = (target_h - new_h) // 2
                    canvas[y_offset:y_offset+new_h, :] = arr
                else:
                    # Resize height to target_h, pad width
                    arr = cv2.resize(arr, (new_w, target_h), interpolation=cv2.INTER_AREA)
                    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
                    canvas[:, :new_w] = arr

        # Apply augmentation if provided (e.g., rotation, noise)
        if self.transform:
            canvas = self.transform(image=canvas)['image']

        # === Normalization and Tensor Conversion ===
        # 1. Convert to [0, 1] range
        tensor = torch.from_numpy(canvas.astype(np.float32)) / 255.0
        
        # 2. Expand to 3 channels (required by Swin backbone)
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)  # (3, 256, 1024)

        # 3. Apply standard ImageNet Normalization
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        img_source = sample.get('image') or sample.get('image_path')
        latex = sample.get('latex', '')
        dataset_name = sample.get('dataset_name', 'unknown')

        # Process Image
        try:
            tensor = self._process_image(img_source)
        except Exception as e:
            logger.warning(f"Failed to load image (idx={idx}): {e}")
            # Return a blank normalized tensor as fallback
            blank = np.full((self.config.img_height, self.config.img_width), 255, dtype=np.uint8)
            tensor = torch.from_numpy(blank.astype(np.float32)) / 255.0
            tensor = tensor.unsqueeze(0).expand(3, -1, -1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            latex = ""

        # Tokenize label
        tokens = self.tokenizer.tokenize(latex)[:self.config.max_seq_len - 2]
        
        # Add special tokens
        ids = [self.tokenizer.sos_id] + self.tokenizer.encode(tokens) + [self.tokenizer.eos_id]

        return {
            'image': tensor,
            'ids': torch.tensor(ids, dtype=torch.long),
            'length': len(ids),
            'latex': latex,
            'dataset_name': dataset_name,
        }


def get_collate_fn(pad_id: int):
    """
    Standard collate function for batching samples.
    Pads token sequences to the longest sequence in the batch.
    """
    def collate_fn(batch):
        # Filter out invalid samples
        batch = [x for x in batch if x['length'] > 0]
        if not batch:
            return None

        # Stack images into a single tensor (B, 3, H, W)
        images = torch.stack([x['image'] for x in batch])
        
        # Collect metadata
        lengths = torch.tensor([x['length'] for x in batch])
        latices = [x['latex'] for x in batch]
        dataset_names = [x['dataset_name'] for x in batch]

        # Pad token IDs
        ids = pad_sequence(
            [x['ids'] for x in batch], 
            batch_first=True, 
            padding_value=pad_id
        )

        return {
            'image': images,
            'ids': ids,
            'length': lengths,
            'latex': latices,
            'dataset_name': dataset_names,
        }
    return collate_fn