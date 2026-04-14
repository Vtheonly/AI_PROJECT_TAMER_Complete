import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import logging
from PIL import Image
from typing import List, Dict, Any, Union
from .tokenizer import LaTeXTokenizer

logger = logging.getLogger("TAMER.Dataset")

class MathDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], config, tokenizer: LaTeXTokenizer, transform=None):
        self.samples = samples
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        self._token_lengths = []
        for s in self.samples:
            tokens = self.tokenizer.tokenize(s.get('latex', ''))
            self._token_lengths.append(len(tokens))

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, img_source: Union[str, Image.Image]) -> torch.Tensor:
        target_h, target_w = self.config.img_height, self.config.img_width

        if isinstance(img_source, str):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        else:
            raise ValueError(f"Unsupported image type: {type(img_source)}")

        img = img.convert('L')
        arr = np.array(img)

        def _blank_tensor():
            canvas = np.full((target_h, target_w), 255.0, dtype=np.float32)
            tensor = torch.from_numpy(canvas) / 255.0
            tensor = tensor.unsqueeze(0).expand(3, -1, -1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (tensor - mean) / std

        if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return _blank_tensor()

        h, w = arr.shape
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > self.config.max_aspect_ratio:
            return _blank_tensor()

        scale = target_h / h
        new_w = int(w * scale)

        if new_w > target_w:
            scale = target_w / w
            new_h = int(h * scale)
            arr = cv2.resize(arr, (target_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, :] = arr
        else:
            arr = cv2.resize(arr, (new_w, target_h), interpolation=cv2.INTER_AREA)
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
            canvas[:, :new_w] = arr

        if self.transform:
            canvas = self.transform(image=canvas)['image']

        # FIX: Standard ImageNet Normalization (No color inversion, expand to 3 channels)
        tensor = torch.from_numpy(canvas.astype(np.float32)) / 255.0
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_source = sample.get('image') or sample.get('image_path')
        latex = sample.get('latex', '')
        dataset_name = sample.get('dataset_name', 'unknown')

        try:
            tensor = self._process_image(img_source)
        except Exception as e:
            logger.warning(f"Failed to load image (idx={idx}): {e}")
            canvas = np.full((self.config.img_height, self.config.img_width), 255.0, dtype=np.float32)
            tensor = torch.from_numpy(canvas) / 255.0
            tensor = tensor.unsqueeze(0).expand(3, -1, -1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            latex = ""

        tokens = self.tokenizer.tokenize(latex)[:self.config.max_seq_len - 2]
        ids = [self.tokenizer.sos_id] + self.tokenizer.encode(tokens) + [self.tokenizer.eos_id]

        return {
            'image': tensor,
            'ids': torch.tensor(ids, dtype=torch.long),
            'length': len(ids),
            'latex': latex,
            'dataset_name': dataset_name,
        }

def get_collate_fn(pad_id: int):
    def collate_fn(batch):
        batch = [x for x in batch if x['length'] > 0]
        if not batch:
            return None
        images = torch.stack([x['image'] for x in batch])
        lengths = torch.tensor([x['length'] for x in batch])
        latices = [x['latex'] for x in batch]
        dataset_names = [x['dataset_name'] for x in batch]
        ids = pad_sequence([x['ids'] for x in batch], batch_first=True, padding_value=pad_id)
        return {'image': images, 'ids': ids, 'length': lengths, 'latex': latices, 'dataset_name': dataset_names}
    return collate_fn