"""
Dataset pipeline for TAMER OCR.

v2.4 Changes:
  - [FIXED] CPU Image Resizing Bottleneck: Removed slow cv2/numpy nested loops
    and replaced with PIL.ImageOps.pad which uses a heavily optimized C-backend.
  - [FIXED] ImageNet Normalization Flaw: Replaced ImageNet stats ([0.485, 0.456, 0.406])
    with mathematically correct B&W math notation normalization ([0.5, 0.5, 0.5]),
    mapping [0.0, 1.0] to [-1.0, 1.0] which is correct for mathematical strokes.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from typing import Dict, Any, Union, Optional

logger = logging.getLogger("TAMER.Dataset")


def preprocess_image(
    img_source: Union[str, Image.Image],
    height: int,
    width: int,
    transform=None,
    max_aspect_ratio: float = 10.0,
) -> torch.Tensor:
    """
    Standalone image preprocessing.
    Used by BOTH training (MathDataset) and inference to guarantee zero mismatch.

    [FIX: CPU Image Resizing Bottleneck]
    Removed slow cv2/numpy nested loops. PIL loads and handles this natively.
    ImageOps.pad does EXACTLY what the old cv2 code did (preserves aspect ratio,
    centers, and pads with white) but uses a heavily optimized C-backend.

    [FIX: ImageNet Normalization Flaw]
    to_tensor() safely moves PIL to PyTorch float [0, 1] without NumPy overhead.
    We map [0.0, 1.0] to [-1.0, 1.0]. Perfect for B&W mathematical strokes.
    """
    target_h, target_w = height, width

    # [FIX: CPU Image Resizing Bottleneck]
    # Removed slow cv2/numpy nested loops. PIL loads and handles this natively.
    if isinstance(img_source, str):
        try:
            img = Image.open(img_source).convert("RGB")
        except Exception:
            img = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    elif isinstance(img_source, Image.Image):
        img = img_source.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(img_source)}")

    w, h = img.size
    aspect_ratio = max(w, h) / max(min(w, h), 1)

    if aspect_ratio > max_aspect_ratio or w == 0 or h == 0:
        # Fallback for corrupted or pathological dimensions
        img = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    else:
        # [FIX: CPU Image Resizing Bottleneck]
        # ImageOps.pad does EXACTLY what the old cv2 code did (preserves aspect
        # ratio, centers, and pads with white) but uses a heavily optimized
        # C-backend, eliminating GIL lockups that starve the Blackwell GPU.
        img = ImageOps.pad(
            img,
            (target_w, target_h),
            method=Image.Resampling.BILINEAR,
            color=(255, 255, 255),
        )

    if transform:
        import numpy as np
        arr = np.array(img)
        arr = transform(image=arr)["image"]
        img = Image.fromarray(arr)

    # [FIX: ImageNet Normalization Flaw]
    # to_tensor() safely moves PIL to PyTorch float [0, 1] without NumPy overhead.
    tensor = TF.to_tensor(img)

    # We map [0.0, 1.0] to [-1.0, 1.0]. Perfect for B&W mathematical strokes.
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return tensor


class MathDataset(Dataset):
    def __init__(self, samples, config, tokenizer, transform=None):
        self.samples = samples
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _process_image(self, img_source: Union[str, Image.Image]) -> torch.Tensor:
        return preprocess_image(
            img_source,
            self.config.img_height,
            self.config.img_width,
            transform=self.transform,
            max_aspect_ratio=getattr(self.config, "max_aspect_ratio", 10.0),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_source = sample.get("image") or sample.get("image_path")
        latex = sample.get("latex", "")
        dataset_name = sample.get("dataset_name", "unknown")

        try:
            tensor = self._process_image(img_source)
        except Exception as e:
            logger.warning(f"Failed to load image (idx={idx}): {e}")
            tensor = preprocess_image(
                Image.new(
                    "RGB",
                    (self.config.img_width, self.config.img_height),
                    color=(255, 255, 255),
                ),
                self.config.img_height,
                self.config.img_width,
            )
            latex = ""

        tokens = self.tokenizer.tokenize(latex)[: self.config.max_seq_len - 2]
        ids = (
            [self.tokenizer.sos_id]
            + self.tokenizer.encode(tokens)
            + [self.tokenizer.eos_id]
        )

        return {
            "image": tensor,
            "ids": torch.tensor(ids, dtype=torch.long),
            "length": len(ids),
            "latex": latex,
            "dataset_name": dataset_name,
        }


def get_collate_fn(pad_id: int):
    def collate_fn(batch):
        batch = [x for x in batch if x["length"] > 0]
        if not batch:
            return None

        images = torch.stack([x["image"] for x in batch])
        lengths = torch.tensor([x["length"] for x in batch])
        latices = [x["latex"] for x in batch]
        dataset_names = [x["dataset_name"] for x in batch]

        ids = pad_sequence(
            [x["ids"] for x in batch],
            batch_first=True,
            padding_value=pad_id,
        )

        return {
            "image": images,
            "ids": ids,
            "length": lengths,
            "latex": latices,
            "dataset_name": dataset_names,
        }

    return collate_fn