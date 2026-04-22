"""
Dataset pipeline for TAMER OCR.

v4.0 Changes (Encoder Padding Mask Edition):
  - [FIXED] Top-Left Anchoring: ImageOps.pad now uses centering=(0,0) so the
    math strokes always start at pixel (0,0). This makes 2D positional encodings
    stable and meaningful across all aspect ratios.
  - [FIXED] Real Dimension Tracking: preprocess_image now returns (tensor, real_w,
    real_h) so the collate function can pass precise bounding boxes to the model.
  - [FIXED] Collate function now stacks real_ws / real_hs into batch tensors so
    the engine can build the encoder padding mask with zero overhead.
  - [RETAINED] White Square of Death fix (recursive resampling).
  - [RETAINED] CPU Image Resizing via PIL C-backend.
  - [RETAINED] Mathematical normalization [-1, 1].
"""

import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from typing import Dict, Any, Union, Tuple

logger = logging.getLogger("TAMER.Dataset")


def preprocess_image(
    img_source: Union[str, Image.Image],
    height: int,
    width: int,
    transform=None,
    max_aspect_ratio: float = 10.0,
) -> Tuple[torch.Tensor, int, int]:
    """
    Standalone image preprocessing function.
    Used by BOTH training (MathDataset) and inference to guarantee zero mismatch.

    Returns:
        tensor:  (3, H, W) float32 tensor normalized to [-1, 1].
        real_w:  Width of the content area (before padding) in pixels at target scale.
        real_h:  Height of the content area (before padding) in pixels at target scale.

    The (real_w, real_h) values are used by TAMERModel.generate_memory_mask() to
    construct a boolean encoder padding mask that tells cross-attention exactly
    which spatial feature tokens correspond to actual ink strokes versus white padding.

    [FIX] Top-Left Anchoring (centering=(0, 0)):
        Previously ImageOps.pad used center-padding (default centering=(0.5, 0.5)).
        A short equation "x=1" placed dead-center in a 384x1280 canvas has its
        top-left ink stroke at a wildly different pixel position for every image.
        The 2D sinusoidal positional encodings in the Swin encoder assign meaning
        to absolute spatial position. Center-padding destroys that meaning entirely.
        Anchoring to top-left keeps ALL images starting at position (0,0), making
        the positional encodings a stable, learnable signal.

    [FIX] Return real dimensions:
        We compute the true scaled (pre-padding) width and height so the model
        can generate a precise boolean mask. Without this, the cross-attention
        softmax distributes probability across all 480 feature patches equally,
        including the hundreds of blank white patches added by padding.
    """
    target_h, target_w = height, width

    # Open the image. Do NOT catch exceptions here.
    # Missing/corrupt files raise immediately so MathDataset can resample.
    if isinstance(img_source, str):
        img = Image.open(img_source).convert("RGB")
    elif isinstance(img_source, Image.Image):
        img = img_source.convert("RGB")
    else:
        raise ValueError(f"Unsupported image source type: {type(img_source)}")

    w, h = img.size
    aspect_ratio = max(w, h) / max(min(w, h), 1)

    if w == 0 or h == 0:
        raise ValueError(
            f"Image has zero dimension: width={w}, height={h}. "
            f"Source: {img_source if isinstance(img_source, str) else 'PIL.Image'}"
        )

    if aspect_ratio > max_aspect_ratio:
        raise ValueError(
            f"Pathological aspect ratio: {aspect_ratio:.2f} exceeds limit of "
            f"{max_aspect_ratio}. Image dimensions: {w}x{h}. "
            f"Source: {img_source if isinstance(img_source, str) else 'PIL.Image'}"
        )

    # Compute the actual scaled dimensions BEFORE padding.
    # This is the bounding box of the real content at the target canvas size.
    # We use the same scale factor that ImageOps.pad uses internally so the
    # values are perfectly consistent with the padded image.
    scale = min(target_w / w, target_h / h)
    real_w = int(round(w * scale))
    real_h = int(round(h * scale))

    # Clamp to canvas size defensively (floating point rounding edge cases)
    real_w = min(real_w, target_w)
    real_h = min(real_h, target_h)

    # [FIX] Top-Left Anchoring: centering=(0, 0) anchors content to top-left.
    # Previously centering=(0.5, 0.5) (the ImageOps.pad default) was used.
    # This caused the ink strokes to appear at a different (cx, cy) pixel for
    # every unique aspect ratio, making absolute positional encodings meaningless.
    img = ImageOps.pad(
        img,
        (target_w, target_h),
        method=Image.Resampling.BILINEAR,
        color=(255, 255, 255),
        centering=(0, 0),   # TOP-LEFT ANCHOR — critical for stable pos encodings
    )

    # Apply albumentations augmentations if provided (training only)
    if transform:
        import numpy as np
        arr = np.array(img)
        arr = transform(image=arr)["image"]
        img = Image.fromarray(arr)

    # TF.to_tensor: PIL HWC uint8 [0,255] -> CHW float32 [0.0, 1.0]
    tensor = TF.to_tensor(img)

    # Map [0.0, 1.0] -> [-1.0, 1.0]. Correct for B&W math strokes.
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return tensor, real_w, real_h


class MathDataset(Dataset):
    def __init__(self, samples, config, tokenizer, transform=None):
        self.samples = samples
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_aspect_ratio = getattr(self.config, "max_aspect_ratio", 10.0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_source = sample.get("image") or sample.get("image_path")
        latex = sample.get("latex", "")
        dataset_name = sample.get("dataset_name", "unknown")

        try:
            # Unpack tensor AND real content dimensions
            tensor, real_w, real_h = preprocess_image(
                img_source=img_source,
                height=self.config.img_height,
                width=self.config.img_width,
                transform=self.transform,
                max_aspect_ratio=self.max_aspect_ratio,
            )
        except Exception as e:
            # Recursive resampling: never pass a blank/broken image to the GPU.
            # Logged at DEBUG to avoid drowning real errors in large datasets.
            logger.debug(
                f"Rejecting broken sample at idx={idx} | "
                f"source='{img_source}' | reason={e}. "
                f"Resampling a random valid index..."
            )
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

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
            # Real content dimensions before padding — used to build encoder mask
            "real_w": real_w,
            "real_h": real_h,
        }


def get_collate_fn(pad_id: int):
    """
    Returns a collate function configured with the correct padding token ID.

    Stacks real_w / real_h into batch tensors so the training engine can pass
    them to TAMERModel.generate_memory_mask() with zero per-step overhead.

    Filters:
      - None values (defensive, should not occur with recursive resampling)
      - length <= 2 (empty formula: only [SOS, EOS] with no real tokens)
    """
    def collate_fn(batch):
        batch = [x for x in batch if x is not None and x["length"] > 2]

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

        # Stack real content dimensions — shape (B,) each, dtype int64
        real_ws = torch.tensor([x["real_w"] for x in batch], dtype=torch.long)
        real_hs = torch.tensor([x["real_h"] for x in batch], dtype=torch.long)

        return {
            "image": images,
            "ids": ids,
            "length": lengths,
            "latex": latices,
            "dataset_name": dataset_names,
            "real_ws": real_ws,   # (B,) pixel width of content before padding
            "real_hs": real_hs,   # (B,) pixel height of content before padding
        }

    return collate_fn