import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import logging
from PIL import Image
from typing import List, Dict, Any, Union, Optional

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
    """
    target_h, target_w = height, width

    if isinstance(img_source, str):
        img = Image.open(img_source)
    elif isinstance(img_source, Image.Image):
        img = img_source
    else:
        raise ValueError(f"Unsupported image type: {type(img_source)}")

    img = img.convert("RGB")
    arr = np.array(img)

    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    else:
        h, w = arr.shape[:2]
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > max_aspect_ratio:
            canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        else:
            img_aspect = w / max(h, 1)
            if img_aspect < 2.0:
                scale_h = target_h / h
                scale_w = target_w / w
                scale = min(scale_h, scale_w)
                new_h = min(int(h * scale), target_h)
                new_w = min(int(w * scale), target_w)
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
                y_off = (target_h - new_h) // 2
                x_off = (target_w - new_w) // 2
                canvas[y_off : y_off + new_h, x_off : x_off + new_w] = arr
            else:
                scale = target_h / h
                new_w = int(w * scale)
                if new_w > target_w:
                    scale = target_w / w
                    new_h = int(h * scale)
                    arr = cv2.resize(arr, (target_w, new_h), interpolation=cv2.INTER_AREA)
                    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
                    y_offset = (target_h - new_h) // 2
                    canvas[y_offset : y_offset + new_h, :, :] = arr
                else:
                    arr = cv2.resize(arr, (new_w, target_h), interpolation=cv2.INTER_AREA)
                    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
                    canvas[:, :new_w, :] = arr

    if transform:
        canvas = transform(image=canvas)["image"]

    tensor = torch.from_numpy(canvas.astype(np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
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
                Image.new("RGB", (self.config.img_width, self.config.img_height), color=(255, 255, 255)),
                self.config.img_height,
                self.config.img_width,
            )
            latex = ""

        tokens = self.tokenizer.tokenize(latex)[: self.config.max_seq_len - 2]
        ids = [self.tokenizer.sos_id] + self.tokenizer.encode(tokens) + [self.tokenizer.eos_id]

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
            [x["ids"] for x in batch], batch_first=True, padding_value=pad_id
        )

        return {
            "image": images,
            "ids": ids,
            "length": lengths,
            "latex": latices,
            "dataset_name": dataset_names,
        }

    return collate_fn