"""
Dataset validator for TAMER OCR training.
Performs integrity checks on a random subset of images and annotations.
"""

import os
import logging
import random
from typing import List, Dict, Any

from PIL import Image

logger = logging.getLogger("TAMER.Validator")


def validate_samples(samples: List[Dict[str, Any]], max_check: int = 100) -> Dict[str, Any]:
    """
    Validate a random subset of samples.
    Checks that images are readable and non-corrupt, and that LaTeX strings are non-empty.
    """
    if not samples:
        return {'is_ok': False, 'error': 'Sample list is empty'}

    valid = 0
    missing_images = 0
    corrupt_images = 0
    empty_latex = 0

    check_indices = random.sample(range(len(samples)), min(len(samples), max_check))

    for idx in check_indices:
        sample = samples[idx]
        img_path = sample.get('image')
        latex = sample.get('latex', '')

        if not isinstance(latex, str) or not latex.strip():
            empty_latex += 1
            continue

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            missing_images += 1
            continue

        # Open once, verify corruption, read dimensions in the same file handle.
        try:
            with Image.open(img_path) as img:
                img.verify()
            # verify() invalidates the file handle; re-open to read size.
            with Image.open(img_path) as img:
                _ = img.size
            valid += 1
        except Exception as e:
            logger.debug(f"Validation failed for {img_path}: {e}")
            corrupt_images += 1

    total_checked = len(check_indices)
    pass_rate = valid / total_checked if total_checked > 0 else 0

    return {
        'total_checked': total_checked,
        'valid': valid,
        'missing_images': missing_images,
        'corrupt_images': corrupt_images,
        'empty_latex': empty_latex,
        'is_ok': pass_rate > 0.9,
    }


def validate_before_training(config, train_samples: List[Dict], val_samples: List[Dict]) -> bool:
    """
    Pre-training integrity check. Runs once and caches the result to a flag file.
    Subsequent calls return immediately without re-scanning any images.
    """
    cache_flag = os.path.join(getattr(config, 'output_dir', ''), "validation_passed.flag")
    if os.path.exists(cache_flag):
        logger.info("Pre-training integrity check passed previously (cached). Skipping.")
        return True

    for path_attr in ['data_dir', 'checkpoint_dir', 'output_dir']:
        path = getattr(config, path_attr)
        if not os.path.exists(path):
            logger.error(f"Required directory does not exist: {path}")
            return False

    train_val = validate_samples(train_samples, max_check=200)
    if not train_val['is_ok']:
        logger.error(f"Training set validation failed: {train_val}")
        return False

    val_val = validate_samples(val_samples, max_check=100)
    if not val_val['is_ok']:
        logger.error(f"Validation set validation failed: {val_val}")
        return False

    logger.info(
        f"Integrity check passed "
        f"(train: {train_val['valid']}/{train_val['total_checked']} valid, "
        f"val: {val_val['valid']}/{val_val['total_checked']} valid)"
    )

    try:
        os.makedirs(os.path.dirname(cache_flag), exist_ok=True)
        with open(cache_flag, 'w') as f:
            f.write("OK")
    except Exception as e:
        logger.warning(f"Could not write validation cache flag: {e}")

    return True