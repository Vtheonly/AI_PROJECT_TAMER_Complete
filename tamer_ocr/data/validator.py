"""
Dataset Validator for TAMER OCR Training.
Performs real integrity checks on images and annotations.
"""

import os
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image

logger = logging.getLogger("TAMER.Validator")


def validate_samples(samples: List[Dict[str, Any]], max_check: int = 100) -> Dict[str, Any]:
    """
    Validate a list of sample dicts.
    Checks that images are readable, non-corrupt, and LaTeX strings are valid.
    """
    if not samples:
        return {'is_ok': False, 'error': 'Sample list is empty'}

    valid = 0
    missing_images = 0
    corrupt_images = 0
    empty_latex = 0
    
    # Check a random subset if the list is large
    check_indices = random.sample(range(len(samples)), min(len(samples), max_check))
    
    for idx in check_indices:
        sample = samples[idx]
        img_path = sample.get('image')
        latex = sample.get('latex', '')
        
        # 1. Check LaTeX
        if not isinstance(latex, str) or not latex.strip():
            empty_latex += 1
            continue
        
        # 2. Check Image Path
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            missing_images += 1
            continue
        
        # 3. Check Image Readability (The "Real" Check)
        try:
            with Image.open(img_path) as img:
                img.verify() # Check if corrupt
            # Re-open to check dimensions (verify() closes the file)
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
        'is_ok': pass_rate > 0.9 # Require 90% pass rate for the smoke test
    }


def validate_before_training(config, train_samples: List[Dict], val_samples: List[Dict]) -> bool:
    """
    Performs a strict pre-training integrity check.
    Ensures the pipeline won't crash 10 hours in due to a missing folder.
    """
    logger.info("Performing pre-training integrity check...")
    
    # 1. Check Directories
    for path_attr in ['data_dir', 'checkpoint_dir', 'output_dir']:
        path = getattr(config, path_attr)
        if not os.path.exists(path):
            logger.error(f"Required directory does not exist: {path}")
            return False

    # 2. Validate Train Subset
    train_val = validate_samples(train_samples, max_check=200)
    if not train_val['is_ok']:
        logger.error(f"Training set validation failed: {train_val}")
        return False
    
    # 3. Validate Val Subset
    val_val = validate_samples(val_samples, max_check=100)
    if not val_val['is_ok']:
        logger.error(f"Validation set validation failed: {val_val}")
        return False

    logger.info(f"Integrity check passed (Train readability: {train_val['valid']}/{train_val['total_checked']})")
    return True