"""
Simplified Dataset Validator for TAMER OCR Training.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image

logger = logging.getLogger("TAMER.Validator")


def validate_dataset_dir(data_dir: str, dataset_name: str) -> bool:
    """Quick check if a dataset directory exists and has content."""
    ds_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(ds_dir):
        return False
    
    # Check for either images or annotations
    has_images = False
    has_annotations = False
    
    for root, dirs, files in os.walk(ds_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                has_images = True
            if f.lower() == 'annotations.json':
                has_annotations = True
    
    return has_images or has_annotations


def validate_samples(samples: List[Dict[str, Any]], max_check: int = 100) -> Dict[str, Any]:
    """
    Validate a list of sample dicts.
    Checks that images exist and LaTeX strings are non-empty.
    """
    valid = 0
    missing_images = 0
    empty_latex = 0
    
    for i, sample in enumerate(samples[:max_check]):
        img = sample.get('image')
        latex = sample.get('latex', '')
        
        if not latex.strip():
            empty_latex += 1
            continue
        
        if isinstance(img, str) and not os.path.exists(img):
            missing_images += 1
            continue
        
        valid += 1
    
    total = min(len(samples), max_check)
    return {
        'total_checked': total,
        'valid': valid,
        'missing_images': missing_images,
        'empty_latex': empty_latex,
        'is_ok': valid > total * 0.5  # At least 50% must be valid
    }


def validate_before_training(config) -> bool:
    """Simple pre-training validation."""
    data_dir = Path(config.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False
    
    logger.info(f"Data directory exists: {data_dir}")
    return True
