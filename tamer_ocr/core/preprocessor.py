"""
Dataset Preprocessor — Forwarded to data module.

The actual implementation has been moved to tamer_ocr/data/preprocessor.py 
which handles VRAM-safe processing and robust downloading.
"""

from ..data.preprocessor import DatasetPreprocessor

__all__ = ['DatasetPreprocessor']