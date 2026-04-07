"""
TAMER OCR Data Module.

This module provides comprehensive data loading capabilities for math formula OCR:
- Tokenization of LaTeX formulas with structural pointer extraction
- Dataset handling with support for both file paths and in-memory PIL Images
- Data augmentation for training robustness
- Curriculum sampling for progressive learning
- Dataset registry with metadata for all supported datasets
- Advanced downloader with support for Kaggle, HuggingFace, Zenodo, and GitHub
- Dataset parser for converting various formats to unified structure
- Data manager for orchestrating multi-stage dataset loading
- Validation for ensuring dataset integrity before training

Supported Datasets:
- Im2LaTeX-100K (Kaggle) - Stage 1: Printed Formulas
- MathWriting (HuggingFace) - Stage 2: Clean Handwritten
- CROHME (Zenodo) - Stage 3a: Competition Handwritten
- HME100K (GitHub/HF) - Stage 3b: Messy Handwritten
"""

# Core data components
from .tokenizer import LaTeXTokenizer, extract_structural_pointers
from .dataset import TreeMathDataset, get_collate_fn
from .augmentation import get_train_augmentation, get_val_augmentation
from .sampler import CurriculumSampler

# Dataset Registry
from .datasets_registry import (
    DatasetRegistry,
    DatasetConfig,
    DatasetFileInfo,
    DatasetCategory,
    get_registry,
    get_dataset_config,
    list_available_datasets,
)

# Advanced Downloader (supports Kaggle, HF, Zenodo, GitHub)
from .advanced_downloader import (
    AdvDownloader,
    DownloadError,
    IntegrityError,
    DiskSpaceError,
    create_downloader,
)

# Original downloader (kept for backward compatibility)
from .downloader import (
    AdvDatasetDownloader,
    create_downloader as create_orig_downloader,
)

# Dataset Parser (converts various formats to unified structure)
from .parser import (
    DatasetParser,
    create_parser,
)

# Data Manager (orchestrates loading of all datasets)
from .data_manager import (
    DataManager,
    create_data_manager,
)

# Validator (ensures dataset integrity before training)
from .validator import (
    DatasetValidator,
    ValidationResult,
    ValidationIssue,
    validate_before_training,
)

__all__ = [
    # Core Data Components
    'LaTeXTokenizer',
    'extract_structural_pointers',
    'TreeMathDataset',
    'get_collate_fn',
    'get_train_augmentation',
    'get_val_augmentation',
    'CurriculumSampler',
    
    # Dataset Registry
    'DatasetRegistry',
    'DatasetConfig',
    'DatasetFileInfo',
    'DatasetCategory',
    'get_registry',
    'get_dataset_config',
    'list_available_datasets',
    
    # Advanced Downloader (Kaggle, HF, Zenodo, GitHub)
    'AdvDownloader',
    'DownloadError',
    'IntegrityError',
    'DiskSpaceError',
    'create_downloader',
    
    # Original Downloader (backward compatibility)
    'AdvDatasetDownloader',
    'create_orig_downloader',
    
    # Dataset Parser
    'DatasetParser',
    'create_parser',
    
    # Data Manager
    'DataManager',
    'create_data_manager',
    
    # Validator
    'DatasetValidator',
    'ValidationResult',
    'ValidationIssue',
    'validate_before_training',
]