"""
Dataset Registry for TAMER OCR Training.

Defines all available datasets with their metadata including:
- Download URLs (with mirrors)
- Expected file counts and sizes
- SHA256 checksums for integrity verification
- License information
- Configuration requirements
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger("TAMER.DatasetRegistry")


class DatasetCategory(Enum):
    """Categories of math datasets."""
    HANDWRITTEN_FORMULA = "handwritten_formula"
    PRINTED_FORMULA = "printed_formula"
    HANDWRITTEN_EXPRESSION = "handwritten_expression"
    PRINTED_EXPRESSION = "printed_expression"
    MIXED = "mixed"


@dataclass
class DatasetFileInfo:
    """Metadata for a single file within a dataset."""
    filename: str
    url: str
    expected_size_bytes: int
    sha256: str = ""  # Empty means checksum not available
    is_archive: bool = False
    extract_to: str = ""


@dataclass
class DatasetConfig:
    """Configuration requirements for a dataset."""
    dataset_name: str
    category: DatasetCategory
    description: str
    
    # Download configuration
    archives: List[DatasetFileInfo] = field(default_factory=list)
    images_dir: str = "images"
    annotations_file: str = "annotations.json"
    
    # Expected statistics for validation
    expected_sample_count: int = 0
    min_sample_count: int = 0
    max_sample_count: int = 0
    
    # File requirements
    required_files: List[str] = field(default_factory=list)
    required_directories: List[str] = field(default_factory=list)
    
    # Metadata
    source_url: str = ""
    license: str = ""
    citation: str = ""
    
    # Compatibility
    min_version: str = "1.0.0"
    
    def __post_init__(self):
        if self.min_sample_count == 0:
            self.min_sample_count = self.expected_sample_count


# =============================================================================
# DATASET REGISTRY
# =============================================================================

class DatasetRegistry:
    """
    Central registry of all available datasets for TAMER OCR training.
    
    Each dataset entry contains:
    - Download URLs (primary and mirrors)
    - Expected file structure and counts
    - SHA256 checksums for integrity
    - Configuration requirements
    
    Usage:
        registry = DatasetRegistry()
        config = registry.get_config("im2latex-100k")
        all_datasets = registry.list_datasets()
    """
    
    # Master list of all registered datasets
    DATASETS: Dict[str, DatasetConfig] = {
        # -----------------------------------------------------------------
        # Im2latex-100k (Stage 1: Printed Formulas - Kaggle)
        # -----------------------------------------------------------------
        "im2latex-100k": DatasetConfig(
            dataset_name="im2latex-100k",
            category=DatasetCategory.PRINTED_FORMULA,
            description="100K printed LaTeX formula images with ground truth (Kaggle)",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=100000,
            min_sample_count=95000,
            required_directories=["images"],
            source_url="https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k/data",
            license="MIT",
        ),
        
        # -----------------------------------------------------------------
        # Im2latex-10k (smaller subset for quick testing)
        # -----------------------------------------------------------------
        "im2latex-10k": DatasetConfig(
            dataset_name="im2latex-10k",
            category=DatasetCategory.PRINTED_FORMULA,
            description="10K printed LaTeX formula images for quick testing",
            images_dir="images",
            annotations_file="im2latex_formulas.norm.lst",
            expected_sample_count=10000,
            min_sample_count=9000,
            required_directories=["images"],
            source_url="https://github.com/harvardnlp/im2markup",
        ),
        
        # -----------------------------------------------------------------
        # MathWriting (Stage 2: Clean Handwritten - Hugging Face)
        # -----------------------------------------------------------------
        "mathwriting": DatasetConfig(
            dataset_name="mathwriting",
            category=DatasetCategory.HANDWRITTEN_FORMULA,
            description="Clean handwritten math formulas from Hugging Face",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=50000,
            min_sample_count=1000,
            required_directories=["images"],
            source_url="https://huggingface.co/datasets/deepcopy/MathWriting-human",
            license="MIT",
        ),
        
        # -----------------------------------------------------------------
        # CROHME (Stage 3a: Competition Handwritten - Zenodo)
        # -----------------------------------------------------------------
        "crohme": DatasetConfig(
            dataset_name="crohme",
            category=DatasetCategory.HANDWRITTEN_EXPRESSION,
            description="CROHME handwritten math expression dataset (Stage 3)",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=10000,
            min_sample_count=100,
            required_directories=["images"],
            source_url="https://zenodo.org/records/8428035/files/CROHME23.zip?download=1",
            license="Academic",
        ),
        
        # -----------------------------------------------------------------
        # HME100K (Stage 3b: Messy Handwritten - GitHub/HF)
        # -----------------------------------------------------------------
        "hme100k": DatasetConfig(
            dataset_name="hme100k",
            category=DatasetCategory.HANDWRITTEN_FORMULA,
            description="100K messy handwritten math formulas (Stage 3)",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=100000,
            min_sample_count=1000,
            required_directories=["images"],
            source_url="https://github.com/Phymond/HME100K",
            license="Academic",
        ),
        
        # -----------------------------------------------------------------
        # HANDMATH (Handwritten Math dataset)
        # -----------------------------------------------------------------
        "handmath": DatasetConfig(
            dataset_name="handmath",
            category=DatasetCategory.HANDWRITTEN_FORMULA,
            description="Handwritten mathematical formulas dataset",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=0,
            min_sample_count=100,
            required_directories=["images"],
            required_files=["annotations.json"],
        ),
        
        # -----------------------------------------------------------------
        # Custom user dataset
        # -----------------------------------------------------------------
        "custom": DatasetConfig(
            dataset_name="custom",
            category=DatasetCategory.MIXED,
            description="Custom user-provided dataset",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=0,
            min_sample_count=1,
            required_directories=["images"],
            required_files=["annotations.json"],
        ),
    }
    
    # Default dataset split configuration
    DEFAULT_SPLITS = {
        "im2latex-100k": {"train": 0.85, "val": 0.05, "test": 0.10},
        "im2latex-10k": {"train": 0.80, "val": 0.10, "test": 0.10},
        "crohme": {"train": 0.80, "val": 0.10, "test": 0.10},
        "handmath": {"train": 0.80, "val": 0.10, "test": 0.10},
        "custom": {"train": 0.80, "val": 0.10, "test": 0.10},
    }
    
    def __init__(self):
        self._configs = self.DATASETS.copy()
    
    def get_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration for a specific dataset."""
        return self._configs.get(dataset_name)
    
    def list_datasets(self, category: Optional[DatasetCategory] = None) -> List[str]:
        """List all available datasets, optionally filtered by category."""
        if category is None:
            return list(self._configs.keys())
        return [
            name for name, config in self._configs.items()
            if config.category == category
        ]
    
    def get_split_config(self, dataset_name: str) -> Dict[str, float]:
        """Get train/val/test split ratios for a dataset."""
        return self.DEFAULT_SPLITS.get(dataset_name, {"train": 0.8, "val": 0.1, "test": 0.1})
    
    def add_custom_dataset(self, name: str, config: DatasetConfig):
        """Register a new dataset at runtime."""
        self._configs[name] = config
        logger.info(f"Registered custom dataset: {name}")
    
    def validate_dataset_name(self, dataset_name: str) -> bool:
        """Check if a dataset is registered."""
        return dataset_name in self._configs


# Global registry instance
_registry = DatasetRegistry()


def get_registry() -> DatasetRegistry:
    """Get the global dataset registry instance."""
    return _registry


def get_dataset_config(dataset_name: str) -> Optional[DatasetConfig]:
    """Convenience function to get a dataset config."""
    return _registry.get_config(dataset_name)


def list_available_datasets(category: Optional[DatasetCategory] = None) -> List[str]:
    """Convenience function to list available datasets."""
    return _registry.list_datasets(category)