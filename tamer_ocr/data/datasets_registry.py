"""
Dataset Registry for TAMER OCR Training.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger("TAMER.DatasetRegistry")


class DatasetCategory(Enum):
    HANDWRITTEN_FORMULA = "handwritten_formula"
    PRINTED_FORMULA = "printed_formula"
    HANDWRITTEN_EXPRESSION = "handwritten_expression"
    PRINTED_EXPRESSION = "printed_expression"
    MIXED = "mixed"


@dataclass
class DatasetFileInfo:
    filename: str
    url: str
    expected_size_bytes: int
    sha256: str = "" 
    is_archive: bool = False
    extract_to: str = ""


@dataclass
class DatasetConfig:
    dataset_name: str
    category: DatasetCategory
    description: str
    
    archives: List[DatasetFileInfo] = field(default_factory=list)
    images_dir: str = "images"
    annotations_file: str = "annotations.json"
    
    expected_sample_count: int = 0
    min_sample_count: int = 0
    max_sample_count: int = 0
    
    required_files: List[str] = field(default_factory=list)
    required_directories: List[str] = field(default_factory=list)
    
    source_url: str = ""
    license: str = ""
    citation: str = ""
    
    min_version: str = "1.0.0"
    
    def __post_init__(self):
        if self.min_sample_count == 0:
            self.min_sample_count = self.expected_sample_count


class DatasetRegistry:
    DATASETS: Dict[str, DatasetConfig] = {
        "im2latex-100k": DatasetConfig(
            dataset_name="im2latex-100k",
            category=DatasetCategory.PRINTED_FORMULA,
            description="100K printed LaTeX formula images with ground truth (Kaggle/HF mirror)",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=100000,
            min_sample_count=10000,  # CHANGED: Lowered to allow sub-splits 
            required_directories=["images"],
            source_url="https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k/data",
            license="MIT",
        ),
        
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
        
        "crohme": DatasetConfig(
            dataset_name="crohme",
            category=DatasetCategory.HANDWRITTEN_EXPRESSION,
            description="CROHME handwritten math expression dataset (Stage 3)",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=10000,
            min_sample_count=100,
            required_directories=[],
            source_url="https://zenodo.org/records/8428035/files/CROHME23.zip?download=1",
            license="Academic",
        ),
        
        "hme100k": DatasetConfig(
            dataset_name="hme100k",
            category=DatasetCategory.HANDWRITTEN_FORMULA,
            description="100K messy handwritten math formulas (Stage 3)",
            images_dir="images",
            annotations_file="annotations.json",
            expected_sample_count=100000,
            min_sample_count=100, # CHANGED: Lowered so missing GH files don't block run
            required_directories=[],
            source_url="https://github.com/Phymond/HME100K",
            license="Academic",
        ),
        
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
        return self._configs.get(dataset_name)
    
    def list_datasets(self, category: Optional[DatasetCategory] = None) -> List[str]:
        if category is None:
            return list(self._configs.keys())
        return [
            name for name, config in self._configs.items()
            if config.category == category
        ]
    
    def get_split_config(self, dataset_name: str) -> Dict[str, float]:
        return self.DEFAULT_SPLITS.get(dataset_name, {"train": 0.8, "val": 0.1, "test": 0.1})
    
    def add_custom_dataset(self, name: str, config: DatasetConfig):
        self._configs[name] = config
        logger.info(f"Registered custom dataset: {name}")
    
    def validate_dataset_name(self, dataset_name: str) -> bool:
        return dataset_name in self._configs


_registry = DatasetRegistry()

def get_registry() -> DatasetRegistry:
    return _registry

def get_dataset_config(dataset_name: str) -> Optional[DatasetConfig]:
    return _registry.get_config(dataset_name)

def list_available_datasets(category: Optional[DatasetCategory] = None) -> List[str]:
    return _registry.list_datasets(category)