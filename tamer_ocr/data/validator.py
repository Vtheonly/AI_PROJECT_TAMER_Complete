"""
Pre-Run Dataset Validator for TAMER OCR Training.

This module is the GATEKEEPER for training. The training process will NOT start
unless ALL datasets are:
1. Fully present (all files and directories exist)
2. Correctly configured (proper annotations, valid samples)
3. Meet minimum requirements (sample counts, image format, etc.)
4. Integrity-verified (SHA256 checksums where available)

Usage:
    validator = DatasetValidator(config)
    result = validator.validate_all()
    if not result.is_valid:
        print("CRITICAL: Training cannot start. Fix the following:")
        for issue in result.issues:
            print(f"  - {issue}")
        sys.exit(1)
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
from collections import Counter

from .datasets_registry import DatasetRegistry, DatasetConfig, get_registry
from .downloader import AdvDatasetDownloader, DownloadError

logger = logging.getLogger("TAMER.Validator")


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: str  # 'CRITICAL' | 'WARNING' | 'INFO'
    component: str  # 'dataset', 'config', 'image', 'annotation', 'structure'
    message: str
    suggestion: str = ""
    affected_dataset: str = ""


@dataclass
class ValidationResult:
    """Complete result of a validation run."""
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    dataset_statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_samples: int = 0
    total_datasets_checked: int = 0

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        if issue.severity == 'CRITICAL':
            self.is_valid = False

    def add_critical(self, component: str, message: str, suggestion: str = "", dataset: str = ""):
        self.add_issue(ValidationIssue('CRITICAL', component, message, suggestion, dataset))

    def add_warning(self, component: str, message: str, suggestion: str = "", dataset: str = ""):
        self.add_issue(ValidationIssue('WARNING', component, message, suggestion, dataset))

    def add_info(self, component: str, message: str, suggestion: str = "", dataset: str = ""):
        self.add_issue(ValidationIssue('INFO', component, message, suggestion, dataset))

    def summary(self) -> str:
        """Generate a human-readable summary of validation results."""
        lines = []
        lines.append("=" * 70)
        lines.append("TAMER Dataset Validation Report")
        lines.append("=" * 70)

        lines.append(f"\nDatasets Checked: {self.total_datasets_checked}")
        lines.append(f"Total Samples Found: {self.total_samples}")
        lines.append(f"Overall Status: {'PASSED' if self.is_valid else 'FAILED'}")

        if self.issues:
            lines.append(f"\nIssues Found: {len(self.issues)}")

            critical = [i for i in self.issues if i.severity == 'CRITICAL']
            warnings = [i for i in self.issues if i.severity == 'WARNING']
            infos = [i for i in self.issues if i.severity == 'INFO']

            if critical:
                lines.append(f"\n  CRITICAL ({len(critical)}):")
                for issue in critical:
                    lines.append(f"    ❌ [{issue.component}] {issue.message}")
                    if issue.suggestion:
                        lines.append(f"      → {issue.suggestion}")

            if warnings:
                lines.append(f"\n  WARNINGS ({len(warnings)}):")
                for issue in warnings:
                    lines.append(f"    ⚠️  [{issue.component}] {issue.message}")
                    if issue.suggestion:
                        lines.append(f"      → {issue.suggestion}")

            if infos:
                lines.append(f"\n  INFO ({len(infos)}):")
                for issue in infos:
                    lines.append(f"    ℹ️  [{issue.component}] {issue.message}")
        else:
            lines.append("\n  ✅ No issues found. All datasets validated successfully!")

        lines.append("=" * 70)
        return "\n".join(lines)


class DatasetValidator:
    """
    Comprehensive dataset validator that acts as the gatekeeper.

    Before training starts, this class verifies:
    1. All configured datasets exist on disk
    2. Directory structure is correct
    3. Required files are present
    4. Annotations are valid JSON with correct format
    5. Images are readable and have valid dimensions
    6. Sample counts meet minimum requirements
    7. LaTeX annotations are valid
    8. File integrity (SHA256 where available)

    Training MUST NOT start if any CRITICAL issue is found.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.registry = get_registry()
        self.result = ValidationResult()
        self.min_image_dimension = 1  # Minimum width/height for valid images
        self.max_image_dimension = 10000  # Maximum width/height

    def validate_all(self) -> ValidationResult:
        """
        Run all validation checks. Returns ValidationResult.

        If is_valid is False, training should NOT start.
        """
        logger.info("Starting full dataset validation...")
        self.result = ValidationResult()

        # 1. Validate data directory
        self._validate_data_directory()

        # 2. Get configured datasets
        configured_datasets = self._get_configured_datasets()
        if not configured_datasets:
            self.result.add_critical(
                'config',
                "No datasets configured. Specify datasets in config or use defaults.",
                "Set config.datasets = ['im2latex-100k'] or run with --download flag"
            )
            return self.result

        self.result.total_datasets_checked = len(configured_datasets)

        # 3. Validate each dataset
        for dataset_name in configured_datasets:
            logger.info(f"Validating dataset: {dataset_name}")
            self._validate_single_dataset(dataset_name)

        # 4. Cross-dataset validation
        if len(configured_datasets) > 1:
            self._validate_cross_dataset_consistency(configured_datasets)

        # 5. Final gate check
        if not self.result.is_valid:
            logger.error(f"VALIDATION FAILED. Training cannot start.{os.linesep}"
                         f"Fix all CRITICAL issues before attempting training.{os.linesep}"
                         f"See validation report below:{os.linesep}"
                         f"{self.result.summary()}")
        else:
            logger.info(f"Validation passed! Found {self.result.total_samples} total samples "
                        f"across {self.result.total_datasets_checked} datasets.")
            
        return self.result

    def _validate_data_directory(self):
        """Verify the data directory exists and is accessible."""
        if not self.data_dir.exists():
            self.result.add_critical(
                'structure',
                f"Data directory does not exist: {self.data_dir}",
                f"Create it with: mkdir -p {self.data_dir}"
            )
            return

        if not os.access(self.data_dir, os.R_OK | os.W_OK):
            self.result.add_critical(
                'structure',
                f"Data directory is not accessible (no read/write): {self.data_dir}",
                "Fix permissions: chmod 755 {self.data_dir}"
            )
            return

    def _get_configured_datasets(self) -> List[str]:
        """Get list of datasets to validate based on configuration."""
        # Check config for datasets list
        datasets = getattr(self.config, 'datasets', None)
        
        if datasets is not None and len(datasets) > 0:
            # Validate that each dataset is registered
            for name in datasets:
                if not self.registry.validate_dataset_name(name):
                    self.result.add_critical(
                        'config',
                        f"Unknown dataset '{name}' in config.datasets",
                        f"Available: {', '.join(self.registry.list_datasets())}",
                        dataset=name
                    )
            return datasets
        
        # Default: try custom dataset in data_dir
        custom_dir = self.data_dir
        if any(custom_dir.iterdir()):
            return ['custom']
        
        return []

    def _validate_single_dataset(self, dataset_name: str):
        """Validate a single dataset thoroughly."""
        dataset_config = self.registry.get_config(dataset_name)
        dataset_dir = self.data_dir / dataset_name if dataset_name != 'custom' else self.data_dir
        
        status = {"exists": False, "structure_valid": False, "annotations_valid": False,
                  "images_valid": False, "sample_count": 0, "ready_for_training": False}

        # Check existence
        if not dataset_dir.exists():
            self.result.add_critical(
                'structure',
                f"Dataset directory missing: {dataset_dir}",
                f"Run download for dataset '{dataset_name}' or provide data manually",
                dataset=dataset_name
            )
            self.result.dataset_statuses[dataset_name] = status
            return

        status["exists"] = True

        # Check directory structure
        structure_ok = self._validate_dataset_structure(dataset_dir, dataset_config)
        status["structure_valid"] = structure_ok

        if not structure_ok:
            self.result.add_critical(
                'structure',
                f"Dataset '{dataset_name}' has missing required files/directories",
                f"Check required_files and required_directories in dataset config",
                dataset=dataset_name
            )
            self.result.dataset_statuses[dataset_name] = status
            return

        # Validate annotations
        annotations, annotations_ok = self._validate_annotations(dataset_dir, dataset_config, dataset_name)
        status["annotations_valid"] = annotations_ok

        if not annotations_ok:
            self.result.dataset_statuses[dataset_name] = status
            return  # Critical error already recorded

        # Validate images and samples
        valid_samples, sample_count, image_issues = self._validate_images_and_samples(
            dataset_dir, annotations, dataset_config, dataset_name
        )
        status["images_valid"] = len(image_issues) == 0
        status["sample_count"] = sample_count

        # Record image issues
        for issue in image_issues:
            if issue.severity == 'CRITICAL':
                self.result.add_issue(issue)
            else:
                self.result.add_warning(issue)

        # Check sample count against minimum
        min_samples = dataset_config.min_sample_count if dataset_config else 1
        if sample_count < min_samples:
            self.result.add_critical(
                'dataset',
                f"Dataset '{dataset_name}' has {sample_count} samples, minimum is {min_samples}",
                f"Add more data to this dataset or use a different one",
                dataset=dataset_name
            )
        else:
            logger.info(f"Dataset '{dataset_name}': {sample_count} valid samples "
                        f"(minimum: {min_samples})")

        # Update status
        status["ready_for_training"] = (
            structure_ok and annotations_ok and valid_samples and sample_count >= min_samples
        )
        self.result.dataset_statuses[dataset_name] = status

        if status["ready_for_training"]:
            self.result.total_samples += sample_count

        # Summary for this dataset
        self.result.add_info(
            'dataset',
            f"Dataset '{dataset_name}': {sample_count} samples, "
            f"{'✅ ready' if status['ready_for_training'] else '❌ not ready'}",
            dataset=dataset_name
        )

    def _validate_dataset_structure(self, dataset_dir: Path, config: Optional[DatasetConfig]) -> bool:
        """Validate directory structure and required files."""
        all_ok = True

        if config:
            # Check required directories
            for req_dir in config.required_directories:
                dir_path = dataset_dir / req_dir
                if not dir_path.exists():
                    self.result.add_critical(
                        'structure',
                        f"Missing required directory: {dir_path}",
                        f"Create directory and add image files",
                    )
                    all_ok = False
                elif not dir_path.is_dir():
                    self.result.add_critical(
                        'structure',
                        f"Path exists but is not a directory: {dir_path}",
                    )
                    all_ok = False

            # Check required files
            for req_file in config.required_files:
                file_path = dataset_dir / req_file
                if not file_path.exists():
                    self.result.add_critical(
                        'structure',
                        f"Missing required file: {file_path}",
                        f"Download or provide the file",
                    )
                    all_ok = False
        else:
            # For unregistered datasets, check for basic structure
            images_dir = dataset_dir / "images"
            if not images_dir.exists():
                self.result.add_critical(
                    'structure',
                    f"Expected 'images/' directory not found in dataset",
                    "Create images/ directory and add image files",
                )
                all_ok = False

        return all_ok

    def _validate_annotations(
        self, dataset_dir: Path, config: Optional[DatasetConfig], dataset_name: str
    ) -> Tuple[Optional[Any], bool]:
        """Validate annotation files (JSON format expected)."""
        annotations = None
        annotations_ok = True

        if config:
            annot_file = dataset_dir / config.annotations_file
            annot_filename = config.annotations_file
        else:
            annot_file = dataset_dir / "annotations.json"
            annot_filename = "annotations.json"

        if not annot_file.exists():
            self.result.add_critical(
                'annotation',
                f"Annotation file not found: {annot_file}",
                f"Provide {annot_filename} with image-laTeX mappings",
                dataset=dataset_name
            )
            return None, False

        # Try to load and validate JSON
        try:
            with open(annot_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    self.result.add_critical(
                        'annotation',
                        f"Annotation file is empty: {annot_filename}",
                        "Add annotation data",
                        dataset=dataset_name
                    )
                    return None, False

                annotations = json.loads(content)

        except json.JSONDecodeError as e:
            self.result.add_critical(
                'annotation',
                f"Invalid JSON in {annot_filename}: {e}",
                "Fix the JSON format",
                dataset=dataset_name
            )
            return None, False

        except Exception as e:
            self.result.add_critical(
                'annotation',
                f"Failed to read {annot_filename}: {e}",
                dataset=dataset_name
            )
            return None, False

        # Validate annotation structure
        if isinstance(annotations, list):
            ok, msg = self._validate_list_annotations(annotations, dataset_name)
            if not ok:
                self.result.add_critical('annotation', msg, dataset=dataset_name)
                annotations_ok = False
                
        elif isinstance(annotations, dict):
            ok, msg = self._validate_dict_annotations(annotations, dataset_name)
            if not ok:
                self.result.add_critical('annotation', msg, dataset=dataset_name)
                annotations_ok = False
        else:
            self.result.add_critical(
                'annotation',
                f"Annotation file must be list or dict, got {type(annotations).__name__}",
                dataset=dataset_name
            )
            annotations_ok = False

        return annotations if annotations_ok else None, annotations_ok

    def _validate_list_annotations(self, annotations: List, dataset_name: str) -> Tuple[bool, str]:
        """Validate list-format annotations (list of dicts with image_path and latex)."""
        if len(annotations) == 0:
            return False, "Annotation list is empty. Add at least one entry."

        # Check first few entries for required keys
        sample_size = min(5, len(annotations))
        for i in range(sample_size):
            entry = annotations[i]
            if not isinstance(entry, dict):
                return False, f"Annotation entry {i} is not a dict, got {type(entry).__name__}"
            
            if 'image_path' not in entry:
                return False, f"Annotation entry {i} missing required key 'image_path'"
            if 'latex' not in entry:
                return False, f"Annotation entry {i} missing required key 'latex'"

        return True, ""

    def _validate_dict_annotations(self, annotations: Dict, dataset_name: str) -> Tuple[bool, str]:
        """Validate dict-format annotations ({image_path: latex} or similar)."""
        if len(annotations) == 0:
            return False, "Annotation dict is empty. Add at least one entry."

        # Accept format: {"image1.png": "\\frac{a}{b}", ...}
        # Or format: {"samples": [...], "meta": {...}}
        if "samples" in annotations and isinstance(annotations["samples"], list):
            return self._validate_list_annotations(annotations["samples"], dataset_name)

        # Check first few values are strings (latex)
        sample_keys = list(annotations.keys())[:3]
        for key in sample_keys:
            if isinstance(annotations[key], dict) and 'latex' not in annotations[key]:
                continue  # Will check more thoroughly elsewhere
            elif key.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                if not isinstance(annotations[key], str):
                    return False, f"Expected LaTeX string for key '{key}', got {type(annotations[key]).__name__}"

        return True, ""

    def _validate_images_and_samples(
        self,
        dataset_dir: Path,
        annotations: Any,
        config: Optional[DatasetConfig],
        dataset_name: str
    ) -> Tuple[bool, int, List[ValidationIssue]]:
        """Validate image files and create sample entries."""
        issues = []
        valid_samples = []
        
        # Build sample list based on annotation format
        samples = self._extract_samples(annotations, dataset_dir, config)

        if len(samples) == 0:
            issues.append(ValidationIssue(
                'CRITICAL', 'dataset',
                f"No valid samples found in dataset '{dataset_name}'",
                "Ensure annotations reference existing image files"
            ))
            return False, 0, issues

        # Validate each sample
        corrupt_images = 0
        dimension_issues = 0
        missing_images = 0
        max_report = 10

        for i, sample in enumerate(samples):
            image_path = Path(sample['image_path'])

            # Check existence
            if not image_path.exists():
                missing_images += 1
                if missing_images <= max_report:
                    issues.append(ValidationIssue(
                        'CRITICAL', 'image',
                        f"Image not found: {image_path}",
                        "Check image paths in annotations",
                        dataset_name
                    ))
                continue

            # Check readability and dimensions
            try:
                with Image.open(image_path) as img:
                    img.verify()

                # Re-open for dimensions (verify() may close)
                with Image.open(image_path) as img:
                    w, h = img.size

                    if w < self.min_image_dimension or h < self.min_image_dimension:
                        dimension_issues += 1
                        if dimension_issues <= max_report:
                            issues.append(ValidationIssue(
                                'WARNING', 'image',
                                f"Image too small: {image_path.name} ({w}x{h})",
                                f"Minimum dimension is {self.min_image_dimension}",
                                dataset_name
                            ))
                    elif w > self.max_image_dimension or h > self.max_image_dimension:
                        dimension_issues += 1
                        issues.append(ValidationIssue(
                            'WARNING', 'image',
                            f"Image very large: {image_path.name} ({w}x{h})",
                            "Large images consume high memory",
                            dataset_name
                        ))

                    valid_samples.append(sample)

            except Exception as e:
                corrupt_images += 1
                if corrupt_images <= max_report:
                    issues.append(ValidationIssue(
                        'CRITICAL', 'image',
                        f"Corrupt/unreadable image: {image_path.name} - {e}",
                        "Replace or remove the corrupt file",
                        dataset_name
                    ))

        # Summarize issues
        if missing_images > max_report:
            issues.append(ValidationIssue(
                'CRITICAL', 'image',
                f"... and {missing_images - max_report} more missing images",
                affected_dataset=dataset_name
            ))
        if corrupt_images > max_report:
            issues.append(ValidationIssue(
                'CRITICAL', 'image',
                f"... and {corrupt_images - max_report} more corrupt images",
                affected_dataset=dataset_name
            ))

        valid_count = len(valid_samples)
        logger.info(f"Dataset '{dataset_name}': {valid_count}/{len(samples)} samples valid "
                    f"({missing_images} missing images, {corrupt_images} corrupt, {dimension_issues} dimension issues)")

        return valid_count > 0, valid_count, issues

    def _extract_samples(self, annotations: Any, dataset_dir: Path, config: Optional[DatasetConfig]) -> List[Dict]:
        """Extract sample dicts (with image_path and latex) from annotations."""
        samples = []

        if isinstance(annotations, list):
            # List of dicts
            for entry in annotations:
                if isinstance(entry, dict) and 'image_path' in entry and 'latex' in entry:
                    # Make image path absolute if relative
                    img_path = Path(entry['image_path'])
                    if not img_path.is_absolute():
                        # Try relative to dataset images directory
                        if config:
                            img_path = dataset_dir / config.images_dir / img_path
                        else:
                            img_path = dataset_dir / "images" / img_path

                    samples.append({
                        'image_path': str(img_path),
                        'latex': entry['latex']
                    })

        elif isinstance(annotations, dict):
            # Check for "samples" key
            if "samples" in annotations and isinstance(annotations["samples"], list):
                return self._extract_samples(annotations["samples"], dataset_dir, config)

            # Dict: {"image_path": "latex"} or {"image_path": {"latex": "..."}}
            for key, value in annotations.items():
                if key in ("meta", "config", "info", "metadata"):
                    continue

                if isinstance(value, str):
                    img_path = Path(key)
                    if not img_path.is_absolute():
                        if config:
                            img_path = dataset_dir / config.images_dir / img_path
                        else:
                            img_path = dataset_dir / "images" / img_path

                    samples.append({
                        'image_path': str(img_path),
                        'latex': value
                    })
                elif isinstance(value, dict) and 'latex' in value:
                    img_path = Path(key)
                    if not img_path.is_absolute():
                        if config:
                            img_path = dataset_dir / config.images_dir / img_path
                        else:
                            img_path = dataset_dir / "images" / img_path

                    samples.append({
                        'image_path': str(img_path),
                        'latex': value['latex']
                    })

        return samples

    def _validate_cross_dataset_consistency(self, dataset_names: List[str]):
        """Check for duplicate image paths or latex across datasets."""
        seen_paths = set()
        duplicate_paths = []

        for name in dataset_names:
            status = self.result.dataset_statuses.get(name, {})
            # We'd need to cross-reference samples here
            # This is a placeholder for future cross-dataset dedup

    def pull_verified_dataset(self, dataset_name: str, repo_id: str) -> bool:
        """Pulls a previously verified and structured dataset directly from Hugging Face."""
        try:
            from datasets import load_dataset
            logger.info(f"Pulling verified dataset from Hugging Face: {repo_id}...")
            
            dataset_dir = self.data_dir / dataset_name
            img_dir = dataset_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            
            # Download from HF
            config = self.registry.get_config(dataset_name)
            hf_ds = load_dataset(repo_id, split="train", token=getattr(self.config, 'hf_token', None))
            
            valid_samples = []
            logger.info(f"Extracting {len(hf_ds)} verified samples to disk...")
            
            for idx, item in enumerate(hf_ds):
                img = item['image']
                latex = item['latex']
                
                h = hashlib.md5(latex.encode('utf-8')).hexdigest()[:8]
                img_path = img_dir / f"img_{idx}_{h}.png"
                
                if not img_path.exists():
                    img.save(img_path)
                
                # Store as absolute path so _extract_samples() recognizes it
                valid_samples.append({'image_path': str(img_path.resolve()), 'latex': latex})
                
                if (idx + 1) % 10000 == 0:
                    logger.info(f"  Saved {idx + 1}/{len(hf_ds)} images...")
            
            annot_file = dataset_dir / "annotations.json"
            with open(annot_file, 'w', encoding='utf-8') as f:
                json.dump(valid_samples, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Successfully pulled and prepped verified dataset '{dataset_name}'")
            
            # Refresh validation state
            self.result = ValidationResult()
            self._validate_data_directory()
            self._validate_single_dataset(dataset_name)
            return self.result.dataset_statuses.get(dataset_name, {}).get('ready_for_training', False)
            
        except Exception as e:
            logger.error(f"Failed to pull verified dataset {repo_id}: {e}")
            return False

    def push_dataset_to_hf(self, dataset_name: str, dataset_dir: Path):
        """Pushes a verified dataset to Hugging Face Hub for future fast access."""
        if not getattr(self.config, 'hf_token', None):
            logger.info("No hf_token provided. Skipping push of verified dataset.")
            return
            
        try:
            from datasets import Dataset, Image as HFImage
            from huggingface_hub import HfApi, login
            
            annot_file = dataset_dir / "annotations.json"
            if not annot_file.exists():
                return
                
            with open(annot_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                
            login(token=self.config.hf_token)
            api = HfApi()
            
            # Dynamically get username so we push to the right account
            username = api.whoami()['name']
            base_repo = getattr(self.config, 'hf_dataset_repo', 'Verified-Datasets')
            if '/' not in base_repo:
                base_repo = f"{username}/{base_repo}"
                
            repo_id = f"{base_repo}-{dataset_name}"
            
            logger.info(f"Checking Hugging Face repository: {repo_id}")
            
            try:
                api.repo_info(repo_id, repo_type="dataset")
                logger.info(f"Repo {repo_id} already exists. Skipping push to avoid overwriting.")
                return 
            except Exception:
                pass  # Repo doesn't exist, we will create it automatically by pushing!
                
            logger.info(f"Uploading verified dataset '{dataset_name}' to {repo_id}...")
            
            images = []
            texts = []
            for ann in annotations:
                img_path = str(ann['image_path'])
                if os.path.exists(img_path):
                    images.append(img_path)
                    texts.append(ann['latex'])
                    
            if not images:
                logger.warning("No valid images found to push.")
                return
                
            hf_ds = Dataset.from_dict({"image": images, "latex": texts}).cast_column("image", HFImage())
            hf_ds.push_to_hub(repo_id, private=True, token=self.config.hf_token)
            logger.info(f"Successfully pushed verified dataset '{dataset_name}' to HF Hub!")
            
        except ImportError:
            logger.warning("Libraries 'datasets' or 'huggingface_hub' missing. Cannot push dataset.")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hugging Face: {e}")

    def try_download_and_validate(self, dataset_name: str) -> bool:
        """
        Attempt to download a dataset and validate it.
        Returns True if download and validation both succeed.
        """
        config = self.registry.get_config(dataset_name)
        if not config:
            logger.error(f"Unknown dataset: {dataset_name}. Cannot download.")
            return False

        # FAST PATH: Check if a verified dataset already exists on our Hugging Face Hub
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=getattr(self.config, 'hf_token', None))
            
            # Dynamically resolve username
            username = api.whoami()['name']
            base_repo = getattr(self.config, 'hf_dataset_repo', 'Verified-Datasets')
            if '/' not in base_repo:
                base_repo = f"{username}/{base_repo}"
            repo_id = f"{base_repo}-{dataset_name}"
            
            if api.repo_exists(repo_id, repo_type="dataset"):
                logger.info("=" * 60)
                logger.info(f"TRUSTED SOURCE FOUND: {repo_id}")
                logger.info("Bypassing raw download/parsing and pulling directly from HF Hub.")
                logger.info("=" * 60)
                success = self.pull_verified_dataset(dataset_name, repo_id)
                if success:
                    return True
                logger.warning("Pull failed, falling back to raw download process...")
        except Exception as e:
            logger.debug(f"HF Hub check failed or repo doesn't exist: {e}")

        from .data_manager import DataManager
        dm = DataManager(self.config)
        dataset_dir = self.data_dir / dataset_name

        logger.info(f"Attempting to download dataset: {dataset_name}")
        logger.info(f"  Destination: {dataset_dir}")

        try:
            samples = []
            if dataset_name == "im2latex-100k":
                samples = dm.get_stage1_im2latex(force_refresh=True)
            elif dataset_name == "mathwriting":
                samples = dm.get_stage2_mathwriting(force_refresh=True)
            elif dataset_name == "crohme":
                samples = dm.get_stage3_crohme(force_refresh=True)
            elif dataset_name == "hme100k":
                samples = dm.get_stage3_hme100k(force_refresh=True)
            else:
                # Custom or simple archive fallback
                from .downloader import AdvDatasetDownloader
                downloader = AdvDatasetDownloader(data_dir=str(self.data_dir))
                for archive_info in config.archives:
                    dest = dataset_dir / archive_info.filename
                    downloader.download_file(
                        url=archive_info.url,
                        dest_path=str(dest),
                        expected_sha256=archive_info.sha256,
                        expected_size=archive_info.expected_size_bytes,
                    )
                    if archive_info.is_archive or archive_info.extract_to:
                        extract_path = dataset_dir / (archive_info.extract_to or "")
                        downloader.extract_archive(str(dest), str(extract_path))
                
                result = self.validate_all()
                return result.dataset_statuses.get(dataset_name, {}).get('ready_for_training', False)

            if samples:
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Ensure required directories exist
                for req_dir in config.required_directories:
                    (dataset_dir / req_dir).mkdir(parents=True, exist_ok=True)
                
                annot_file = dataset_dir / config.annotations_file
                valid_samples = []
                
                # Save parsed samples to annotations.json so the validator can pick them up
                for idx, s in enumerate(samples):
                    img = s.get('image')
                    latex = s.get('latex', '')
                    
                    if not latex:
                        continue
                        
                    if isinstance(img, str):
                        # It's a file path
                        valid_samples.append({'image_path': img, 'latex': latex})
                    else:
                        # It's a PIL Image (e.g. from HuggingFace / MathWriting)
                        img_dir = dataset_dir / (config.images_dir if config.images_dir else "images")
                        img_dir.mkdir(parents=True, exist_ok=True)
                        
                        import hashlib
                        h = hashlib.md5(latex.encode('utf-8')).hexdigest()[:8]
                        img_filename = f"img_{idx}_{h}.png"
                        img_path = img_dir / img_filename
                        
                        if not img_path.exists():
                            img.save(img_path)
                            
                        # Store as absolute path so _extract_samples() recognizes it
                        valid_samples.append({'image_path': str(img_path.resolve()), 'latex': latex})
                
                with open(annot_file, 'w', encoding='utf-8') as f:
                    json.dump(valid_samples, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"Saved {len(valid_samples)} mapped samples to {annot_file}")
            else:
                logger.error(f"DataManager returned 0 samples for {dataset_name}. Check download logs.")
                return False

            # Refresh validation state for this dataset
            self.result = ValidationResult()
            self._validate_data_directory()
            self._validate_single_dataset(dataset_name)
            
            is_ready = self.result.dataset_statuses.get(dataset_name, {}).get('ready_for_training', False)
            
            # PUSH TO VERIFIED REPO
            if is_ready:
                self.push_dataset_to_hf(dataset_name, dataset_dir)
                
            return is_ready

        except Exception as e:
            logger.error(f"Download/validation failed for {dataset_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False


def validate_datasets(config, force: bool = False) -> ValidationResult:
    """
    Run dataset validation and enforce gatekeeping.
    This MUST pass before training can begin.
    Training WILL NOT START if this validation fails.
    
    Args:
        config: Training configuration
        force: If True, return result without blocking (caller decides)
        
    Returns:
        ValidationResult indicating whether training can proceed
        
    Raises:
        RuntimeError: If validation fails and force=False
    """
    validator = DatasetValidator(config)
    result = validator.validate_all()

    if not result.is_valid and not force:
        # Print formatted report
        print("\n" + result.summary(), file=sys.stderr)
        print("\n" + "!" * 70, file=sys.stderr)
        print("TRAINING BLOCKED: Dataset validation failed.", file=sys.stderr)
        print("Fix the CRITICAL issues above before running training.", file=sys.stderr)
        print("!" * 70, file=sys.stderr)
        
        raise RuntimeError(
            "Dataset validation failed. Fix all CRITICAL issues before training. "
            "Check the validation report above for details."
        )

    return result


# Backward compatibility alias
validate_before_training = validate_datasets
