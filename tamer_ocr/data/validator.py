"""
Pre-Run Dataset Validator for TAMER OCR Training.
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
    severity: str  # 'CRITICAL' | 'WARNING' | 'INFO'
    component: str
    message: str
    suggestion: str = ""
    affected_dataset: str = ""


@dataclass
class ValidationResult:
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
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.registry = get_registry()
        self.result = ValidationResult()
        self.min_image_dimension = 1
        self.max_image_dimension = 10000

    def validate_all(self) -> ValidationResult:
        logger.info("Starting full dataset validation...")
        self.result = ValidationResult()

        self._validate_data_directory()
        configured_datasets = self._get_configured_datasets()
        
        if not configured_datasets:
            self.result.add_critical(
                'config',
                "No datasets configured. Specify datasets in config or use defaults.",
                "Set config.datasets = ['im2latex-100k'] or run with --download flag"
            )
            return self.result

        self.result.total_datasets_checked = len(configured_datasets)

        for dataset_name in configured_datasets:
            logger.info(f"Validating dataset: {dataset_name}")
            self._validate_single_dataset(dataset_name)

        if len(configured_datasets) > 1:
            self._validate_cross_dataset_consistency(configured_datasets)

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
                f"Fix permissions: chmod 755 {self.data_dir}"
            )
            return

    def _get_configured_datasets(self) -> List[str]:
        datasets = getattr(self.config, 'datasets', None)
        
        if datasets is not None and len(datasets) > 0:
            for name in datasets:
                if not self.registry.validate_dataset_name(name):
                    self.result.add_critical(
                        'config',
                        f"Unknown dataset '{name}' in config.datasets",
                        f"Available: {', '.join(self.registry.list_datasets())}",
                        dataset=name
                    )
            return datasets
        
        custom_dir = self.data_dir
        if any(custom_dir.iterdir()):
            return ['custom']
        
        return []

    def _validate_single_dataset(self, dataset_name: str):
        dataset_config = self.registry.get_config(dataset_name)
        dataset_dir = self.data_dir / dataset_name if dataset_name != 'custom' else self.data_dir
        
        status = {"exists": False, "structure_valid": False, "annotations_valid": False,
                  "images_valid": False, "sample_count": 0, "ready_for_training": False}

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

        annotations, annotations_ok = self._validate_annotations(dataset_dir, dataset_config, dataset_name)
        status["annotations_valid"] = annotations_ok

        if not annotations_ok:
            self.result.dataset_statuses[dataset_name] = status
            return 

        valid_samples, sample_count, image_issues = self._validate_images_and_samples(
            dataset_dir, annotations, dataset_config, dataset_name
        )
        status["images_valid"] = len(image_issues) == 0
        status["sample_count"] = sample_count

        for issue in image_issues:
            if issue.severity == 'CRITICAL':
                self.result.add_issue(issue)
            else:
                self.result.add_warning(issue.component, issue.message, issue.suggestion, issue.affected_dataset)

        min_samples = dataset_config.min_sample_count if dataset_config else 1
        if sample_count < min_samples:
            self.result.add_critical(
                'dataset',
                f"Dataset '{dataset_name}' has {sample_count} samples, minimum is {min_samples}",
                f"Add more data to this dataset or use a different one",
                dataset=dataset_name
            )
        else:
            logger.info(f"Dataset '{dataset_name}': {sample_count} valid samples (minimum: {min_samples})")

        status["ready_for_training"] = (
            structure_ok and annotations_ok and valid_samples and sample_count >= min_samples
        )
        self.result.dataset_statuses[dataset_name] = status

        if status["ready_for_training"]:
            self.result.total_samples += sample_count

        self.result.add_info(
            'dataset',
            f"Dataset '{dataset_name}': {sample_count} samples, "
            f"{'✅ ready' if status['ready_for_training'] else '❌ not ready'}",
            dataset=dataset_name
        )

    def _validate_dataset_structure(self, dataset_dir: Path, config: Optional[DatasetConfig]) -> bool:
        all_ok = True
        if config:
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
            self.result.add_critical('annotation', f"Failed to read {annot_filename}: {e}", dataset=dataset_name)
            return None, False

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
        if len(annotations) == 0:
            return False, "Annotation list is empty. Add at least one entry."
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
        if len(annotations) == 0:
            return False, "Annotation dict is empty. Add at least one entry."
        if "samples" in annotations and isinstance(annotations["samples"], list):
            return self._validate_list_annotations(annotations["samples"], dataset_name)
        sample_keys = list(annotations.keys())[:3]
        for key in sample_keys:
            if isinstance(annotations[key], dict) and 'latex' not in annotations[key]:
                continue 
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
        issues = []
        valid_samples = []
        
        samples = self._extract_samples(annotations, dataset_dir, config)

        if len(samples) == 0:
            issues.append(ValidationIssue(
                'CRITICAL', 'dataset',
                f"No valid samples found in dataset '{dataset_name}'",
                "Ensure annotations reference existing image files"
            ))
            return False, 0, issues

        corrupt_images = 0
        dimension_issues = 0
        missing_images = 0
        max_report = 10

        for i, sample in enumerate(samples):
            image_path = Path(sample['image_path'])

            if not image_path.exists():
                missing_images += 1
                if missing_images <= max_report:
                    # CHANGED: Missing images are WARNINGS so they drop gracefully without blocking
                    issues.append(ValidationIssue(
                        'WARNING', 'image',
                        f"Image not found: {image_path}",
                        "Check image paths in annotations",
                        dataset_name
                    ))
                continue

            try:
                with Image.open(image_path) as img:
                    img.verify()
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
                    # CHANGED: Corrupt images are WARNINGS, allowing skipping instead of blocking
                    issues.append(ValidationIssue(
                        'WARNING', 'image',
                        f"Corrupt/unreadable image: {image_path.name} - {e}",
                        "Replace or remove the corrupt file",
                        dataset_name
                    ))

        if missing_images > max_report:
            issues.append(ValidationIssue(
                'WARNING', 'image',
                f"... and {missing_images - max_report} more missing images (skipped)",
                affected_dataset=dataset_name
            ))
        if corrupt_images > max_report:
            issues.append(ValidationIssue(
                'WARNING', 'image',
                f"... and {corrupt_images - max_report} more corrupt images (skipped)",
                affected_dataset=dataset_name
            ))

        valid_count = len(valid_samples)
        logger.info(f"Dataset '{dataset_name}': {valid_count}/{len(samples)} samples valid "
                    f"({missing_images} missing images, {corrupt_images} corrupt, {dimension_issues} dimension issues)")

        return valid_count > 0, valid_count, issues

    def _extract_samples(self, annotations: Any, dataset_dir: Path, config: Optional[DatasetConfig]) -> List[Dict]:
        """Extract sample dicts (with image_path and latex) from annotations."""
        samples = []

        # CHANGED: Robust path resolution function to completely eliminate double path appending.
        def resolve_img_path(raw_path_str: str) -> Path:
            p = Path(raw_path_str)
            if p.is_absolute():
                return p
            
            # Check 1: Is it already correct relative to the current working dir?
            if p.is_file():
                return p
            # Check 2: Relative to the dataset root?
            if (dataset_dir / p).is_file():
                return dataset_dir / p
            # Check 3: Relative to the configured images_dir?
            if config and (dataset_dir / config.images_dir / p).is_file():
                return dataset_dir / config.images_dir / p
            # Check 4: Relative to the default 'images' dir?
            if (dataset_dir / "images" / p).is_file():
                return dataset_dir / "images" / p
            
            # Default fallback (will throw a missing file warning later if incorrect)
            if config:
                return dataset_dir / config.images_dir / p
            return dataset_dir / "images" / p

        if isinstance(annotations, list):
            for entry in annotations:
                if isinstance(entry, dict) and 'image_path' in entry and 'latex' in entry:
                    samples.append({
                        'image_path': str(resolve_img_path(entry['image_path'])),
                        'latex': entry['latex']
                    })

        elif isinstance(annotations, dict):
            if "samples" in annotations and isinstance(annotations["samples"], list):
                return self._extract_samples(annotations["samples"], dataset_dir, config)

            for key, value in annotations.items():
                if key in ("meta", "config", "info", "metadata"):
                    continue

                if isinstance(value, str):
                    samples.append({
                        'image_path': str(resolve_img_path(key)),
                        'latex': value
                    })
                elif isinstance(value, dict) and 'latex' in value:
                    samples.append({
                        'image_path': str(resolve_img_path(key)),
                        'latex': value['latex']
                    })

        return samples

    def _validate_cross_dataset_consistency(self, dataset_names: List[str]):
        pass



   
    def pull_verified_dataset(self, dataset_name: str, repo_id: str) -> bool:
        try:
            from datasets import load_dataset
            logger.info(f"Pulling verified dataset from Hugging Face: {repo_id}...")
            
            dataset_dir = self.data_dir / dataset_name
            img_dir = dataset_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            
            hf_ds = load_dataset(repo_id, split="train", token=getattr(self.config, 'hf_token', None))
            
            valid_samples = []
            logger.info(f"Extracting {len(hf_ds)} verified samples to disk...")
            
            for idx, item in enumerate(hf_ds):
                try:
                    img = item.get('image')
                    latex = item.get('latex')
                    if img is None or not latex: continue

                    h = hashlib.md5(latex.encode('utf-8')).hexdigest()[:8]
                    img_path = img_dir / f"img_{idx}_{h}.png"
                    
                    if not img_path.exists():
                        # --- THE FIX: Wrap the save in a try/except ---
                        try:
                            img.save(img_path)
                        except Exception:
                            # Skip corrupt images on the HF Hub (like index 130,000)
                            continue
                    
                    valid_samples.append({'image_path': str(img_path.resolve()), 'latex': latex})
                except Exception:
                    continue
                
                if (idx + 1) % 10000 == 0:
                    logger.info(f"  Processed {idx + 1}/{len(hf_ds)} samples...")
            
            annot_file = dataset_dir / "annotations.json"
            with open(annot_file, 'w', encoding='utf-8') as f:
                json.dump(valid_samples, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Successfully pulled verified dataset '{dataset_name}'")
            return True # Logic: If we pulled data, we are ready.
            
        except Exception as e:
            logger.error(f"Failed to pull verified dataset {repo_id}: {e}")
            return False
   
   
   
    def push_dataset_to_hf(self, dataset_name: str, dataset_dir: Path):
        token = getattr(self.config, 'hf_token', None) or os.getenv("HF_TOKEN")
        if not token:
            logger.info("No Hugging Face token provided. Skipping push of verified dataset.")
            return

        try:
            from datasets import Dataset, Image as HFImage
            from huggingface_hub import HfApi
            
            annot_file = dataset_dir / "annotations.json"
            if not annot_file.exists():
                return
                
            with open(annot_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            api = HfApi(token=token)
            try:
                username = api.whoami()["name"]
            except Exception:
                # Token invalid / network issue: don't crash training or preparation.
                logger.warning("Hugging Face auth check failed. Skipping push of verified dataset.")
                return

            base_repo = getattr(self.config, 'hf_dataset_repo', 'Verified-Datasets')
            if "/" not in base_repo:
                base_repo = f"{username}/{base_repo}"

            repo_id = f"{base_repo}-{dataset_name}"
            
            logger.info(f"Checking Hugging Face repository: {repo_id}")
            
            try:
                api.repo_info(repo_id, repo_type="dataset")
                logger.info(f"Repo {repo_id} already exists. Skipping push to avoid overwriting.")
                return 
            except Exception:
                pass
                
            logger.info(f"Uploading verified dataset '{dataset_name}' to {repo_id}...")

            # Use generator to avoid loading all paths/strings into RAM at once.
            def gen():
                if isinstance(annotations, dict) and "samples" in annotations:
                    items = annotations["samples"]
                else:
                    items = annotations
                for ann in items:
                    try:
                        img_path = str(ann["image_path"])
                        latex = ann["latex"]
                    except Exception:
                        continue
                    if img_path and latex and os.path.exists(img_path):
                        yield {"image": img_path, "latex": latex}

            hf_ds = Dataset.from_generator(gen).cast_column("image", HFImage())
            if len(hf_ds) == 0:
                logger.warning("No valid samples found to push.")
                return

            hf_ds.push_to_hub(repo_id, private=True, token=token)
            logger.info(f"Successfully pushed verified dataset '{dataset_name}' to HF Hub!")
            
        except ImportError:
            logger.warning("Libraries 'datasets' or 'huggingface_hub' missing. Cannot push dataset.")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hugging Face: {e}")

    def try_download_and_validate(self, dataset_name: str) -> bool:
        config = self.registry.get_config(dataset_name)
        if not config:
            logger.error(f"Unknown dataset: {dataset_name}. Cannot download.")
            return False

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=getattr(self.config, 'hf_token', None))
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
                for req_dir in config.required_directories:
                    (dataset_dir / req_dir).mkdir(parents=True, exist_ok=True)
                
                annot_file = dataset_dir / config.annotations_file
                valid_samples = []
                
                for idx, s in enumerate(samples):
                    img = s.get('image')
                    latex = s.get('latex', '')
                    
                    if not latex:
                        continue
                        
                    if isinstance(img, str):
                        valid_samples.append({'image_path': img, 'latex': latex})
                    else:
                        img_dir = dataset_dir / (config.images_dir if config.images_dir else "images")
                        img_dir.mkdir(parents=True, exist_ok=True)
                        
                        import hashlib
                        h = hashlib.md5(latex.encode('utf-8')).hexdigest()[:8]
                        img_filename = f"img_{idx}_{h}.png"
                        img_path = img_dir / img_filename
                        
                        if not img_path.exists():
                            img.save(img_path)
                            
                        valid_samples.append({'image_path': str(img_path.resolve()), 'latex': latex})
                
                with open(annot_file, 'w', encoding='utf-8') as f:
                    json.dump(valid_samples, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"Saved {len(valid_samples)} mapped samples to {annot_file}")
            else:
                logger.error(f"DataManager returned 0 samples for {dataset_name}. Check download logs.")
                return False

            self.result = ValidationResult()
            self._validate_data_directory()
            self._validate_single_dataset(dataset_name)
            
            is_ready = self.result.dataset_statuses.get(dataset_name, {}).get('ready_for_training', False)
            
            if is_ready:
                self.push_dataset_to_hf(dataset_name, dataset_dir)
                
            return is_ready

        except Exception as e:
            logger.error(f"Download/validation failed for {dataset_name}: {e}")
            return False


def validate_datasets(config, force: bool = False) -> ValidationResult:
    validator = DatasetValidator(config)
    result = validator.validate_all()

    if not result.is_valid and not force:
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

validate_before_training = validate_datasets