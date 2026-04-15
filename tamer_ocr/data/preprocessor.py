"""
Dataset Preprocessor for TAMER OCR v2.3.

v2.3 Changes:
  - FIXED: Zip/push/pull path safety completely rewritten.
    All paths stored in JSONL files are now RELATIVE to data_dir.
    On pull, paths are reconstructed using data_dir as the base.
    No more hardcoded directory name heuristics.

  - FIXED: Post-extraction verification. After unzipping, every image
    path in every JSONL is checked. Samples with missing images are
    dropped with a warning, not silently kept.

  - FIXED: Zip creation now uses sorted, deterministic file ordering
    and verifies the archive with testzip() before pushing.

  - FIXED: data_manager.py cache files also use relative paths now.
    (See data_manager.py changes.)
"""

import os
import gc
import json
import time
import shutil
import logging
import zipfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .latex_normalizer import normalize_latex, normalize_corpus, should_discard, get_complexity
from .tokenizer import LaTeXTokenizer
from .parser import DatasetParser
from .advanced_downloader import AdvDownloader
from .validator import validate_samples

logger = logging.getLogger("TAMER.Preprocessor")

def _get_memory_usage_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except (ImportError, Exception):
        return 0.0

def _log_memory(context: str):
    mem = _get_memory_usage_mb()
    if mem > 0:
        logger.info(f"  Memory ({context}): {mem:.0f} MB")


def _make_relative(path: str, base_dir: str) -> str:
    """
    Convert an absolute path to a relative path based on data_dir.
    
    If the path is already relative, return as-is.
    If the path can't be made relative (different drive, etc), return basename.
    """
    if not path or not isinstance(path, str):
        return path
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        # Different drive on Windows, or other edge case
        # Fall back to just the filename
        return os.path.basename(path)


def _resolve_path(rel_path: str, base_dir: str) -> str:
    """
    Convert a relative path back to an absolute path based on data_dir.
    
    Handles:
    - Already-absolute paths (returned as-is if they exist)
    - Relative paths (joined with base_dir)
    - Paths with stale absolute prefixes from other machines
    """
    if not rel_path or not isinstance(rel_path, str):
        return rel_path

    # Case 1: It's already absolute AND exists — use it directly
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path

    # Case 2: It's relative — join with base_dir
    if not os.path.isabs(rel_path):
        candidate = os.path.join(base_dir, rel_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        # Try normalizing separators (Windows/Linux cross-compat)
        candidate = os.path.join(base_dir, rel_path.replace('\\', '/'))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # Case 3: It's absolute but from a DIFFERENT machine (stale Kaggle/Colab path)
    # Try to extract the relative part by finding common directory names
    if os.path.isabs(rel_path):
        parts = Path(rel_path).parts
        # Walk backwards through the path, trying each suffix as a relative path
        for i in range(len(parts)):
            suffix = os.path.join(*parts[i:])
            candidate = os.path.join(base_dir, suffix)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    # Case 4: Nothing worked — return best-effort path (will fail at load time)
    return os.path.join(base_dir, rel_path)


class DatasetPreprocessor:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.parser = DatasetParser()
        self.downloader = AdvDownloader(config)
        self.tokenizer = LaTeXTokenizer()

        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.image_dir = os.path.join(self.processed_dir, "images")
        
        os.makedirs(self.image_dir, exist_ok=True)

        self.manifest_path = os.path.join(self.processed_dir, "manifest.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "datasets": {},
            "all_preprocessed": False,
            "pushed_to_hf": False,
            "tokenizer_built": False,
            "archived": False
        }

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------------
    # PULL FROM HUGGINGFACE
    # ----------------------------------------------------------------

    def pull_from_huggingface(self) -> bool:
        """
        Download and extract the processed dataset archive from HuggingFace.
        
        After extraction, ALL image paths in JSONL files are resolved
        against self.data_dir with full verification.
        """
        hf_repo = self.config.hf_dataset_repo_id
        hf_token = self.config.hf_token

        if not hf_repo or not hf_token:
            return False

        logger.info(f"🔍 Checking Hugging Face for processed image archive: {hf_repo}")
        try:
            from huggingface_hub import hf_hub_download
            zip_path = hf_hub_download(
                repo_id=hf_repo,
                filename="processed_images.zip",
                repo_type="dataset",
                token=hf_token
            )
            
            logger.info("📦 Archive found! Extracting...")
            
            # Verify zip integrity BEFORE extracting
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                bad_file = zip_ref.testzip()
                if bad_file is not None:
                    logger.error(f"❌ Corrupt file in archive: {bad_file}")
                    return False
                    
                zip_ref.extractall(self.data_dir)
            
            logger.info("📂 Extraction complete. Resolving paths...")
            
            # Load tokenizer
            tok_path = os.path.join(self.processed_dir, "tokenizer.json")
            if os.path.exists(tok_path):
                self.tokenizer.load(tok_path)
            
            self.manifest = self._load_manifest()
            logger.info("✅ Recovery complete. All images and metadata restored.")
            return True
            
        except Exception as e:
            logger.info(f"ℹ️ Cloud archive not available ({e}). Fresh processing required.")
            return False

    def _load_and_resolve_all_jsonl(self) -> Dict[str, List[Dict]]:
        """
        Load ALL JSONL files from processed_dir and resolve every image path.
        
        Returns dict of {dataset_name: [resolved_samples]}.
        Drops samples where the image file doesn't exist after resolution.
        """
        all_processed = {}
        total_loaded = 0
        total_missing = 0
        
        for jsonl_path in sorted(Path(self.processed_dir).glob("*.jsonl")):
            dataset_name = jsonl_path.stem
            raw_samples = self._load_jsonl(str(jsonl_path))
            
            resolved = []
            missing = 0
            
            for s in raw_samples:
                img = s.get('image') or s.get('image_path', '')
                
                if not img:
                    missing += 1
                    continue
                
                # Resolve path against data_dir
                resolved_path = _resolve_path(img, self.data_dir)
                
                if os.path.exists(resolved_path):
                    s['image'] = resolved_path
                    s.pop('image_path', None)
                    resolved.append(s)
                else:
                    missing += 1
            
            if missing > 0:
                logger.warning(
                    f"  {dataset_name}: {missing}/{len(raw_samples)} samples had missing images "
                    f"(dropped). {len(resolved)} samples OK."
                )
            else:
                logger.info(f"  {dataset_name}: {len(resolved)} samples loaded, all images verified ✓")
            
            all_processed[dataset_name] = resolved
            total_loaded += len(resolved)
            total_missing += missing
        
        logger.info(
            f"Total: {total_loaded} samples loaded, {total_missing} dropped (missing images)"
        )
        return all_processed

    # ----------------------------------------------------------------
    # PUSH TO HUGGINGFACE
    # ----------------------------------------------------------------

    def push_to_huggingface(self, all_processed: Dict[str, List[Dict]]) -> bool:
        """
        Zip the entire data directory and push to HuggingFace.
        
        Safety measures:
        1. All JSONL paths are made relative BEFORE zipping
        2. Zip file is verified with testzip() before pushing
        3. File list is sorted for deterministic ordering
        """
        hf_token = self.config.hf_token
        hf_repo = self.config.hf_dataset_repo_id

        if not hf_token or not hf_repo:
            logger.warning("⚠️ HF credentials missing. Skipping cloud archive push.")
            return False

        # Step 1: Re-save all JSONL with RELATIVE paths to ensure portability
        logger.info("📝 Ensuring all JSONL files use relative paths...")
        for dataset_name, samples in all_processed.items():
            self._save_processed_cache(dataset_name, samples)

        # Step 2: Create the zip archive
        zip_filename = os.path.join(self.data_dir, "processed_images.zip")
        logger.info(f"🤐 Zipping ALL datasets and metadata into {zip_filename}...")
        
        # Collect all files first, sort for determinism
        files_to_zip = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file == "processed_images.zip": 
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, self.data_dir)
                files_to_zip.append((file_path, arcname))
        
        files_to_zip.sort(key=lambda x: x[1])  # Sort by archive name
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in files_to_zip:
                zipf.write(file_path, arcname)
        
        # Step 3: Verify the zip before pushing
        logger.info("🔍 Verifying zip integrity...")
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            bad_file = zipf.testzip()
            if bad_file is not None:
                logger.error(f"❌ Zip verification failed! Corrupt file: {bad_file}")
                return False
            file_count = len(zipf.namelist())
            zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
            logger.info(f"✅ Zip verified: {file_count} files, {zip_size_mb:.1f} MB")

        # Step 4: Push to HuggingFace
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="dataset", private=True)

            logger.info(f"📤 Pushing archive to Hugging Face: {hf_repo}...")
            api.upload_file(
                path_or_fileobj=zip_filename,
                path_in_repo="processed_images.zip",
                repo_id=hf_repo,
                repo_type="dataset"
            )
            
            self.manifest['pushed_to_hf'] = True
            self.manifest['archived'] = True
            self._save_manifest()
            logger.info("✅ Cloud Sync Complete. Archive stored safely.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to push archive to HF: {e}")
            return False

    # ----------------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------------

    def run_full_pipeline(self, force_refresh: bool = False) -> Tuple[Dict[str, List[Dict]], LaTeXTokenizer]:
        """
        Run the complete data pipeline:
        1. Try pulling from HuggingFace first (skip download+preprocess if available)
        2. Otherwise download → preprocess → push
        
        Returns (all_processed, tokenizer)
        """
        if not force_refresh and self.pull_from_huggingface():
            # Load and resolve ALL paths from JSONL files
            all_processed = self._load_and_resolve_all_jsonl()
            
            if not all_processed:
                logger.warning("Pull succeeded but no valid samples found. Falling through to fresh processing.")
            else:
                return all_processed, self.tokenizer

        dataset_sources = self.download_all_datasets()
        all_processed = self.preprocess_all_datasets(dataset_sources)

        if not self.verify_dataset(all_processed):
            raise RuntimeError("Dataset verification failed after local processing.")

        self.push_to_huggingface(all_processed)
        return all_processed, self.tokenizer

    # ----------------------------------------------------------------
    # DOWNLOAD
    # ----------------------------------------------------------------

    def download_all_datasets(self) -> Dict[str, Any]:
        logger.info("STEP 1: Downloading raw datasets (Zenodo, Kaggle, HF)")
        dataset_sources = {}
        for ds in self.config.datasets:
            name, ds_type = ds.get('name'), ds.get('type')
            try:
                if ds_type == 'huggingface':
                    dataset_sources[name] = self.downloader.get_hf_dataset(ds.get('hf_repo'), split="train")
                elif ds_type == 'kaggle':
                    path = os.path.join(self.data_dir, name)
                    self.downloader.download_kaggle(ds.get('kaggle_slug'), path)
                    dataset_sources[name] = path
                elif ds_type == 'url':
                    path = os.path.join(self.data_dir, name)
                    self.downloader.download_zenodo_zip(ds.get('url'), path)
                    dataset_sources[name] = path
            except Exception as e:
                logger.error(f"Download failed for {name}: {e}")
                dataset_sources[name] = None
        return dataset_sources

    # ----------------------------------------------------------------
    # PREPROCESS
    # ----------------------------------------------------------------

    def preprocess_all_datasets(self, dataset_sources: Dict[str, Any]) -> Dict[str, List[Dict]]:
        logger.info("STEP 2: Processing and Rendering Datasets")
        all_processed = {}
        
        for dataset_name, source in dataset_sources.items():
            if source is None: continue
            
            logger.info(f"--- Preprocessing: {dataset_name} ---")
            _log_memory(f"start {dataset_name}")

            raw_samples = []
            if isinstance(source, str):
                if dataset_name == 'crohme': raw_samples = self.parser.parse_crohme(source)
                elif dataset_name == 'hme100k': raw_samples = self.parser.parse_hme100k(source)
                elif dataset_name == 'im2latex': raw_samples = self.parser.parse_im2latex(source)
                else: raw_samples = self.parser.parse_crohme(source)
            else:
                raw_samples = self.parser.parse_mathwriting(source, extract_dir=self.processed_dir)

            processed = normalize_corpus(raw_samples)
            
            valid_samples = []
            for s in processed:
                latex = s.get('latex', '')
                if not latex: continue
                if len(self.tokenizer.tokenize(latex)) > self.config.max_token_length:
                    continue
                s['dataset_name'] = dataset_name
                
                # Verify image exists
                img = s.get('image') or s.get('image_path', '')
                if isinstance(img, str) and os.path.exists(img):
                    # Normalize to 'image' key with absolute path
                    s['image'] = os.path.abspath(img)
                    s.pop('image_path', None)
                    valid_samples.append(s)

            all_processed[dataset_name] = valid_samples
            self._save_processed_cache(dataset_name, valid_samples)
            
            self.manifest['datasets'][dataset_name] = {'preprocessed': True, 'count': len(valid_samples)}
            _log_memory(f"end {dataset_name}")
            gc.collect()

        logger.info("Building Global Tokenizer...")
        flat_list = [s for sublist in all_processed.values() for s in sublist]
        self.tokenizer.build_from_samples(flat_list)
        self.tokenizer.save(os.path.join(self.processed_dir, "tokenizer.json"))
        
        self.manifest['all_preprocessed'] = True
        self.manifest['vocab_size'] = len(self.tokenizer)
        self._save_manifest()
        
        return all_processed

    def verify_dataset(self, all_processed: Dict[str, List[Dict]]) -> bool:
        for name, samples in all_processed.items():
            if not samples: return False
            if not validate_samples(samples, max_check=50)['is_ok']: return False
        return True

    # ----------------------------------------------------------------
    # JSONL I/O — The critical path safety layer
    # ----------------------------------------------------------------

    def _save_jsonl(self, samples: List[Dict], path: str):
        """Write samples to a JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Read samples from a JSONL file."""
        samples = []
        if not os.path.exists(path): return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): samples.append(json.loads(line))
        return samples

    def _save_processed_cache(self, dataset_name: str, samples: List[Dict]):
        """
        Save processed samples to a JSONL file with RELATIVE paths.
        
        This is the CRITICAL function for zip/push/pull safety.
        All image paths are converted to paths relative to self.data_dir.
        When loaded on another machine, _resolve_path() converts them back.
        """
        path = os.path.join(self.processed_dir, f"{dataset_name}.jsonl")
        rel_samples = []
        
        for s in samples:
            s2 = dict(s)
            img = s2.get('image') or s2.get('image_path', '')
            
            if isinstance(img, str) and img:
                # Always store as relative path from data_dir
                s2['image'] = _make_relative(img, self.data_dir)
                s2.pop('image_path', None)
            
            rel_samples.append(s2)
        
        self._save_jsonl(rel_samples, path)
        logger.debug(f"Saved {len(rel_samples)} samples to {path} (all paths relative)")