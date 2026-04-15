"""
Dataset Preprocessor for TAMER OCR v2.2.

STRICT WORKFLOW:
  1. Recovery: Checks HuggingFace for 'processed_images.zip'.
     - If found: Downloads, extracts, and loads metadata. (Saves hours)
  2. Processing (if no cloud ZIP):
     - Downloads raw datasets (Kaggle, Zenodo, HF).
     - Renders InkML to PNG images.
     - Saves all images into 'data/processed/images/'.
     - Generates JSONL metadata.
  3. Archiving:
     - Zips the entire 'processed' folder (images + metadata).
     - Pushes the ZIP to HuggingFace for future session recovery.
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

from .latex_normalizer import normalize_latex, normalize_corpus, should_discard
from .tokenizer import LaTeXTokenizer
from .parser import DatasetParser
from .advanced_downloader import AdvDownloader
from .validator import validate_samples

logger = logging.getLogger("TAMER.Preprocessor")

def _get_memory_usage_mb() -> float:
    """Safe memory check to prevent OOM during large image processing."""
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

class DatasetPreprocessor:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.parser = DatasetParser()
        self.downloader = AdvDownloader(config)
        self.tokenizer = LaTeXTokenizer()

        # Define paths for the processed output
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.image_dir = os.path.join(self.processed_dir, "images")
        
        # Ensure directories exist
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

    # ============================================================
    # CLOUD SYNC: ZIP & HF STORAGE
    # ============================================================

    def pull_from_huggingface(self) -> bool:
        """Downloads the image archive from HF to skip all processing steps."""
        hf_repo = self.config.hf_dataset_repo_id
        hf_token = self.config.hf_token

        if not hf_repo or not hf_token:
            return False

        logger.info(f"🔍 Checking Hugging Face for processed image archive: {hf_repo}")
        try:
            from huggingface_hub import hf_hub_download
            
            # Download the ZIP
            zip_path = hf_hub_download(
                repo_id=hf_repo,
                filename="processed_images.zip",
                repo_type="dataset",
                token=hf_token
            )
            
            logger.info("📦 Archive found! Extracting images and metadata...")
            # Extract to data_dir so that the 'processed' folder is recreated
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Load the extracted tokenizer
            tok_path = os.path.join(self.processed_dir, "tokenizer.json")
            if os.path.exists(tok_path):
                self.tokenizer.load(tok_path)
            
            self.manifest = self._load_manifest()
            logger.info("✅ Recovery complete. All images and metadata restored.")
            return True
        except Exception as e:
            logger.info(f"ℹ️ Cloud archive not available ({e}). Fresh processing required.")
            return False

    def push_to_huggingface(self, all_processed: Dict[str, List[Dict]]) -> bool:
        """Zips the processed/images folder and metadata, then pushes to HF."""
        hf_token = self.config.hf_token
        hf_repo = self.config.hf_dataset_repo_id

        if not hf_token or not hf_repo:
            logger.warning("⚠️ HF credentials missing. Skipping cloud archive push.")
            return False

        # 1. Create the ZIP archive of the entire processed folder
        zip_filename = os.path.join(self.data_dir, "processed_images.zip")
        logger.info(f"🤐 Zipping processed images and metadata into {zip_filename}...")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.processed_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Maintain the folder structure relative to the data_dir
                    arcname = os.path.relpath(file_path, self.data_dir)
                    zipf.write(file_path, arcname)

        # 2. Upload the ZIP to HF
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="dataset", private=True)

            logger.info(f"📤 Pushing 1.5GB+ archive to Hugging Face: {hf_repo}...")
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

    # ============================================================
    # CORE PROCESSING LOGIC
    # ============================================================

    def run_full_pipeline(self) -> Tuple[Dict[str, List[Dict]], LaTeXTokenizer]:
        if self.pull_from_huggingface():
            all_processed = {}
            for jsonl in Path(self.processed_dir).glob("*.jsonl"):
                raw = self._load_jsonl(str(jsonl))
                # Re-expand relative paths to absolute using current data_dir
                for s in raw:
                    img = s.get('image') or s.get('image_path')
                    if isinstance(img, str) and not os.path.isabs(img):
                        s['image'] = os.path.join(self.data_dir, img)
                all_processed[jsonl.stem] = raw
            return all_processed, self.tokenizer
        # ... rest unchanged

        # 2. Local fallback: If no cloud ZIP, we must download and render everything
        dataset_sources = self.download_all_datasets()
        all_processed = self.preprocess_all_datasets(dataset_sources)

        if not self.verify_dataset(all_processed):
            raise RuntimeError("Dataset verification failed after local processing.")

        # 3. Save everything and push to cloud so we don't do this again
        self.push_to_huggingface(all_processed)
        
        return all_processed, self.tokenizer

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

    def preprocess_all_datasets(self, dataset_sources: Dict[str, Any]) -> Dict[str, List[Dict]]:
        logger.info("STEP 2: Processing and Rendering Datasets")
        all_processed = {}
        
        for dataset_name, source in dataset_sources.items():
            if source is None: continue
            
            logger.info(f"--- Preprocessing: {dataset_name} ---")
            _log_memory(f"start {dataset_name}")

            # 1. Parse/Render using Parser (Saves PNGs to processed/images/)
            raw_samples = []
            if isinstance(source, str):
                if dataset_name == 'crohme': raw_samples = self.parser.parse_crohme(source)
                elif dataset_name == 'hme100k': raw_samples = self.parser.parse_hme100k(source)
                elif dataset_name == 'im2latex': raw_samples = self.parser.parse_im2latex(source)
                else: raw_samples = self.parser.parse_crohme(source)
            else:
                # MathWriting from HF - saves PNGs into processed_dir
                raw_samples = self.parser.parse_mathwriting(source, extract_dir=self.processed_dir)

            # 2. Normalize and Filter
            processed = normalize_corpus(raw_samples)
            
            valid_samples = []
            for s in processed:
                latex = s.get('latex', '')
                if not latex: continue
                
                # Token length check
                if len(self.tokenizer.tokenize(latex)) > self.config.max_token_length:
                    continue
                
                s['dataset_name'] = dataset_name
                
                # Verify that the image was actually created on disk
                if isinstance(s.get('image'), str) and os.path.exists(s['image']):
                    valid_samples.append(s)

            all_processed[dataset_name] = valid_samples
            self._save_processed_cache(dataset_name, valid_samples)
            
            self.manifest['datasets'][dataset_name] = {'preprocessed': True, 'count': len(valid_samples)}
            _log_memory(f"end {dataset_name}")
            gc.collect()

        # 3. Finalize Tokenizer
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

    def _save_jsonl(self, samples: List[Dict], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    def _load_jsonl(self, path: str) -> List[Dict]:
        samples = []
        if not os.path.exists(path): return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): samples.append(json.loads(line))
        return samples

    def _save_processed_cache(self, dataset_name: str, samples: List[Dict]):
        path = os.path.join(self.processed_dir, f"{dataset_name}.jsonl")
        rel_samples = []
        for s in samples:
            s2 = dict(s)
            img = s2.get('image') or s2.get('image_path')
            if isinstance(img, str) and os.path.isabs(img):
                try:
                    # Store path relative to data_dir — portable across environments
                    s2['image'] = os.path.relpath(img, self.data_dir)
                    s2.pop('image_path', None)
                except ValueError:
                    pass  # Windows cross-drive edge case
            rel_samples.append(s2)
        self._save_jsonl(rel_samples, path)    