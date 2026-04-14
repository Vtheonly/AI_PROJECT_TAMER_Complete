"""
Dataset Preprocessor for TAMER OCR v2.2.

Implements the STRICT pipeline:
  1. Download all datasets from source (HF, Kaggle, URL)
  2. Preprocess ENTIRE dataset (normalize LaTeX, filter, render InkML via Parser)
  3. Verify clean datasets and build tokenizer
  4. Archive processed data (ZIP) and Push to HuggingFace
  5. Recovery: Can pull the ZIP from HF to skip Step 1-3 on fresh runs.
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
    """Safe memory check handling missing psutil."""
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

    # --- CLOUD SYNC LOGIC ---

    def pull_from_huggingface(self) -> bool:
        """Attempts to download and extract the processed archive to save time."""
        hf_repo = self.config.hf_dataset_repo_id
        if not hf_repo or not self.config.hf_token:
            return False

        logger.info(f"Checking Hugging Face for pre-processed archive: {hf_repo}")
        try:
            from huggingface_hub import hf_hub_download
            zip_path = hf_hub_download(
                repo_id=hf_repo,
                filename="processed_data.zip",
                repo_type="dataset",
                token=self.config.hf_token
            )
            
            logger.info("Cloud archive found. Extracting...")
            # We extract to the data_dir. The zip contains the 'processed' folder structure.
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Re-initialize state from extracted files
            tok_path = os.path.join(self.processed_dir, "tokenizer.json")
            if os.path.exists(tok_path):
                self.tokenizer.load(tok_path)
            
            self.manifest = self._load_manifest()
            logger.info("Cloud recovery complete. Ready for training.")
            return True
        except Exception as e:
            logger.info(f"Cloud archive not available ({e}). Proceeding with fresh processing.")
            return False

    def push_to_huggingface(self, all_processed: Dict[str, List[Dict]]) -> bool:
        """Packs the processed images and metadata into a ZIP and pushes to HF."""
        hf_token = self.config.hf_token
        hf_repo = self.config.hf_dataset_repo_id

        if not hf_token or not hf_repo:
            logger.warning("No HF credentials. Data will only be saved locally.")
            return False

        # 1. Create the ZIP archive
        zip_filename = os.path.join(self.data_dir, "processed_data.zip")
        logger.info(f"Archiving processed data to {zip_filename}...")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add images, jsonl files, tokenizer, and manifest
            for root, _, files in os.walk(self.processed_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.data_dir)
                    zipf.write(file_path, arcname)

        # 2. Upload to HF
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="dataset", private=True)

            logger.info(f"Pushing archive to HF: {hf_repo}")
            api.upload_file(
                path_or_fileobj=zip_filename,
                path_in_repo="processed_data.zip",
                repo_id=hf_repo,
                repo_type="dataset"
            )
            
            self.manifest['pushed_to_hf'] = True
            self.manifest['archived'] = True
            self._save_manifest()
            return True
        except Exception as e:
            logger.error(f"Failed to push to HF: {e}")
            return False

    # --- CORE PREPROCESSING LOGIC ---

    def download_all_datasets(self) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("STEP 1: Downloading all datasets")
        logger.info("=" * 70)

        dataset_sources = {}
        for ds in self.config.datasets:
            name = ds.get('name')
            ds_type = ds.get('type')
            
            try:
                if ds_type == 'huggingface':
                    repo = ds.get('hf_repo')
                    logger.info(f"Downloading {name} from HF ({repo})...")
                    dataset_sources[name] = self.downloader.get_hf_dataset(repo, split="train")
                elif ds_type == 'kaggle':
                    slug = ds.get('kaggle_slug')
                    extract_dir = os.path.join(self.data_dir, name)
                    logger.info(f"Downloading {name} from Kaggle ({slug})...")
                    self.downloader.download_kaggle(slug, extract_dir)
                    dataset_sources[name] = extract_dir
                elif ds_type == 'url':
                    url = ds.get('url')
                    extract_dir = os.path.join(self.data_dir, name)
                    logger.info(f"Downloading {name} from URL ({url})...")
                    self.downloader.download_zenodo_zip(url, extract_dir)
                    dataset_sources[name] = extract_dir
            except Exception as e:
                logger.error(f"Download failed for {name}: {e}")
                dataset_sources[name] = None
                    
        return dataset_sources

    def preprocess_all_datasets(self, dataset_sources: Dict[str, Any]) -> Dict[str, List[Dict]]:
        logger.info("=" * 70)
        logger.info("STEP 2: Preprocessing ALL datasets")
        logger.info("=" * 70)

        all_processed = {}
        for dataset_name, source in dataset_sources.items():
            if source is None:
                all_processed[dataset_name] = []
                continue

            # Check local jsonl cache
            if self.manifest['datasets'].get(dataset_name, {}).get('preprocessed', False):
                cached = self._load_processed_cache(dataset_name)
                if cached:
                    logger.info(f"{dataset_name}: using cached data ({len(cached)} samples)")
                    all_processed[dataset_name] = cached
                    continue

            logger.info(f"\n--- Preprocessing: {dataset_name} ---")
            _log_memory(f"before {dataset_name}")

            samples = self._preprocess_single_dataset(dataset_name, source)

            logger.info(f"{dataset_name}: {len(samples)} valid samples")
            _log_memory(f"after {dataset_name}")

            self._save_processed_cache(dataset_name, samples)
            self.manifest['datasets'][dataset_name] = {
                'preprocessed': True,
                'sample_count': len(samples),
            }
            self._save_manifest()
            gc.collect()
            all_processed[dataset_name] = samples

        self._build_tokenizer(all_processed)
        self.manifest['all_preprocessed'] = True
        self._save_manifest()
        return all_processed

    def _preprocess_single_dataset(self, dataset_name: str, source: Any) -> List[Dict]:
        samples = []
        try:
            # Setup image output dir for this dataset inside processed/images
            # 1. Parse raw data
            if isinstance(source, str):
                # Local directory parsing
                if dataset_name == 'crohme':
                    raw_samples = self.parser.parse_crohme(source)
                elif dataset_name == 'hme100k':
                    raw_samples = self.parser.parse_hme100k(source)
                elif dataset_name == 'im2latex':
                    raw_samples = self.parser.parse_im2latex(source)
                else:
                    raw_samples = self.parser.parse_crohme(source)
            else:
                # HuggingFace dataset object (MathWriting)
                # This also renders/saves images to disk via parser
                raw_samples = self.parser.parse_mathwriting(source, extract_dir=self.processed_dir)

            # 2. Normalize LaTeX
            processed = normalize_corpus(raw_samples)
            
            # 3. Filter
            filtered = []
            for s in processed:
                latex = s.get('latex', '')
                if not latex: continue
                
                # Tag with dataset name
                s['dataset_name'] = dataset_name
                
                # Check if image path exists (should be inside processed/images)
                img_path = s.get('image')
                if isinstance(img_path, str) and os.path.exists(img_path):
                    filtered.append(s)

            samples = filtered
            del raw_samples, processed
        except Exception as e:
            logger.error(f"Preprocessing {dataset_name} failed: {e}")
        return samples

    def _build_tokenizer(self, all_processed: Dict[str, List[Dict]]):
        logger.info("Building global tokenizer...")
        all_samples = []
        for samples in all_processed.values():
            all_samples.extend(samples)

        self.tokenizer = LaTeXTokenizer()
        self.tokenizer.build_from_samples(all_samples)
        
        tokenizer_path = os.path.join(self.processed_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        self.manifest['tokenizer_built'] = True
        self.manifest['vocab_size'] = len(self.tokenizer)

    def verify_dataset(self, all_processed: Dict[str, List[Dict]]) -> bool:
        logger.info("=" * 70)
        logger.info("STEP 3: Verifying dataset integrity")
        logger.info("=" * 70)

        for name, samples in all_processed.items():
            if not samples:
                logger.error(f"  {name}: EMPTY dataset")
                return False
            validation = validate_samples(samples, max_check=100)
            logger.info(f"  {name}: {len(samples)} samples — validation: {'OK' if validation['is_ok'] else 'FAILED'}")
            if not validation['is_ok']: return False
        return True

    # --- HELPERS ---

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
        self._save_jsonl(samples, path)

    def _load_processed_cache(self, dataset_name: str) -> Optional[List[Dict]]:
        path = os.path.join(self.processed_dir, f"{dataset_name}.jsonl")
        return self._load_jsonl(path)

    def run_full_pipeline(self) -> Tuple[Dict[str, List[Dict]], LaTeXTokenizer]:
        """Entry point that manages the entire lifecycle."""
        # 1. Try cloud recovery
        if self.pull_from_huggingface():
            all_processed = {}
            for jsonl in Path(self.processed_dir).glob("*.jsonl"):
                all_processed[jsonl.stem] = self._load_jsonl(str(jsonl))
            return all_processed, self.tokenizer

        # 2. Hard Work (Download + Process)
        sources = self.download_all_datasets()
        all_processed = self.preprocess_all_datasets(sources)

        if not self.verify_dataset(all_processed):
            raise RuntimeError("Dataset verification failed.")

        # 3. Archive & Sync
        self.push_to_huggingface(all_processed)
        
        return all_processed, self.tokenizer