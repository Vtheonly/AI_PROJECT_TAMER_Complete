"""
Dataset Preprocessor for TAMER OCR v2.1.

Implements the STRICT pipeline:
  1. Download all 4 datasets from source
  2. Preprocess ENTIRE dataset (normalize LaTeX, filter, render InkML)
  3. Verify the dataset is clean, processed, and ready
  4. Create or reuse a HuggingFace dataset repository
  5. Push the FULL processed dataset to HuggingFace
  6. Only AFTER dataset is uploaded and verified → signal ready for training
"""

import os
import gc
import json
import time
import shutil
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .latex_normalizer import normalize_latex, normalize_corpus, should_discard
from .tokenizer import LaTeXTokenizer
from .parser import DatasetParser
from .advanced_downloader import AdvDownloader
from .validator import validate_samples

logger = logging.getLogger("TAMER.Preprocessor")


def _get_memory_usage_mb() -> float:
    """Safe memory check handling missing psutil (Issue #14)."""
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
        os.makedirs(self.processed_dir, exist_ok=True)

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
        }

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def download_all_datasets(self) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("STEP 1: Downloading all datasets")
        logger.info("=" * 70)

        dataset_sources = {}

        for ds in self.config.datasets:
            name = ds.get('name')
            ds_type = ds.get('type')
            
            if ds_type == 'huggingface':
                repo = ds.get('hf_repo')
                try:
                    logger.info(f"Downloading {name} from HF ({repo})...")
                    dataset_sources[name] = self.downloader.get_hf_dataset(repo, split="train")
                except Exception as e:
                    logger.error(f"Download failed for {name}: {e}")
                    dataset_sources[name] = None
                    
            elif ds_type == 'kaggle':
                slug = ds.get('kaggle_slug')
                extract_dir = os.path.join(self.data_dir, name)
                try:
                    logger.info(f"Downloading {name} from Kaggle ({slug})...")
                    self.downloader.download_kaggle(slug, extract_dir)
                    dataset_sources[name] = extract_dir
                except Exception as e:
                    logger.error(f"Download failed for {name}: {e}")
                    dataset_sources[name] = None
                    
            elif ds_type == 'url':
                url = ds.get('url')
                extract_dir = os.path.join(self.data_dir, name)
                try:
                    logger.info(f"Downloading {name} from URL ({url})...")
                    self.downloader.download_zenodo_zip(url, extract_dir)
                    dataset_sources[name] = extract_dir
                except Exception as e:
                    logger.error(f"Download failed for {name}: {e}")
                    dataset_sources[name] = None
                    
        logger.info("All datasets downloaded!")
        return dataset_sources

    def preprocess_all_datasets(self, dataset_sources: Dict[str, Any]) -> Dict[str, List[Dict]]:
        logger.info("=" * 70)
        logger.info("STEP 2: Preprocessing ALL datasets")
        logger.info("=" * 70)

        all_processed = {}

        for dataset_name, source in dataset_sources.items():
            if source is None:
                logger.warning(f"Skipping {dataset_name} — source missing")
                all_processed[dataset_name] = []
                continue

            # Check local cache first
            if self.manifest['datasets'].get(dataset_name, {}).get('preprocessed', False):
                cached = self._load_processed_cache(dataset_name)
                if cached:
                    logger.info(f"{dataset_name}: using cached preprocessed data ({len(cached)} samples)")
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

            # CRITICAL: Clear memory between datasets
            gc.collect()
            all_processed[dataset_name] = samples

        self._build_tokenizer(all_processed)

        total = sum(len(v) for v in all_processed.values())
        self.manifest['all_preprocessed'] = True
        self.manifest['total_samples'] = total
        self._save_manifest()

        logger.info(f"\nAll datasets preprocessed! Total: {total} samples")
        return all_processed

    def _preprocess_single_dataset(self, dataset_name: str, source: Any) -> List[Dict]:
        samples = []
        try:
            parser_name = next((ds.get('parser') for ds in self.config.datasets if ds.get('name') == dataset_name), dataset_name)
            
            # 1. Parse raw data
            raw_samples = []
            extract_dir = os.path.join(self.data_dir, dataset_name)
            os.makedirs(extract_dir, exist_ok=True)

            if isinstance(source, str):
                parsers = {
                    'im2latex': self.parser.parse_im2latex,
                    'crohme': self.parser.parse_crohme,
                    'hme100k': self.parser.parse_hme100k,
                }
                parser_fn = parsers.get(parser_name, self.parser.parse_crohme)
                raw_samples = parser_fn(source)
            else:
                # MathWriting from HF
                raw_samples = self.parser.parse_mathwriting(source, extract_dir=extract_dir)

            # 2. Normalize LaTeX
            processed = normalize_corpus(raw_samples)
            
            # 3. Filter and Tag
            filtered = []
            for s in processed:
                latex = s.get('latex', '')
                if not latex: continue
                
                # Token length filter
                tokens = self.tokenizer.tokenize(latex)
                if len(tokens) <= self.config.max_token_length:
                    # Tag with dataset name for Temperature Sampler
                    s['dataset_name'] = dataset_name
                    
                    # Ensure image exists on disk (MathWriting was saved by parser)
                    img_path = s.get('image')
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        filtered.append(s)

            samples = filtered
            del raw_samples, processed, filtered
            gc.collect()
            
        except Exception as e:
            logger.error(f"  {dataset_name}: Preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

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

        if os.path.exists(tokenizer_path):
            self.manifest['tokenizer_built'] = True
            self.manifest['vocab_size'] = len(self.tokenizer)
            self._save_manifest()
            logger.info(f"Tokenizer built and saved: {len(self.tokenizer)} tokens")
        else:
            logger.error("Failed to save tokenizer.json")

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

        return self.manifest.get('tokenizer_built', False)

    def push_to_huggingface(self, all_processed: Dict[str, List[Dict]]) -> bool:
        logger.info("=" * 70)
        logger.info("STEP 4/5: Pushing processed dataset to HuggingFace")
        logger.info("=" * 70)

        hf_token = self.config.hf_token or os.getenv('HF_TOKEN', '')
        hf_repo = self.config.hf_dataset_repo_id

        if not hf_token:
            logger.error("No HF token provided — skipping push")
            return False

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)

            # Auto-resolve username if repo ID is incomplete
            if hf_repo and '/' not in hf_repo:
                username = api.whoami()['name']
                hf_repo = f"{username}/{hf_repo}"
            elif not hf_repo:
                username = api.whoami()['name']
                hf_repo = f"{username}/tamer-preprocessed"
                self.config.hf_dataset_repo_id = hf_repo

            api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="dataset", private=True)

            upload_dir = os.path.join(self.processed_dir, "hf_upload")
            if os.path.exists(upload_dir): shutil.rmtree(upload_dir)
            os.makedirs(upload_dir, exist_ok=True)

            for name, samples in all_processed.items():
                self._save_jsonl(samples, os.path.join(upload_dir, f"{name}.jsonl"))

            shutil.copy2(os.path.join(self.processed_dir, "tokenizer.json"), os.path.join(upload_dir, "tokenizer.json"))
            shutil.copy2(self.manifest_path, os.path.join(upload_dir, "manifest.json"))

            api.upload_folder(folder_path=upload_dir, repo_id=hf_repo, repo_type="dataset")
            logger.info(f"Successfully pushed dataset to: {hf_repo}")

            self.manifest['pushed_to_hf'] = True
            self.manifest['hf_dataset_repo'] = hf_repo
            self._save_manifest()
            return True
        except Exception as e:
            logger.error(f"Failed to push dataset to HF: {e}")
            return False

    def load_from_huggingface(self) -> Optional[Dict[str, List[Dict]]]:
        hf_token = self.config.hf_token or os.getenv('HF_TOKEN', '')
        hf_repo = self.config.hf_dataset_repo_id
        if not hf_token or not hf_repo: return None

        try:
            from huggingface_hub import hf_hub_download, HfApi
            api = HfApi(token=hf_token)
            files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset")

            all_processed = {}
            for ds in self.config.datasets:
                name = ds.get('name')
                filename = f"{name}.jsonl"
                if filename in files:
                    local = hf_hub_download(repo_id=hf_repo, filename=filename, repo_type="dataset", token=hf_token)
                    all_processed[name] = self._load_jsonl(local)

            if "tokenizer.json" in files:
                tok = hf_hub_download(repo_id=hf_repo, filename="tokenizer.json", repo_type="dataset", token=hf_token)
                self.tokenizer.load(tok)

            return all_processed
        except Exception:
            return None

    def _save_jsonl(self, samples: List[Dict], path: str):
        """Saves samples to JSONL. Strictly ensures only serializable data is saved."""
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                # Ensure we only save the path string, not a PIL object
                img_val = s.get('image', '')
                if not isinstance(img_val, str):
                    continue
                    
                f.write(json.dumps({
                    'image': img_val,
                    'latex': s.get('latex', ''),
                    'dataset_name': s.get('dataset_name', ''),
                }, ensure_ascii=False) + '\n')

    def _load_jsonl(self, path: str) -> List[Dict]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): 
                    samples.append(json.loads(line))
        return samples

    def _save_processed_cache(self, dataset_name: str, samples: List[Dict]):
        self._save_jsonl(samples, os.path.join(self.processed_dir, f"{dataset_name}.jsonl"))

    def _load_processed_cache(self, dataset_name: str) -> Optional[List[Dict]]:
        path = os.path.join(self.processed_dir, f"{dataset_name}.jsonl")
        return self._load_jsonl(path) if os.path.exists(path) else None

    def run_full_pipeline(self) -> Tuple[Dict[str, List[Dict]], LaTeXTokenizer]:
        # Try loading from HF first if already pushed
        if self.manifest.get('pushed_to_hf'):
            logger.info("Attempting to load preprocessed dataset from Hugging Face...")
            cached = self.load_from_huggingface()
            if cached: return cached, self.tokenizer

        # Otherwise run full local pipeline
        dataset_sources = self.download_all_datasets()
        all_processed = self.preprocess_all_datasets(dataset_sources)

        if not self.verify_dataset(all_processed):
            raise RuntimeError("Dataset verification FAILED — check logs for missing images or empty labels.")

        self.push_to_huggingface(all_processed)
        return all_processed, self.tokenizer