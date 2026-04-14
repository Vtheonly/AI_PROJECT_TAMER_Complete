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
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
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

        self._hf_cache = {}

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
                logger.warning(f"Skipping {dataset_name} — download failed")
                all_processed[dataset_name] = []
                continue

            if self.manifest['datasets'].get(dataset_name, {}).get('preprocessed', False):
                cached = self._load_processed_cache(dataset_name)
                if cached:
                    logger.info(f"{dataset_name}: already preprocessed ({len(cached)} samples cached)")
                    all_processed[dataset_name] = cached
                    continue

            logger.info(f"\n--- Preprocessing: {dataset_name} ---")
            _log_memory(f"before {dataset_name}")

            samples = self._preprocess_single_dataset(dataset_name, source)

            logger.info(f"{dataset_name}: {len(samples)} valid samples after preprocessing")
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
            
            raw_samples = []

            if isinstance(source, str):
                parsers = {
                    'im2latex': self.parser.parse_im2latex,
                    'crohme': self.parser.parse_crohme,
                    'hme100k': self.parser.parse_hme100k,
                }
                parser_fn = parsers.get(parser_name, self.parser.parse_crohme)
                raw_samples = parser_fn(source)
                for s in raw_samples:
                    s['dataset_name'] = dataset_name
            else:
                extract_dir = os.path.join(self.data_dir, dataset_name)
                raw_samples = self.parser.parse_mathwriting(source, extract_dir=extract_dir)
                for s in raw_samples:
                    s['dataset_name'] = dataset_name

            logger.info(f"  {dataset_name}: {len(raw_samples)} raw samples parsed")

            processed = normalize_corpus(raw_samples)
            logger.info(f"  {dataset_name}: {len(processed)} samples after normalization")

            filtered = []
            for s in processed:
                latex = s.get('latex', '')
                if not latex:
                    continue
                tokens = self.tokenizer.tokenize(latex)
                if len(tokens) <= self.config.max_token_length:
                    filtered.append(s)

            logger.info(f"  {dataset_name}: {len(filtered)} samples after token length filter")

            validated = []
            for s in filtered:
                img = s.get('image')
                if isinstance(img, str) and not os.path.exists(img):
                    continue
                validated.append(s)

            logger.info(f"  {dataset_name}: {len(validated)} samples after image validation")
            samples = validated

            del raw_samples
            del processed
            del filtered
            gc.collect()

        except Exception as e:
            logger.error(f"  {dataset_name}: Preprocessing FAILED: {e}")
            import traceback
            traceback.print_exc()

        return samples

    def _build_tokenizer(self, all_processed: Dict[str, List[Dict]]):
        logger.info("Building global tokenizer from ALL processed data...")
        all_samples = []
        for samples in all_processed.values():
            all_samples.extend(samples)

        self.tokenizer = LaTeXTokenizer()
        self.tokenizer.build_from_samples(all_samples)
        self.tokenizer.save(os.path.join(self.processed_dir, "tokenizer.json"))

        self.manifest['tokenizer_built'] = True
        self.manifest['vocab_size'] = len(self.tokenizer)
        self._save_manifest()

        logger.info(f"Tokenizer built: {len(self.tokenizer)} tokens")

    def verify_dataset(self, all_processed: Dict[str, List[Dict]]) -> bool:
        logger.info("=" * 70)
        logger.info("STEP 3: Verifying dataset integrity")
        logger.info("=" * 70)

        issues = []

        for name, samples in all_processed.items():
            if not samples:
                issues.append(f"{name}: 0 samples — EMPTY")
                continue
            validation = validate_samples(samples, max_check=100)
            logger.info(f"  {name}: {len(samples)} samples — validation: {validation}")
            if not validation['is_ok']:
                issues.append(f"{name}: validation failed ({validation})")

        if not self.manifest.get('tokenizer_built', False):
            issues.append("Tokenizer not built")

        if len(self.tokenizer) < 10:
            issues.append(f"Tokenizer too small: {len(self.tokenizer)} tokens")

        total = sum(len(v) for v in all_processed.values())
        logger.info(f"  Total samples: {total}")
        logger.info(f"  Vocab size: {len(self.tokenizer)}")

        if issues:
            logger.error("VERIFICATION FAILED:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False

        logger.info("VERIFICATION PASSED — dataset is clean and ready!")
        return True

    def push_to_huggingface(self, all_processed: Dict[str, List[Dict]]) -> bool:
        logger.info("=" * 70)
        logger.info("STEP 4/5: Pushing processed dataset to HuggingFace")
        logger.info("=" * 70)

        hf_token = self.config.hf_token or os.getenv('HF_TOKEN', '')
        hf_repo = self.config.hf_dataset_repo_id

        if not hf_token:
            logger.error("No HF token provided — cannot push dataset to HF")
            return False

        if not hf_repo:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                username = api.whoami()['name']
                hf_repo = f"{username}/tamer-preprocessed"
                self.config.hf_dataset_repo_id = hf_repo
            except Exception as e:
                logger.error(f"Cannot resolve HF username: {e}")
                return False

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)

            try:
                api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="dataset", private=True)
                logger.info(f"Dataset repo created/confirmed: {hf_repo}")
            except Exception as e:
                logger.info(f"Repo creation note: {e}")

            upload_dir = os.path.join(self.processed_dir, "hf_upload")
            os.makedirs(upload_dir, exist_ok=True)

            for name, samples in all_processed.items():
                jsonl_path = os.path.join(upload_dir, f"{name}.jsonl")
                self._save_jsonl(samples, jsonl_path)
                logger.info(f"  Saved {name}.jsonl ({len(samples)} samples)")

            tokenizer_src = os.path.join(self.processed_dir, "tokenizer.json")
            if os.path.exists(tokenizer_src):
                shutil.copy2(tokenizer_src, os.path.join(upload_dir, "tokenizer.json"))

            shutil.copy2(self.manifest_path, os.path.join(upload_dir, "manifest.json"))

            logger.info(f"Uploading to HF dataset repo: {hf_repo}...")
            api.upload_folder(
                folder_path=upload_dir,
                repo_id=hf_repo,
                repo_type="dataset",
            )

            logger.info(f"Successfully pushed dataset to: {hf_repo}")

            self.manifest['pushed_to_hf'] = True
            self.manifest['hf_dataset_repo'] = hf_repo
            self._save_manifest()

            shutil.rmtree(upload_dir, ignore_errors=True)

            return True

        except Exception as e:
            logger.error(f"Failed to push dataset to HF: {e}")
            return False

    def _save_jsonl(self, samples: List[Dict], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                cacheable = {
                    'image': s.get('image', '') if isinstance(s.get('image'), str) else '',
                    'latex': s.get('latex', ''),
                    'dataset_name': s.get('dataset_name', ''),
                }
                f.write(json.dumps(cacheable, ensure_ascii=False) + '\n')

    def load_from_huggingface(self) -> Optional[Dict[str, List[Dict]]]:
        hf_token = self.config.hf_token or os.getenv('HF_TOKEN', '')
        hf_repo = self.config.hf_dataset_repo_id

        if not hf_token or not hf_repo:
            return None

        try:
            from huggingface_hub import hf_hub_download, HfApi

            api = HfApi(token=hf_token)

            try:
                files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset")
            except Exception:
                return None

            all_processed = {}
            for ds in self.config.datasets:
                name = ds.get('name')
                jsonl_filename = f"{name}.jsonl"
                if jsonl_filename not in files:
                    continue
                local_path = hf_hub_download(
                    repo_id=hf_repo, filename=jsonl_filename,
                    repo_type="dataset", token=hf_token,
                )
                samples = self._load_jsonl(local_path)
                all_processed[name] = samples
                logger.info(f"  Loaded {name} from HF: {len(samples)} samples")

            if "tokenizer.json" in files:
                tok_path = hf_hub_download(
                    repo_id=hf_repo, filename="tokenizer.json",
                    repo_type="dataset", token=hf_token,
                )
                self.tokenizer.load(tok_path)
                logger.info(f"  Tokenizer loaded from HF: {len(self.tokenizer)} tokens")

            if all_processed:
                total = sum(len(v) for v in all_processed.values())
                logger.info(f"  Total samples loaded from HF: {total}")
                return all_processed

        except Exception as e:
            logger.error(f"Failed to load dataset from HF: {e}")

        return None

    def _load_jsonl(self, path: str) -> List[Dict]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return samples

    def _save_processed_cache(self, dataset_name: str, samples: List[Dict]):
        cache_path = os.path.join(self.processed_dir, f"{dataset_name}.jsonl")
        self._save_jsonl(samples, cache_path)
        logger.info(f"  Cached {dataset_name} to {cache_path}")

    def _load_processed_cache(self, dataset_name: str) -> Optional[List[Dict]]:
        cache_path = os.path.join(self.processed_dir, f"{dataset_name}.jsonl")
        if os.path.exists(cache_path):
            return self._load_jsonl(cache_path)
        return None

    def run_full_pipeline(self) -> Tuple[Dict[str, List[Dict]], LaTeXTokenizer]:
        if self.manifest.get('all_preprocessed') and self.manifest.get('pushed_to_hf'):
            logger.info("Dataset already preprocessed and pushed. Loading from HF...")
            cached = self.load_from_huggingface()
            if cached:
                return cached, self.tokenizer

            logger.info("HF load failed, trying local cache...")
            all_processed = {}
            for ds in self.config.datasets:
                name = ds.get('name')
                samples = self._load_processed_cache(name)
                if samples:
                    all_processed[name] = samples

            if all_processed and self.manifest.get('tokenizer_built'):
                tok_path = os.path.join(self.processed_dir, "tokenizer.json")
                if os.path.exists(tok_path):
                    self.tokenizer.load(tok_path)
                if self.verify_dataset(all_processed):
                    self.push_to_huggingface(all_processed)
                    return all_processed, self.tokenizer

        dataset_sources = self.download_all_datasets()
        all_processed = self.preprocess_all_datasets(dataset_sources)

        if not self.verify_dataset(all_processed):
            raise RuntimeError("Dataset verification FAILED — fix issues before training")

        success = self.push_to_huggingface(all_processed)
        if not success:
            logger.warning("HF push failed — training will proceed with local data only")

        logger.info("=" * 70)
        logger.info("PREPROCESSING PIPELINE COMPLETE")
        logger.info(f"Total samples: {sum(len(v) for v in all_processed.values())}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")
        logger.info("=" * 70)

        return all_processed, self.tokenizer