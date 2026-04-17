"""
TAMER OCR - 100% OFFLINE Preprocessor for Kaggle
Reads directly from the uploaded 'processed/' folder.
Bypasses ALL network calls (HuggingFace, Kaggle API, Zenodo).

KEY FIX: manifest.json is NEVER written to /kaggle/input/ (read-only).
All manifest operations are skipped silently in offline mode.
"""

import os
import gc
import json
import logging
import zipfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .latex_normalizer import normalize_latex, normalize_corpus, should_discard, get_complexity
from .tokenizer import LaTeXTokenizer
from .parser import DatasetParser
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
    If the path cannot be made relative, return basename.
    """
    if not path or not isinstance(path, str):
        return path
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return os.path.basename(path)


def _resolve_path(rel_path: str, base_dir: str) -> str:
    """
    Convert a relative or stale-absolute path to a valid absolute path.

    Strategy (in order):
      1. Already absolute AND exists on this machine  -> use it directly.
      2. Relative -> join with base_dir and check both slash variants.
      3. Absolute but stale (from a different machine) -> walk path
         suffixes against base_dir until one resolves.
      4. Nothing worked -> return best-effort join (will fail at image load).
    """
    if not rel_path or not isinstance(rel_path, str):
        return rel_path

    
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path

    
    if not os.path.isabs(rel_path):
        candidate = os.path.join(base_dir, rel_path.replace('\\', '/'))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        candidate2 = os.path.join(base_dir, rel_path)
        if os.path.exists(candidate2):
            return os.path.abspath(candidate2)

    
    if os.path.isabs(rel_path):
        parts = Path(rel_path).parts
        for i in range(len(parts)):
            suffix = os.path.join(*parts[i:])
            candidate = os.path.join(base_dir, suffix)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    
    return os.path.join(base_dir, rel_path)


class DatasetPreprocessor:
    """
    Offline-only preprocessor for Kaggle.

    Expects that processed/ folder already contains:
      - tokenizer.json
      - one or more *.jsonl files (one per dataset)
      - the actual image files referenced by those JSONL files

    run_full_pipeline() simply loads these files and resolves image paths.
    No network calls are made under any circumstance.

    CRITICAL: The processed/ folder lives inside /kaggle/input/ which is
    READ-ONLY. This class NEVER attempts to write anything to that folder.
    The manifest is kept in memory only.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.parser = DatasetParser()
        self.tokenizer = LaTeXTokenizer()

        self.processed_dir = os.path.join(self.data_dir, "processed")

        
        
        self.image_dir = os.path.join(self.processed_dir, "images")

        
        
        self.manifest_path = os.path.join(self.processed_dir, "manifest.json")
        self.manifest = self._load_manifest()

    
    
    
    
    
    
    

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest from disk (read-only safe). Returns empty dict if absent."""
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
            "archived": False,
        }

    def _save_manifest(self):
        """
        Attempt to save manifest to disk.

        In Kaggle offline mode the processed/ folder is inside /kaggle/input/
        which is strictly read-only. Writing there raises:
            OSError: [Errno 30] Read-only file system

        This method catches that error silently so training is never interrupted
        by a manifest write. The manifest is kept up-to-date in self.manifest
        (in memory) for the duration of the run.
        """
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        except OSError as e:
            
            logger.debug(
                f"Manifest write skipped (read-only filesystem): {e}"
            )
        except Exception as e:
            
            logger.warning(f"Manifest write failed (non-fatal): {e}")

    
    
    

    def _save_jsonl(self, samples: List[Dict], path: str):
        """Write samples to a JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Read samples from a JSONL file."""
        samples = []
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _save_processed_cache(self, dataset_name: str, samples: List[Dict]):
        """
        Save processed samples to JSONL using RELATIVE image paths.
        Relative to self.data_dir for portability across machines.

        In offline Kaggle mode this is only called by preprocess_all_datasets()
        which itself is never called (we go straight to _load_and_resolve_all_jsonl).
        Kept here for completeness and future local-machine use.
        """
        path = os.path.join(self.processed_dir, f"{dataset_name}.jsonl")
        rel_samples = []
        for s in samples:
            s2 = dict(s)
            img = s2.get('image') or s2.get('image_path', '')
            if isinstance(img, str) and img:
                s2['image'] = _make_relative(img, self.data_dir)
                s2.pop('image_path', None)
            rel_samples.append(s2)
        self._save_jsonl(rel_samples, path)
        logger.debug(f"Saved {len(rel_samples)} samples to {path} (paths relative)")

    
    
    

    def _load_and_resolve_all_jsonl(self) -> Dict[str, List[Dict]]:
        """
        Load every *.jsonl from processed_dir and resolve image paths.

        Drops samples whose image file cannot be found after resolution.
        Returns {dataset_name: [resolved_sample, ...]}

        This is the ONLY data-loading path used in Kaggle offline mode.
        It is entirely read-only — it never writes anything to disk.
        """
        all_processed = {}
        total_loaded = 0
        total_missing = 0

        jsonl_files = sorted(Path(self.processed_dir).glob("*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(
                f"No *.jsonl files found in {self.processed_dir}. "
                "Make sure your processed/ dataset was uploaded correctly."
            )

        for jsonl_path in jsonl_files:
            dataset_name = jsonl_path.stem
            logger.info(f"  Loading {dataset_name} from {jsonl_path.name} ...")
            raw_samples = self._load_jsonl(str(jsonl_path))

            resolved = []
            missing = 0

            for s in raw_samples:
                img = s.get('image') or s.get('image_path', '')

                if not img:
                    missing += 1
                    continue

                resolved_path = _resolve_path(img, self.data_dir)

                if os.path.exists(resolved_path):
                    s['image'] = resolved_path
                    s.pop('image_path', None)
                    resolved.append(s)
                else:
                    missing += 1

            if missing > 0:
                logger.warning(
                    f"  {dataset_name}: {missing}/{len(raw_samples)} samples "
                    f"had missing images (dropped). {len(resolved)} samples OK."
                )
            else:
                logger.info(
                    f"  {dataset_name}: {len(resolved)} samples loaded, "
                    f"all images verified ✓"
                )

            all_processed[dataset_name] = resolved
            total_loaded += len(resolved)
            total_missing += missing

        logger.info(
            f"Total: {total_loaded} samples loaded, "
            f"{total_missing} dropped (missing images)"
        )
        return all_processed

    
    
    

    def pull_from_huggingface(self) -> bool:
        """Disabled in offline Kaggle mode. Always returns False."""
        logger.info(
            "pull_from_huggingface() called but OFFLINE MODE is active — skipped."
        )
        return False

    def push_to_huggingface(self, all_processed: Dict[str, List[Dict]]) -> bool:
        """Disabled in offline Kaggle mode. Always returns False."""
        logger.info(
            "push_to_huggingface() called but OFFLINE MODE is active — skipped."
        )
        return False

    
    
    

    def download_all_datasets(self) -> Dict[str, Any]:
        """Disabled in offline Kaggle mode. Always returns empty dict."""
        logger.info(
            "download_all_datasets() called but OFFLINE MODE is active — skipped."
        )
        return {}

    def preprocess_all_datasets(
        self, dataset_sources: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """Disabled in offline Kaggle mode. Always returns empty dict."""
        logger.info(
            "preprocess_all_datasets() called but OFFLINE MODE is active — skipped."
        )
        return {}

    def verify_dataset(self, all_processed: Dict[str, List[Dict]]) -> bool:
        """Light verification — checks that every dataset has at least one sample."""
        for name, samples in all_processed.items():
            if not samples:
                logger.warning(f"verify_dataset: {name} has 0 samples!")
                return False
            result = validate_samples(samples, max_check=50)
            if not result['is_ok']:
                logger.warning(f"verify_dataset: {name} failed validation!")
                return False
        return True

    
    
    

    def run_full_pipeline(
        self, force_refresh: bool = False
    ) -> Tuple[Dict[str, List[Dict]], LaTeXTokenizer]:
        """
        OFFLINE MODE: load pre-processed data directly from processed/.

        Steps:
          1. Load tokenizer.json        (read-only, safe)
          2. Load + path-resolve all    (read-only, safe)
             *.jsonl files
          3. Update self.manifest       (in memory only, never written
             in memory                  to /kaggle/input/)
          4. Return (all_processed, tokenizer)

        No downloads, no HuggingFace calls, no network traffic.
        No writes to /kaggle/input/ at any point.
        """
        logger.info("=" * 70)
        logger.info("OFFLINE MODE: Bypassing all network downloads")
        logger.info(f"Reading from: {self.processed_dir}")
        logger.info("=" * 70)

        
        tok_path = os.path.join(self.processed_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            self.tokenizer.load(tok_path)
            logger.info(
                f"Tokenizer loaded — vocab size: {len(self.tokenizer)}"
            )
        else:
            raise FileNotFoundError(
                f"tokenizer.json not found in {self.processed_dir}. "
                "Upload it as part of your processed/ dataset."
            )

        
        all_processed = self._load_and_resolve_all_jsonl()

        if not all_processed:
            raise RuntimeError(
                "No valid samples found in the processed/ folder. "
                "Check that your *.jsonl files and image files were "
                "uploaded correctly."
            )

        
        
        
        
        
        
        for ds_name, samples in all_processed.items():
            self.manifest['datasets'][ds_name] = {
                'preprocessed': True,
                'count': len(samples),
            }
        self.manifest['all_preprocessed'] = True
        self.manifest['tokenizer_built'] = True
        

        total = sum(len(v) for v in all_processed.values())
        logger.info(
            f"Offline load complete — {total} total samples ready for training."
        )
        return all_processed, self.tokenizer