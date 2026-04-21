"""
Stateless Kaggle Offline Utilities.

Handles complex path resolution for read-only /kaggle/input environments.
All functions are pure/stateless — no trainer state is touched here.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("TAMER.OfflineUtils")

_DATASET_FILES = {
    "crohme":      "crohme.jsonl",
    "hme100k":     "hme100k.jsonl",
    "im2latex":    "im2latex.jsonl",
    "mathwriting": "mathwriting.jsonl",
}

_MARKER_DIRS = {"crohme", "hme100k", "im2latex"}


# ──────────────────────────────────────────────────────────────────────
# Path Resolution
# ──────────────────────────────────────────────────────────────────────

def _resolve_image_path(img_path: str, data_dir: str, sanitized_dir: str) -> str:
    """
    Attempt to locate an image file using several heuristics.

    Resolution order:
      1. Absolute path that already exists on disk.
      2. Relative to data_dir (forward + original separators).
      3. Relative to sanitized_dir.
      4. Walk the suffix tree of an absolute path against data_dir
         (handles stale /kaggle/input/old-dataset/... absolute paths).

    Returns the resolved absolute path, or "" if not found.
    """
    if not img_path:
        return ""

    # 1. Already a valid absolute path.
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path

    # 2. Relative to data_dir.
    if data_dir:
        for sep_normalised in (img_path.replace("\\", "/"), img_path):
            candidate = os.path.join(data_dir, sep_normalised)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    # 3. Relative to sanitized_dir.
    candidate = os.path.join(sanitized_dir, img_path.replace("\\", "/"))
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # 4. Walk suffix tree of a stale absolute path against data_dir.
    if os.path.isabs(img_path) and data_dir:
        parts = Path(img_path).parts
        for i in range(len(parts)):
            suffix = os.path.join(*parts[i:])
            candidate = os.path.join(data_dir, suffix)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    return ""


def _find_image_root(*candidates: str) -> str:
    """
    Discover the directory that contains at least 2 of the marker
    sub-directories {"crohme", "hme100k", "im2latex"}.

    Checks explicit candidates first, then walks /kaggle/input up to
    depth 6 to avoid traversing the entire filesystem.

    Returns the first matching path, or "" if none found.
    """

    def _has_image_subdirs(path: str) -> bool:
        if not path or not os.path.isdir(path):
            return False
        try:
            entries = set(os.listdir(path))
        except OSError:
            return False
        return len(_MARKER_DIRS & entries) >= 2

    for c in candidates:
        if _has_image_subdirs(c):
            return c

    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for dirpath, dirnames, _ in os.walk(kaggle_input):
            depth = dirpath.replace(kaggle_input, "").count(os.sep)
            if depth > 6:
                dirnames.clear()
                continue
            if _has_image_subdirs(dirpath):
                return dirpath

    return ""


# ──────────────────────────────────────────────────────────────────────
# Sanitized JSONL Loading with Pickle Cache
# ──────────────────────────────────────────────────────────────────────

def load_sanitized_samples(sanitized_dir: str, data_dir: str = "") -> Dict[str, List]:
    """
    Load pre-sanitized JSONL files from ``sanitized_dir``.

    A pickle cache (``resolved_samples_cache.pkl``) is written alongside
    the JSONL files so subsequent runs skip expensive path resolution.
    The cache is invalidated automatically when any source JSONL is
    newer than the cache file.

    Args:
        sanitized_dir: Directory containing the ``*.jsonl`` files and
                       the optional ``resolved_samples_cache.pkl``.
        data_dir:      Root to search for relative image paths.

    Returns:
        Dict mapping dataset name → list of sample dicts.  Each sample
        dict has an ``"image"`` key containing a resolved absolute path.
    """
    cache_file = os.path.join(sanitized_dir, "resolved_samples_cache.pkl")

    source_files = [
        os.path.join(sanitized_dir, fname)
        for fname in _DATASET_FILES.values()
        if os.path.exists(os.path.join(sanitized_dir, fname))
    ]

    # ── Cache validity check ──────────────────────────────────────────
    if os.path.exists(cache_file) and source_files:
        cache_mtime = os.path.getmtime(cache_file)
        if all(os.path.getmtime(src) <= cache_mtime for src in source_files):
            logger.info(f"Loading resolved samples from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                logger.warning(f"Cache read failed ({exc}) — rebuilding.")

    # ── Parse JSONL files ─────────────────────────────────────────────
    all_processed: Dict[str, List] = {}

    for ds_name, filename in _DATASET_FILES.items():
        fpath = os.path.join(sanitized_dir, filename)
        if not os.path.exists(fpath):
            logger.warning(f"Sanitized file not found — skipping {ds_name}: {fpath}")
            continue

        samples: List[dict] = []
        missing_count = 0

        with open(fpath, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.debug(f"JSON decode error in {filename}: {exc}")
                    continue

                sample["dataset_name"] = ds_name

                img = sample.get("image") or sample.get("image_path", "")
                if img and isinstance(img, str):
                    resolved = _resolve_image_path(img, data_dir, sanitized_dir)
                    if resolved:
                        sample["image"] = resolved
                        sample.pop("image_path", None)
                    else:
                        missing_count += 1
                        continue  # drop samples whose image cannot be found

                samples.append(sample)

        if missing_count:
            logger.warning(f"  {ds_name}: {missing_count:,} samples dropped (image not found)")

        logger.info(f"  Loaded {ds_name}: {len(samples):,} samples")
        all_processed[ds_name] = samples

    # ── Write cache ───────────────────────────────────────────────────
    logger.info(f"Caching resolved samples → {cache_file}")
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(all_processed, fh)
    except Exception as exc:
        logger.warning(f"Failed to write cache (read-only FS?): {exc}")

    return all_processed