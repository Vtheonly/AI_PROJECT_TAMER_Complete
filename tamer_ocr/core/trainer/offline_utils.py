"""
Stateless Kaggle Offline Utilities.

Handles complex path resolution for read-only /kaggle/input environments.
All functions are pure/stateless — no trainer state is touched here.

Changelog (v2):
    Bug 1 — NFS freeze (FIXED in v1, maintained):
        Single os.walk builds an O(1) index. Zero NFS round-trips during parsing.

    Bug 2 — Silent cache failure (FIXED in v1, maintained):
        Cache is written to /kaggle/working/ (writable) instead of
        /kaggle/input/ (read-only).

    Bug 3 — Cross-dataset basename collision (FIXED in v2):
        The index now maps basename -> List[str] instead of basename -> str.
        When multiple datasets share an identical filename (e.g. "1.png"),
        the resolver uses the known dataset_name to pick the correct path
        instantly — still O(1), no extra NFS calls.
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








_FILENAME_INDEX: Dict[str, List[str]] = {}
_INDEX_BUILT: bool = False






def _build_filename_index(data_dir: str) -> None:
    """
    Traverse *data_dir* exactly once with ``os.walk`` and build a flat
    dictionary mapping  ``basename -> [absolute_path, ...]``  for every
    image file found.

    Called once before any JSONL is parsed.  All subsequent lookups are
    O(1) dict hits — no further NFS traffic.

    v2 change: values are *lists* so that files from different datasets
    that share the same basename are never silently overwritten.
    """
    
    
    
    global _INDEX_BUILT, _FILENAME_INDEX
    if _INDEX_BUILT or not data_dir or not os.path.exists(data_dir):
        return

    logger.info(
        f"Building fast O(1) filename index for '{data_dir}' "
        f"to bypass slow Kaggle NFS ..."
    )

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                
                
                if f not in _FILENAME_INDEX:
                    _FILENAME_INDEX[f] = []
                _FILENAME_INDEX[f].append(os.path.join(root, f))

    _INDEX_BUILT = True
    total_files = sum(len(paths) for paths in _FILENAME_INDEX.values())
    logger.info(
        f"Index built: {total_files:,} images mapped across "
        f"{len(_FILENAME_INDEX):,} unique basenames."
    )


def _resolve_image_path(
    img_path: str,
    data_dir: str,
    dataset_name: str = "",
) -> str:
    """
    Locate an image file with zero NFS ping storms.

    Resolution order
    ────────────────
    1. Absolute path that already exists on disk  →  return as-is.
    2. O(1) basename lookup in the pre-built filename index:
       a. Single candidate  →  return immediately (no collision).
       b. Multiple candidates  →  disambiguate with *dataset_name* by
          checking whether the dataset name appears as a path component.
       c. Fallback: check whether any candidate ends with the full
          relative path recorded in the JSONL (handles nested sub-dirs).
       d. Desperate fallback: return the first candidate and log a warning.
    3. Direct join of the relative tail against *data_dir*  →  safety net
       for paths that are genuinely relative and not yet in the index.

    Returns the resolved absolute path, or ``""`` when the file cannot
    be found.

    Args:
        img_path:     Raw path string from the JSONL ``"image"`` field.
        data_dir:     Root directory used to resolve relative paths.
        dataset_name: Name of the owning dataset (e.g. ``"crohme"``).
                      Used to disambiguate basename collisions in step 2b.
    """
    if not img_path:
        return ""

    
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path

    
    basename = os.path.basename(img_path.replace("\\", "/"))
    if basename in _FILENAME_INDEX:
        candidates = _FILENAME_INDEX[basename]

        
        if len(candidates) == 1:
            return candidates[0]

        
        
        
        normalized_ds = dataset_name.lower()
        if normalized_ds:
            for cand in candidates:
                cand_unix = cand.replace("\\", "/").lower()
                
                if f"/{normalized_ds}/" in cand_unix:
                    return cand

        
        
        
        normalized_img = img_path.replace("\\", "/")
        for cand in candidates:
            if cand.replace("\\", "/").endswith(normalized_img):
                return cand

        
        logger.warning(
            f"Basename collision for '{basename}' could not be resolved "
            f"(dataset='{dataset_name}', {len(candidates)} candidates). "
            f"Returning first match: {candidates[0]}"
        )
        return candidates[0]

    
    if data_dir:
        for normalised in (img_path.replace("\\", "/"), img_path):
            candidate = os.path.join(data_dir, normalised)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    return ""


def _find_image_root(*candidates: str) -> str:
    """
    Discover the directory that contains at least 2 of the marker
    sub-directories ``{"crohme", "hme100k", "im2latex"}``.

    Checks explicit *candidates* first, then walks ``/kaggle/input`` up
    to depth 6 to avoid traversing the entire filesystem.

    Returns the first matching path, or ``""`` if none is found.
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






def load_sanitized_samples(sanitized_dir: str, data_dir: str = "") -> Dict[str, List]:
    """
    Load pre-sanitized JSONL files from *sanitized_dir*.

    Bugs fixed
    ──────────
    Bug 1 — NFS freeze (v1):
        ``_build_filename_index`` does a single ``os.walk`` before any
        parsing begins.  Every image lookup is then an O(1) dict access.

    Bug 2 — Silent cache failure (v1):
        Cache is written to ``/kaggle/working/`` (writable) instead of
        inside ``sanitized_dir`` which lives under the read-only
        ``/kaggle/input/`` mount.

    Bug 3 — Cross-dataset basename collision (v2):
        The index now stores *all* paths per basename.  ``_resolve_image_path``
        uses the known ``dataset_name`` to pick the correct one without any
        extra NFS calls.

    The cache is invalidated automatically whenever any source JSONL is
    newer than the cache file.

    Args:
        sanitized_dir: Directory that contains the ``*.jsonl`` files.
        data_dir:      Root directory used to resolve relative image paths.

    Returns:
        ``Dict`` mapping dataset name → list of sample dicts.  Each dict
        has an ``"image"`` key holding a resolved absolute path.
    """
    
    cache_dir  = "/kaggle/working" if os.path.isdir("/kaggle/working") else sanitized_dir
    cache_file = os.path.join(cache_dir, "resolved_samples_cache.pkl")

    source_files = [
        os.path.join(sanitized_dir, fname)
        for fname in _DATASET_FILES.values()
        if os.path.exists(os.path.join(sanitized_dir, fname))
    ]

    
    if os.path.exists(cache_file) and source_files:
        cache_mtime = os.path.getmtime(cache_file)
        if all(os.path.getmtime(src) <= cache_mtime for src in source_files):
            logger.info(f"Loading resolved samples from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                logger.warning(f"Cache read failed ({exc}) — rebuilding.")

    
    _build_filename_index(data_dir)

    
    all_processed: Dict[str, List] = {}

    for ds_name, filename in _DATASET_FILES.items():
        fpath = os.path.join(sanitized_dir, filename)
        if not os.path.exists(fpath):
            logger.warning(
                f"Sanitized file not found — skipping {ds_name}: {fpath}"
            )
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
                    
                    
                    resolved = _resolve_image_path(
                        img, data_dir, dataset_name=ds_name
                    )
                    if resolved:
                        sample["image"] = resolved
                        sample.pop("image_path", None)
                    else:
                        missing_count += 1
                        continue  

                samples.append(sample)

        if missing_count:
            logger.warning(
                f"  {ds_name}: {missing_count:,} samples dropped "
                f"(image not found)"
            )

        logger.info(f"  Loaded {ds_name}: {len(samples):,} samples")
        all_processed[ds_name] = samples

    
    logger.info(f"Caching resolved samples → {cache_file}")
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(all_processed, fh)
        logger.info("Cache saved successfully.")
    except Exception as exc:
        logger.warning(f"Failed to write cache: {exc}")

    return all_processed