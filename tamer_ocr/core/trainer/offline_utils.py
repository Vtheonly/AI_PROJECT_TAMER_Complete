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

# Global O(1) lookup table to bypass Kaggle's slow Network File System (NFS).
# The old code called os.path.exists() ~1.5 million times against the NFS,
# causing 10-15 minute freezes. A single os.walk + dict eliminates that entirely.
_FILENAME_INDEX: Dict[str, str] = {}
_INDEX_BUILT: bool = False


# ──────────────────────────────────────────────────────────────────────
# Path Resolution
# ──────────────────────────────────────────────────────────────────────

def _build_filename_index(data_dir: str) -> None:
    """
    Traverse data_dir exactly once with os.walk and build a flat
    dictionary mapping  basename -> absolute_path  for every image file.

    This is called once before any JSONL is parsed.  All subsequent
    lookups are O(1) dict hits — no further NFS traffic.
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
                # If two datasets share an identical basename the last one
                # wins, but cross-dataset basename collisions are very rare
                # for these specific subsets.
                _FILENAME_INDEX[f] = os.path.join(root, f)

    _INDEX_BUILT = True
    logger.info(f"Index built: {len(_FILENAME_INDEX):,} images mapped instantly.")


def _resolve_image_path(img_path: str, data_dir: str) -> str:
    """
    Locate an image file with zero NFS ping storms.

    Resolution order
    ────────────────
    1. Absolute path that already exists on disk  →  return as-is.
    2. O(1) basename lookup in the pre-built filename index  →  instant.
    3. Direct join of the relative tail against data_dir  →  safety net.

    Returns the resolved absolute path, or "" when the file cannot be found.

    NOTE: The old 'suffix-tree' fallback (step 4 in the previous version)
    that iterated over every path prefix with os.path.exists() has been
    removed.  It was responsible for ~1.5 M NFS round-trips and the
    10-15 minute freeze described in the article.  The filename index
    covers that use-case instantly.
    """
    if not img_path:
        return ""

    # 1. Already a valid absolute path — nothing to resolve.
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path

    # 2. O(1) lookup by exact basename (covers stale Windows/Linux absolute
    #    paths from the sanitisation machine — the common failure case).
    basename = os.path.basename(img_path.replace("\\", "/"))
    if basename in _FILENAME_INDEX:
        return _FILENAME_INDEX[basename]

    # 3. Direct relative-path join against data_dir (safety net for paths
    #    that are genuinely relative and not yet in the index).
    if data_dir:
        for normalised in (img_path.replace("\\", "/"), img_path):
            candidate = os.path.join(data_dir, normalised)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    return ""


def _find_image_root(*candidates: str) -> str:
    """
    Discover the directory that contains at least 2 of the marker
    sub-directories {"crohme", "hme100k", "im2latex"}.

    Checks explicit candidates first, then walks /kaggle/input up to
    depth 6 to avoid traversing the entire filesystem.

    Returns the first matching path, or "" if none is found.
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

    Two critical bugs from the original version are fixed here:

    Bug 1 — NFS freeze
        The old ``_resolve_image_path`` walked a suffix tree and called
        ``os.path.exists()`` up to 6 times per image.  With ~250 k images
        that is 1.5 M NFS round-trips → 10-15 minute hang.
        Fix: ``_build_filename_index`` does a single ``os.walk`` before
        parsing begins; every lookup is then an O(1) dict access.

    Bug 2 — Silent cache failure
        The old code wrote the cache to
        ``<sanitized_dir>/resolved_samples_cache.pkl``.
        ``sanitized_dir`` lives inside ``/kaggle/input/`` which is
        strictly read-only.  The write silently raised
        ``[Errno 30] Read-only file system``, the cache was never saved,
        and the 15-minute freeze repeated on every kernel restart.
        Fix: write the cache to ``/kaggle/working/`` which is writable.

    The cache is invalidated automatically whenever any source JSONL is
    newer than the cache file.

    Args:
        sanitized_dir: Directory that contains the ``*.jsonl`` files.
        data_dir:      Root directory used to resolve relative image paths.

    Returns:
        Dict mapping dataset name → list of sample dicts.  Each dict has
        an ``"image"`` key holding a resolved absolute path.
    """
    # FIX (Bug 2): use /kaggle/working/ so the cache is actually written.
    cache_dir  = "/kaggle/working" if os.path.isdir("/kaggle/working") else sanitized_dir
    cache_file = os.path.join(cache_dir, "resolved_samples_cache.pkl")

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

    # FIX (Bug 1): build the O(1) index once before touching any JSONL.
    _build_filename_index(data_dir)

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
                    resolved = _resolve_image_path(img, data_dir)
                    if resolved:
                        sample["image"] = resolved
                        sample.pop("image_path", None)
                    else:
                        missing_count += 1
                        continue  # drop samples whose image cannot be found

                samples.append(sample)

        if missing_count:
            logger.warning(
                f"  {ds_name}: {missing_count:,} samples dropped (image not found)"
            )

        logger.info(f"  Loaded {ds_name}: {len(samples):,} samples")
        all_processed[ds_name] = samples

    # ── Write cache to writable location ──────────────────────────────
    logger.info(f"Caching resolved samples → {cache_file}")
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(all_processed, fh)
        logger.info("Cache saved successfully.")
    except Exception as exc:
        logger.warning(f"Failed to write cache: {exc}")

    return all_processed