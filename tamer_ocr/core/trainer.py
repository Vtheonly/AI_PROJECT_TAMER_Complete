"""
Main Training Pipeline for TAMER OCR v2.4.

Changes from v2.3:
  - Multi-GPU support via torch.nn.DataParallel.
  - BFloat16 AMP. Prevents NaN losses at large batch sizes.
  - prefetch_factor=4 on train loader to saturate 175GB RAM.
  - _unwrap_model() helper so DataParallel-wrapped models can still
    access .encoder / .decoder without AttributeError.
  - Val DataLoader keeps prefetch_factor=2.
  - sanitized_data_dir support: when set in config, the preprocessor
    loads the clean JSONL files instead of re-downloading raw datasets.
  - All 4 datasets supported: CROHME, HME100K, Im2LaTeX, MathWriting.
  - FIXED: LaTeXTokenizer.load() called as instance method not classmethod.

All v2.3 features retained:
  - Curriculum learning, structure-aware loss, structural accuracy
  - Encoder freeze/unfreeze, gradient checkpointing
  - persistent_workers, HF push throttling
  - eval_every, eval_warmup, torch.compile
"""

import os
import time
import math
import logging
import random
import threading
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from ..config import Config
from ..models.tamer import TAMERModel
from ..data.dataset import MathDataset, get_collate_fn
from ..data.tokenizer import LaTeXTokenizer
from ..data.sampler import (
    MultiDatasetBatchSampler,
    get_temperature_for_step,
)
from ..data.preprocessor import DatasetPreprocessor
from ..data.augmentation import get_train_augmentation, get_val_augmentation

from .losses import LabelSmoothedCELoss, StructureAwareLoss
from .engine import train_step, optimizer_step, evaluate_full
from ..utils.checkpoint import (
    save_checkpoint, load_checkpoint,
    find_latest_checkpoint, cleanup_old_checkpoints,
    push_checkpoint_to_hf,
)
from ..utils.metrics import evaluate_structural_accuracy
from ..data.latex_normalizer import get_complexity
from ..logger import setup_logger

logger = logging.getLogger("TAMER.Trainer")


# ---------------------------------------------------------------------------
# Background HuggingFace Push
# ---------------------------------------------------------------------------

def _push_hf_background(
    checkpoint_path: str,
    config: Config,
    epoch: int,
    is_best: bool,
) -> threading.Thread:
    """
    Push a checkpoint to HuggingFace Hub in a background daemon thread.
    Training is never blocked by network latency.
    """
    def _worker():
        try:
            push_checkpoint_to_hf(
                checkpoint_path, config, epoch, is_best=is_best
            )
            logger.info(f"HF push complete (epoch {epoch}, best={is_best})")
        except Exception as e:
            logger.warning(f"Background HF push failed (epoch {epoch}): {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Helper: unwrap DataParallel or compiled model
# ---------------------------------------------------------------------------

def _unwrap_model(model: nn.Module) -> TAMERModel:
    """
    Return the underlying TAMERModel regardless of whether it has been
    wrapped by DataParallel or torch.compile.

    DataParallel stores the original module as .module.
    torch.compile wraps the module in ._orig_mod (PyTorch >= 2.0).
    Both wrappers shadow .encoder / .decoder attribute access, so
    freeze/unfreeze and checkpoint logic must go through this helper.
    """
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


# ---------------------------------------------------------------------------
# Helper: load sanitized JSONL files (all 4 datasets)
# ---------------------------------------------------------------------------

def _load_sanitized_samples(
    sanitized_dir: str,
    data_dir: str = "",
) -> Dict[str, List]:
    """
    Load pre-sanitized JSONL files from sanitized_dir.

    Expects the following files (produced by Cell 2.5):
      crohme.jsonl
      hme100k.jsonl
      im2latex.jsonl
      mathwriting.jsonl

    Image path resolution:
      The sanitizer writes image paths exactly as they appear in the
      original JSONL files — typically relative to the hf_data folder
      (e.g. "crohme/images/foo.png").  MathDataset needs an absolute
      path (or one relative to cwd) to actually open images.

      We resolve each image path by checking, in order:
        1. Already absolute AND exists → use as-is.
        2. Relative to data_dir → join and use if the file exists.
        3. Relative to sanitized_dir → join and use if the file exists.
        4. None of the above → keep original (will fail at image load
           but MathDataset has a blank-fallback so training continues).

    Args:
        sanitized_dir: Directory containing the sanitized .jsonl files.
        data_dir:      The root directory where images actually live
                       (e.g. /kaggle/input/.../hf_data).  When unset,
                       image paths are used as-is.

    Returns a dict: {dataset_name: [sample, ...]}
    Missing files are skipped with a warning — training continues on
    the datasets that ARE present.
    """
    import json
    import pickle

    cache_file = os.path.join(sanitized_dir, "resolved_samples_cache.pkl")
    if os.path.exists(cache_file):
        logger.info(f"✅ Fast path: loading fully resolved samples from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    dataset_files = {
        "crohme":      "crohme.jsonl",
        "hme100k":     "hme100k.jsonl",
        "im2latex":    "im2latex.jsonl",
        "mathwriting": "mathwriting.jsonl",
    }

    all_processed = {}

    for ds_name, filename in dataset_files.items():
        fpath = os.path.join(sanitized_dir, filename)
        if not os.path.exists(fpath):
            logger.warning(
                f"Sanitized file not found — skipping {ds_name}: {fpath}"
            )
            continue

        samples = []
        missing_count = 0
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue

                s['dataset_name'] = ds_name  # ensure tag is present

                # ── Resolve image path ────────────────────────────
                img = s.get('image') or s.get('image_path', '')
                if img and isinstance(img, str):
                    resolved = _resolve_image_path(img, data_dir, sanitized_dir)
                    if resolved:
                        s['image'] = resolved
                        s.pop('image_path', None)
                    else:
                        missing_count += 1
                        continue  # drop samples with unfindable images

                samples.append(s)

        if missing_count > 0:
            logger.warning(
                f"  {ds_name}: {missing_count:,} samples dropped "
                f"(image not found after path resolution)"
            )
        logger.info(
            f"  Loaded sanitized {ds_name}: {len(samples):,} samples"
        )
        all_processed[ds_name] = samples

    logger.info(f"💾 Caching resolved dictionary to {cache_file} for instant loads...")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(all_processed, f)
    except Exception as e:
        logger.warning(f"Failed to cache solved dictionary: {e}")

    return all_processed



def _resolve_image_path(
    img_path: str,
    data_dir: str,
    sanitized_dir: str,
) -> str:
    """
    Try to resolve an image path to an existing file on disk.

    Returns the resolved absolute path, or empty string if not found.
    """
    if not img_path:
        return ""

    # 1. Already absolute and exists
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path

    # 2. Relative to data_dir (most common case)
    if data_dir:
        candidate = os.path.join(data_dir, img_path.replace('\\', '/'))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        # Also try without slash normalisation
        candidate2 = os.path.join(data_dir, img_path)
        if os.path.exists(candidate2):
            return os.path.abspath(candidate2)

    # 3. Relative to sanitized_dir
    candidate = os.path.join(sanitized_dir, img_path.replace('\\', '/'))
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # 4. Stale absolute path — try suffix matching against data_dir
    if os.path.isabs(img_path) and data_dir:
        from pathlib import Path as _Path
        parts = _Path(img_path).parts
        for i in range(len(parts)):
            suffix = os.path.join(*parts[i:])
            candidate = os.path.join(data_dir, suffix)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

    # Nothing worked
    return ""


def _find_image_root(*candidates: str) -> str:
    """
    Auto-discover the directory that contains the actual image
    subdirectories (crohme/, hme100k/, im2latex/).

    Checks each candidate path first.  If none of them contain image
    subdirs, falls back to scanning /kaggle/input/ (the standard
    Kaggle dataset mount) up to depth 6.

    Returns the discovered path, or empty string if nothing found.
    """
    # Known subdirectories that indicate this is the image root
    _MARKER_DIRS = {"crohme", "hme100k", "im2latex"}

    def _has_image_subdirs(path: str) -> bool:
        if not path or not os.path.isdir(path):
            return False
        try:
            entries = set(os.listdir(path))
        except OSError:
            return False
        # At least 2 of 3 marker dirs must be present
        return len(_MARKER_DIRS & entries) >= 2

    # 1. Check explicit candidates
    for c in candidates:
        if _has_image_subdirs(c):
            return c

    # 2. Scan /kaggle/input/ (standard Kaggle mount)
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for dirpath, dirnames, _ in os.walk(kaggle_input):
            depth = dirpath.replace(kaggle_input, "").count(os.sep)
            if depth > 6:
                dirnames.clear()  # stop descending
                continue
            if _has_image_subdirs(dirpath):
                return dirpath

    return ""


class Trainer:
    """
    Orchestrates the full TAMER OCR training pipeline.

    Pipeline stages (called in order by run()):
      1. preprocess_data()     — load sanitized JSONL or run preprocessor
      2. create_dataloaders()  — build Dataset + DataLoader objects
      3. build_model()         — init model, optimiser, scheduler
      4. _auto_resume()        — load latest checkpoint if present
      5. train()               — main epoch loop
      6. _evaluate(beam=True)  — final beam-search evaluation
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.use_amp = self.device.type == 'cuda'

        self.logger = setup_logger("TAMER.Trainer", config.log_dir)
        self.logger.info(f"Device: {self.device} (AMP: {self.use_amp})")

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"GPUs available: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                self.logger.info(
                    f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                    f"| VRAM: {vram_gb:.1f} GB"
                )
            total_vram = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(num_gpus)
            ) / 1e9
            self.logger.info(
                f"Total VRAM across all GPUs: {total_vram:.1f} GB"
            )
            self.logger.info(
                f"Image resolution: "
                f"{config.img_height}×{config.img_width} → "
                f"{(config.img_height // 4) * (config.img_width // 4):,} "
                f"patches per image"
            )
            self.logger.info(
                f"TF32 matmul: "
                f"{torch.backends.cuda.matmul.allow_tf32} | "
                f"cuDNN benchmark: {torch.backends.cudnn.benchmark}"
            )

        self.tokenizer = LaTeXTokenizer()
        self.model: Optional[nn.Module] = None

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.criterion = LabelSmoothedCELoss(
            pad_id=0,
            label_smoothing=config.label_smoothing,
        )

        # ── Training State ─────────────────────────────────────────
        self.current_epoch = 0
        self.global_step = 0
        self.best_exp_rate = 0.0
        self.best_edit_dist = float('inf')
        self.epochs_without_improvement = 0

        # ── Data ───────────────────────────────────────────────────
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.dataset_ranges: Dict = {}
        self.train_samples: List = []
        self.val_samples: List = []
        self.all_train_samples: List = []

        # ── Timing ─────────────────────────────────────────────────
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

        # ── HF Push Throttle ───────────────────────────────────────
        self._last_hf_push_epoch: int = -1

        # ── Curriculum State ───────────────────────────────────────
        self._current_curriculum_stage: str = 'simple'

    # ----------------------------------------------------------------
    # PHASE 1: Data Preprocessing
    # ----------------------------------------------------------------

    def preprocess_data(self):
        """
        Load data for all 4 datasets.

        Fast path (Beast Mode):
          If config.sanitized_data_dir points to a directory that
          contains the 4 sanitized JSONL files from Cell 2.5, load
          those directly. This skips all downloading, rendering, and
          normalisation.

        Slow path (fallback):
          If sanitized_data_dir is not set or does not exist, run the
          full DatasetPreprocessor pipeline.

        FIXED: tokenizer is loaded via instance method:
            tok = LaTeXTokenizer()
            tok.load(path)
        NOT as a classmethod (LaTeXTokenizer.load(path) is wrong).
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: Data Preprocessing  (4 datasets)")
        self.logger.info("=" * 70)

        # ── Decide: sanitized fast-path or full pipeline ───────────
        sdir = getattr(self.config, 'sanitized_data_dir', '')
        use_sanitized = (
            sdir
            and os.path.isdir(sdir)
            and any(
                os.path.exists(os.path.join(sdir, f))
                for f in [
                    "crohme.jsonl", "hme100k.jsonl",
                    "im2latex.jsonl", "mathwriting.jsonl",
                ]
            )
        )

        if use_sanitized:
            self.logger.info(
                f"✅ Fast path: loading sanitized JSONL files from "
                f"{sdir}"
            )

            # ── Find the real image root ───────────────────────────
            # config.data_dir may point to the sanitized JSONL folder
            # (e.g. /kaggle/working/sanitized_processed) rather than
            # the original hf_data folder where images actually live
            # (e.g. /kaggle/input/.../hf_data).
            #
            # We auto-discover the image root by checking, in order:
            #   1. config.data_dir   — if it contains image subdirs
            #   2. config.data_root  — same check
            #   3. /kaggle/input/    — recursive scan (max depth 6)
            image_root = _find_image_root(
                self.config.data_dir,
                getattr(self.config, 'data_root', ''),
            )
            if image_root:
                self.logger.info(
                    f"  Image root discovered: {image_root}"
                )
            else:
                self.logger.warning(
                    "  Could not auto-discover image root — "
                    "image paths may fail to resolve. "
                    "Set config.data_dir to the folder containing "
                    "crohme/, hme100k/, im2latex/ subdirs."
                )
                image_root = self.config.data_dir

            all_processed = _load_sanitized_samples(
                sdir, data_dir=image_root
            )

            # ── Tokenizer load — FIXED ─────────────────────────────
            # LaTeXTokenizer.load() is an INSTANCE method, not a
            # classmethod. We must create an instance first, then
            # call load() on that instance to populate its vocab.
            tok_path = os.path.join(sdir, "tokenizer.json")
            if os.path.exists(tok_path):
                try:
                    # CORRECT pattern: instance first, then load
                    tok = LaTeXTokenizer()
                    tok.load(tok_path)
                    self.tokenizer = tok
                    self.logger.info(
                        f"Tokenizer loaded from {tok_path} "
                        f"({len(self.tokenizer)} tokens)"
                    )
                except Exception as e:
                    # If load() still fails for any reason, log and
                    # fall back to building from scratch — training
                    # will still work, vocab will be rebuilt.
                    self.logger.warning(
                        f"Tokenizer load failed ({e}) — "
                        f"will build from corpus instead."
                    )
            else:
                self.logger.info(
                    "tokenizer.json not found in sanitized dir — "
                    "will build from scratch after split."
                )
        else:
            self.logger.info(
                "No sanitized dir found — running full preprocessor."
            )
            preprocessor = DatasetPreprocessor(self.config)
            all_processed, self.tokenizer = preprocessor.run_full_pipeline()

        # ── Flatten all datasets ───────────────────────────────────
        all_samples = []
        for dataset_name, samples in all_processed.items():
            all_samples.extend(samples)

        self.logger.info(
            f"Total samples loaded across all 4 datasets: "
            f"{len(all_samples):,}"
        )
        for ds, samples in all_processed.items():
            self.logger.info(f"  {ds:<14}: {len(samples):,} samples")

        # ── Token Length Filter ────────────────────────────────────
        filtered = []
        for s in all_samples:
            latex = s.get('latex', '')
            if not latex:
                continue
            tokens = self.tokenizer.tokenize(latex)
            if len(tokens) <= self.config.max_token_length:
                filtered.append(s)

        self.logger.info(
            f"After token length filter "
            f"(≤{self.config.max_token_length}): "
            f"{len(filtered):,} samples"
        )

        # ── Stratified Train / Val Split ───────────────────────────
        grouped: Dict[str, List] = {}
        for s in filtered:
            ds = s.get('dataset_name', 'unknown')
            grouped.setdefault(ds, []).append(s)

        self.train_samples = []
        self.val_samples = []

        rng = random.Random(42)
        for ds, ds_samples in grouped.items():
            rng.shuffle(ds_samples)
            split_idx  = int(len(ds_samples) * 0.9)
            train_part = ds_samples[:split_idx]
            val_part   = ds_samples[split_idx:]
            self.train_samples.extend(train_part)
            self.val_samples.extend(val_part)
            self.logger.info(
                f"  {ds:<14}: {len(train_part):,} train | "
                f"{len(val_part):,} val"
            )

        self.train_samples.sort(
            key=lambda x: x.get('dataset_name', 'unknown')
        )
        self.val_samples.sort(
            key=lambda x: x.get('dataset_name', 'unknown')
        )

        self.logger.info(
            f"Split totals → Train: {len(self.train_samples):,} | "
            f"Val: {len(self.val_samples):,}"
        )
        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        # ── Complexity Distribution ────────────────────────────────
        complexity_counts = {'simple': 0, 'medium': 0, 'complex': 0}
        for s in self.train_samples:
            c = s.get(
                'complexity',
                get_complexity(s.get('latex', ''))
            )
            complexity_counts[c] = complexity_counts.get(c, 0) + 1
        self.logger.info(
            f"Train complexity — "
            f"simple: {complexity_counts['simple']:,} | "
            f"medium: {complexity_counts['medium']:,} | "
            f"complex: {complexity_counts['complex']:,}"
        )

        self.all_train_samples = list(self.train_samples)

        # ── Loss Function ──────────────────────────────────────────
        if self.config.structure_aware_loss:
            self.criterion = StructureAwareLoss(
                tokenizer=self.tokenizer,
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.config.label_smoothing,
                structural_weight=self.config.structural_token_weight,
            ).to(self.device)
            self.logger.info(
                f"Loss: StructureAwareLoss "
                f"(structural_weight="
                f"{self.config.structural_token_weight})"
            )
        else:
            self.criterion = LabelSmoothedCELoss(
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.config.label_smoothing,
            ).to(self.device)
            self.logger.info("Loss: LabelSmoothedCELoss")

        # ── Persist Tokenizer ──────────────────────────────────────
        tokenizer_path = os.path.join(
            self.config.output_dir, "tokenizer.json"
        )
        self.tokenizer.save(tokenizer_path)
        self.logger.info(f"Tokenizer saved → {tokenizer_path}")

    # ----------------------------------------------------------------
    # PHASE 2: Create DataLoaders
    # ----------------------------------------------------------------

    def create_dataloaders(self):
        """
        Build MathDataset instances and wrap them in DataLoaders.

        Train loader — RTX 6000 Ada Beast Mode:
          prefetch_factor=4 + num_workers=32:
            32 workers × 4 batches each = 128 batches always pre-loaded
            in the 175GB of system RAM. GPU never stalls for data.
          pin_memory=True:
            Tensors live in page-locked RAM so the GPU DMA engine
            transfers them without CPU involvement.
          persistent_workers=True:
            Worker processes survive between epochs, eliminating the
            ~10s worker spawn overhead at the start of each epoch.

        Val loader:
          prefetch_factor=2: no augmentation bottleneck, 2 is enough.
          batch_size = batch_size // 2: eval has no gradient
          checkpointing so peak VRAM per-sample is higher — halving
          prevents OOM on the validation pass.
        """
        self.logger.info("Creating DataLoaders (all 4 datasets)...")
        self._compute_dataset_ranges(self.train_samples)

        train_transform = get_train_augmentation(
            self.config.img_height, self.config.img_width
        )
        val_transform = get_val_augmentation()

        self.train_dataset = MathDataset(
            self.train_samples,
            self.config,
            self.tokenizer,
            train_transform,
        )
        self.val_dataset = MathDataset(
            self.val_samples,
            self.config,
            self.tokenizer,
            val_transform,
        )

        collate_fn = get_collate_fn(self.tokenizer.pad_id)

        # ── Train DataLoader ───────────────────────────────────────
        # Beast Mode: 32 workers × prefetch 4 = 128 batches in RAM
        if self.dataset_ranges:
            batch_sampler = MultiDatasetBatchSampler(
                dataset_ranges=self.dataset_ranges,
                batch_size=self.config.batch_size,
                temperature=self.config.temp_start,
                drop_last=True,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,    # 32
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,                      # 128 batches in RAM
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,    # 32
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,                      # 128 batches in RAM
                drop_last=True,
            )

        # ── Val DataLoader ─────────────────────────────────────────
        val_batch_size = max(self.config.batch_size // 2, 1)

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        self.logger.info(
            f"Train: {len(self.train_dataset):,} samples | "
            f"Val: {len(self.val_dataset):,} samples"
        )
        self.logger.info(
            f"Train batch: {self.config.batch_size} | "
            f"Val batch: {val_batch_size} | "
            f"Workers: {self.config.num_workers} | "
            f"Prefetch: 4 (train) / 2 (val)"
        )

    def _compute_dataset_ranges(self, samples: List):
        """
        Compute per-dataset contiguous index ranges for temperature
        sampling. Covers all 4 datasets automatically.
        """
        self.dataset_ranges = {}
        current_idx = 0

        for s in samples:
            name = s.get('dataset_name', 'unknown')
            if name not in self.dataset_ranges:
                self.dataset_ranges[name] = [current_idx, current_idx + 1]
            else:
                self.dataset_ranges[name][1] = current_idx + 1
            current_idx += 1

        for name, rng in self.dataset_ranges.items():
            self.dataset_ranges[name] = tuple(rng)
            self.logger.info(
                f"  {name:<14}: {rng[1] - rng[0]:,} samples "
                f"(idx {rng[0]}–{rng[1] - 1})"
            )

    # ----------------------------------------------------------------
    # PHASE 3: Model Initialization
    # ----------------------------------------------------------------

    def build_model(self):
        """
        Initialize Swin-v2 encoder + Transformer decoder.

        Multi-GPU:
          torch.nn.DataParallel splits each batch across all GPUs.
          Must be applied BEFORE torch.compile.

        Wrapping order:
          raw TAMERModel → DataParallel → torch.compile
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 2: Model Initialization")
        self.logger.info("=" * 70)

        vocab_size = len(self.tokenizer)
        self.model = TAMERModel(vocab_size, self.config)
        self.model = self.model.to(self.device)

        # ── Multi-GPU: DataParallel ────────────────────────────────
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.logger.info(
                f"🔥 MULTI-GPU: DataParallel across {num_gpus} GPUs | "
                f"batch {self.config.batch_size} → "
                f"{self.config.batch_size // num_gpus} images/GPU"
            )
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.logger.info(
                "Single GPU — DataParallel not applied."
            )

        # ── torch.compile ──────────────────────────────────────────
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.logger.info(
                "🔥 torch.compile() enabled — first epoch pauses "
                "~3-5 min for JIT. Subsequent epochs ~30% faster."
            )
            self.model = torch.compile(self.model)

        # ── Parameter counts ───────────────────────────────────────
        raw_model = _unwrap_model(self.model)
        total_p   = sum(p.numel() for p in raw_model.parameters())
        trainable = sum(
            p.numel() for p in raw_model.parameters()
            if p.requires_grad
        )
        self.logger.info(f"Total parameters     : {total_p:,}")
        self.logger.info(f"Trainable parameters : {trainable:,}")
        self.logger.info(
            f"Model size (fp32)    : ~{total_p * 4 / 1e9:.2f} GB"
        )
        self.logger.info(
            f"Model size (bf16)    : ~{total_p * 2 / 1e9:.2f} GB"
        )

        # ── Encoder Freeze Warmup ──────────────────────────────────
        raw_model = _unwrap_model(self.model)
        if self.config.freeze_encoder_epochs > 0:
            for p in raw_model.encoder.parameters():
                p.requires_grad = False
            frozen  = sum(
                p.numel() for p in raw_model.encoder.parameters()
            )
            active  = trainable - frozen
            self.logger.info(
                f"Encoder FROZEN for epochs "
                f"1-{self.config.freeze_encoder_epochs} "
                f"({frozen:,} frozen | {active:,} active)"
            )
        else:
            self.logger.info(
                "Encoder freeze disabled (freeze_encoder_epochs=0)"
            )

        # ── Differential Learning Rates ────────────────────────────
        encoder_params = list(raw_model.encoder.parameters())
        decoder_params = list(raw_model.decoder.parameters())

        param_groups = [
            {
                'params': encoder_params,
                'lr': self.config.encoder_lr,
                'name': 'encoder',
            },
            {
                'params': decoder_params,
                'lr': self.config.decoder_lr,
                'name': 'decoder',
            },
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        # ── OneCycleLR ─────────────────────────────────────────────
        steps_per_epoch = math.ceil(
            len(self.train_loader) / self.config.accumulation_steps
        )
        steps_per_epoch = max(steps_per_epoch, 1)
        self.config.total_training_steps = (
            steps_per_epoch * self.config.num_epochs
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.config.encoder_lr, self.config.decoder_lr],
            total_steps=self.config.total_training_steps,
            pct_start=self.config.pct_start,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        self.logger.info(
            f"Encoder LR: {self.config.encoder_lr:.1e} | "
            f"Decoder LR: {self.config.decoder_lr:.1e}"
        )
        self.logger.info(
            f"Effective batch: "
            f"{self.config.batch_size * self.config.accumulation_steps} | "
            f"Steps/epoch: {steps_per_epoch} | "
            f"Total steps: {self.config.total_training_steps:,}"
        )

    # ----------------------------------------------------------------
    # PHASE 4: Training Loop
    # ----------------------------------------------------------------

    def train(self):
        """
        Main training loop.

        self.model is the DataParallel (or compiled) wrapper.
        train_step / optimizer_step / evaluate_full all receive
        self.model directly — DataParallel handles scatter/gather.
        Only encoder freeze/unfreeze touches .encoder, so only
        those lines need _unwrap_model().
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: Training Loop")
        self.logger.info("=" * 70)
        self.logger.info(
            f"Epochs: {self.config.num_epochs} | "
            f"Batch: {self.config.batch_size} | "
            f"Accum: {self.config.accumulation_steps} | "
            f"Effective: "
            f"{self.config.batch_size * self.config.accumulation_steps} | "
            f"Eval every: {self.config.eval_every} epochs"
        )
        if self.config.freeze_encoder_epochs > 0:
            self.logger.info(
                f"Encoder freeze: epochs "
                f"1-{self.config.freeze_encoder_epochs}"
            )
        if self.config.curriculum_enabled:
            self.logger.info(
                f"Curriculum: simple → epoch "
                f"{self.config.curriculum_simple_until} | "
                f"medium → epoch "
                f"{self.config.curriculum_medium_until} | "
                f"complex → remainder"
            )

        self._profile_dataloader()

        self.model.train()
        self.step_start_time = time.time()
        self.optimizer.zero_grad()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1

            # ── Encoder Unfreeze ───────────────────────────────────
            if (self.config.freeze_encoder_epochs > 0 and
                    self.current_epoch ==
                    self.config.freeze_encoder_epochs + 1):
                raw_model = _unwrap_model(self.model)
                for p in raw_model.encoder.parameters():
                    p.requires_grad = True
                newly = sum(
                    p.numel() for p in raw_model.encoder.parameters()
                )
                self.logger.info(
                    f"*** Epoch {self.current_epoch}: Encoder UNFROZEN "
                    f"— {newly:,} params now training ***"
                )

            # ── Curriculum ────────────────────────────────────────
            if self.config.curriculum_enabled:
                new_stage = self._get_curriculum_stage(self.current_epoch)
                if new_stage != self._current_curriculum_stage:
                    self.logger.info(
                        f"*** Curriculum: "
                        f"{self._current_curriculum_stage} → "
                        f"{new_stage} ***"
                    )
                    self._current_curriculum_stage = new_stage
                    self._rebuild_train_loader_for_curriculum(new_stage)

            self.epoch_start_time = time.time()
            epoch_loss  = 0.0
            epoch_steps = 0

            self.logger.info(
                f"\n{'='*30} "
                f"Epoch {self.current_epoch}/{self.config.num_epochs} "
                f"{'='*30}"
            )

            # ── Batch Loop ─────────────────────────────────────────
            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                current_temp = get_temperature_for_step(
                    self.global_step,
                    self.config.total_training_steps,
                    self.config.temp_start,
                    self.config.temp_end,
                )
                if (hasattr(self.train_loader, 'batch_sampler') and
                        hasattr(
                            self.train_loader.batch_sampler,
                            'set_temperature',
                        )):
                    self.train_loader.batch_sampler.set_temperature(
                        current_temp
                    )

                loss = train_step(
                    model=self.model,
                    batch=batch,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    device=self.device,
                    accumulation_steps=self.config.accumulation_steps,
                    max_grad_norm=self.config.max_grad_norm,
                )
                epoch_loss  += loss
                epoch_steps += 1

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    optimizer_step(
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        scheduler=self.scheduler,
                        max_grad_norm=self.config.max_grad_norm,
                    )
                    self.global_step += 1

                    if self.global_step % 10 == 0:
                        elapsed    = time.time() - self.step_start_time
                        current_lr = self.scheduler.get_last_lr()
                        avg_loss   = epoch_loss / max(epoch_steps, 1)
                        self.logger.info(
                            f"Step {self.global_step:>6d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR enc={current_lr[0]:.2e} "
                            f"dec={current_lr[1]:.2e} | "
                            f"Temp: {current_temp:.3f} | "
                            f"10-step: {elapsed:.1f}s"
                        )
                        self.step_start_time = time.time()

            # ── Flush Remaining Gradients ──────────────────────────
            if (epoch_steps > 0 and
                    epoch_steps % self.config.accumulation_steps != 0):
                optimizer_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=self.scheduler,
                    max_grad_norm=self.config.max_grad_norm,
                )
                self.global_step += 1

            epoch_time = time.time() - self.epoch_start_time
            avg_loss   = epoch_loss / max(epoch_steps, 1)
            enc_status = (
                "FROZEN"
                if (self.config.freeze_encoder_epochs > 0 and
                    self.current_epoch <=
                    self.config.freeze_encoder_epochs)
                else "active"
            )
            self.logger.info(
                f"Epoch {self.current_epoch} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time / 60:.1f} min | "
                f"Step: {self.global_step} | "
                f"Encoder: {enc_status}"
            )

            # ── Evaluation ─────────────────────────────────────────
            is_best = False
            if self.current_epoch % self.config.eval_every == 0:
                in_warmup   = (
                    self.current_epoch <=
                    self.config.eval_warmup_epochs
                )
                max_samples = (
                    self.config.eval_warmup_max_samples
                    if in_warmup else None
                )
                _, is_best = self._evaluate(
                    use_beam_search=False,
                    max_samples=max_samples,
                )

            # ── Early Stopping Counter ─────────────────────────────
            if is_best:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # ── Epoch Checkpoint ───────────────────────────────────
            if (self.current_epoch %
                    self.config.checkpoint_every_epochs == 0):
                self._save_epoch_checkpoint()

            # ── Early Stopping ─────────────────────────────────────
            if (self.epochs_without_improvement >=
                    self.config.early_stopping_patience):
                self.logger.info(
                    f"Early stopping after "
                    f"{self.epochs_without_improvement} epochs "
                    f"without improvement."
                )
                break

        self.logger.info("=" * 70)
        self.logger.info("Training complete!")
        self.logger.info(f"Best ExpRate  : {self.best_exp_rate:.4f}")
        self.logger.info(f"Best EditDist : {self.best_edit_dist:.2f}")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def _evaluate(
        self,
        use_beam_search: bool = False,
        max_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Evaluate on the full validation set (all 4 datasets mixed).
        """
        sample_desc = (
            f" (capped at {max_samples})" if max_samples else ""
        )
        self.logger.info(
            f"Evaluating{sample_desc} | beam={use_beam_search} ..."
        )

        self.model.eval()

        metrics, all_preds, all_targets = evaluate_full(
            model=self.model,
            dataloader=self.val_loader,
            criterion=self.criterion,
            tokenizer=self.tokenizer,
            device=self.device,
            use_beam_search=use_beam_search,
            beam_width=self.config.beam_width,
            max_len=self.config.max_seq_len,
            length_penalty=self.config.length_penalty,
            max_samples=max_samples,
        )

        self.model.train()

        self.logger.info(
            f"  EVAL | Loss: {metrics.get('val_loss', 0.0):.4f} | "
            f"ExpRate: {metrics['exact_match']:.4f} | "
            f"EditDist: {metrics['edit_dist']:.2f} | "
            f"SER: {metrics['ser']:.4f} | "
            f"Leq1: {metrics['leq1']:.4f}"
        )

        # ── Structural Accuracy ────────────────────────────────────
        try:
            struct_metrics = evaluate_structural_accuracy(
                all_preds, all_targets
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to compute structural metrics: {e}"
            )
            struct_metrics = {}

        if struct_metrics:
            self.logger.info(
                f"  STRUCT | "
                f"simple={struct_metrics.get('exprate_simple', 0):.3f}"
                f" (n={struct_metrics.get('count_simple', 0)}) | "
                f"medium={struct_metrics.get('exprate_medium', 0):.3f}"
                f" (n={struct_metrics.get('count_medium', 0)}) | "
                f"complex="
                f"{struct_metrics.get('exprate_complex', 0):.3f}"
                f" (n={struct_metrics.get('count_complex', 0)}) | "
                f"struct_recall="
                f"{struct_metrics.get('structural_token_recall', 0):.3f}"
            )
            metrics.update(struct_metrics)

        # ── Best Model ─────────────────────────────────────────────
        is_best = metrics['edit_dist'] < self.best_edit_dist
        if is_best:
            self.best_exp_rate  = metrics['exact_match']
            self.best_edit_dist = metrics['edit_dist']
            self.logger.info(
                f"  *** New best | "
                f"EditDist: {self.best_edit_dist:.2f} | "
                f"ExpRate: {self.best_exp_rate:.4f} ***"
            )

            best_path = os.path.join(
                self.config.checkpoint_dir, "best.pt"
            )
            save_checkpoint(
                _unwrap_model(self.model),
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.current_epoch,
                self.global_step,
                {
                    'exp_rate':  self.best_exp_rate,
                    'edit_dist': self.best_edit_dist,
                },
                best_path,
            )

            should_push = (
                self.current_epoch - self._last_hf_push_epoch
                >= self.config.hf_push_every_n_epochs
            )
            if should_push:
                self._last_hf_push_epoch = self.current_epoch
                _push_hf_background(
                    best_path,
                    self.config,
                    self.current_epoch,
                    is_best=True,
                )

        return metrics, is_best

    def evaluate_with_beam_search(
        self, max_samples: int = 500
    ) -> Dict[str, float]:
        """Public beam-search eval. Called in --eval-only mode."""
        metrics, _ = self._evaluate(
            use_beam_search=True,
            max_samples=max_samples,
        )
        return metrics

    # ----------------------------------------------------------------
    # Curriculum Learning
    # ----------------------------------------------------------------

    def _get_curriculum_stage(self, epoch: int) -> str:
        if not self.config.curriculum_enabled:
            return 'complex'
        if epoch <= self.config.curriculum_simple_until:
            return 'simple'
        if epoch <= self.config.curriculum_medium_until:
            return 'medium'
        return 'complex'

    def _rebuild_train_loader_for_curriculum(self, stage: str):
        """
        Rebuild train DataLoader filtered to the curriculum stage.
        Works across all 4 datasets — each dataset contributes
        samples at the appropriate complexity level.
        """
        allowed = {
            'simple':  {'simple'},
            'medium':  {'simple', 'medium'},
            'complex': {'simple', 'medium', 'complex'},
        }[stage]

        filtered_samples = [
            s for s in self.all_train_samples
            if s.get(
                'complexity',
                get_complexity(s.get('latex', ''))
            ) in allowed
        ]

        self.logger.info(
            f"Curriculum rebuild ({stage}): "
            f"{len(filtered_samples):,} / "
            f"{len(self.all_train_samples):,} train samples"
        )

        # Per-dataset breakdown
        ds_counts: Dict[str, int] = {}
        for s in filtered_samples:
            ds = s.get('dataset_name', 'unknown')
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
        for ds, cnt in sorted(ds_counts.items()):
            self.logger.info(f"  {ds:<14}: {cnt:,} in stage")

        self.train_samples = filtered_samples
        self._compute_dataset_ranges(self.train_samples)

        train_transform = get_train_augmentation(
            self.config.img_height, self.config.img_width
        )

        self.train_dataset = MathDataset(
            samples=self.train_samples,
            config=self.config,
            tokenizer=self.tokenizer,
            transform=train_transform,
        )

        collate_fn = get_collate_fn(self.tokenizer.pad_id)

        if self.dataset_ranges:
            batch_sampler = MultiDatasetBatchSampler(
                dataset_ranges=self.dataset_ranges,
                batch_size=self.config.batch_size,
                temperature=self.config.temp_start,
                drop_last=True,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
                drop_last=True,
            )

        self.logger.info(
            f"Train DataLoader rebuilt | "
            f"Stage: {stage} | "
            f"Batches/epoch: ~{len(self.train_loader)}"
        )

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_epoch_checkpoint(self):
        """
        Save portable checkpoint (unwrapped state_dict).
        No 'module.' prefix — loads on any GPU count.
        """
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            f"epoch_{self.current_epoch}.pt",
        )
        save_checkpoint(
            _unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.current_epoch,
            self.global_step,
            {
                'exp_rate':  self.best_exp_rate,
                'edit_dist': self.best_edit_dist,
            },
            ckpt_path,
        )
        self.logger.info(f"Checkpoint saved → {ckpt_path}")

        cleanup_old_checkpoints(
            self.config.checkpoint_dir,
            self.config.keep_last_n_checkpoints,
        )

        should_push = (
            self.current_epoch - self._last_hf_push_epoch
            >= self.config.hf_push_every_n_epochs
        )
        if should_push:
            self._last_hf_push_epoch = self.current_epoch
            _push_hf_background(
                ckpt_path,
                self.config,
                self.current_epoch,
                is_best=False,
            )

    def _auto_resume(self) -> bool:
        latest = find_latest_checkpoint(self.config.checkpoint_dir)
        if latest is None:
            self.logger.info("No checkpoint found — starting from scratch")
            return False

        self.logger.info(f"Auto-resuming from: {latest}")
        epoch, step, metrics = load_checkpoint(
            latest,
            _unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            self.scaler,
            device=str(self.device),
        )
        self.current_epoch  = epoch
        self.global_step    = step
        self.best_exp_rate  = metrics.get('exp_rate', 0.0)
        self.best_edit_dist = metrics.get('edit_dist', float('inf'))
        self.logger.info(
            f"Resumed: epoch={epoch}, step={step}, "
            f"best_edit_dist={self.best_edit_dist:.2f}, "
            f"best_exp_rate={self.best_exp_rate:.4f}"
        )
        return True

    def resume_from_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        self.logger.info(f"Resuming from: {checkpoint_path}")
        epoch, step, metrics = load_checkpoint(
            checkpoint_path,
            _unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            self.scaler,
            device=str(self.device),
        )
        self.current_epoch  = epoch
        self.global_step    = step
        self.best_exp_rate  = metrics.get('exp_rate', 0.0)
        self.best_edit_dist = metrics.get('edit_dist', float('inf'))
        self.logger.info(
            f"Loaded: epoch={epoch}, step={step}, "
            f"best_edit_dist={self.best_edit_dist:.2f}"
        )

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def _profile_dataloader(self):
        """
        Time the first batch.
        With prefetch_factor=4 + 32 workers this should be < 1s on SSD.
        > 2s means I/O bottleneck not GPU bottleneck.
        """
        self.logger.info("Profiling DataLoader (timing first batch)...")
        t0          = time.time()
        first_batch = None
        for batch in self.train_loader:
            if batch is not None:
                first_batch = batch
                break
        elapsed = time.time() - t0

        if elapsed > 2.0:
            self.logger.warning(
                f"First batch: {elapsed:.2f}s — SLOW. "
                f"Check: num_workers={self.config.num_workers}, "
                f"SSD speed, augmentation complexity."
            )
        else:
            self.logger.info(
                f"First batch: {elapsed:.2f}s — OK "
                f"(workers={self.config.num_workers}, prefetch=4)"
            )

        if first_batch is not None:
            images = first_batch.get(
                'images',
                first_batch[0]
                if isinstance(first_batch, (list, tuple)) else None,
            )
            if images is not None and hasattr(images, 'shape'):
                self.logger.info(
                    f"Batch shape: {tuple(images.shape)} | "
                    f"dtype: {images.dtype}"
                )

    # ----------------------------------------------------------------
    # MAIN: Full Pipeline
    # ----------------------------------------------------------------

    def run(self, resume_from: Optional[str] = None):
        """
        Complete pipeline in strict order:
          1. preprocess_data()   — load 4 sanitized datasets
          2. create_dataloaders()
          3. build_model()       — DataParallel → compile
          4. resume / auto_resume
          5. train()
          6. Final beam-search eval (500 samples)
        """
        self.preprocess_data()
        self.create_dataloaders()
        self.build_model()

        if resume_from and os.path.exists(resume_from):
            self.resume_from_checkpoint(resume_from)
        else:
            self._auto_resume()

        self.train()

        self.logger.info(
            "Running final beam-search evaluation (500 samples)..."
        )
        self._evaluate(use_beam_search=True, max_samples=500)

        self.logger.info("=" * 70)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 70)