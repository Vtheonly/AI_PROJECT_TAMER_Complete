"""
Trainer Mixin: Data Preprocessing, DataLoaders, and Curriculum Scheduling.

Responsibilities:
  - Run (or fast-path) dataset preprocessing via DatasetPreprocessor.
  - Load pre-sanitized JSONL files when available.
  - Audit dataset health via DatasetAuditor.
  - Perform stratified 90/10 train/val splits per dataset.
  - Build MathDataset + DataLoader objects.
  - Reconstruct the train DataLoader on curriculum stage transitions
    (with explicit GC to free worker memory before re-allocating).

State written:
  self.tokenizer, self.criterion,
  self.train_samples, self.val_samples, self.all_train_samples,
  self.train_dataset, self.val_dataset,
  self.train_loader, self.val_loader,
  self.dataset_ranges
"""

import os
import gc
import random
import logging
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ...data.dataset import MathDataset, get_collate_fn
from ...data.tokenizer import LaTeXTokenizer
from ...data.sampler import MultiDatasetBatchSampler
from ...data.preprocessor import DatasetPreprocessor
from ...data.augmentation import get_train_augmentation, get_val_augmentation
from ...data.audit import DatasetAuditor
from ...data.latex_normalizer import get_complexity
from ..losses import LabelSmoothedCELoss, StructureAwareLoss
from .offline_utils import _find_image_root, load_sanitized_samples

logger = logging.getLogger("TAMER.Data")


class TrainerDataMixin:

    # ──────────────────────────────────────────────────────────────────
    # PHASE 1 — Preprocessing
    # ──────────────────────────────────────────────────────────────────

    def preprocess_data(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: Data Preprocessing  (4 datasets)")
        self.logger.info("=" * 70)

        sdir = getattr(self.config, "sanitized_data_dir", "")
        tok_path = os.path.join(sdir, "tokenizer.json") if sdir else ""

        # ── Fast path: sanitized JSONL already on disk ────────────────
        if sdir and os.path.isdir(sdir) and os.path.exists(tok_path):
            self.logger.info(f"Fast path: loading sanitized JSONL from {sdir}")

            image_root = _find_image_root(
                self.config.data_dir,
                getattr(self.config, "data_root", ""),
            )
            if image_root:
                self.logger.info(f"  Image root discovered: {image_root}")
            else:
                self.logger.warning(
                    "Could not auto-discover image root — falling back to config.data_dir"
                )
                image_root = self.config.data_dir

            all_processed = load_sanitized_samples(sdir, data_dir=image_root)

            try:
                tok = LaTeXTokenizer()
                tok.load(tok_path)
                self.tokenizer = tok
                self.logger.info(
                    f"Tokenizer loaded from {tok_path} ({len(self.tokenizer)} tokens)"
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"tokenizer.json found at {tok_path} but failed to load: {exc}"
                ) from exc

        # ── Slow path: full preprocessing pipeline ────────────────────
        else:
            self.logger.info("Sanitized data not found — running full preprocessor.")
            preprocessor = DatasetPreprocessor(self.config)
            all_processed, self.tokenizer = preprocessor.run_full_pipeline()

        # ── Dataset health audit ──────────────────────────────────────
        DatasetAuditor(
            self.tokenizer,
            self.config.data_dir,
            sdir or self.config.data_dir,
        ).audit(list(all_processed.keys()))

        # ── Flatten & filter by token length ─────────────────────────
        all_samples: List[dict] = []
        for samples in all_processed.values():
            all_samples.extend(samples)

        self.logger.info(f"Total samples loaded: {len(all_samples):,}")

        filtered: List[dict] = []
        for s in all_samples:
            latex = s.get("latex", "")
            if not latex:
                continue
            if len(self.tokenizer.tokenize(latex)) <= self.config.max_token_length:
                filtered.append(s)

        self.logger.info(
            f"After token-length filter (≤{self.config.max_token_length}): {len(filtered):,}"
        )

        # ── Stratified 90/10 split per dataset ───────────────────────
        grouped: Dict[str, List] = {}
        for s in filtered:
            grouped.setdefault(s.get("dataset_name", "unknown"), []).append(s)

        self.train_samples: List[dict] = []
        self.val_samples:   List[dict] = []
        rng = random.Random(42)

        for ds_name, ds_samples in grouped.items():
            rng.shuffle(ds_samples)
            split_idx = int(len(ds_samples) * 0.9)
            train_part = ds_samples[:split_idx]
            val_part   = ds_samples[split_idx:]
            self.train_samples.extend(train_part)
            self.val_samples.extend(val_part)
            self.logger.info(
                f"  {ds_name:<14}: {len(train_part):,} train | {len(val_part):,} val"
            )

        # Sort by dataset name so MultiDatasetBatchSampler sees
        # contiguous ranges (required by _compute_dataset_ranges).
        self.train_samples.sort(key=lambda x: x.get("dataset_name", "unknown"))
        self.val_samples.sort(  key=lambda x: x.get("dataset_name", "unknown"))

        self.logger.info(
            f"Split totals → Train: {len(self.train_samples):,} | "
            f"Val: {len(self.val_samples):,}"
        )
        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        # ── Complexity stats ──────────────────────────────────────────
        complexity_counts: Dict[str, int] = {"simple": 0, "medium": 0, "complex": 0}
        for s in self.train_samples:
            c = s.get("complexity") or get_complexity(s.get("latex", ""))
            complexity_counts[c] = complexity_counts.get(c, 0) + 1
        self.logger.info(
            f"Train complexity — "
            f"simple: {complexity_counts['simple']:,} | "
            f"medium: {complexity_counts['medium']:,} | "
            f"complex: {complexity_counts['complex']:,}"
        )

        # Snapshot for curriculum resets
        self.all_train_samples: List[dict] = list(self.train_samples)

        # ── Loss function ─────────────────────────────────────────────
        if self.config.structure_aware_loss:
            self.criterion = StructureAwareLoss(
                tokenizer=self.tokenizer,
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.config.label_smoothing,
                structural_weight=self.config.structural_token_weight,
            ).to(self.device)
            self.logger.info(
                f"Loss: StructureAwareLoss "
                f"(structural_weight={self.config.structural_token_weight})"
            )
        else:
            self.criterion = LabelSmoothedCELoss(
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.config.label_smoothing,
            ).to(self.device)
            self.logger.info("Loss: LabelSmoothedCELoss")

        # ── Persist tokenizer ─────────────────────────────────────────
        tok_out = os.path.join(self.config.output_dir, "tokenizer.json")
        self.tokenizer.save(tok_out)
        self.logger.info(f"Tokenizer saved → {tok_out}")

    # ──────────────────────────────────────────────────────────────────
    # PHASE 2 — DataLoaders
    # ──────────────────────────────────────────────────────────────────

    def create_dataloaders(self) -> None:
        """Build train and validation DataLoaders from the current sample lists."""
        self.logger.info("Creating DataLoaders...")
        self._compute_dataset_ranges(self.train_samples)

        train_transform = get_train_augmentation(
            self.config.img_height, self.config.img_width
        )
        val_transform = get_val_augmentation()

        self.train_dataset = MathDataset(
            self.train_samples, self.config, self.tokenizer, train_transform
        )
        self.val_dataset = MathDataset(
            self.val_samples, self.config, self.tokenizer, val_transform
        )

        collate_fn = get_collate_fn(self.tokenizer.pad_id)

        self.train_loader = self._make_train_loader(collate_fn)

        val_batch_size = max(self.config.batch_size // 2, 1)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=(
                max(self.config.prefetch_factor // 2, 2)
                if self.config.num_workers > 0 else None
            ),
        )

        self.logger.info(
            f"Train: {len(self.train_dataset):,} samples | "
            f"Val: {len(self.val_dataset):,} samples"
        )
        self.logger.info(
            f"Train batch: {self.config.batch_size} | Val batch: {val_batch_size}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _make_train_loader(self, collate_fn) -> DataLoader:
        """
        Return a DataLoader for self.train_dataset.

        Uses MultiDatasetBatchSampler when dataset_ranges are available
        (temperature-based inter-dataset sampling), otherwise falls back
        to a standard shuffled loader.
        """
        common_kwargs = dict(
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=(
                self.config.prefetch_factor if self.config.num_workers > 0 else None
            ),
        )

        if self.dataset_ranges:
            batch_sampler = MultiDatasetBatchSampler(
                dataset_ranges=self.dataset_ranges,
                batch_size=self.config.batch_size,
                temperature=self.config.temp_start,
                drop_last=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                **common_kwargs,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs,
        )

    def _compute_dataset_ranges(self, samples: List[dict]) -> None:
        """
        Populate self.dataset_ranges with {name: (start_idx, end_idx)}.

        Assumes samples are sorted by dataset_name so that each dataset
        occupies a contiguous index range — a pre-condition required by
        MultiDatasetBatchSampler.
        """
        self.dataset_ranges: Dict[str, tuple] = {}
        current_idx = 0
        for s in samples:
            name = s.get("dataset_name", "unknown")
            if name not in self.dataset_ranges:
                self.dataset_ranges[name] = [current_idx, current_idx + 1]
            else:
                self.dataset_ranges[name][1] = current_idx + 1
            current_idx += 1

        for name in self.dataset_ranges:
            rng = self.dataset_ranges[name]
            self.dataset_ranges[name] = tuple(rng)
            self.logger.info(
                f"  {name:<14}: {rng[1] - rng[0]:,} samples "
                f"(idx {rng[0]}–{rng[1] - 1})"
            )

    # ──────────────────────────────────────────────────────────────────
    # Curriculum
    # ──────────────────────────────────────────────────────────────────

    def _get_curriculum_stage(self, epoch: int) -> str:
        """Map an epoch number to a curriculum complexity stage."""
        if not self.config.curriculum_enabled:
            return "complex"
        if epoch <= self.config.curriculum_simple_until:
            return "simple"
        if epoch <= self.config.curriculum_medium_until:
            return "medium"
        return "complex"

    def _rebuild_train_loader_for_curriculum(self, stage: str) -> None:
        """
        Filter self.all_train_samples to the allowed complexities for
        ``stage``, then rebuild the train DataLoader.

        The old DataLoader is explicitly deleted before construction so
        that its persistent worker processes are terminated and the
        associated memory is reclaimed before we allocate new workers.
        This is critical on 96 GB H100s where worker pools can hold
        several GB of pinned memory.
        """
        allowed = {
            "simple":  {"simple"},
            "medium":  {"simple", "medium"},
            "complex": {"simple", "medium", "complex"},
        }[stage]

        self.train_samples = [
            s for s in self.all_train_samples
            if (s.get("complexity") or get_complexity(s.get("latex", ""))) in allowed
        ]

        self.logger.info(
            f"Curriculum rebuild ({stage}): "
            f"{len(self.train_samples):,} / {len(self.all_train_samples):,} samples"
        )

        # ── Explicit teardown of the old loader ───────────────────────
        if getattr(self, "train_loader", None) is not None:
            del self.train_loader
            self.train_loader = None
            gc.collect()

        # Rebuild dataset + loader from the new sample list.
        # We do NOT call create_dataloaders() here to avoid recreating
        # the val loader unnecessarily.
        self._compute_dataset_ranges(self.train_samples)

        train_transform = get_train_augmentation(
            self.config.img_height, self.config.img_width
        )
        self.train_dataset = MathDataset(
            self.train_samples, self.config, self.tokenizer, train_transform
        )
        collate_fn = get_collate_fn(self.tokenizer.pad_id)
        self.train_loader = self._make_train_loader(collate_fn)