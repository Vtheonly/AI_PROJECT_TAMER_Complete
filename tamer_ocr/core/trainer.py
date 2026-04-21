"""
Main Training Pipeline for TAMER OCR v2.4 — Kaggle Offline Edition.

Zero-contradiction fixes:
  - Removed duplicate _resolve_image_path.
  - Added DatasetAuditor call in preprocess_data().
  - Uses official engine.train_step / engine.optimizer_step (NO manual shift).
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
from ..data.sampler import MultiDatasetBatchSampler, get_temperature_for_step
from ..data.preprocessor import DatasetPreprocessor
from ..data.augmentation import get_train_augmentation, get_val_augmentation
from ..data.audit import DatasetAuditor
from ..data.latex_normalizer import get_complexity

from .losses import LabelSmoothedCELoss, StructureAwareLoss
from .engine import train_step, optimizer_step, evaluate_full
from ..utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    push_checkpoint_to_hf,
)
from ..utils.metrics import evaluate_structural_accuracy
from ..logger import setup_logger

logger = logging.getLogger("TAMER.Trainer")


def _push_hf_background(checkpoint_path: str, config: Config, epoch: int, is_best: bool) -> threading.Thread:
    def _worker():
        try:
            push_checkpoint_to_hf(checkpoint_path, config, epoch, is_best=is_best)
            logger.info(f"HF push complete (epoch {epoch}, best={is_best})")
        except Exception as e:
            logger.warning(f"Background HF push failed (epoch {epoch}): {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def _unwrap_model(model: nn.Module) -> TAMERModel:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _resolve_image_path(img_path: str, data_dir: str, sanitized_dir: str) -> str:
    if not img_path:
        return ""
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path
    if data_dir:
        candidate = os.path.join(data_dir, img_path.replace("\\", "/"))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        candidate2 = os.path.join(data_dir, img_path)
        if os.path.exists(candidate2):
            return os.path.abspath(candidate2)
    candidate = os.path.join(sanitized_dir, img_path.replace("\\", "/"))
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    if os.path.isabs(img_path) and data_dir:
        from pathlib import Path as _Path

        parts = _Path(img_path).parts
        for i in range(len(parts)):
            suffix = os.path.join(*parts[i:])
            candidate = os.path.join(data_dir, suffix)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return ""


def _find_image_root(*candidates: str) -> str:
    _MARKER_DIRS = {"crohme", "hme100k", "im2latex"}

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


def _load_sanitized_samples(sanitized_dir: str, data_dir: str = "") -> Dict[str, List]:
    import json
    import pickle

    cache_file = os.path.join(sanitized_dir, "resolved_samples_cache.pkl")
    dataset_files = {
        "crohme": "crohme.jsonl",
        "hme100k": "hme100k.jsonl",
        "im2latex": "im2latex.jsonl",
        "mathwriting": "mathwriting.jsonl",
    }

    source_files = [
        os.path.join(sanitized_dir, fname)
        for fname in dataset_files.values()
        if os.path.exists(os.path.join(sanitized_dir, fname))
    ]

    cache_is_valid = False
    if os.path.exists(cache_file) and source_files:
        cache_mtime = os.path.getmtime(cache_file)
        if all(os.path.getmtime(src) <= cache_mtime for src in source_files):
            cache_is_valid = True
        else:
            logger.info("Source JSONL files changed since last cache — rebuilding.")

    if cache_is_valid:
        logger.info(f"Loading resolved samples from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    all_processed = {}
    for ds_name, filename in dataset_files.items():
        fpath = os.path.join(sanitized_dir, filename)
        if not os.path.exists(fpath):
            logger.warning(f"Sanitized file not found — skipping {ds_name}: {fpath}")
            continue

        samples = []
        missing_count = 0
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue

                s["dataset_name"] = ds_name
                img = s.get("image") or s.get("image_path", "")
                if img and isinstance(img, str):
                    resolved = _resolve_image_path(img, data_dir, sanitized_dir)
                    if resolved:
                        s["image"] = resolved
                        s.pop("image_path", None)
                    else:
                        missing_count += 1
                        continue

                samples.append(s)

        if missing_count > 0:
            logger.warning(
                f"  {ds_name}: {missing_count:,} samples dropped (image not found)"
            )
        logger.info(f"  Loaded sanitized {ds_name}: {len(samples):,} samples")
        all_processed[ds_name] = samples

    logger.info(f"Caching resolved dictionary to {cache_file}")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(all_processed, f)
    except Exception as e:
        logger.warning(f"Failed to write cache: {e}")

    return all_processed


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        self.logger = setup_logger("TAMER.Trainer", config.log_dir)
        self.logger.info(f"Device: {self.device} (AMP: {self.use_amp})")

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"GPUs available: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                self.logger.info(
                    f"  GPU {i}: {torch.cuda.get_device_name(i)} | VRAM: {vram_gb:.1f} GB"
                )

        self.tokenizer = LaTeXTokenizer()
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.criterion = LabelSmoothedCELoss(
            pad_id=0, label_smoothing=config.label_smoothing
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_exp_rate = 0.0
        self.best_edit_dist = float("inf")
        self.epochs_without_improvement = 0

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.dataset_ranges: Dict = {}
        self.train_samples: List = []
        self.val_samples: List = []
        self.all_train_samples: List = []

        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self._last_hf_push_epoch: int = -1
        self._current_curriculum_stage: str = "simple"

    # ------------------------------------------------------------------
    # PHASE 1: Data Preprocessing
    # ------------------------------------------------------------------
    def preprocess_data(self):
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: Data Preprocessing  (4 datasets)")
        self.logger.info("=" * 70)

        sdir = getattr(self.config, "sanitized_data_dir", "")
        use_sanitized = (
            sdir
            and os.path.isdir(sdir)
            and any(
                os.path.exists(os.path.join(sdir, f))
                for f in [
                    "crohme.jsonl",
                    "hme100k.jsonl",
                    "im2latex.jsonl",
                    "mathwriting.jsonl",
                ]
            )
        )

        if use_sanitized:
            self.logger.info(f"Fast path: loading sanitized JSONL from {sdir}")
            image_root = _find_image_root(
                self.config.data_dir,
                getattr(self.config, "data_root", ""),
            )
            if image_root:
                self.logger.info(f"  Image root discovered: {image_root}")
            else:
                self.logger.warning(
                    "Could not auto-discover image root — using config.data_dir"
                )
                image_root = self.config.data_dir

            all_processed = _load_sanitized_samples(sdir, data_dir=image_root)

            tok_path = os.path.join(sdir, "tokenizer.json")
            if os.path.exists(tok_path):
                try:
                    tok = LaTeXTokenizer()
                    tok.load(tok_path)
                    self.tokenizer = tok
                    self.logger.info(
                        f"Tokenizer loaded from {tok_path} ({len(self.tokenizer)} tokens)"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Tokenizer load failed ({e}) — will build from corpus."
                    )
            else:
                raise FileNotFoundError(
                    f"tokenizer.json not found in {sdir}. Upload it with your dataset."
                )
        else:
            self.logger.info("No sanitized dir found — running full preprocessor.")
            preprocessor = DatasetPreprocessor(self.config)
            all_processed, self.tokenizer = preprocessor.run_full_pipeline()

        # ── DATASET HEALTH AUDIT ──
        auditor = DatasetAuditor(
            self.tokenizer, self.config.data_dir, sdir or self.config.data_dir
        )
        auditor.audit(list(all_processed.keys()))

        # ── Flatten & Filter ──
        all_samples = []
        for dataset_name, samples in all_processed.items():
            all_samples.extend(samples)

        self.logger.info(
            f"Total samples loaded across all 4 datasets: {len(all_samples):,}"
        )

        filtered = []
        for s in all_samples:
            latex = s.get("latex", "")
            if not latex:
                continue
            tokens = self.tokenizer.tokenize(latex)
            if len(tokens) <= self.config.max_token_length:
                filtered.append(s)

        self.logger.info(
            f"After token length filter (≤{self.config.max_token_length}): {len(filtered):,}"
        )

        # ── Stratified Split ──
        grouped: Dict[str, List] = {}
        for s in filtered:
            ds = s.get("dataset_name", "unknown")
            grouped.setdefault(ds, []).append(s)

        self.train_samples = []
        self.val_samples = []
        rng = random.Random(42)
        for ds, ds_samples in grouped.items():
            rng.shuffle(ds_samples)
            split_idx = int(len(ds_samples) * 0.9)
            train_part = ds_samples[:split_idx]
            val_part = ds_samples[split_idx:]
            self.train_samples.extend(train_part)
            self.val_samples.extend(val_part)
            self.logger.info(
                f"  {ds:<14}: {len(train_part):,} train | {len(val_part):,} val"
            )

        self.train_samples.sort(key=lambda x: x.get("dataset_name", "unknown"))
        self.val_samples.sort(key=lambda x: x.get("dataset_name", "unknown"))

        self.logger.info(
            f"Split totals → Train: {len(self.train_samples):,} | Val: {len(self.val_samples):,}"
        )
        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        complexity_counts = {"simple": 0, "medium": 0, "complex": 0}
        for s in self.train_samples:
            c = s.get("complexity", get_complexity(s.get("latex", "")))
            complexity_counts[c] = complexity_counts.get(c, 0) + 1
        self.logger.info(
            f"Train complexity — simple: {complexity_counts['simple']:,} | "
            f"medium: {complexity_counts['medium']:,} | complex: {complexity_counts['complex']:,}"
        )

        self.all_train_samples = list(self.train_samples)

        # ── Loss ──
        if self.config.structure_aware_loss:
            self.criterion = StructureAwareLoss(
                tokenizer=self.tokenizer,
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.config.label_smoothing,
                structural_weight=self.config.structural_token_weight,
            ).to(self.device)
            self.logger.info(
                f"Loss: StructureAwareLoss (weight={self.config.structural_token_weight})"
            )
        else:
            self.criterion = LabelSmoothedCELoss(
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.config.label_smoothing,
            ).to(self.device)
            self.logger.info("Loss: LabelSmoothedCELoss")

        tokenizer_path = os.path.join(self.config.output_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        self.logger.info(f"Tokenizer saved → {tokenizer_path}")

    # ------------------------------------------------------------------
    # PHASE 2: DataLoaders
    # ------------------------------------------------------------------
    def create_dataloaders(self):
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
                prefetch_factor=self.config.prefetch_factor,
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
                prefetch_factor=self.config.prefetch_factor,
                drop_last=True,
            )

        val_batch_size = max(self.config.batch_size // 2, 1)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=max(self.config.prefetch_factor // 2, 2),
        )

        self.logger.info(
            f"Train: {len(self.train_dataset):,} | Val: {len(self.val_dataset):,}"
        )
        self.logger.info(
            f"Train batch: {self.config.batch_size} | Val batch: {val_batch_size}"
        )

    def _compute_dataset_ranges(self, samples: List):
        self.dataset_ranges = {}
        current_idx = 0
        for s in samples:
            name = s.get("dataset_name", "unknown")
            if name not in self.dataset_ranges:
                self.dataset_ranges[name] = [current_idx, current_idx + 1]
            else:
                self.dataset_ranges[name][1] = current_idx + 1
            current_idx += 1
        for name, rng in self.dataset_ranges.items():
            self.dataset_ranges[name] = tuple(rng)
            self.logger.info(
                f"  {name:<14}: {rng[1] - rng[0]:,} samples (idx {rng[0]}–{rng[1] - 1})"
            )

    # ------------------------------------------------------------------
    # PHASE 3: Model
    # ------------------------------------------------------------------
    def build_model(self):
        self.logger.info("=" * 70)
        self.logger.info("PHASE 2: Model Initialization")
        self.logger.info("=" * 70)

        vocab_size = len(self.tokenizer)
        self.model = TAMERModel(vocab_size, self.config)
        self.model = self.model.to(self.device)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.logger.info(
                f"MULTI-GPU: DataParallel across {num_gpus} GPUs"
            )
            self.model = torch.nn.DataParallel(self.model)

        if self.config.compile_model and hasattr(torch, "compile"):
            self.logger.info("torch.compile() enabled")
            self.model = torch.compile(self.model)

        raw_model = _unwrap_model(self.model)
        total_p = sum(p.numel() for p in raw_model.parameters())
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters     : {total_p:,}")
        self.logger.info(f"Trainable parameters : {trainable:,}")

        if self.config.freeze_encoder_epochs > 0:
            for p in raw_model.encoder.parameters():
                p.requires_grad = False
            frozen = sum(p.numel() for p in raw_model.encoder.parameters())
            active = trainable - frozen
            self.logger.info(
                f"Encoder FROZEN for epochs 1-{self.config.freeze_encoder_epochs} "
                f"({frozen:,} frozen | {active:,} active)"
            )

        encoder_params = list(raw_model.encoder.parameters())
        decoder_params = list(raw_model.decoder.parameters())

        param_groups = [
            {"params": encoder_params, "lr": self.config.encoder_lr, "name": "encoder"},
            {"params": decoder_params, "lr": self.config.decoder_lr, "name": "decoder"},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.config.weight_decay
        )

        steps_per_epoch = math.ceil(
            len(self.train_loader) / self.config.accumulation_steps
        )
        steps_per_epoch = max(steps_per_epoch, 1)
        self.config.total_training_steps = steps_per_epoch * self.config.num_epochs

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.config.encoder_lr, self.config.decoder_lr],
            total_steps=self.config.total_training_steps,
            pct_start=self.config.pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        self.logger.info(
            f"Encoder LR: {self.config.encoder_lr:.1e} | Decoder LR: {self.config.decoder_lr:.1e}"
        )
        self.logger.info(
            f"Effective batch: {self.config.batch_size * self.config.accumulation_steps} | "
            f"Steps/epoch: {steps_per_epoch} | Total: {self.config.total_training_steps:,}"
        )

    # ------------------------------------------------------------------
    # PHASE 4: Training Loop
    # ------------------------------------------------------------------
    def train(self):
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: Training Loop")
        self.logger.info("=" * 70)

        self.model.train()
        self.step_start_time = time.time()
        self.optimizer.zero_grad()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1

            # Encoder unfreeze
            if (
                self.config.freeze_encoder_epochs > 0
                and self.current_epoch == self.config.freeze_encoder_epochs + 1
            ):
                raw_model = _unwrap_model(self.model)
                for p in raw_model.encoder.parameters():
                    p.requires_grad = True
                newly = sum(p.numel() for p in raw_model.encoder.parameters())
                self.logger.info(
                    f"*** Epoch {self.current_epoch}: Encoder UNFROZEN — {newly:,} params now training ***"
                )

            # Curriculum
            if self.config.curriculum_enabled:
                new_stage = self._get_curriculum_stage(self.current_epoch)
                if new_stage != self._current_curriculum_stage:
                    self.logger.info(
                        f"*** Curriculum: {self._current_curriculum_stage} → {new_stage} ***"
                    )
                    self._current_curriculum_stage = new_stage
                    self._rebuild_train_loader_for_curriculum(new_stage)

            self.epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            self.logger.info(
                f"\n{'='*30} Epoch {self.current_epoch}/{self.config.num_epochs} {'='*30}"
            )

            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                current_temp = get_temperature_for_step(
                    self.global_step,
                    self.config.total_training_steps,
                    self.config.temp_start,
                    self.config.temp_end,
                )
                if (
                    hasattr(self.train_loader, "batch_sampler")
                    and hasattr(self.train_loader.batch_sampler, "set_temperature")
                ):
                    self.train_loader.batch_sampler.set_temperature(current_temp)

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
                epoch_loss += loss
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
                        elapsed = time.time() - self.step_start_time
                        current_lr = self.scheduler.get_last_lr()
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        self.logger.info(
                            f"Step {self.global_step:>6d} | Loss: {avg_loss:.4f} | "
                            f"LR enc={current_lr[0]:.2e} dec={current_lr[1]:.2e} | "
                            f"Temp: {current_temp:.3f} | 10-step: {elapsed:.1f}s"
                        )
                        self.step_start_time = time.time()

            # Flush remaining
            if epoch_steps > 0 and epoch_steps % self.config.accumulation_steps != 0:
                optimizer_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=self.scheduler,
                    max_grad_norm=self.config.max_grad_norm,
                )
                self.global_step += 1

            epoch_time = time.time() - self.epoch_start_time
            avg_loss = epoch_loss / max(epoch_steps, 1)
            enc_status = (
                "FROZEN"
                if (
                    self.config.freeze_encoder_epochs > 0
                    and self.current_epoch <= self.config.freeze_encoder_epochs
                )
                else "active"
            )
            self.logger.info(
                f"Epoch {self.current_epoch} | Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time / 60:.1f} min | Step: {self.global_step} | Encoder: {enc_status}"
            )

            # Evaluation
            is_best = False
            if self.current_epoch % self.config.eval_every == 0:
                in_warmup = self.current_epoch <= self.config.eval_warmup_epochs
                max_samples = (
                    self.config.eval_warmup_max_samples if in_warmup else None
                )
                _, is_best = self._evaluate(use_beam_search=False, max_samples=max_samples)

            if is_best:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.current_epoch % self.config.checkpoint_every_epochs == 0:
                self._save_epoch_checkpoint()

            if (
                self.epochs_without_improvement
                >= self.config.early_stopping_patience
            ):
                self.logger.info(
                    f"Early stopping after {self.epochs_without_improvement} epochs without improvement."
                )
                break

        self.logger.info("=" * 70)
        self.logger.info("Training complete!")
        self.logger.info(f"Best ExpRate  : {self.best_exp_rate:.4f}")
        self.logger.info(f"Best EditDist : {self.best_edit_dist:.2f}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        use_beam_search: bool = False,
        max_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, float], bool]:
        sample_desc = f" (capped at {max_samples})" if max_samples else ""
        self.logger.info(f"Evaluating{sample_desc} | beam={use_beam_search} ...")

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
            f"ExpRate: {metrics['exact_match']:.4f} | EditDist: {metrics['edit_dist']:.2f} | "
            f"SER: {metrics['ser']:.4f} | Leq1: {metrics['leq1']:.4f}"
        )

        try:
            struct_metrics = evaluate_structural_accuracy(all_preds, all_targets)
        except Exception as e:
            self.logger.warning(f"Failed structural metrics: {e}")
            struct_metrics = {}

        if struct_metrics:
            self.logger.info(
                f"  STRUCT | simple={struct_metrics.get('exprate_simple', 0):.3f} | "
                f"medium={struct_metrics.get('exprate_medium', 0):.3f} | "
                f"complex={struct_metrics.get('exprate_complex', 0):.3f}"
            )
            metrics.update(struct_metrics)

        is_best = metrics["edit_dist"] < self.best_edit_dist
        if is_best:
            self.best_exp_rate = metrics["exact_match"]
            self.best_edit_dist = metrics["edit_dist"]
            self.logger.info(
                f"  *** New best | EditDist: {self.best_edit_dist:.2f} | ExpRate: {self.best_exp_rate:.4f} ***"
            )
            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            save_checkpoint(
                _unwrap_model(self.model),
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.current_epoch,
                self.global_step,
                {"exp_rate": self.best_exp_rate, "edit_dist": self.best_edit_dist},
                best_path,
            )
            should_push = (
                self.current_epoch - self._last_hf_push_epoch
                >= self.config.hf_push_every_n_epochs
            )
            if should_push and self.config.hf_repo_id:
                self._last_hf_push_epoch = self.current_epoch
                _push_hf_background(best_path, self.config, self.current_epoch, is_best=True)

        return metrics, is_best

    def evaluate_with_beam_search(self, max_samples: int = 500) -> Dict[str, float]:
        metrics, _ = self._evaluate(use_beam_search=True, max_samples=max_samples)
        return metrics

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def _get_curriculum_stage(self, epoch: int) -> str:
        if not self.config.curriculum_enabled:
            return "complex"
        if epoch <= self.config.curriculum_simple_until:
            return "simple"
        if epoch <= self.config.curriculum_medium_until:
            return "medium"
        return "complex"

    def _rebuild_train_loader_for_curriculum(self, stage: str):
        allowed = {
            "simple": {"simple"},
            "medium": {"simple", "medium"},
            "complex": {"simple", "medium", "complex"},
        }[stage]

        filtered_samples = [
            s
            for s in self.all_train_samples
            if s.get("complexity", get_complexity(s.get("latex", ""))) in allowed
        ]

        self.logger.info(
            f"Curriculum rebuild ({stage}): {len(filtered_samples):,} / {len(self.all_train_samples):,}"
        )

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
                prefetch_factor=self.config.prefetch_factor,
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
                prefetch_factor=self.config.prefetch_factor,
                drop_last=True,
            )

        self.logger.info(
            f"Train DataLoader rebuilt | Stage: {stage} | Batches/epoch: ~{len(self.train_loader)}"
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_epoch_checkpoint(self):
        ckpt_path = os.path.join(
            self.config.checkpoint_dir, f"epoch_{self.current_epoch}.pt"
        )
        save_checkpoint(
            _unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.current_epoch,
            self.global_step,
            {"exp_rate": self.best_exp_rate, "edit_dist": self.best_edit_dist},
            ckpt_path,
        )
        self.logger.info(f"Checkpoint saved → {ckpt_path}")

        cleanup_old_checkpoints(
            self.config.checkpoint_dir, self.config.keep_last_n_checkpoints
        )

        should_push = (
            self.current_epoch - self._last_hf_push_epoch
            >= self.config.hf_push_every_n_epochs
        )
        if should_push and self.config.hf_repo_id:
            self._last_hf_push_epoch = self.current_epoch
            _push_hf_background(ckpt_path, self.config, self.current_epoch, is_best=False)

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
        self.current_epoch = epoch
        self.global_step = step
        self.best_exp_rate = metrics.get("exp_rate", 0.0)
        self.best_edit_dist = metrics.get("edit_dist", float("inf"))
        self.logger.info(
            f"Resumed: epoch={epoch}, step={step}, best_edit_dist={self.best_edit_dist:.2f}"
        )
        return True

    def resume_from_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.logger.info(f"Resuming from: {checkpoint_path}")
        epoch, step, metrics = load_checkpoint(
            checkpoint_path,
            _unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            self.scaler,
            device=str(self.device),
        )
        self.current_epoch = epoch
        self.global_step = step
        self.best_exp_rate = metrics.get("exp_rate", 0.0)
        self.best_edit_dist = metrics.get("edit_dist", float("inf"))
        self.logger.info(f"Loaded: epoch={epoch}, step={step}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _profile_dataloader(self):
        self.logger.info("Profiling DataLoader (timing first batch)...")
        t0 = time.time()
        first_batch = None
        for batch in self.train_loader:
            if batch is not None:
                first_batch = batch
                break
        elapsed = time.time() - t0

        if elapsed > 2.0:
            self.logger.warning(f"First batch: {elapsed:.2f}s — SLOW")
        else:
            self.logger.info(f"First batch: {elapsed:.2f}s — OK")

        if first_batch is not None:
            images = first_batch.get("image")
            if images is not None and hasattr(images, "shape"):
                self.logger.info(f"Batch shape: {tuple(images.shape)} | dtype: {images.dtype}")

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def run(self, resume_from: Optional[str] = None):
        self.preprocess_data()
        self.create_dataloaders()
        self.build_model()

        if resume_from and os.path.exists(resume_from):
            self.resume_from_checkpoint(resume_from)
        else:
            self._auto_resume()

        self._profile_dataloader()
        self.train()

        self.logger.info("Running final beam-search evaluation (500 samples)...")
        self._evaluate(use_beam_search=True, max_samples=500)

        self.logger.info("=" * 70)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 70)