"""
Main Training Pipeline for TAMER OCR v2.2.

Changes from v2.1:
  - build_model(): Encoder freeze for the first config.freeze_encoder_epochs
    epochs (default: 5). The decoder bootstraps on its own; encoder unfreezes
    automatically at the right epoch with a log message confirming it.
  - train(): Unfreeze logic injected at the top of the epoch loop — one clean
    check, no extra state needed.
  - Swin-v2 backbone is handled transparently: encoder.py reads channel counts
    dynamically, so no changes are needed there.

All v2.1 changes retained:
  - DataLoaders: persistent_workers=True, prefetch_factor=2
  - HuggingFace push: throttled to hf_push_every_n_epochs, background thread
  - Evaluation: every config.eval_every epochs (default: 3)
  - Evaluation: val set capped during eval_warmup_epochs
  - evaluate_with_beam_search(): public method for --eval-only mode
  - torch.compile(): optional, controlled by config.compile_model
  - DataLoader profiler at training start
  - Removed gc.collect() + cuda.empty_cache() from epoch loop

Strict pipeline order:
  1. Preprocess ALL datasets (via DatasetPreprocessor)
  2. Push to HuggingFace dataset repo
  3. Build model
  4. Auto-resume from latest checkpoint if available
  5. Train with epoch-based checkpointing (every 3 epochs)
  6. Push checkpoints to HuggingFace model repo (throttled)
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

from .losses import LabelSmoothedCELoss
from .engine import train_step, optimizer_step, evaluate_full
from ..utils.checkpoint import (
    save_checkpoint, load_checkpoint,
    find_latest_checkpoint, cleanup_old_checkpoints,
    push_checkpoint_to_hf,
)
from ..logger import setup_logger

logger = logging.getLogger("TAMER.Trainer")


# ---------------------------------------------------------------------------
# Background HuggingFace Push
# Runs in a daemon thread so the training loop never blocks on a network call.
# ---------------------------------------------------------------------------

def _push_hf_background(checkpoint_path: str, config: Config, epoch: int, is_best: bool):
    """Push a checkpoint to HuggingFace in a background thread."""
    def _worker():
        try:
            push_checkpoint_to_hf(checkpoint_path, config, epoch, is_best=is_best)
        except Exception as e:
            logger.warning(f"Background HF push failed (epoch {epoch}): {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


class Trainer:
    """
    Orchestrates the full TAMER OCR training pipeline.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = self.device.type == 'cuda'

        self.logger = setup_logger("TAMER.Trainer", config.log_dir)
        self.logger.info(f"Device: {self.device} (AMP: {self.use_amp})")

        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram_gb = getattr(props, 'total_memory', 0) / 1e9
            self.logger.info(f"VRAM: {vram_gb:.1f} GB")
            self.logger.info(
                f"Image resolution: {config.img_height}x{config.img_width} "
                f"→ {(config.img_height // 4) * (config.img_width // 4):,} patches per image"
            )

        self.tokenizer = LaTeXTokenizer()
        self.model: Optional[TAMERModel] = None

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.criterion = LabelSmoothedCELoss(
            pad_id=0,
            label_smoothing=config.label_smoothing,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_exp_rate = 0.0
        self.best_edit_dist = float('inf')
        self.epochs_without_improvement = 0

        # Data
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.dataset_ranges: Dict = {}
        self.train_samples: List = []
        self.val_samples: List = []

        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

        # Track the last epoch we pushed to HF (prevents repeated pushes)
        self._last_hf_push_epoch: int = -1

    # ----------------------------------------------------------------
    # PHASE 1: Data Preprocessing
    # ----------------------------------------------------------------

    def preprocess_data(self):
        """Run the full preprocessing pipeline."""
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: Data Preprocessing")
        self.logger.info("=" * 70)

        preprocessor = DatasetPreprocessor(self.config)
        all_processed, self.tokenizer = preprocessor.run_full_pipeline()

        all_samples = []
        for dataset_name, samples in all_processed.items():
            all_samples.extend(samples)

        self.logger.info(f"Total preprocessed samples: {len(all_samples)}")

        # Filter by token length
        filtered = []
        for s in all_samples:
            latex = s.get('latex', '')
            if not latex:
                continue
            tokens = self.tokenizer.tokenize(latex)
            if len(tokens) <= self.config.max_token_length:
                filtered.append(s)

        self.logger.info(f"After token length filter: {len(filtered)}")

        # Stratified split per dataset to maintain distribution
        grouped: Dict[str, List] = {}
        for s in filtered:
            ds = s.get('dataset_name', 'unknown')
            grouped.setdefault(ds, []).append(s)

        self.train_samples = []
        self.val_samples = []

        rng = random.Random(42)
        for ds, ds_samples in grouped.items():
            rng.shuffle(ds_samples)
            split_idx = int(len(ds_samples) * 0.9)
            self.train_samples.extend(ds_samples[:split_idx])
            self.val_samples.extend(ds_samples[split_idx:])

        # Sort by dataset_name so MultiDatasetBatchSampler gets contiguous ranges
        self.train_samples.sort(key=lambda x: x.get('dataset_name', 'unknown'))
        self.val_samples.sort(key=lambda x: x.get('dataset_name', 'unknown'))

        self.logger.info(f"Train: {len(self.train_samples)}, Val: {len(self.val_samples)}")
        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        self.criterion = LabelSmoothedCELoss(
            pad_id=self.tokenizer.pad_id,
            label_smoothing=self.config.label_smoothing,
        )

        self.tokenizer.save(os.path.join(self.config.output_dir, "tokenizer.json"))

    # ----------------------------------------------------------------
    # PHASE 2: Create DataLoaders
    # ----------------------------------------------------------------

    def create_dataloaders(self):
        """Create train and val DataLoaders from preprocessed samples."""
        self.logger.info("Creating DataLoaders...")
        self._compute_dataset_ranges(self.train_samples)

        train_transform = get_train_augmentation(self.config.img_height, self.config.img_width)
        val_transform = get_val_augmentation()

        self.train_dataset = MathDataset(
            self.train_samples, self.config, self.tokenizer, train_transform
        )
        self.val_dataset = MathDataset(
            self.val_samples, self.config, self.tokenizer, val_transform
        )

        collate_fn = get_collate_fn(self.tokenizer.pad_id)

        # ----------------------------------------------------------------
        # persistent_workers=True — workers stay alive across epochs.
        #   Without this, all 4 workers are killed and respawned at the end
        #   of every epoch. At 70 epochs, that's 280 unnecessary process
        #   spawns, each adding several seconds of startup overhead.
        #
        # prefetch_factor=2 — each worker pre-loads 2 batches ahead.
        #   The GPU never sits idle waiting for the CPU to finish augmentation.
        # ----------------------------------------------------------------

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
                prefetch_factor=2,
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
                prefetch_factor=2,
                drop_last=True,
            )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        self.logger.info(
            f"Train: {len(self.train_dataset)} samples | "
            f"Val: {len(self.val_dataset)} samples | "
            f"Workers: {self.config.num_workers} | "
            f"Batch: {self.config.batch_size}"
        )

    def _compute_dataset_ranges(self, samples: List):
        """Compute per-dataset contiguous index ranges for temperature sampling."""
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
                f"  {name}: {rng[1] - rng[0]} samples (range {rng[0]}–{rng[1]})"
            )

    # ----------------------------------------------------------------
    # PHASE 3: Model Initialization
    # ----------------------------------------------------------------

    def build_model(self):
        """
        Initialize Swin-v2 encoder + Transformer decoder (8 layers).

        Applies:
          - Differential learning rates (encoder 1e-5, decoder 1e-4)
          - Optional encoder freeze for the first freeze_encoder_epochs epochs
          - Optional torch.compile()
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 2: Model Initialization")
        self.logger.info("=" * 70)

        vocab_size = len(self.tokenizer)
        self.model = TAMERModel(vocab_size, self.config)
        self.model = self.model.to(self.device)

        # Optional: torch.compile for 10-30% speed gain on PyTorch 2.x.
        # Adds a one-time JIT warmup cost on the first epoch (~2-3 min).
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.logger.info("Applying torch.compile() — first epoch will be slower (JIT warmup)")
            self.model = torch.compile(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters:     {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable:,}")
        self.logger.info(f"Model size (fp32):    ~{total_params * 4 / 1e9:.2f} GB")
        self.logger.info(f"Model size (fp16):    ~{total_params * 2 / 1e9:.2f} GB")

        # ----------------------------------------------------------------
        # Encoder Freeze Warmup
        #
        # Freeze encoder weights for the first freeze_encoder_epochs epochs.
        # Benefit: the randomly-initialised decoder can learn to read the
        # encoder's existing feature map without destabilising it. Once the
        # decoder has a reasonable representation, the encoder unfreezes and
        # fine-tunes at its lower LR alongside the decoder.
        #
        # The optimizer still tracks frozen encoder params (so LR scheduling
        # is consistent), but gradients for them are zero during the freeze.
        # When unfreezing happens in train(), requires_grad is simply set back
        # to True — no optimizer rebuild needed.
        # ----------------------------------------------------------------
        if self.config.freeze_encoder_epochs > 0:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            trainable_after_freeze = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            frozen_count = trainable - trainable_after_freeze
            self.logger.info(
                f"Encoder FROZEN for epochs 1-{self.config.freeze_encoder_epochs} "
                f"({frozen_count:,} params frozen, {trainable_after_freeze:,} active)"
            )
        else:
            self.logger.info("Encoder freeze disabled (freeze_encoder_epochs=0)")

        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        param_groups = [
            {'params': encoder_params, 'lr': self.config.encoder_lr, 'name': 'encoder'},
            {'params': decoder_params, 'lr': self.config.decoder_lr, 'name': 'decoder'},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        # math.ceil prevents undercounting when len(loader) % accumulation_steps != 0,
        # which would cause OneCycleLR to end before the scheduler completes its anneal.
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
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        self.logger.info(
            f"Encoder LR: {self.config.encoder_lr} | "
            f"Decoder LR: {self.config.decoder_lr} | "
            f"Effective batch: {self.config.batch_size * self.config.accumulation_steps} | "
            f"Total steps: {self.config.total_training_steps}"
        )

    # ----------------------------------------------------------------
    # PHASE 4: Training Loop
    # ----------------------------------------------------------------

    def train(self):
        """Main training loop."""
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: Training Loop")
        self.logger.info("=" * 70)
        self.logger.info(
            f"Epochs: {self.config.num_epochs} | "
            f"Batch: {self.config.batch_size} | "
            f"Accum: {self.config.accumulation_steps} | "
            f"Effective batch: {self.config.batch_size * self.config.accumulation_steps} | "
            f"Eval every: {self.config.eval_every} epochs"
        )
        if self.config.freeze_encoder_epochs > 0:
            self.logger.info(
                f"Encoder freeze: epochs 1-{self.config.freeze_encoder_epochs}, "
                f"then full fine-tuning"
            )

        # Profile the DataLoader before training starts.
        self._profile_dataloader()

        self.model.train()
        self.step_start_time = time.time()
        self.optimizer.zero_grad()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1  # 1-indexed

            # ----------------------------------------------------------------
            # Encoder Unfreeze
            # At the start of epoch (freeze_encoder_epochs + 1), re-enable
            # gradients for the encoder. The optimizer already tracks its
            # params, so no rebuild is needed — just flip requires_grad.
            # ----------------------------------------------------------------
            if (self.config.freeze_encoder_epochs > 0 and
                    self.current_epoch == self.config.freeze_encoder_epochs + 1):
                for p in self.model.encoder.parameters():
                    p.requires_grad = True
                newly_trainable = sum(
                    p.numel() for p in self.model.encoder.parameters()
                )
                self.logger.info(
                    f"*** Epoch {self.current_epoch}: Encoder UNFROZEN — "
                    f"{newly_trainable:,} encoder params now training at "
                    f"LR={self.config.encoder_lr:.1e} ***"
                )

            self.epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            self.logger.info(
                f"\n{'='*40} Epoch {self.current_epoch}/{self.config.num_epochs} {'='*40}"
            )

            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                # Update temperature for the multi-dataset sampler
                current_temp = get_temperature_for_step(
                    self.global_step,
                    self.config.total_training_steps,
                    self.config.temp_start,
                    self.config.temp_end,
                )
                if (hasattr(self.train_loader, 'batch_sampler') and
                        hasattr(self.train_loader.batch_sampler, 'set_temperature')):
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

                # Optimizer step every accumulation_steps batches
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
                            f"Step {self.global_step:>6d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: enc={current_lr[0]:.2e} dec={current_lr[1]:.2e} | "
                            f"Temp: {current_temp:.3f} | "
                            f"Epoch: {self.current_epoch} | "
                            f"10-step time: {elapsed:.1f}s"
                        )
                        self.step_start_time = time.time()

            # Flush remaining accumulated gradients at end of epoch
            if epoch_steps % self.config.accumulation_steps != 0:
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

            # Indicate encoder freeze status in the epoch summary line
            enc_status = (
                "FROZEN" if (
                    self.config.freeze_encoder_epochs > 0 and
                    self.current_epoch <= self.config.freeze_encoder_epochs
                ) else "active"
            )
            self.logger.info(
                f"Epoch {self.current_epoch} done | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time / 60:.1f} min | "
                f"Step: {self.global_step} | "
                f"Encoder: {enc_status}"
            )

            # ----------------------------------------------------------------
            # Evaluation — run every eval_every epochs, not every epoch.
            # During warmup epochs, cap the val set to avoid long waits.
            # ----------------------------------------------------------------
            is_best = False
            if self.current_epoch % self.config.eval_every == 0:
                in_warmup = self.current_epoch <= self.config.eval_warmup_epochs
                max_samples = (
                    self.config.eval_warmup_max_samples if in_warmup else None
                )
                _, is_best = self._evaluate(
                    use_beam_search=False,
                    max_samples=max_samples,
                )

            # Early stopping counter
            if is_best:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Epoch checkpoint (every N epochs)
            if self.current_epoch % self.config.checkpoint_every_epochs == 0:
                self._save_epoch_checkpoint()

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(
                    f"Early stopping after {self.epochs_without_improvement} epochs without improvement."
                )
                break

        self.logger.info("=" * 70)
        self.logger.info("Training complete!")
        self.logger.info(f"Best ExpRate:    {self.best_exp_rate:.4f}")
        self.logger.info(f"Best EditDist:   {self.best_edit_dist:.2f}")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def _evaluate(
        self,
        use_beam_search: bool = False,
        max_samples: int = None,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Run evaluation on the validation set via engine.evaluate_full().

        Returns (metrics_dict, is_best).
        """
        sample_desc = f" (capped at {max_samples})" if max_samples else ""
        self.logger.info(
            f"Evaluating{sample_desc} (beam={use_beam_search})..."
        )

        metrics = evaluate_full(
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

        self.logger.info(
            f"  EVAL | Loss: {metrics.get('val_loss', 0.0):.4f} | "
            f"ExpRate: {metrics['exact_match']:.4f} | "
            f"EditDist: {metrics['edit_dist']:.2f} | "
            f"SER: {metrics['ser']:.4f} | "
            f"Leq1: {metrics['leq1']:.4f}"
        )

        # Early stopping on edit distance (lower is better).
        is_best = metrics['edit_dist'] < self.best_edit_dist
        if is_best:
            self.best_exp_rate = metrics['exact_match']
            self.best_edit_dist = metrics['edit_dist']
            self.logger.info(
                f"  *** New best | EditDist: {self.best_edit_dist:.2f} | "
                f"ExpRate: {self.best_exp_rate:.4f} ***"
            )

            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                self.current_epoch, self.global_step,
                {'exp_rate': self.best_exp_rate, 'edit_dist': self.best_edit_dist},
                best_path,
            )

            should_push = (
                self.current_epoch - self._last_hf_push_epoch
                >= self.config.hf_push_every_n_epochs
            )
            if should_push:
                self._last_hf_push_epoch = self.current_epoch
                _push_hf_background(best_path, self.config, self.current_epoch, is_best=True)

        return metrics, is_best

    def evaluate_with_beam_search(self, max_samples: int = 500) -> Dict[str, float]:
        """
        Public method for beam-search evaluation.
        Called by train.py in eval-only mode (--eval-only flag).
        """
        metrics, _ = self._evaluate(use_beam_search=True, max_samples=max_samples)
        return metrics

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_epoch_checkpoint(self):
        """Save an epoch checkpoint and push to HF (throttled, background)."""
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            f"epoch_{self.current_epoch}.pt"
        )
        save_checkpoint(
            self.model, self.optimizer, self.scheduler, self.scaler,
            self.current_epoch, self.global_step,
            {'exp_rate': self.best_exp_rate, 'edit_dist': self.best_edit_dist},
            ckpt_path,
        )
        cleanup_old_checkpoints(self.config.checkpoint_dir, self.config.keep_last_n_checkpoints)

        should_push = (
            self.current_epoch - self._last_hf_push_epoch
            >= self.config.hf_push_every_n_epochs
        )
        if should_push:
            self._last_hf_push_epoch = self.current_epoch
            _push_hf_background(ckpt_path, self.config, self.current_epoch, is_best=False)

    def _auto_resume(self) -> bool:
        """Auto-resume from the latest checkpoint if one exists."""
        latest = find_latest_checkpoint(self.config.checkpoint_dir)
        if latest is None:
            self.logger.info("No checkpoint found — starting from scratch")
            return False

        self.logger.info(f"Auto-resuming from: {latest}")
        epoch, step, metrics = load_checkpoint(
            latest,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            device=str(self.device),
        )
        self.current_epoch = epoch
        self.global_step = step
        self.best_exp_rate = metrics.get('exp_rate', 0.0)
        self.best_edit_dist = metrics.get('edit_dist', float('inf'))
        self.logger.info(
            f"Resumed: epoch={epoch}, step={step}, "
            f"best_edit_dist={self.best_edit_dist:.2f}"
        )
        return True

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a specific checkpoint path."""
        self.logger.info(f"Resuming from: {checkpoint_path}")
        epoch, step, metrics = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            device=str(self.device),
        )
        self.current_epoch = epoch
        self.global_step = step
        self.best_exp_rate = metrics.get('exp_rate', 0.0)
        self.best_edit_dist = metrics.get('edit_dist', float('inf'))

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def _profile_dataloader(self):
        """
        Time one batch from the DataLoader before training starts.
        If this takes > 2 seconds, your bottleneck is I/O, not GPU.
        """
        self.logger.info("Profiling DataLoader (timing first batch)...")
        t0 = time.time()
        for batch in self.train_loader:
            if batch is not None:
                break
        elapsed = time.time() - t0

        if elapsed > 2.0:
            self.logger.warning(
                f"DataLoader first batch: {elapsed:.2f}s — "
                f"SLOW. Your bottleneck is I/O, not GPU. "
                f"Check augmentation complexity and disk read speed."
            )
        else:
            self.logger.info(f"DataLoader first batch: {elapsed:.2f}s — OK")

    # ----------------------------------------------------------------
    # MAIN: Full Pipeline
    # ----------------------------------------------------------------

    def run(self, resume_from: str = None):
        """Run the complete pipeline in strict order."""
        self.preprocess_data()
        self.create_dataloaders()
        self.build_model()

        if resume_from and os.path.exists(resume_from):
            self.resume_from_checkpoint(resume_from)
        else:
            self._auto_resume()

        self.train()

        # Final evaluation with beam search (capped to 500 to avoid multi-hour wait)
        self.logger.info("Running final beam-search evaluation (500 samples)...")
        self._evaluate(use_beam_search=True, max_samples=500)

        self.logger.info("=" * 70)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 70)