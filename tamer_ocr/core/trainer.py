"""
Main Training Pipeline for TAMER OCR.

Key features:
- AMP (float16) with GradScaler
- Differential learning rates: Encoder 1e-5, Decoder 1e-4
- OneCycleLR scheduler with pct_start=0.1
- Label smoothing 0.1 in CrossEntropyLoss
- Gradient accumulation for effective batch size 32-64
- Step-based checkpointing with scaler/scheduler/optimizer states
- Google Drive backup every 1000 steps
- Dynamic temperature sampling decay
- 72-hour training schedule support
"""

import os
import time
import math
import logging
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import OneCycleLR

from ..config import Config
from ..models.tamer import TAMERModel
from ..data.dataset import MathDataset, get_collate_fn
from ..data.tokenizer import LaTeXTokenizer
from ..data.sampler import (
    TemperatureSampler,
    MultiDatasetBatchSampler,
    get_temperature_for_step,
)
from ..data.data_manager import DataManager
from ..data.augmentation import get_train_augmentation, get_val_augmentation
from ..data.latex_normalizer import normalize_latex
from .losses import LabelSmoothedCELoss
from .inference import beam_search, greedy_decode
from ..utils.checkpoint import save_checkpoint, load_checkpoint, backup_to_drive, push_to_huggingface
from ..utils.metrics import calculate_metrics, compute_batch_metrics
from ..logger import setup_logger

logger = logging.getLogger("TAMER.Trainer")


class Trainer:
    """
    Orchestrates the entire training pipeline.

    All logic lives here — the notebook just calls trainer.train().
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup logging
        self.logger = setup_logger("TAMER.Trainer", config.log_dir)
        self.logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        # Tokenizer (built from data or loaded)
        self.tokenizer = LaTeXTokenizer()

        # Model (initialized after tokenizer is built)
        self.model: Optional[TAMERModel] = None

        # Optimizer & Scheduler
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # AMP GradScaler

        # Loss
        self.criterion = LabelSmoothedCELoss(
            pad_id=0,  # Will be updated after tokenizer build
            label_smoothing=config.label_smoothing,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_exp_rate = 0.0
        self.best_edit_dist = float('inf')

        # Data
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.data_manager = DataManager(config)

        # Datasets ranges for temperature sampling
        self.dataset_ranges = {}

        # Timing
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

    # ----------------------------------------------------------------
    # PHASE 1: Data Preparation
    # ----------------------------------------------------------------

    def prepare_data(self, force_refresh: bool = False):
        """
        Load, normalize, and split all datasets.
        Builds the global tokenizer from the combined training corpus.
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: Data Preparation")
        self.logger.info("=" * 70)

        # Load all stages
        s1_im2latex, s2_mathwriting, s3_combined = self.data_manager.load_all_stages(force_refresh)

        # Filter by max token length
        all_samples = s1_im2latex + s2_mathwriting + s3_combined
        self.logger.info(f"Total samples before filtering: {len(all_samples)}")

        filtered = []
        for s in all_samples:
            latex = s.get('latex', '')
            if not latex:
                continue
            # Quick token length estimate (space-separated)
            approx_tokens = len(latex.split())
            if approx_tokens <= self.config.max_token_length:
                filtered.append(s)

        self.logger.info(f"Samples after max_token_length filter: {len(filtered)}")

        # Split: 90% train, 10% val
        import random
        random.seed(42)
        random.shuffle(filtered)

        split_idx = int(len(filtered) * 0.9)
        train_samples = filtered[:split_idx]
        val_samples = filtered[split_idx:]

        self.logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

        # Build global tokenizer from training data
        self.logger.info("Building global tokenizer from training corpus...")
        self.tokenizer = LaTeXTokenizer()
        self.tokenizer.build_from_samples(train_samples)
        self.tokenizer.save(os.path.join(self.config.output_dir, "tokenizer.json"))

        # Update criterion pad_id
        self.criterion = LabelSmoothedCELoss(
            pad_id=self.tokenizer.pad_id,
            label_smoothing=self.config.label_smoothing,
        )

        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        # Compute dataset ranges for temperature sampling
        self._compute_dataset_ranges(train_samples)

        # Create datasets
        train_transform = get_train_augmentation(self.config.img_height, self.config.img_width)
        val_transform = get_val_augmentation()

        self.train_dataset = MathDataset(train_samples, self.config, self.tokenizer, train_transform)
        self.val_dataset = MathDataset(val_samples, self.config, self.tokenizer, val_transform)

        # Create data loaders
        self._create_data_loaders()

        self.logger.info(f"Data preparation complete. Train dataset: {len(self.train_dataset)} samples")

    def _compute_dataset_ranges(self, samples):
        """Compute per-dataset index ranges for temperature sampling."""
        self.dataset_ranges = {}
        dataset_counts = {}

        for s in samples:
            name = s.get('dataset_name', 'unknown')
            dataset_counts[name] = dataset_counts.get(name, 0) + 1

        offset = 0
        for name, count in dataset_counts.items():
            self.dataset_ranges[name] = (offset, offset + count)
            offset += count
            self.logger.info(f"  {name}: {count} samples (range {offset-count}-{offset})")

    def _create_data_loaders(self):
        """Create train and val data loaders with temperature-based sampling."""
        collate_fn = get_collate_fn(self.tokenizer.pad_id)

        # Training loader with temperature-based batch sampling
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
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True,
            )

        # Validation loader — simple sequential
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    # ----------------------------------------------------------------
    # PHASE 2: Model Initialization
    # ----------------------------------------------------------------

    def build_model(self):
        """
        Initialize the Swin-Base + Transformer Decoder model.
        Sets up differential learning rates and OneCycleLR scheduler.
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 2: Model Initialization")
        self.logger.info("=" * 70)

        vocab_size = len(self.tokenizer)
        self.model = TAMERModel(vocab_size, self.config)
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model size: ~{total_params * 4 / 1e9:.2f} GB (float32)")

        # Differential learning rates
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        param_groups = [
            {'params': encoder_params, 'lr': self.config.encoder_lr, 'name': 'encoder'},
            {'params': decoder_params, 'lr': self.config.decoder_lr, 'name': 'decoder'},
        ]

        # AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        self.logger.info(f"Encoder LR: {self.config.encoder_lr}, Decoder LR: {self.config.decoder_lr}")
        self.logger.info(f"Weight decay: {self.config.weight_decay}")

        # OneCycleLR scheduler
        total_steps = self.config.total_training_steps
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.config.encoder_lr, self.config.decoder_lr],
            total_steps=total_steps,
            pct_start=self.config.pct_start,
            anneal_strategy='cos',
            div_factor=25.0,  # Initial LR = max_lr/25
            final_div_factor=10000.0,
        )

        self.logger.info(f"OneCycleLR: total_steps={total_steps}, pct_start={self.config.pct_start}")

        # AMP GradScaler
        self.scaler = torch.amp.GradScaler('cuda')

        self.logger.info("Model initialization complete.")

    # ----------------------------------------------------------------
    # PHASE 3: Training Loop
    # ----------------------------------------------------------------

    def train(self):
        """
        Main training loop.

        Handles:
        - Epoch/step-based iteration
        - AMP autocast
        - Gradient accumulation
        - Dynamic temperature decay
        - Step-based checkpointing
        - Periodic evaluation
        - Google Drive backups
        - Hugging Face pushes
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: Training Loop")
        self.logger.info("=" * 70)
        self.logger.info(f"Effective batch size: {self.config.batch_size * self.config.accumulation_steps}")
        self.logger.info(f"Total training steps: {self.config.total_training_steps}")
        self.logger.info(f"Checkpoint every {self.config.checkpoint_every_steps} steps")

        self.model.train()
        self.step_start_time = time.time()
        self.optimizer.zero_grad()

        while self.global_step < self.config.total_training_steps:
            self.current_epoch += 1
            self.epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            self.logger.info(f"\n--- Epoch {self.current_epoch} ---")

            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                if self.global_step >= self.config.total_training_steps:
                    break

                # Dynamic temperature update
                current_temp = get_temperature_for_step(
                    self.global_step,
                    self.config.total_training_steps,
                    self.config.temp_start,
                    self.config.temp_end,
                )
                # Update batch sampler temperature if applicable
                if hasattr(self.train_loader, 'batch_sampler') and \
                   hasattr(self.train_loader.batch_sampler, 'set_temperature'):
                    self.train_loader.batch_sampler.set_temperature(current_temp)

                loss = self._train_step(batch)
                epoch_loss += loss
                epoch_steps += 1

                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    # Unscale gradients
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Step optimizer
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Step scheduler (step-based, not epoch-based)
                    self.scheduler.step()

                    # Zero gradients
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % 10 == 0:
                        elapsed = time.time() - self.step_start_time
                        steps_per_sec = self.config.accumulation_steps / max(elapsed, 0.001)
                        current_lr = self.scheduler.get_last_lr()
                        avg_loss = epoch_loss / max(epoch_steps, 1)

                        self.logger.info(
                            f"Step {self.global_step:>6d}/{self.config.total_training_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: enc={current_lr[0]:.2e} dec={current_lr[1]:.2e} | "
                            f"Temp: {current_temp:.3f} | "
                            f"Speed: {steps_per_sec:.2f} steps/s"
                        )
                        self.step_start_time = time.time()

                    # Step-based checkpointing
                    if self.global_step % self.config.checkpoint_every_steps == 0:
                        self._save_step_checkpoint()

                        # Evaluate
                        self._evaluate()

                        # Backup to Drive
                        if self.config.drive_backup_dir:
                            ckpt_path = os.path.join(
                                self.config.checkpoint_dir,
                                f"step_{self.global_step}.pt"
                            )
                            backup_to_drive(ckpt_path, self.config.drive_backup_dir,
                                            self.config.keep_last_n_checkpoints)

            # End of epoch
            epoch_time = time.time() - self.epoch_start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self.logger.info(
                f"Epoch {self.current_epoch} complete: "
                f"Avg Loss={avg_epoch_loss:.4f}, "
                f"Time={epoch_time/60:.1f}min, "
                f"Global Step={self.global_step}"
            )

        self.logger.info("=" * 70)
        self.logger.info("Training complete!")
        self.logger.info(f"Best ExpRate: {self.best_exp_rate:.4f}")
        self.logger.info(f"Best Edit Distance: {self.best_edit_dist:.2f}")

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """
        Single training step with AMP.

        Returns:
            Scalar loss value
        """
        images = batch['image'].to(self.device, non_blocking=True)
        ids = batch['ids'].to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = self.model(images, ids)  # (B, L, V)
            loss = self.criterion(logits, ids)
            loss = loss / self.config.accumulation_steps  # Scale for accumulation

        # Backward pass with scaler
        self.scaler.scale(loss).backward()

        return loss.item() * self.config.accumulation_steps  # Unscale for logging

    # ----------------------------------------------------------------
    # PHASE 4: Evaluation
    # ----------------------------------------------------------------

    def _evaluate(self):
        """Run evaluation on the validation set."""
        self.logger.info("Running evaluation...")
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue

                images = batch['image'].to(self.device)
                ids = batch['ids'].to(self.device)

                # Compute loss
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = self.model(images, ids)
                    loss = self.criterion(logits, ids)
                total_loss += loss.item()
                num_batches += 1

                # Decode predictions (greedy for speed during training eval)
                for i in range(images.size(0)):
                    pred_tokens = greedy_decode(
                        self.model,
                        images[i:i+1],
                        self.tokenizer.sos_id,
                        self.tokenizer.eos_id,
                        max_len=self.config.max_seq_len,
                        device=self.device,
                    )
                    pred_latex = self.tokenizer.decode(pred_tokens, skip_special=True)

                    # Ground truth
                    gt_ids = ids[i].cpu().tolist()
                    gt_latex = self.tokenizer.decode(gt_ids, skip_special=True)

                    all_preds.append(pred_latex)
                    all_targets.append(gt_latex)

        # Compute metrics
        metrics = compute_batch_metrics(all_preds, all_targets)
        avg_loss = total_loss / max(num_batches, 1)

        self.logger.info(
            f"  EVAL | Loss: {avg_loss:.4f} | "
            f"ExpRate: {metrics['exact_match']:.4f} | "
            f"EditDist: {metrics['edit_dist']:.2f} | "
            f"SER: {metrics['ser']:.4f} | "
            f"Leq1: {metrics['leq1']:.4f}"
        )

        # Track best model
        is_best = metrics['exact_match'] > self.best_exp_rate
        if is_best:
            self.best_exp_rate = metrics['exact_match']
            self.best_edit_dist = metrics['edit_dist']
            self.logger.info(f"  *** New best ExpRate: {self.best_exp_rate:.4f} ***")

            # Save best checkpoint
            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                self.current_epoch, self.global_step,
                {'exp_rate': self.best_exp_rate, 'edit_dist': self.best_edit_dist},
                best_path,
            )

            # Push best to HuggingFace
            if self.config.hf_repo_id:
                push_to_huggingface(best_path, self.config, self.global_step, is_best=True)

        self.model.train()
        return metrics

    def evaluate_with_beam_search(self, max_samples: int = 200):
        """
        Full evaluation with beam search (slower but more accurate).
        Use this for final evaluation, not during training.
        """
        self.logger.info(f"Running beam search evaluation (max {max_samples} samples)...")
        self.model.eval()

        all_preds = []
        all_targets = []
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue

                images = batch['image'].to(self.device)
                ids = batch['ids'].to(self.device)

                for i in range(images.size(0)):
                    if count >= max_samples:
                        break

                    pred_tokens = beam_search(
                        self.model,
                        images[i:i+1],
                        self.tokenizer.sos_id,
                        self.tokenizer.eos_id,
                        self.tokenizer.pad_id,
                        beam_width=self.config.beam_width,
                        max_len=self.config.max_seq_len,
                        length_penalty=self.config.length_penalty,
                        device=self.device,
                    )
                    pred_latex = self.tokenizer.decode(pred_tokens, skip_special=True)

                    gt_ids = ids[i].cpu().tolist()
                    gt_latex = self.tokenizer.decode(gt_ids, skip_special=True)

                    all_preds.append(pred_latex)
                    all_targets.append(gt_latex)
                    count += 1

                if count >= max_samples:
                    break

        metrics = compute_batch_metrics(all_preds, all_targets)
        self.logger.info(
            f"BEAM SEARCH EVAL | "
            f"ExpRate: {metrics['exact_match']:.4f} | "
            f"EditDist: {metrics['edit_dist']:.2f} | "
            f"SER: {metrics['ser']:.4f} | "
            f"Leq1: {metrics['leq1']:.4f}"
        )

        self.model.train()
        return metrics

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_step_checkpoint(self):
        """Save a step-based checkpoint with all training state."""
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            f"step_{self.global_step}.pt"
        )
        metrics = {
            'exp_rate': self.best_exp_rate,
            'edit_dist': self.best_edit_dist,
        }
        save_checkpoint(
            self.model, self.optimizer, self.scheduler, self.scaler,
            self.current_epoch, self.global_step,
            metrics, ckpt_path,
        )

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        # Push to HuggingFace
        if self.config.hf_repo_id:
            push_to_huggingface(ckpt_path, self.config, self.global_step, is_best=False)

    def _cleanup_old_checkpoints(self):
        """Keep only the last N step checkpoints (not counting best.pt)."""
        import glob
        ckpt_files = sorted(
            glob.glob(os.path.join(self.config.checkpoint_dir, "step_*.pt")),
            key=os.path.getmtime,
        )
        keep = self.config.keep_last_n_checkpoints
        if len(ckpt_files) > keep:
            for old_ckpt in ckpt_files[:-keep]:
                try:
                    os.remove(old_ckpt)
                except OSError:
                    pass

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
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
        self.logger.info(
            f"Resumed: epoch={epoch}, step={step}, "
            f"best_exp_rate={self.best_exp_rate:.4f}"
        )

    # ----------------------------------------------------------------
    # Convenience: Full Pipeline
    # ----------------------------------------------------------------

    def run(self, resume_from: str = None):
        """
        Run the complete training pipeline.

        Args:
            resume_from: Optional path to a checkpoint to resume from
        """
        # Step 1: Prepare data
        self.prepare_data()

        # Step 2: Build model
        self.build_model()

        # Step 2.5: Resume if specified
        if resume_from and os.path.exists(resume_from):
            self.resume_from_checkpoint(resume_from)

        # Step 3: Train
        self.train()

        # Step 4: Final beam search evaluation
        self.evaluate_with_beam_search(max_samples=500)

        self.logger.info("=" * 70)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 70)
