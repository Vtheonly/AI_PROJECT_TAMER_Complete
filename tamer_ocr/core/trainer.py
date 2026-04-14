"""
Main Training Pipeline for TAMER OCR v2.1.

Strict pipeline order:
  1. Preprocess ALL datasets (via DatasetPreprocessor)
  2. Push to HuggingFace dataset repo
  3. Build model
  4. Auto-resume from latest checkpoint if available
  5. Train with epoch-based checkpointing (every 3 epochs)
  6. Push checkpoints to HuggingFace model repo
"""

import os
import time
import gc
import math
import logging
import random
from typing import Dict, Optional, Any, List, Tuple

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


class Trainer:
    """
    Orchestrates the entire training pipeline.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = self.device.type == 'cuda'

        # Setup logging
        self.logger = setup_logger("TAMER.Trainer", config.log_dir)
        self.logger.info(f"Device: {self.device} (AMP Enabled: {self.use_amp})")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram_gb = getattr(props, 'total_memory', 0) / 1e9
            self.logger.info(f"VRAM: {vram_gb:.1f} GB")

        # Tokenizer — built during preprocessing
        self.tokenizer = LaTeXTokenizer()

        # Model (initialized after tokenizer is built)
        self.model: Optional[TAMERModel] = None

        # Optimizer & Scheduler
        self.optimizer = None
        self.scheduler = None
        
        # FIX: Modern PyTorch 2.0+ AMP GradScaler API
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Loss
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

        # Dataset ranges for temperature sampling
        self.dataset_ranges = {}

        # Processed samples
        self.train_samples = []
        self.val_samples = []

        # Timing
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

    # ----------------------------------------------------------------
    # PHASE 1: Data Preprocessing
    # ----------------------------------------------------------------

    def preprocess_data(self):
        """
        Run the FULL preprocessing pipeline.
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: Data Preprocessing (STRICT — no training until complete)")
        self.logger.info("=" * 70)

        preprocessor = DatasetPreprocessor(self.config)
        all_processed, self.tokenizer = preprocessor.run_full_pipeline()

        # Flatten all datasets into a single list
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
        grouped = {}
        for s in filtered:
            ds = s.get('dataset_name', 'unknown')
            if ds not in grouped:
                grouped[ds] = []
            grouped[ds].append(s)

        self.train_samples = []
        self.val_samples = []

        # FIX: Instantiate RNG once outside the loop
        rng = random.Random(42)
        for ds, ds_samples in grouped.items():
            rng.shuffle(ds_samples)
            split_idx = int(len(ds_samples) * 0.9)
            self.train_samples.extend(ds_samples[:split_idx])
            self.val_samples.extend(ds_samples[split_idx:])

        # FIX: Sort by dataset_name to guarantee contiguous ranges for MultiDatasetBatchSampler
        self.train_samples.sort(key=lambda x: x.get('dataset_name', 'unknown'))
        self.val_samples.sort(key=lambda x: x.get('dataset_name', 'unknown'))

        self.logger.info(f"Train: {len(self.train_samples)}, Val: {len(self.val_samples)}")
        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        # Update criterion pad_id
        self.criterion = LabelSmoothedCELoss(
            pad_id=self.tokenizer.pad_id,
            label_smoothing=self.config.label_smoothing,
        )

        # Save tokenizer to output dir
        self.tokenizer.save(os.path.join(self.config.output_dir, "tokenizer.json"))

    # ----------------------------------------------------------------
    # PHASE 2: Create DataLoaders
    # ----------------------------------------------------------------

    def create_dataloaders(self):
        """Create train and val data loaders from preprocessed samples."""
        self.logger.info("Creating data loaders...")

        # Compute dataset ranges for temperature sampling
        self._compute_dataset_ranges(self.train_samples)

        # Create datasets
        train_transform = get_train_augmentation(self.config.img_height, self.config.img_width)
        val_transform = get_val_augmentation()

        self.train_dataset = MathDataset(self.train_samples, self.config, self.tokenizer, train_transform)
        self.val_dataset = MathDataset(self.val_samples, self.config, self.tokenizer, val_transform)

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

        # Validation loader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self.logger.info(f"Train dataset: {len(self.train_dataset)} samples, Val: {len(self.val_dataset)} samples")

    def _compute_dataset_ranges(self, samples):
        """Compute per-dataset index ranges for temperature sampling."""
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
            self.logger.info(f"  {name}: {rng[1]-rng[0]} samples (range {rng[0]}-{rng[1]})")

    # ----------------------------------------------------------------
    # PHASE 3: Model Initialization
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

        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        self.logger.info(f"Encoder LR: {self.config.encoder_lr}, Decoder LR: {self.config.decoder_lr}")

        # FIX: Use math.ceil to prevent undercounting steps and crashing OneCycleLR
        steps_per_epoch = math.ceil(len(self.train_loader) / self.config.accumulation_steps)
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        self.config.total_training_steps = steps_per_epoch * self.config.num_epochs

        # OneCycleLR scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.config.encoder_lr, self.config.decoder_lr],
            total_steps=self.config.total_training_steps,
            pct_start=self.config.pct_start,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        self.logger.info(f"OneCycleLR: total_steps={self.config.total_training_steps}, pct_start={self.config.pct_start}")
        self.logger.info("Model initialization complete.")

    # ----------------------------------------------------------------
    # PHASE 4: Training Loop
    # ----------------------------------------------------------------

    def train(self):
        """Main training loop."""
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: Training Loop")
        self.logger.info("=" * 70)
        self.logger.info(f"Effective batch size: {self.config.batch_size * self.config.accumulation_steps}")
        self.logger.info(f"Num epochs: {self.config.num_epochs}")
        self.logger.info(f"Checkpoint every {self.config.checkpoint_every_epochs} epochs")

        self.model.train()
        self.step_start_time = time.time()
        self.optimizer.zero_grad()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1  # 1-indexed
            self.epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            self.logger.info(f"\n{'='*40} Epoch {self.current_epoch}/{self.config.num_epochs} {'='*40}")

            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                # Dynamic temperature update
                current_temp = get_temperature_for_step(
                    self.global_step,
                    self.config.total_training_steps,
                    self.config.temp_start,
                    self.config.temp_end,
                )
                if hasattr(self.train_loader, 'batch_sampler') and \
                   hasattr(self.train_loader.batch_sampler, 'set_temperature'):
                    self.train_loader.batch_sampler.set_temperature(current_temp)

                # FIX: Use engine.py train_step
                loss = train_step(
                    model=self.model,
                    batch=batch,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    device=self.device,
                    accumulation_steps=self.config.accumulation_steps,
                    max_grad_norm=self.config.max_grad_norm
                )
                epoch_loss += loss
                epoch_steps += 1

                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    # FIX: Use engine.py optimizer_step
                    optimizer_step(
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        scheduler=self.scheduler,
                        max_grad_norm=self.config.max_grad_norm
                    )
                    self.global_step += 1

                    # Logging every 10 optimizer steps
                    if self.global_step % 10 == 0:
                        elapsed = time.time() - self.step_start_time
                        current_lr = self.scheduler.get_last_lr()
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        self.logger.info(
                            f"Step {self.global_step:>6d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: enc={current_lr[0]:.2e} dec={current_lr[1]:.2e} | "
                            f"Temp: {current_temp:.3f} | "
                            f"Epoch: {self.current_epoch}"
                        )
                        self.step_start_time = time.time()

            # FIX: Step optimizer for remaining accumulated gradients at the end of epoch
            if epoch_steps % self.config.accumulation_steps != 0:
                optimizer_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=self.scheduler,
                    max_grad_norm=self.config.max_grad_norm
                )
                self.global_step += 1

            # End of epoch — always evaluate
            epoch_time = time.time() - self.epoch_start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self.logger.info(
                f"Epoch {self.current_epoch} complete: "
                f"Avg Loss={avg_epoch_loss:.4f}, "
                f"Time={epoch_time/60:.1f}min, "
                f"Global Step={self.global_step}"
            )

            # Evaluate
            metrics, is_best = self._evaluate(use_beam_search=False)

            # Early Stopping Logic
            if is_best:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Checkpoint every N epochs
            if self.current_epoch % self.config.checkpoint_every_epochs == 0:
                self._save_epoch_checkpoint()

            # Free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Early Stopping Check
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement.")
                break

        self.logger.info("=" * 70)
        self.logger.info("Training complete!")
        self.logger.info(f"Best ExpRate: {self.best_exp_rate:.4f}")
        self.logger.info(f"Best Edit Distance: {self.best_edit_dist:.2f}")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def _evaluate(self, use_beam_search: bool = False, max_samples: int = None) -> Tuple[Dict[str, float], bool]:
        """Run evaluation on the validation set using engine.py."""
        self.logger.info(f"Running evaluation (Beam Search: {use_beam_search})...")
        
        # FIX: Use the fully batched evaluate_full from engine.py
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
            max_samples=max_samples
        )

        self.logger.info(
            f"  EVAL | Loss: {metrics.get('val_loss', 0.0):.4f} | "
            f"ExpRate: {metrics['exact_match']:.4f} | "
            f"EditDist: {metrics['edit_dist']:.2f} | "
            f"SER: {metrics['ser']:.4f} | "
            f"Leq1: {metrics['leq1']:.4f}"
        )

        # FIX: Early stopping on Edit Distance (lower is better) instead of Exact Match
        is_best = metrics['edit_dist'] < self.best_edit_dist
        if is_best:
            self.best_exp_rate = metrics['exact_match']
            self.best_edit_dist = metrics['edit_dist']
            self.logger.info(f"  *** New best EditDist: {self.best_edit_dist:.2f} (ExpRate: {self.best_exp_rate:.4f}) ***")

            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                self.current_epoch, self.global_step,
                {'exp_rate': self.best_exp_rate, 'edit_dist': self.best_edit_dist},
                best_path,
            )

            # Push best to HuggingFace model repo
            push_checkpoint_to_hf(best_path, self.config, self.current_epoch, is_best=True)

        return metrics, is_best

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    def _save_epoch_checkpoint(self):
        """Save an epoch-based checkpoint and push to HF."""
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            f"epoch_{self.current_epoch}.pt"
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
        cleanup_old_checkpoints(self.config.checkpoint_dir, self.config.keep_last_n_checkpoints)

        # Push to HuggingFace model repo
        push_checkpoint_to_hf(ckpt_path, self.config, self.current_epoch, is_best=False)

    def _auto_resume(self) -> bool:
        """
        Auto-resume from the latest checkpoint if one exists.
        """
        latest = find_latest_checkpoint(self.config.checkpoint_dir)
        if latest is None:
            self.logger.info("No checkpoint found — starting from scratch")
            return False

        self.logger.info(f"Auto-resuming from latest checkpoint: {latest}")
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
        """Resume training from a specific checkpoint."""
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

    # ----------------------------------------------------------------
    # MAIN: Full Pipeline
    # ----------------------------------------------------------------

    def run(self, resume_from: str = None):
        """
        Run the complete pipeline in strict order.
        """
        self.preprocess_data()
        self.create_dataloaders()
        self.build_model()

        if resume_from and os.path.exists(resume_from):
            self.resume_from_checkpoint(resume_from)
        else:
            self._auto_resume()

        self.train()
        
        # Final evaluation with beam search
        self._evaluate(use_beam_search=True, max_samples=500)

        self.logger.info("=" * 70)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 70)