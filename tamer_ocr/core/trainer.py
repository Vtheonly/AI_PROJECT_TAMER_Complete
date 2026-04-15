"""
Main Training Pipeline for TAMER OCR v2.3.

Changes from v2.2:
  - Curriculum learning: progressively introduces harder samples.
    Phase 1 (simple only) → Phase 2 (+aligned/cases) → Phase 3 (+matrices).
    Controlled by config.curriculum_enabled and epoch thresholds.
    DataLoaders are rebuilt when the curriculum stage advances.

  - Structure-aware loss: structural tokens (\\\\, &, \\begin{env}, \\end{env})
    are weighted 3× in the loss. Controlled by config.structure_aware_loss.

  - Structural accuracy metrics: evaluation now reports per-complexity
    ExpRate (simple/medium/complex) and structural token recall.

  - Normalizer no longer discards matrix/aligned/cases samples.
    The total dataset is ~15-25% larger than v2.2.

  - Tokenizer now handles \\\\, \\begin{env}, \\end{env} as atomic tokens.

  - H100 optimizations: val DataLoader uses batch_size // 2 to prevent
    OOM during evaluation at large batch sizes. TF32 flags and
    torch.compile are set in train.py at process startup.

All v2.2 features retained:
  - Encoder freeze/unfreeze, gradient checkpointing
  - persistent_workers, prefetch_factor, HF push throttling
  - eval_every, eval_warmup, torch.compile

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
# Runs in a daemon thread so the training loop never blocks on a network call.
# ---------------------------------------------------------------------------

def _push_hf_background(
    checkpoint_path: str,
    config: Config,
    epoch: int,
    is_best: bool,
) -> threading.Thread:
    """
    Push a checkpoint to HuggingFace Hub in a background daemon thread.

    Using a daemon thread means:
      - Training is never blocked by network latency.
      - If the main process exits (e.g. training finishes), the push
        thread is killed automatically — no zombie processes.
      - Failures are logged as warnings, not exceptions, so training
        continues uninterrupted even if HF is temporarily unreachable.
    """
    def _worker():
        try:
            push_checkpoint_to_hf(checkpoint_path, config, epoch, is_best=is_best)
            logger.info(f"HF push complete (epoch {epoch}, best={is_best})")
        except Exception as e:
            logger.warning(f"Background HF push failed (epoch {epoch}): {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


class Trainer:
    """
    Orchestrates the full TAMER OCR training pipeline.

    Pipeline stages (called in order by run()):
      1. preprocess_data()     — download, parse, normalise, split
      2. create_dataloaders()  — build Dataset + DataLoader objects
      3. build_model()         — init model, optimiser, scheduler
      4. _auto_resume()        — load latest checkpoint if present
      5. train()               — main epoch loop
      6. _evaluate(beam=True)  — final beam-search evaluation
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
            vram_gb = props.total_memory / 1e9
            self.logger.info(f"VRAM: {vram_gb:.1f} GB")
            self.logger.info(
                f"Image resolution: {config.img_height}×{config.img_width} "
                f"→ {(config.img_height // 4) * (config.img_width // 4):,} "
                f"patches per image"
            )
            # Log TF32 status so users can verify the flags are active
            self.logger.info(
                f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32} | "
                f"cuDNN benchmark: {torch.backends.cudnn.benchmark}"
            )

        self.tokenizer = LaTeXTokenizer()
        self.model: Optional[TAMERModel] = None

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Placeholder criterion — replaced after tokenizer is built in
        # preprocess_data() once we know the vocabulary and pad_id.
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
        self.all_train_samples: List = []  # Full unfiltered train set for curriculum

        # ── Timing ─────────────────────────────────────────────────
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

        # ── HF Push Throttle ───────────────────────────────────────
        # Track the last epoch we pushed to HF to prevent repeated pushes.
        self._last_hf_push_epoch: int = -1

        # ── Curriculum State ───────────────────────────────────────
        self._current_curriculum_stage: str = 'simple'

    # ----------------------------------------------------------------
    # PHASE 1: Data Preprocessing
    # ----------------------------------------------------------------

    def preprocess_data(self):
        """
        Run the full preprocessing pipeline via DatasetPreprocessor.

        Steps performed here:
          1. Download / locate raw datasets
          2. Parse each dataset into (image_path, latex) pairs
          3. Normalise LaTeX strings
          4. Build / extend the tokenizer vocabulary
          5. Filter by token length
          6. Stratified 90/10 train/val split per dataset
          7. Annotate each sample with its complexity bucket
          8. Instantiate the appropriate loss function
          9. Save the tokenizer to disk
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: Data Preprocessing")
        self.logger.info("=" * 70)

        preprocessor = DatasetPreprocessor(self.config)
        all_processed, self.tokenizer = preprocessor.run_full_pipeline()

        all_samples = []
        for dataset_name, samples in all_processed.items():
            all_samples.extend(samples)

        self.logger.info(f"Total preprocessed samples: {len(all_samples)}")

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
            f"After token length filter (≤{self.config.max_token_length}): "
            f"{len(filtered)} samples"
        )

        # ── Stratified Train / Val Split ───────────────────────────
        # Group by dataset name so each source gets its own 90/10 split.
        # This prevents any single large dataset from dominating the val set.
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
            train_part = ds_samples[:split_idx]
            val_part = ds_samples[split_idx:]
            self.train_samples.extend(train_part)
            self.val_samples.extend(val_part)
            self.logger.info(
                f"  {ds}: {len(train_part)} train, {len(val_part)} val"
            )

        # Sort by dataset_name so MultiDatasetBatchSampler gets contiguous ranges.
        self.train_samples.sort(key=lambda x: x.get('dataset_name', 'unknown'))
        self.val_samples.sort(key=lambda x: x.get('dataset_name', 'unknown'))

        self.logger.info(
            f"Split totals → Train: {len(self.train_samples)}, "
            f"Val: {len(self.val_samples)}"
        )
        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        # ── Complexity Distribution ────────────────────────────────
        complexity_counts = {'simple': 0, 'medium': 0, 'complex': 0}
        for s in self.train_samples:
            c = s.get('complexity', get_complexity(s.get('latex', '')))
            complexity_counts[c] = complexity_counts.get(c, 0) + 1
        self.logger.info(
            f"Train complexity distribution: "
            f"simple={complexity_counts['simple']:,}, "
            f"medium={complexity_counts['medium']:,}, "
            f"complex={complexity_counts['complex']:,}"
        )

        # Store the full train set for curriculum learning.
        # curriculum logic will filter this down each phase.
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
                f"(structural_weight={self.config.structural_token_weight})"
            )
        else:
            self.criterion = LabelSmoothedCELoss(
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.config.label_smoothing,
            ).to(self.device)
            self.logger.info("Loss: LabelSmoothedCELoss")

        # ── Persist Tokenizer ──────────────────────────────────────
        tokenizer_path = os.path.join(self.config.output_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        self.logger.info(f"Tokenizer saved → {tokenizer_path}")

    # ----------------------------------------------------------------
    # PHASE 2: Create DataLoaders
    # ----------------------------------------------------------------

    def create_dataloaders(self):
        """
        Build MathDataset instances and wrap them in DataLoaders.

        Train loader:
          - Uses MultiDatasetBatchSampler for temperature-based
            per-dataset sampling when multiple datasets are present.
          - Falls back to a standard shuffled DataLoader for single-
            dataset scenarios.
          - persistent_workers=True keeps worker processes alive
            between epochs (avoids 280+ process spawns over 70 epochs).
          - prefetch_factor=2 keeps 2 batches pre-loaded per worker
            so the GPU never waits on CPU augmentation.

        Val loader:
          - batch_size = config.batch_size // 2
            During validation we run forward passes without the memory
            savings of gradient checkpointing. At batch_size=192 this
            would OOM on most GPUs; halving it keeps peak VRAM safe
            while still being fast (no augmentation overhead).
          - No shuffling — deterministic evaluation order.
        """
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

        # ── Train DataLoader ───────────────────────────────────────
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

        # ── Val DataLoader ─────────────────────────────────────────
        # SAFETY FIX: halve batch size for validation.
        # Validation runs without gradient checkpointing, so peak VRAM
        # per-sample is higher than during training. At config.batch_size
        # = 192 this would OOM; // 2 = 96 keeps us safely in budget.
        # Minimum of 1 prevents edge cases with tiny batch configs.
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
            f"Val: {len(self.val_dataset):,} samples | "
            f"Train batch: {self.config.batch_size} | "
            f"Val batch: {val_batch_size} | "
            f"Workers: {self.config.num_workers}"
        )

    def _compute_dataset_ranges(self, samples: List):
        """
        Compute per-dataset contiguous index ranges for temperature sampling.

        MultiDatasetBatchSampler needs to know where each dataset's
        samples start and end in the flat sample list. Since we sorted
        samples by dataset_name in preprocess_data(), these ranges are
        guaranteed to be contiguous.
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
                f"  {name}: {rng[1] - rng[0]:,} samples "
                f"(idx {rng[0]}–{rng[1] - 1})"
            )

    # ----------------------------------------------------------------
    # PHASE 3: Model Initialization
    # ----------------------------------------------------------------

    def build_model(self):
        """
        Initialize Swin-v2 encoder + Transformer decoder (10 layers).

        Applies:
          - Differential learning rates (encoder 1e-5, decoder 1e-4)
          - Optional encoder freeze for the first freeze_encoder_epochs epochs
          - Optional torch.compile()
          - OneCycleLR scheduler sized to total_training_steps

        OneCycleLR note:
          math.ceil is used for steps_per_epoch to prevent undercounting
          when len(loader) % accumulation_steps != 0. Undercounting would
          cause the scheduler to exhaust its budget before training ends,
          leaving the last steps at an extremely low LR.
        """
        self.logger.info("=" * 70)
        self.logger.info("PHASE 2: Model Initialization")
        self.logger.info("=" * 70)

        vocab_size = len(self.tokenizer)
        self.model = TAMERModel(vocab_size, self.config)
        self.model = self.model.to(self.device)

        # ── torch.compile ──────────────────────────────────────────
        # torch.compile() traces the model's forward pass and compiles
        # it into optimised GPU kernels (Triton on CUDA). On H100 this
        # gives 20-40% throughput improvement. The first forward pass
        # triggers JIT compilation — expect a 3-5 minute pause on
        # epoch 1, step 1. Every subsequent step runs the cached kernel.
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.logger.info(
                "Applying torch.compile() — first epoch will pause ~3-5 min "
                "for JIT compilation (H100 Tensor Core kernel generation)"
            )
            self.model = torch.compile(self.model)

        # ── Parameter counts ───────────────────────────────────────
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Total parameters     : {total_params:,}")
        self.logger.info(f"Trainable parameters : {trainable:,}")
        self.logger.info(f"Model size (fp32)    : ~{total_params * 4 / 1e9:.2f} GB")
        self.logger.info(f"Model size (fp16)    : ~{total_params * 2 / 1e9:.2f} GB")

        # ── Encoder Freeze Warmup ──────────────────────────────────
        # Freeze encoder weights for the first freeze_encoder_epochs
        # epochs. The decoder (randomly initialised) learns to interpret
        # the encoder's existing feature map before we allow gradients
        # to flow back into it. This prevents the strong gradient signal
        # from a random decoder from corrupting pre-trained encoder weights.
        #
        # The optimizer still receives encoder params so the LR schedule
        # is consistent. Gradients are zeroed by requires_grad=False.
        # Unfreezing in train() just flips requires_grad back to True —
        # no optimizer rebuild required.
        if self.config.freeze_encoder_epochs > 0:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            frozen_count = sum(
                p.numel() for p in self.model.encoder.parameters()
            )
            active_count = trainable - frozen_count
            self.logger.info(
                f"Encoder FROZEN for epochs 1-{self.config.freeze_encoder_epochs} "
                f"({frozen_count:,} params frozen, {active_count:,} active)"
            )
        else:
            self.logger.info("Encoder freeze disabled (freeze_encoder_epochs=0)")

        # ── Differential Learning Rates ────────────────────────────
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

        # ── OneCycleLR ─────────────────────────────────────────────
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
            f"Encoder LR      : {self.config.encoder_lr:.1e} | "
            f"Decoder LR      : {self.config.decoder_lr:.1e}"
        )
        self.logger.info(
            f"Effective batch : "
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

        Epoch structure:
          1. (Maybe) unfreeze encoder
          2. (Maybe) advance curriculum stage → rebuild DataLoader
          3. Iterate over train_loader, accumulating gradients
          4. Every accumulation_steps batches: optimizer + scheduler step
          5. Log every 10 global steps
          6. End of epoch: flush remaining gradients, log epoch summary
          7. Every eval_every epochs: run validation
          8. Every checkpoint_every_epochs epochs: save + push checkpoint
          9. Early stopping check
        """
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
        if self.config.curriculum_enabled:
            self.logger.info(
                f"Curriculum: simple only until epoch "
                f"{self.config.curriculum_simple_until}, "
                f"+medium until epoch {self.config.curriculum_medium_until}, "
                f"then all data"
            )

        # Profile I/O before training starts.
        self._profile_dataloader()

        self.model.train()
        self.step_start_time = time.time()
        self.optimizer.zero_grad()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1  # 1-indexed for human-readable logs

            # ── Encoder Unfreeze ───────────────────────────────────
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

            # ── Curriculum Learning ────────────────────────────────
            if self.config.curriculum_enabled:
                new_stage = self._get_curriculum_stage(self.current_epoch)
                if new_stage != self._current_curriculum_stage:
                    self.logger.info(
                        f"*** Curriculum advance: "
                        f"{self._current_curriculum_stage} → {new_stage} ***"
                    )
                    self._current_curriculum_stage = new_stage
                    self._rebuild_train_loader_for_curriculum(new_stage)

            self.epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            self.logger.info(
                f"\n{'='*35} "
                f"Epoch {self.current_epoch}/{self.config.num_epochs} "
                f"{'='*35}"
            )

            # ── Batch Loop ─────────────────────────────────────────
            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                # Update temperature for the multi-dataset sampler.
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

                # Optimizer step every accumulation_steps micro-batches.
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
                            f"LR: enc={current_lr[0]:.2e} "
                            f"dec={current_lr[1]:.2e} | "
                            f"Temp: {current_temp:.3f} | "
                            f"Epoch: {self.current_epoch} | "
                            f"10-step: {elapsed:.1f}s"
                        )
                        self.step_start_time = time.time()

            # ── Flush Remaining Gradients ──────────────────────────
            # If the last batch doesn't land on an accumulation boundary,
            # we still need to step the optimizer before the epoch ends.
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
                if (self.config.freeze_encoder_epochs > 0 and
                    self.current_epoch <= self.config.freeze_encoder_epochs)
                else "active"
            )
            self.logger.info(
                f"Epoch {self.current_epoch} complete | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time / 60:.1f} min | "
                f"Step: {self.global_step} | "
                f"Encoder: {enc_status}"
            )

            # ── Evaluation ─────────────────────────────────────────
            # Run every eval_every epochs. During warmup, cap samples
            # to avoid multi-minute waits on early (poor) models.
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

            # ── Early Stopping Counter ─────────────────────────────
            if is_best:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # ── Epoch Checkpoint ───────────────────────────────────
            if self.current_epoch % self.config.checkpoint_every_epochs == 0:
                self._save_epoch_checkpoint()

            # ── Early Stopping ─────────────────────────────────────
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after "
                    f"{self.epochs_without_improvement} epochs without improvement."
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
        Run evaluation on the validation set via engine.evaluate_full().

        Args:
            use_beam_search: If True, uses beam search decoding (slower,
                more accurate). If False, uses greedy decoding.
            max_samples: If set, evaluation stops after this many samples.
                Used during warmup epochs to cap evaluation time.

        Returns:
            (metrics_dict, is_best) where is_best is True if this run
            achieved a new minimum edit distance.
        """
        sample_desc = f" (capped at {max_samples})" if max_samples else ""
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

        # ── Structural Accuracy Breakdown ──────────────────────────
        try:
            struct_metrics = evaluate_structural_accuracy(all_preds, all_targets)
        except Exception as e:
            self.logger.warning(f"Failed to compute structural metrics: {e}")
            struct_metrics = {}

        if struct_metrics:
            self.logger.info(
                f"  STRUCT | "
                f"simple={struct_metrics.get('exprate_simple', 0):.3f} "
                f"(n={struct_metrics.get('count_simple', 0)}) | "
                f"medium={struct_metrics.get('exprate_medium', 0):.3f} "
                f"(n={struct_metrics.get('count_medium', 0)}) | "
                f"complex={struct_metrics.get('exprate_complex', 0):.3f} "
                f"(n={struct_metrics.get('count_complex', 0)}) | "
                f"struct_recall="
                f"{struct_metrics.get('structural_token_recall', 0):.3f}"
            )
            metrics.update(struct_metrics)

        # ── Best Model Update ──────────────────────────────────────
        # Primary metric: edit distance (lower is better).
        # ExpRate is recorded for logging but not used as the stopping criterion
        # because it's binary (exact match) and less informative on long sequences.
        is_best = metrics['edit_dist'] < self.best_edit_dist
        if is_best:
            self.best_exp_rate = metrics['exact_match']
            self.best_edit_dist = metrics['edit_dist']
            self.logger.info(
                f"  *** New best | "
                f"EditDist: {self.best_edit_dist:.2f} | "
                f"ExpRate: {self.best_exp_rate:.4f} ***"
            )

            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                self.current_epoch, self.global_step,
                {
                    'exp_rate': self.best_exp_rate,
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
                    best_path, self.config, self.current_epoch, is_best=True
                )

        return metrics, is_best

    def evaluate_with_beam_search(self, max_samples: int = 500) -> Dict[str, float]:
        """
        Public convenience method for beam-search evaluation.

        Called by train.py in --eval-only mode. Internally delegates to
        _evaluate() with use_beam_search=True.

        Args:
            max_samples: Cap on validation samples (default 500 to keep
                beam-search evaluation under ~5 minutes).

        Returns:
            metrics dict from evaluate_full().
        """
        metrics, _ = self._evaluate(
            use_beam_search=True,
            max_samples=max_samples,
        )
        return metrics

    # ----------------------------------------------------------------
    # Curriculum Learning
    # ----------------------------------------------------------------

    def _get_curriculum_stage(self, epoch: int) -> str:
        """
        Map the current epoch to a curriculum stage name.

        Stages:
          'simple'  — only samples with simple complexity (single-line formulas)
          'medium'  — simple + medium (aligned environments, cases)
          'complex' — all data (matrices, arrays, nested environments)

        If curriculum_enabled is False, always returns 'complex' so the
        full dataset is used from epoch 1.
        """
        if not self.config.curriculum_enabled:
            return 'complex'
        if epoch <= self.config.curriculum_simple_until:
            return 'simple'
        if epoch <= self.config.curriculum_medium_until:
            return 'medium'
        return 'complex'

    def _rebuild_train_loader_for_curriculum(self, stage: str):
        """
        Rebuild the train DataLoader with samples filtered to the given stage.

        Called whenever the curriculum stage advances. Tears down the old
        DataLoader (allowing worker processes to die) and creates a fresh
        one with the new filtered sample set.

        Note: The val DataLoader is intentionally never filtered — we always
        evaluate on the full val set regardless of curriculum stage so that
        metrics are comparable across epochs.

        Args:
            stage: One of 'simple', 'medium', 'complex'.
        """
        if stage == 'simple':
            allowed = {'simple'}
        elif stage == 'medium':
            allowed = {'simple', 'medium'}
        else:  # 'complex' — all data
            allowed = {'simple', 'medium', 'complex'}

        filtered_samples = [
            s for s in self.all_train_samples
            if s.get('complexity', get_complexity(s.get('latex', ''))) in allowed
        ]

        self.logger.info(
            f"Curriculum rebuild ({stage}): "
            f"{len(filtered_samples):,} / {len(self.all_train_samples):,} "
            f"train samples available"
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

        # Use MultiDatasetBatchSampler if we have multiple datasets,
        # otherwise fall back to a standard shuffled loader.
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
        Save an epoch checkpoint and push to HuggingFace (throttled, background).

        Checkpoint filename: epoch_{N}.pt
        Old checkpoints are cleaned up to keep only the last N on disk.
        HF push is throttled by hf_push_every_n_epochs to avoid hammering
        the API on every checkpoint save.
        """
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            f"epoch_{self.current_epoch}.pt",
        )
        save_checkpoint(
            self.model, self.optimizer, self.scheduler, self.scaler,
            self.current_epoch, self.global_step,
            {
                'exp_rate': self.best_exp_rate,
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
                ckpt_path, self.config, self.current_epoch, is_best=False
            )

    def _auto_resume(self) -> bool:
        """
        Auto-resume from the latest checkpoint if one exists.

        Scans config.checkpoint_dir for epoch_*.pt files and loads
        the most recent one. Returns True if a checkpoint was loaded,
        False if starting from scratch.
        """
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
            f"best_edit_dist={self.best_edit_dist:.2f}, "
            f"best_exp_rate={self.best_exp_rate:.4f}"
        )
        return True

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a specific checkpoint path.

        Unlike _auto_resume(), this method is called explicitly with a
        user-specified path (via --resume flag or direct API call).

        Args:
            checkpoint_path: Absolute or relative path to a .pt checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

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
        self.logger.info(
            f"Loaded: epoch={epoch}, step={step}, "
            f"best_edit_dist={self.best_edit_dist:.2f}"
        )

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def _profile_dataloader(self):
        """
        Time one batch from the DataLoader before training starts.

        This is a quick I/O health check. If it takes > 2 seconds to
        load the first batch, the training bottleneck is disk/CPU, not
        GPU — and you should investigate augmentation complexity or
        storage speed before committing to a long training run.

        At batch_size=192 with 16 workers, a healthy H100 setup should
        load the first batch in < 1 second.
        """
        self.logger.info("Profiling DataLoader (timing first batch)...")
        t0 = time.time()
        first_batch = None
        for batch in self.train_loader:
            if batch is not None:
                first_batch = batch
                break
        elapsed = time.time() - t0

        if elapsed > 2.0:
            self.logger.warning(
                f"DataLoader first batch: {elapsed:.2f}s — SLOW. "
                f"Bottleneck is I/O, not GPU. "
                f"Check: augmentation complexity, num_workers ({self.config.num_workers}), "
                f"disk read speed (SSD vs HDD), and image decode overhead."
            )
        else:
            self.logger.info(
                f"DataLoader first batch: {elapsed:.2f}s — OK "
                f"(workers={self.config.num_workers})"
            )

        if first_batch is not None:
            images = first_batch.get('images', first_batch[0] if isinstance(first_batch, (list, tuple)) else None)
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
        Run the complete training pipeline in strict order.

        Order is fixed and intentional:
          1. preprocess_data() — must run before create_dataloaders()
             because it builds self.tokenizer and self.train/val_samples.
          2. create_dataloaders() — must run before build_model()
             because build_model() reads len(self.train_loader) to size
             the OneCycleLR scheduler.
          3. build_model() — must run before any checkpoint loading
             because load_checkpoint() restores state into existing objects.
          4. resume/auto_resume — loads weights + optimizer state.
          5. train() — main loop.
          6. Final beam-search eval — capped at 500 samples to avoid
             a multi-hour wait at the end of a long training run.

        Args:
            resume_from: If provided and the path exists, resume from
                this specific checkpoint. Otherwise, auto-resume from
                the latest checkpoint in checkpoint_dir (if any).
        """
        self.preprocess_data()
        self.create_dataloaders()
        self.build_model()

        if resume_from and os.path.exists(resume_from):
            self.resume_from_checkpoint(resume_from)
        else:
            self._auto_resume()

        self.train()

        # Final evaluation with beam search.
        # Capped at 500 samples — full beam-search eval on a large val
        # set can take hours; 500 samples gives a reliable estimate.
        self.logger.info(
            "Running final beam-search evaluation (500 samples)..."
        )
        self._evaluate(use_beam_search=True, max_samples=500)

        self.logger.info("=" * 70)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 70)