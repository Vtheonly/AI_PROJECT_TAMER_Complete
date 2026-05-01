"""
Trainer Mixin: Model Initialization, Compilation, and Optimizer/Scheduler.

Responsibilities:
  - Instantiate TAMERModel and move it to the correct device.
  - Optionally wrap with DataParallel and torch.compile.
  - Build the two-group AdamW optimizer (encoder / decoder LRs).
  - Build OneCycleLR scheduler sized to the actual train-loader length.
  - Handle encoder freeze/unfreeze bookkeeping.

This mixin reads from self.config, self.device, self.train_loader, and
self.tokenizer. It writes self.model, self.optimizer, and self.scheduler.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from ...models.tamer import TAMERModel

logger = logging.getLogger("TAMER.Model")


# ──────────────────────────────────────────────────────────────────────
# Module unwrapping helper (module-level so other mixins can import it)
# ──────────────────────────────────────────────────────────────────────

def unwrap_model(model: nn.Module) -> TAMERModel:
    """
    Strip DataParallel and torch.compile wrappers to reach the raw
    TAMERModel instance.  Safe to call on an already-unwrapped model.
    """
    if hasattr(model, "module"):       # nn.DataParallel
        model = model.module
    if hasattr(model, "_orig_mod"):    # torch.compile
        model = model._orig_mod
    return model  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────
# Mixin
# ──────────────────────────────────────────────────────────────────────

class TrainerModelMixin:
    """
    Provides build_model() which must be called after create_dataloaders()
    because the scheduler is sized against len(self.train_loader).
    """

    def build_model(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("PHASE 2: Model Initialization")
        self.logger.info("=" * 70)

        vocab_size = len(self.tokenizer)
        self.model: nn.Module = TAMERModel(vocab_size, self.config).to(self.device)

        # ── Multi-GPU ────────────────────────────────────────────────
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.logger.info(f"MULTI-GPU: DataParallel across {num_gpus} GPUs")
            self.model = nn.DataParallel(self.model)

        # ── torch.compile ────────────────────────────────────────────
        if self.config.compile_model and hasattr(torch, "compile"):
            self.logger.info("torch.compile() enabled — first step will be slow")
            self.model = torch.compile(self.model)

        # ── Parameter counts ─────────────────────────────────────────
        raw_model = unwrap_model(self.model)
        total_params    = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters     : {total_params:,}")
        self.logger.info(f"Trainable parameters : {trainable_params:,}")

        # ── Encoder freeze ───────────────────────────────────────────
        if self.config.freeze_encoder_epochs > 0:
            for p in raw_model.encoder.parameters():
                p.requires_grad = False
            frozen = sum(p.numel() for p in raw_model.encoder.parameters())
            active = trainable_params - frozen
            self.logger.info(
                f"Encoder FROZEN for epochs 1–{self.config.freeze_encoder_epochs} "
                f"({frozen:,} frozen | {active:,} active)"
            )

        # ── Optimizer ────────────────────────────────────────────────
        # Two separate LR groups allow the pre-trained encoder to warm
        # up at a much lower rate than the randomly initialised decoder.
        encoder_params = list(raw_model.encoder.parameters())
        decoder_params = list(raw_model.decoder.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.config.encoder_lr, "name": "encoder"},
                {"params": decoder_params, "lr": self.config.decoder_lr, "name": "decoder"},
            ],
            weight_decay=self.config.weight_decay,
        )

        # ── Scheduler ────────────────────────────────────────────────
        # Calculate exact optimizer steps across ALL curriculum epochs.
        # Essential to prevent LR collapse when dataset size increases.
        if getattr(self.config, "curriculum_enabled", False):
            from ...data.latex_normalizer import get_complexity
            simple_count = 0
            medium_count = 0
            complex_count = len(self.all_train_samples)
            
            for s in self.all_train_samples:
                c = s.get("complexity") or get_complexity(s.get("latex", ""))
                if c == "simple":
                    simple_count += 1
                    medium_count += 1
                elif c == "medium":
                    medium_count += 1
                    
            total_steps = 0
            for ep in range(1, self.config.num_epochs + 1):
                stage = self._get_curriculum_stage(ep)
                if stage == "simple":
                    stage_samples = simple_count
                elif stage == "medium":
                    stage_samples = medium_count
                else:
                    stage_samples = complex_count
                
                stage_batches = stage_samples // self.config.batch_size
                stage_steps = max(math.ceil(stage_batches / self.config.accumulation_steps), 1)
                total_steps += stage_steps
                
            self.config.total_training_steps = total_steps
            self.logger.info(f"Curriculum exact total steps: {total_steps:,}")
        else:
            steps_per_epoch = max(math.ceil(len(self.train_loader) / self.config.accumulation_steps), 1)
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
            f"Encoder LR  : {self.config.encoder_lr:.1e} | "
            f"Decoder LR  : {self.config.decoder_lr:.1e}"
        )
        self.logger.info(f"Total steps : {self.config.total_training_steps:,}")