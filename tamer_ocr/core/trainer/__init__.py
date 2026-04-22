"""
TAMER OCR Trainer — Mixin Orchestrator.

The Trainer class is assembled from four focused mixins:

  TrainerDataMixin       — preprocessing, dataloaders, curriculum
  TrainerModelMixin      — model build, optimizer, scheduler
  TrainerLoopMixin       — train/eval loops
  TrainerCheckpointMixin — saving, resuming, HF push

All mixins share state through a single __init__ defined here.
Python's MRO guarantees that attribute lookups follow the order
listed in the class definition, so there are no ambiguities.

MRO (left → right, most-specific first):
  Trainer → TrainerDataMixin → TrainerModelMixin
          → TrainerLoopMixin → TrainerCheckpointMixin
"""

import time
import logging
from typing import Optional

import torch

from ...logger import setup_logger
from .data import TrainerDataMixin
from .model import TrainerModelMixin
from .loop import TrainerLoopMixin
from .checkpoint import TrainerCheckpointMixin


class Trainer(
    TrainerDataMixin,
    TrainerModelMixin,
    TrainerLoopMixin,
    TrainerCheckpointMixin,
):
    """
    Unified trainer for TAMER OCR.

    Usage:
        trainer = Trainer(config)
        trainer.run()                        # full pipeline
        trainer.run(resume_from="ckpt.pt")  # resume training
    """

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        
        
        
        # ── Logging ───────────────────────────────────────────────────
        self.logger = setup_logger("TAMER.Trainer", config.log_dir)
        self.logger.info(f"Device : {self.device} | AMP : {self.use_amp}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props   = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                self.logger.info(
                    f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
                    f"VRAM: {vram_gb:.1f} GB"
                )

        # ── Model / optimisation state ────────────────────────────────
        self.model      = None
        self.optimizer  = None
        self.scheduler  = None
        self.scaler     = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Training counters ─────────────────────────────────────────
        self.current_epoch             = 0
        self.global_step               = 0
        self.best_exp_rate             = 0.0
        self.best_edit_dist            = float("inf")
        self.epochs_without_improvement = 0

        # ── Data state ────────────────────────────────────────────────
        self.tokenizer         = None   # set in preprocess_data()
        self.criterion         = None   # set in preprocess_data()
        self.train_samples     = []
        self.val_samples       = []
        self.all_train_samples = []     # immutable snapshot for curriculum
        self.train_dataset     = None
        self.val_dataset       = None
        self.train_loader      = None
        self.val_loader        = None
        self.dataset_ranges    = {}

        # ── Curriculum state ──────────────────────────────────────────
        self._current_curriculum_stage = "simple"

        # ── HF push tracking ─────────────────────────────────────────
        self._last_hf_push_epoch = -1



        
        # Register the signal handlers for SIGTERM (Kaggle timeout) and SIGINT (Ctrl+C)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self._stop_requested = False
    # ──────────────────────────────────────────────────────────────────
    # Pipeline entry points
    # ──────────────────────────────────────────────────────────────────

    def run(self, resume_from: Optional[str] = None) -> None:
        """
        Execute the full training pipeline:
          1. Preprocess data
          2. Create dataloaders
          3. Build model + optimizer + scheduler
          4. Resume from checkpoint (explicit path or auto-detect)
          5. Profile the first dataloader batch
          6. Train
          7. Final beam-search evaluation (500 samples)
        """
        self.preprocess_data()
        self.create_dataloaders()
        self.build_model()

        if resume_from and os.path.exists(resume_from):
            self.resume_from_checkpoint(resume_from)
        else:
            self._auto_resume()

        self._profile_dataloader()
        self.train()

        self.logger.info("Running final beam-search evaluation (500 samples)…")
        self._evaluate(use_beam_search=True, max_samples=500)

        self.logger.info("=" * 70)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────

    def _profile_dataloader(self) -> None:
        """Time the first batch fetch to catch slow worker configurations."""
        self.logger.info("Profiling DataLoader (first batch)…")
        t0 = time.time()
        first_batch = None
        for batch in self.train_loader:
            if batch is not None:
                first_batch = batch
                break
        elapsed = time.time() - t0

        if elapsed > 2.0:
            self.logger.warning(
                f"First batch took {elapsed:.2f}s — consider increasing "
                f"num_workers or prefetch_factor."
            )
        else:
            self.logger.info(f"First batch: {elapsed:.2f}s — OK")

        if first_batch is not None:
            images = first_batch.get("image")
            if images is not None and hasattr(images, "shape"):
                self.logger.info(
                    f"Batch tensor: shape={tuple(images.shape)} | "
                    f"dtype={images.dtype}"
                )
    def _signal_handler(self, sig, frame):
        self.logger.warning(f"Signal {sig} received! Attempting emergency save...")
        self._stop_requested = True

# Re-export so `from tamer_ocr.core.trainer import Trainer` works.
__all__ = ["Trainer"]