"""
Trainer Mixin: Training and Evaluation Loops.

Responsibilities:
  - Run the per-epoch training loop with gradient accumulation.
  - Handle encoder unfreeze at the configured epoch boundary.
  - Drive curriculum stage transitions via the data mixin.
  - Run full validation (greedy or beam) via engine.evaluate_full.
  - Compute and log structural accuracy metrics.
  - Delegate best-checkpoint saving to the checkpoint mixin.

This mixin reads: self.model, self.optimizer, self.scheduler,
self.scaler, self.criterion, self.tokenizer, self.train_loader,
self.val_loader, self.config, self.device, and all tracking counters.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple

import torch

from ..engine import train_step, optimizer_step, evaluate_full
from ...data.sampler import get_temperature_for_step
from ...utils.metrics import evaluate_structural_accuracy
from .model import unwrap_model

logger = logging.getLogger("TAMER.Loop")


class TrainerLoopMixin:

    # ──────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: Training Loop")
        self.logger.info("=" * 70)

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            # ── Encoder unfreeze ──────────────────────────────────────
            if (
                self.config.freeze_encoder_epochs > 0
                and self.current_epoch == self.config.freeze_encoder_epochs + 1
            ):
                raw = unwrap_model(self.model)
                for p in raw.encoder.parameters():
                    p.requires_grad = True
                newly_active = sum(p.numel() for p in raw.encoder.parameters())
                self.logger.info(
                    f"*** Epoch {self.current_epoch}: Encoder UNFROZEN "
                    f"({newly_active:,} params now training) ***"
                )

            # ── Curriculum transition ─────────────────────────────────
            if self.config.curriculum_enabled:
                new_stage = self._get_curriculum_stage(self.current_epoch)
                if new_stage != self._current_curriculum_stage:
                    self.logger.info(
                        f"*** Curriculum: {self._current_curriculum_stage} "
                        f"→ {new_stage} ***"
                    )
                    self._current_curriculum_stage = new_stage
                    self._rebuild_train_loader_for_curriculum(new_stage)

            # ── Epoch banner ──────────────────────────────────────────
            self.logger.info(
                f"\n{'='*30} "
                f"Epoch {self.current_epoch}/{self.config.num_epochs} "
                f"{'='*30}"
            )

            epoch_loss, epoch_steps = 0.0, 0
            step_timer = time.time()

            # ── Batch loop ────────────────────────────────────────────
            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                # Update inter-dataset sampling temperature.
                current_temp = get_temperature_for_step(
                    self.global_step,
                    self.config.total_training_steps,
                    self.config.temp_start,
                    self.config.temp_end,
                )
                sampler = getattr(
                    getattr(self.train_loader, "batch_sampler", None),
                    "set_temperature",
                    None,
                )
                if callable(sampler):
                    sampler(current_temp)

                # Forward + backward (accumulation handled inside).
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

                # Optimizer step every `accumulation_steps` iterations.
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
                        elapsed   = time.time() - step_timer
                        avg_loss  = epoch_loss / max(epoch_steps, 1)
                        last_lr   = self.scheduler.get_last_lr()
                        enc_lr    = last_lr[0] if len(last_lr) > 0 else 0.0
                        dec_lr    = last_lr[1] if len(last_lr) > 1 else 0.0
                        self.logger.info(
                            f"Step {self.global_step:>6d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR enc={enc_lr:.2e} dec={dec_lr:.2e} | "
                            f"Temp: {current_temp:.3f} | "
                            f"10-step: {elapsed:.1f}s"
                        )
                        step_timer = time.time()

            # ── Flush remaining accumulated gradients ──────────────────
            # If the total number of batches is not divisible by
            # accumulation_steps the last partial accumulation would be
            # silently discarded without this flush.
            if epoch_steps > 0 and epoch_steps % self.config.accumulation_steps != 0:
                optimizer_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=self.scheduler,
                    max_grad_norm=self.config.max_grad_norm,
                )
                self.global_step += 1

            # ── Epoch summary ──────────────────────────────────────────
            epoch_time  = time.time() - epoch_start
            avg_loss    = epoch_loss / max(epoch_steps, 1)
            enc_status  = (
                "FROZEN"
                if self.config.freeze_encoder_epochs > 0
                and self.current_epoch <= self.config.freeze_encoder_epochs
                else "active"
            )
            self.logger.info(
                f"Epoch {self.current_epoch} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time / 60:.1f} min | "
                f"Step: {self.global_step} | "
                f"Encoder: {enc_status}"
            )

            # ── Evaluation ─────────────────────────────────────────────
            is_best = False
            if self.current_epoch % self.config.eval_every == 0:
                in_warmup   = self.current_epoch <= self.config.eval_warmup_epochs
                max_samples = (
                    self.config.eval_warmup_max_samples if in_warmup else None
                )
                _, is_best = self._evaluate(
                    use_beam_search=False,
                    max_samples=max_samples,
                )

            # ── Patience counter ───────────────────────────────────────
            if is_best:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # ── Periodic checkpoint ────────────────────────────────────
            if self.current_epoch % self.config.checkpoint_every_epochs == 0:
                self._save_epoch_checkpoint()

            # ── Early stopping ─────────────────────────────────────────
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(
                    f"Early stopping: {self.epochs_without_improvement} epochs "
                    "without improvement."
                )
                break

        # ── Training complete ──────────────────────────────────────────
        self.logger.info("=" * 70)
        self.logger.info("Training complete!")
        self.logger.info(f"Best ExpRate  : {self.best_exp_rate:.4f}")
        self.logger.info(f"Best EditDist : {self.best_edit_dist:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        use_beam_search: bool = False,
        max_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Run a full validation pass and return (metrics_dict, is_best).

        ``is_best`` is True when the current edit distance is strictly
        lower than the historical best.  When that happens the best
        checkpoint is saved automatically via _save_best_checkpoint().
        """
        sample_desc = f" (capped at {max_samples})" if max_samples else ""
        self.logger.info(
            f"Evaluating{sample_desc} | beam={use_beam_search} ..."
        )

        self.model.eval()
        try:
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
        finally:
            # Always restore train mode even if evaluate_full raises.
            self.model.train()

        self.logger.info(
            f"  EVAL | "
            f"Loss: {metrics.get('val_loss', 0.0):.4f} | "
            f"ExpRate: {metrics.get('exact_match', 0.0):.4f} | "
            f"EditDist: {metrics.get('edit_dist', 0.0):.2f} | "
            f"SER: {metrics.get('ser', 0.0):.4f} | "
            f"Leq1: {metrics.get('leq1', 0.0):.4f}"
        )

        # ── Structural accuracy breakdown ──────────────────────────────
        try:
            struct_metrics = evaluate_structural_accuracy(all_preds, all_targets)
        except Exception as exc:
            self.logger.warning(f"Structural metrics failed: {exc}")
            struct_metrics = {}

        if struct_metrics:
            self.logger.info(
                f"  STRUCT | "
                f"simple={struct_metrics.get('exprate_simple', 0.0):.3f} | "
                f"medium={struct_metrics.get('exprate_medium', 0.0):.3f} | "
                f"complex={struct_metrics.get('exprate_complex', 0.0):.3f}"
            )
            metrics.update(struct_metrics)

        # ── Best-model tracking ────────────────────────────────────────
        is_best = metrics.get("edit_dist", float("inf")) < self.best_edit_dist
        if is_best:
            self.best_edit_dist = metrics["edit_dist"]
            self.best_exp_rate  = metrics.get("exact_match", 0.0)
            self.logger.info(
                f"  *** New best | "
                f"EditDist: {self.best_edit_dist:.4f} | "
                f"ExpRate: {self.best_exp_rate:.4f} ***"
            )
            self._save_best_checkpoint()

        return metrics, is_best

    def evaluate_with_beam_search(
        self,
        max_samples: int = 500,
    ) -> Dict[str, float]:
        """Public helper used by eval-only mode in train.py."""
        metrics, _ = self._evaluate(
            use_beam_search=True,
            max_samples=max_samples,
        )
        return metrics