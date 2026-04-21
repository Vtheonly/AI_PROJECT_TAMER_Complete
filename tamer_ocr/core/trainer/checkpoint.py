"""
Trainer Mixin: Checkpointing and HuggingFace Hub Synchronisation.

Design constraints
──────────────────
1.  OFFLINE-FIRST.  Kaggle competition accelerators (RTX Blackwell Pro
    6000, 96 GB VRAM / 170 GB RAM) have network access fully blocked.
    Every HF push call will fail with a socket / connection error.
    Failures must be silent warnings — the training loop must never
    be interrupted by a network error.

2.  ATOMIC WRITES.  torch.save() serialises via pickle in multiple
    internal write() calls.  A reader that opens the file before
    torch.save() returns will see a truncated, un-loadable file.
    Fix: write to <path>.tmp, fsync, then os.replace() → final path.
    os.replace() is guaranteed atomic on POSIX when src and dst share
    the same filesystem (they always do here — same directory).

3.  NO BACKGROUND THREADS FOR UPLOAD.
    The original code span a daemon thread to push the checkpoint to
    HF while the main thread continued training.  This has two failure
    modes:
      a) The daemon thread starts reading <path> before torch.save()
         has flushed all bytes → corrupted upload.
      b) On Kaggle, the thread blocks on a TCP connect that never
         completes, leaking the thread and its file handle for the
         remainder of the kernel lifetime.
    Because network is blocked, synchronous push costs essentially
    zero extra wall-clock time (connection refused / unreachable is
    returned immediately by the kernel).  Synchronous execution also
    means the file is guaranteed complete before any read attempt.

Write sequence (guaranteed ordering)
─────────────────────────────────────
    torch.save()   → <path>.tmp          (all bytes in tmp)
    fh.flush() +
    os.fsync()     → <path>.tmp          (bytes committed to storage)
    os.replace()   → <path>              (atomic rename, POSIX §4.6)
    push_to_hf()   → HF Hub (no-op offline, silent warning on failure)
"""

import os
import logging
from typing import Optional

from ...utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    push_checkpoint_to_hf,
)
from .model import unwrap_model

logger = logging.getLogger("TAMER.Checkpoint")


class TrainerCheckpointMixin:

    # ──────────────────────────────────────────────────────────────────
    # Public API (called from train.py / Trainer.run)
    # ──────────────────────────────────────────────────────────────────

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights, optimizer, scheduler and scaler state from
        ``checkpoint_path``, then update all epoch/step/metric counters.

        Raises
        ------
        FileNotFoundError
            Raised immediately if the path does not exist so the caller
            gets a clear error instead of a cryptic KeyError deep inside
            load_checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        self.logger.info(f"Resuming from: {checkpoint_path}")
        epoch, step, metrics = load_checkpoint(
            checkpoint_path,
            unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            self.scaler,
            device=str(self.device),
        )

        self.current_epoch              = epoch
        self.global_step                = step
        self.best_exp_rate              = metrics.get("exp_rate",  0.0)
        self.best_edit_dist             = metrics.get("edit_dist", float("inf"))

        self.logger.info(
            f"Resumed  epoch={epoch}  step={step}  "
            f"best_edit_dist={self.best_edit_dist:.4f}  "
            f"best_exp_rate={self.best_exp_rate:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers — called exclusively from loop.py
    # ──────────────────────────────────────────────────────────────────

    def _auto_resume(self) -> bool:
        """
        Scan checkpoint_dir for the latest epoch checkpoint and resume
        if one is found.

        Returns
        -------
        bool
            True  — resumed from an existing checkpoint.
            False — no checkpoint found; training starts from scratch.
        """
        latest = find_latest_checkpoint(self.config.checkpoint_dir)
        if latest is None:
            self.logger.info("No checkpoint found — starting from scratch.")
            return False

        self.logger.info(f"Auto-resuming from: {latest}")
        self.resume_from_checkpoint(latest)
        return True

    def _save_epoch_checkpoint(self) -> None:
        """
        Write a numbered epoch checkpoint atomically, run housekeeping,
        then attempt a silent HF push (no-op when offline).

        Steps
        ─────
        1. Atomic write  → checkpoints/epoch_N.pt
        2. Cleanup       → delete oldest checkpoints beyond keep_last_n
        3. HF push       → silent warning on any network failure
        """
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            f"epoch_{self.current_epoch}.pt",
        )

        self._write_checkpoint_atomic(ckpt_path)
        self.logger.info(f"Epoch checkpoint saved → {ckpt_path}")

        cleanup_old_checkpoints(
            self.config.checkpoint_dir,
            self.config.keep_last_n_checkpoints,
        )

        self._attempt_hf_push(ckpt_path, is_best=False)

    def _save_best_checkpoint(self) -> None:
        """
        Atomically overwrite best.pt with the current model state, then
        attempt a silent HF push.

        best.pt is always pushed unconditionally (ignoring the
        hf_push_every_n_epochs gate) because a new best model is always
        worth uploading the moment connectivity is restored.
        """
        best_path = os.path.join(self.config.checkpoint_dir, "best.pt")

        self._write_checkpoint_atomic(best_path)
        self.logger.info(f"Best checkpoint saved → {best_path}")

        self._attempt_hf_push(best_path, is_best=True)

    # ──────────────────────────────────────────────────────────────────
    # Atomic write implementation
    # ──────────────────────────────────────────────────────────────────

    def _write_checkpoint_atomic(self, final_path: str) -> None:
        """
        Serialize the full training state to ``final_path`` atomically.

        Procedure
        ─────────
        1.  Write to   ``final_path + ".tmp"``  via save_checkpoint().
        2.  Open the   .tmp file and call os.fsync() to commit all
            kernel write-back buffers to storage.  This is important on
            Kaggle's NFS-backed /kaggle/working where dirty pages can
            linger in the page cache for seconds.
        3.  Call       os.replace(tmp, final_path).
            POSIX guarantees this rename is atomic on the same
            filesystem: any concurrent reader sees either the complete
            old file or the complete new file — never a partial write.

        Error handling
        ──────────────
        If any step fails the .tmp file is removed and a RuntimeError is
        raised.  We do NOT silently swallow write errors — a corrupted
        or missing checkpoint is worse than a crash with a clear message.
        """
        # Ensure the checkpoint directory exists (first epoch edge case).
        os.makedirs(
            os.path.dirname(final_path) or ".",
            exist_ok=True,
        )

        tmp_path = final_path + ".tmp"

        try:
            # ── Step 1: write all bytes to the temporary file ─────────
            save_checkpoint(
                unwrap_model(self.model),
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.current_epoch,
                self.global_step,
                {
                    "exp_rate":  self.best_exp_rate,
                    "edit_dist": self.best_edit_dist,
                },
                tmp_path,
            )

            # ── Step 2: fsync — flush kernel buffers to disk ──────────
            # Open in "rb+" (read-write binary) so we do not truncate.
            with open(tmp_path, "rb+") as fh:
                fh.flush()
                os.fsync(fh.fileno())

            # ── Step 3: atomic rename tmp → final ─────────────────────
            # os.replace() is the correct call on Python 3; it wraps
            # rename(2) on POSIX and MoveFileEx on Windows (with
            # MOVEFILE_REPLACE_EXISTING).
            os.replace(tmp_path, final_path)

        except Exception as exc:
            # Best-effort cleanup of the partial temporary file.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as rm_err:
                    self.logger.warning(
                        f"Could not remove partial tmp file {tmp_path}: {rm_err}"
                    )

            raise RuntimeError(
                f"Atomic checkpoint write failed for {final_path}: {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────
    # HF push — synchronous, fully offline-safe
    # ──────────────────────────────────────────────────────────────────

    def _attempt_hf_push(self, path: str, is_best: bool) -> None:
        """
        Attempt a synchronous push of the completed checkpoint at
        ``path`` to HF Hub.

        Gate logic
        ──────────
        - config.hf_repo_id must be non-empty.
        - For non-best checkpoints: at least hf_push_every_n_epochs
          epochs must have elapsed since the last successful push.
        - For best-model checkpoints: the epoch gate is bypassed.

        Offline behaviour
        ─────────────────
        On Kaggle competition accelerators network is fully blocked.
        The underlying huggingface_hub call will raise a
        requests.exceptions.ConnectionError (or similar) within
        milliseconds of the TCP connect attempt failing.  We catch
        *all* exceptions and log a WARNING so the training loop is
        never interrupted.

        Why not a background thread?
        ─────────────────────────────
        This method is only ever called *after* _write_checkpoint_atomic
        returns, which means the file is fully flushed and renamed into
        its final location before any read attempt begins.  Running the
        push in the main thread:
          • eliminates the torch.save() / reader race entirely;
          • avoids leaked daemon threads blocking on a dead socket;
          • costs ≈0 ms extra when offline (connection refused is
            returned instantly by the kernel).
        """
        if not getattr(self.config, "hf_repo_id", ""):
            return

        # Epoch-interval gate for non-best periodic checkpoints.
        epochs_since_last_push = self.current_epoch - self._last_hf_push_epoch
        if (
            not is_best
            and epochs_since_last_push < self.config.hf_push_every_n_epochs
        ):
            return

        self.logger.info(
            f"HF push attempt: {os.path.basename(path)}  "
            f"epoch={self.current_epoch}  best={is_best}"
        )

        try:
            push_checkpoint_to_hf(
                path,
                self.config,
                self.current_epoch,
                is_best=is_best,
            )
            # Only advance the watermark on success so a failed push is
            # retried at the next checkpoint opportunity.
            self._last_hf_push_epoch = self.current_epoch
            self.logger.info(
                f"HF push succeeded  "
                f"epoch={self.current_epoch}  best={is_best}"
            )

        except Exception as exc:
            # Expected on offline Kaggle kernels — demote to WARNING.
            self.logger.warning(
                f"HF push failed (offline or auth error) — "
                f"will retry at next checkpoint.  Reason: {exc}"
            )
            # Intentionally do NOT update self._last_hf_push_epoch so
            # the next checkpoint triggers another attempt.