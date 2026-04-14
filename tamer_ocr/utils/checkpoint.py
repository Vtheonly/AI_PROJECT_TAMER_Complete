"""
Checkpoint utilities for TAMER training v2.1.

- Saves full training state: model, optimizer, scheduler, scaler, epoch, step
- Pushes checkpoints to Hugging Face Hub
- Finds latest checkpoint for auto-resume
- Keeps last N local checkpoints to avoid filling disk
"""

import os
import glob
import torch
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("TAMER.Checkpoint")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    metrics: Dict[str, Any],
    path: str,
):
    """Save a full training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    # Save scaler state (critical for AMP)
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Save scheduler state
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    try:
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path} (epoch={epoch}, step={step})")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: str = 'cpu',
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Load a training checkpoint.
    FIX: weights_only=True for security.
    """
    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return 0, 0, {}

    try:
        # FIX: Security risk fix - set weights_only=True
        ckpt = torch.load(path, map_location=device, weights_only=True)

        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            # Handle cases where the whole model might have been saved
            logger.warning("model_state_dict not found in checkpoint, attempting full load.")
            model.load_state_dict(ckpt)

        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}. This is normal if total_steps changed.")

        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        epoch = ckpt.get('epoch', 0)
        step = ckpt.get('step', 0)
        metrics = ckpt.get('metrics', {})

        logger.info(f"Resumed from checkpoint {path} (epoch={epoch}, step={step})")
        return epoch, step, metrics

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, 0, {}


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))

    if not ckpt_files:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            return best_path
        return None

    # Sort by modification time (newest first)
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0]


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3):
    """Keep only the last N epoch checkpoints."""
    ckpt_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")),
        key=os.path.getmtime,
    )
    if len(ckpt_files) > keep_last_n:
        for old_ckpt in ckpt_files[:-keep_last_n]:
            try:
                os.remove(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")
            except OSError:
                pass


def push_checkpoint_to_hf(checkpoint_path: str, config, epoch: int, is_best: bool = False):
    """Push a checkpoint to Hugging Face Hub (model repo)."""
    hf_token = getattr(config, 'hf_token', '') or os.getenv('HF_TOKEN', '')
    hf_repo = getattr(config, 'hf_repo_id', '')

    if not hf_token or not hf_repo:
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)

        # Auto-resolve username if needed
        if '/' not in hf_repo:
            username = api.whoami()['name']
            hf_repo = f"{username}/{hf_repo}"

        api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model", private=True)

        remote_name = "best.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=remote_name,
            repo_id=hf_repo,
            repo_type="model"
        )
        logger.info(f"Pushed {remote_name} to Hugging Face Hub ({hf_repo})")
    except Exception as e:
        logger.error(f"Failed to push checkpoint to HF: {e}")