"""
Checkpoint utilities for TAMER training.

Key improvements:
- Saves scaler_state_dict (for AMP)
- Saves scheduler_state_dict (for OneCycleLR)
- Saves step count (for step-based training)
- Google Drive backup support for Colab
- Keeps last N checkpoints to avoid filling disk
"""

import os
import glob
import shutil
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("TAMER.Checkpoint")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,  # torch.amp.GradScaler
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
) -> tuple:
    """
    Load a training checkpoint.
    
    Returns:
        (epoch, step, metrics) tuple
    """
    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return 0, 0, {}
    
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        
        model.load_state_dict(ckpt['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
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


def backup_to_drive(checkpoint_path: str, drive_dir: str, keep_last_n: int = 3):
    """
    Copy checkpoint to Google Drive for Colab session hopping.
    Keeps only the last N checkpoints to avoid filling Drive.
    """
    if not drive_dir or not os.path.exists(checkpoint_path):
        return
    
    try:
        os.makedirs(drive_dir, exist_ok=True)
        
        # Copy the checkpoint
        dest = os.path.join(drive_dir, os.path.basename(checkpoint_path))
        shutil.copy2(checkpoint_path, dest)
        logger.info(f"Backup saved to Drive: {dest}")
        
        # Clean up old checkpoints
        checkpoints = sorted(
            glob.glob(os.path.join(drive_dir, "*.pt")),
            key=os.path.getmtime,
            reverse=True
        )
        
        for old_ckpt in checkpoints[keep_last_n:]:
            os.remove(old_ckpt)
            logger.info(f"Removed old backup: {old_ckpt}")
    
    except Exception as e:
        logger.error(f"Drive backup failed: {e}")


def push_to_huggingface(checkpoint_path: str, config, step: int, is_best: bool = False):
    """Push a checkpoint to Hugging Face Hub."""
    hf_token = getattr(config, 'hf_token', '') or os.getenv('HF_TOKEN', '')
    hf_repo = getattr(config, 'hf_repo_id', '')
    
    if not hf_token or not hf_repo:
        return
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        
        # Auto-resolve username
        if '/' not in hf_repo:
            username = api.whoami()['name']
            hf_repo = f"{username}/{hf_repo}"
        
        api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model", private=True)
        
        remote_name = "best.pt" if is_best else f"checkpoint_step_{step}.pt"
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=remote_name,
            repo_id=hf_repo,
            repo_type="model"
        )
        logger.info(f"Pushed {remote_name} to Hugging Face Hub ({hf_repo})")
    except Exception as e:
        logger.error(f"Failed to push checkpoint to HF: {e}")
