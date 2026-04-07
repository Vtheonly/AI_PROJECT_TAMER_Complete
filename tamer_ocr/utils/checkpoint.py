import os
import torch
import json
import logging
from huggingface_hub import HfApi, login

logger = logging.getLogger("TAMER.Checkpoint")

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics  # <-- This contains your "gains"
        }, path)
        logger.info(f"Checkpoint saved locally to {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def push_checkpoint_to_hf(checkpoint_path, config, epoch, is_best=False):
    """Pushes a local checkpoint to Hugging Face Hub."""
    if not config.hf_repo_id or config.hf_repo_id == "your-username/TAMER-Checkpoints":
        logger.warning("hf_repo_id is not set. Skipping Hugging Face upload.")
        return

    if not config.hf_token:
        logger.warning("hf_token is not set. Skipping Hugging Face upload.")
        return

    try:
        # Authenticate with Hugging Face
        login(token=config.hf_token)
        api = HfApi()

        # Ensure the repository exists (creates it if it doesn't, sets it to private)
        api.create_repo(repo_id=config.hf_repo_id, exist_ok=True, repo_type="model", private=True)

        # Name the file on Hugging Face
        remote_file_name = "best.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"

        logger.info(f"Uploading {remote_file_name} to Hugging Face Hub ({config.hf_repo_id})...")

        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=remote_file_name,
            repo_id=config.hf_repo_id,
            repo_type="model"
        )
        logger.info(f"Successfully pushed {remote_file_name} to Hugging Face.")
    except Exception as e:
        logger.error(f"Failed to push checkpoint to Hugging Face: {e}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return 0, {}

    try:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        logger.info(f"Resumed from checkpoint {path} (Epoch {ckpt['epoch']})")
        return ckpt.get('epoch', 0), ckpt.get('metrics', {})
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, {}