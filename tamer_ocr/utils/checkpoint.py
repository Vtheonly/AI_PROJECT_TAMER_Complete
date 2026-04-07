import os
import torch
import json
import logging

logger = logging.getLogger("TAMER.Checkpoint")

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

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