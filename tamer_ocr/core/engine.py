"""
Training and Evaluation Engine Functions — v2.4 (BFloat16 + Multi-GPU)

Key change from v2.3:
  - dtype=torch.float16 → dtype=torch.bfloat16 in BOTH train_step and eval_step.
    BFloat16 has the same dynamic range as FP32, preventing NaN/Inf losses
    at massive batch s/kaggle/working/check.zipizes that float16 overflows.
  - RTX 6000 Ada has dedicated BF16 Tensor Cores — this is the optimal dtype.
"""



"""
Training and Evaluation Engine Functions.

Lower-level train/eval step functions that can be called independently
or composed into custom training loops. The Trainer class uses these internally.
"""

import torch
import logging
from typing import Dict, Any, List, Tuple

from ..models.tamer import TAMERModel
from ..data.tokenizer import LaTeXTokenizer
from .losses import LabelSmoothedCELoss
from .inference import beam_search, greedy_decode
from ..utils.metrics import compute_batch_metrics

logger = logging.getLogger("TAMER.Engine")


def train_step(
    model: TAMERModel,
    batch: Dict[str, Any],
    criterion: LabelSmoothedCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    device: torch.device,
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Single training step with AMP and gradient accumulation.

    Args:
        model: TAMERModel instance
        batch: Dict with 'image' and 'ids' tensors
        criterion: Loss function
        optimizer: Optimizer
        scaler: AMP GradScaler (PyTorch 2.0+)
        device: torch device
        accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Max gradient norm for clipping

    Returns:
        Unscaled loss value for logging
    """
    model.train()

    images = batch['image'].to(device, non_blocking=True)
    ids = batch['ids'].to(device, non_blocking=True)

    use_amp = device.type == 'cuda'

    
    
    
    
    
    
    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
        logits = model(images, ids)
        loss = criterion(logits, ids)
        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    return loss.item() * accumulation_steps


def optimizer_step(
    model: TAMERModel,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    scheduler=None,
    max_grad_norm: float = 1.0,
):
    """
    Perform optimizer step with gradient clipping and AMP scaler.

    Call this after accumulation_steps of train_step().
    """
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    optimizer.zero_grad()


def eval_step(
    model: TAMERModel,
    batch: Dict[str, Any],
    criterion: LabelSmoothedCELoss,
    tokenizer: LaTeXTokenizer,
    device: torch.device,
    use_beam_search: bool = False,
    beam_width: int = 5,
    max_len: int = 200,
    length_penalty: float = 0.6,
) -> Tuple[float, List[str], List[str]]:
    """
    Single evaluation step.

    Args:
        model: TAMERModel instance
        batch: Dict with 'image' and 'ids' tensors
        criterion: Loss function
        tokenizer: LaTeXTokenizer for decoding
        device: torch device
        use_beam_search: Whether to use beam search (slower)
        beam_width: Beam width for beam search
        max_len: Max output sequence length
        length_penalty: Length penalty for beam search

    Returns:
        (loss, predictions, ground_truths)
    """
    model.eval()
    images = batch['image'].to(device)
    ids = batch['ids'].to(device)
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        
        
        
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, ids)
            loss = criterion(logits, ids)

    preds = []
    gts = []

    with torch.no_grad():
        if use_beam_search:
            
            for i in range(images.size(0)):
                pred_tokens = beam_search(
                    model, images[i:i+1],
                    tokenizer.sos_id, tokenizer.eos_id, tokenizer.pad_id,
                    beam_width=beam_width, max_len=max_len,
                    length_penalty=length_penalty, device=device,
                )
                pred_latex = tokenizer.decode(pred_tokens, skip_special=True)
                preds.append(pred_latex)
        else:
            
            pred_tokens_batch = greedy_decode(
                model, images,
                tokenizer.sos_id, tokenizer.eos_id,
                max_len=max_len, device=device,
            )
            for pred_tokens in pred_tokens_batch:
                preds.append(tokenizer.decode(pred_tokens, skip_special=True))

        
        for i in range(images.size(0)):
            gt_ids = ids[i].cpu().tolist()
            gt_latex = tokenizer.decode(gt_ids, skip_special=True)
            gts.append(gt_latex)

    return loss.item(), preds, gts


def evaluate_full(
    model: TAMERModel,
    dataloader,
    criterion: LabelSmoothedCELoss,
    tokenizer: LaTeXTokenizer,
    device: torch.device,
    use_beam_search: bool = False,
    beam_width: int = 5,
    max_len: int = 200,
    length_penalty: float = 0.6,
    max_samples: int = None,
) -> Dict[str, float]:
    """
    Full evaluation over a dataset.

    Args:
        model: TAMERModel instance
        dataloader: Validation DataLoader
        criterion: Loss function
        tokenizer: LaTeXTokenizer
        device: torch device
        use_beam_search: Use beam search (slower but more accurate)
        beam_width: Beam width
        max_len: Max output length
        length_penalty: Length penalty for beam search
        max_samples: Limit number of samples to evaluate

    Returns:
        Dict of metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            
            
            batch_size = batch['image'].size(0)
            if max_samples and sample_count + batch_size > max_samples:
                limit = max_samples - sample_count
                batch['image'] = batch['image'][:limit]
                batch['ids'] = batch['ids'][:limit]
                batch_size = limit

            loss, preds, gts = eval_step(
                model, batch, criterion, tokenizer, device,
                use_beam_search=use_beam_search,
                beam_width=beam_width, max_len=max_len,
                length_penalty=length_penalty,
            )

            total_loss += loss
            num_batches += 1
            all_preds.extend(preds)
            all_targets.extend(gts)
            sample_count += batch_size

            if max_samples and sample_count >= max_samples:
                break

    metrics = compute_batch_metrics(all_preds, all_targets)
    metrics['val_loss'] = total_loss / max(num_batches, 1)

    model.train()
    return metrics, all_preds, all_targets