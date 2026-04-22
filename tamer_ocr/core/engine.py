"""
Training and Evaluation Engine — v3.0 (Encoder Padding Mask Edition)

Changes from v2.5:
  - [FIXED] train_step() and eval_step() now extract real_ws / real_hs from
    the batch dict and pass them to model() so the encoder padding mask is
    built correctly on every forward pass.
  - [FIXED] evaluate_full() passes real_ws / real_hs through eval_step().
  - [FIXED] greedy_decode() and beam_search() in inference.py now receive
    memory_mask from encode() during inference (wired in eval_step).
  - [RETAINED] Strict teacher forcing shift.
  - [RETAINED] FP32 loss to prevent BF16 NaN/underflow.
  - [RETAINED] BFloat16 AMP for forward pass.
"""

import torch
import logging
from typing import Dict, Any, List, Tuple, Optional

from ..models.tamer import TAMERModel
from ..data.tokenizer import LaTeXTokenizer
from .losses import LabelSmoothedCELoss
from .inference import beam_search, greedy_decode
from ..utils.metrics import compute_batch_metrics

logger = logging.getLogger("TAMER.Engine")


def train_step(
    model: TAMERModel,
    batch: Dict[str, Any],
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    device: torch.device,
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Single gradient-accumulation training step.

    Returns:
        Unscaled loss value (float) for logging.
    """
    model.train()

    images = batch["image"].to(device, non_blocking=True)
    ids = batch["ids"].to(device, non_blocking=True)

    # Extract real content dimensions for encoder padding mask.
    # .get() is used defensively in case an older dataloader without the fix
    # is still in use — the model silently falls back to no mask.
    real_ws = batch.get("real_ws")
    real_hs = batch.get("real_hs")
    if real_ws is not None:
        real_ws = real_ws.to(device, non_blocking=True)
    if real_hs is not None:
        real_hs = real_hs.to(device, non_blocking=True)

    # Strict teacher forcing:
    #   Input  to model: [SOS, t1, t2, ..., tN]   (drop last token = EOS)
    #   Target for loss: [t1,  t2, ..., tN, EOS]  (drop first token = SOS)
    tgt_in = ids[:, :-1].contiguous()
    tgt_out = ids[:, 1:].contiguous()

    use_amp = device.type == "cuda"

    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
        logits = model(images, tgt_in, real_ws=real_ws, real_hs=real_hs)

    # Cast to FP32 before loss to prevent silent NaN / underflow in BF16
    logits = logits.float()
    loss = criterion(logits, tgt_out)
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
    Apply accumulated gradients: unscale → clip → step → update → zero.

    Call this after every `accumulation_steps` calls to train_step().
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
    criterion,
    tokenizer,
    device: torch.device,
    use_beam_search: bool = False,
    beam_width: int = 5,
    max_len: int = 200,
    length_penalty: float = 0.6,
):
    """
    Single evaluation step: compute loss + generate predictions.

    Returns:
        loss:  float scalar
        preds: List[str] — predicted LaTeX strings
        gts:   List[str] — ground-truth LaTeX strings
    """
    model.eval()

    images = batch["image"].to(device)
    ids = batch["ids"].to(device)
    use_amp = device.type == "cuda"

    # Extract real content dimensions (same as train_step)
    real_ws = batch.get("real_ws")
    real_hs = batch.get("real_hs")
    if real_ws is not None:
        real_ws = real_ws.to(device)
    if real_hs is not None:
        real_hs = real_hs.to(device)

    tgt_in = ids[:, :-1].contiguous()
    tgt_out = ids[:, 1:].contiguous()

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, tgt_in, real_ws=real_ws, real_hs=real_hs)

        logits = logits.float()
        loss = criterion(logits, tgt_out)

    preds: List[str] = []
    gts: List[str] = []

    with torch.no_grad():
        if use_beam_search:
            for i in range(images.size(0)):
                # Pass per-image real dimensions so beam search builds the mask
                single_real_ws = real_ws[i : i + 1] if real_ws is not None else None
                single_real_hs = real_hs[i : i + 1] if real_hs is not None else None

                pred_tokens = beam_search(
                    model,
                    images[i : i + 1],
                    tokenizer.sos_id,
                    tokenizer.eos_id,
                    tokenizer.pad_id,
                    beam_width=beam_width,
                    max_len=max_len,
                    length_penalty=length_penalty,
                    device=device,
                    real_ws=single_real_ws,
                    real_hs=single_real_hs,
                )
                preds.append(tokenizer.decode(pred_tokens, skip_special=True))
        else:
            pred_tokens_batch = greedy_decode(
                model,
                images,
                tokenizer.sos_id,
                tokenizer.eos_id,
                max_len=max_len,
                device=device,
                real_ws=real_ws,
                real_hs=real_hs,
            )
            for pred_tokens in pred_tokens_batch:
                preds.append(tokenizer.decode(pred_tokens, skip_special=True))

        for i in range(images.size(0)):
            gt_ids = ids[i].cpu().tolist()
            gts.append(tokenizer.decode(gt_ids, skip_special=True))

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
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Full evaluation loop over an entire dataloader.

    Returns:
        metrics:     Dict with val_loss, exact_match, etc.
        all_preds:   List of all predicted LaTeX strings
        all_targets: List of all ground-truth LaTeX strings
    """
    model.eval()
    all_preds: List[str] = []
    all_targets: List[str] = []
    total_loss = 0.0
    num_batches = 0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            batch_size = batch["image"].size(0)

            if max_samples is not None and sample_count + batch_size > max_samples:
                limit = max_samples - sample_count
                batch["image"] = batch["image"][:limit]
                batch["ids"] = batch["ids"][:limit]
                if batch.get("real_ws") is not None:
                    batch["real_ws"] = batch["real_ws"][:limit]
                if batch.get("real_hs") is not None:
                    batch["real_hs"] = batch["real_hs"][:limit]
                batch_size = limit

            loss, preds, gts = eval_step(
                model,
                batch,
                criterion,
                tokenizer,
                device,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
                max_len=max_len,
                length_penalty=length_penalty,
            )

            total_loss += loss
            num_batches += 1
            all_preds.extend(preds)
            all_targets.extend(gts)
            sample_count += batch_size

            if max_samples is not None and sample_count >= max_samples:
                break

    metrics = compute_batch_metrics(all_preds, all_targets)
    metrics["val_loss"] = total_loss / max(num_batches, 1)

    model.train()
    return metrics, all_preds, all_targets