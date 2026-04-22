"""
Loss functions for TAMER model.

v3.0 Changes (Global Token Averaging Edition):
  - [FIXED] Sequence-Length Bias in StructureAwareLoss:
    Replaced flawed "per-sequence averaging then batch mean" with
    "global token averaging". Every token in the batch — whether it belongs
    to a 5-token equation or a 150-token matrix — now contributes exactly
    1/N_valid to the gradient. The model can no longer ignore complex equations
    by suppressing their gradients via the sequence-length denominator.

  - [FIXED] PyTorch native weight= argument trap avoided:
    We do NOT pass token_weights to nn.CrossEntropyLoss(weight=...) because
    PyTorch divides by sum(weights) instead of count(valid_tokens), which
    dilutes structural multipliers when a batch contains many structural tokens.
    Instead we multiply manually and divide by valid token count.

  - [RETAINED] LabelSmoothedCELoss unchanged — uses nn.CrossEntropyLoss
    reduction='mean' which is already globally correct (averages over all
    non-ignored tokens across the flattened batch, not per-sequence).

  - [RETAINED] StructureAwareLoss structural token weighting (3.0x multiplier).
  - [RETAINED] register_buffer for device-safe weight tensor.
  - [RETAINED] clamp(min=1.0) NaN guard on denominator.

Mathematical proof of correctness:
  Old (flawed):
    Loss = (1/B) * sum_b [ sum_j(loss_b_j * w_b_j) / L_b ]
    A 1-token error in a 5-token seq  contributes: 1/5  = 0.20 per batch item
    A 1-token error in a 50-token seq contributes: 1/50 = 0.02 per batch item
    -> Model learns to ignore long equations (10x gradient suppression)

  New (correct):
    Loss = sum_{b,j}(loss_b_j * w_b_j) / N_valid
    A 1-token error in a 5-token seq  contributes: 1/N_valid
    A 1-token error in a 50-token seq contributes: 1/N_valid
    -> Every token in the batch is equal. Structural tokens get exactly 3x gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("TAMER.Losses")


class LabelSmoothedCELoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for autoregressive sequence modeling.

    Uses PyTorch's native nn.CrossEntropyLoss with reduction='mean'.
    PyTorch's 'mean' reduction correctly averages over all non-ignored
    (non-padding) token positions across the entire flattened batch,
    which is mathematically equivalent to global token averaging.

    This class is mathematically correct as-is and requires no changes.

    Args:
        pad_id:          Token ID to ignore in loss (typically 0).
        label_smoothing: Smoothing factor. 0.1 = 10% uniform label distribution.
    """

    def __init__(self, pad_id: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
            reduction='mean',   # Averages over ALL non-ignored tokens globally
        )
        self.pad_id = pad_id

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, L, vocab_size) — model predictions, pre-aligned.
            targets: (B, L)             — target token indices, pre-aligned.

        Returns:
            Scalar loss.
        """
        # Flatten: (B, L, V) -> (B*L, V) and (B, L) -> (B*L,)
        # PyTorch CrossEntropyLoss with ignore_index averages only over
        # the non-ignored positions, giving us correct global averaging.
        pred_logits   = logits.view(-1, logits.size(-1))   # (B*L, V)
        target_tokens = targets.view(-1)                   # (B*L,)

        return self.criterion(pred_logits, target_tokens)


class StructureAwareLoss(nn.Module):
    """
    Structure-aware cross-entropy loss for multi-line and matrix equations.

    Assigns a higher loss weight to structural tokens that control 2D layout.
    Getting a row separator (\\\\) or column separator (&) wrong does not
    produce a single-token error — it destroys the entire matrix structure.
    The 3x gradient multiplier forces the optimizer to aggressively correct
    these high-impact mistakes first.

    Structural tokens weighted at structural_weight (default 3.0x):
      - Row separator:    \\\\
      - Column separator: &
      - \\begin{env} and \\end{env} for all matrix/alignment environments

    Mathematical guarantee:
      Final loss = sum_{all valid tokens}(ce_loss * weight) / N_valid_tokens

      Every token contributes 1/N_valid to the total gradient regardless of
      which sequence it came from. Structural tokens contribute exactly
      structural_weight/N_valid. No sequence-length bias exists.

    Args:
        tokenizer:         LaTeXTokenizer — used to look up structural token IDs.
        pad_id:            Token ID to ignore (typically 0).
        label_smoothing:   Label smoothing factor.
        structural_weight: Loss multiplier for structural tokens (default 3.0).
    """

    def __init__(
        self,
        tokenizer,
        pad_id: int = 0,
        label_smoothing: float = 0.1,
        structural_weight: float = 3.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.structural_weight = structural_weight

        # -----------------------------------------------------------------
        # Build the per-token weight vector.
        # All tokens start at weight 1.0. Structural tokens are raised to
        # structural_weight. This vector is registered as a buffer so it
        # automatically moves to the correct device with model.to(device).
        # -----------------------------------------------------------------
        vocab_size = len(tokenizer)
        weights = torch.ones(vocab_size, dtype=torch.float32)

        # Inline structural tokens
        STRUCTURAL_TOKENS = ['\\\\', '&']

        # Environment names whose \begin{} and \end{} tokens are structural
        ENV_NAMES = [
            'aligned', 'align', 'cases', 'gathered', 'split',
            'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix',
            'smallmatrix', 'array', 'eqnarray', 'multline',
        ]

        structural_count = 0

        for token in STRUCTURAL_TOKENS:
            if token in tokenizer.vocab:
                weights[tokenizer.vocab[token]] = structural_weight
                structural_count += 1

        for env in ENV_NAMES:
            for prefix in ['\\begin', '\\end']:
                env_token = f'{prefix}{{{env}}}'
                if env_token in tokenizer.vocab:
                    weights[tokenizer.vocab[env_token]] = structural_weight
                    structural_count += 1

        logger.info(
            f"StructureAwareLoss initialized: {structural_count} structural "
            f"tokens weighted at {structural_weight}x | vocab_size={vocab_size}"
        )

        # register_buffer guarantees device safety across DataParallel / DDP
        self.register_buffer('token_weights', weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, L, V) — model predictions.
            targets: (B, L)    — target token indices.

        Returns:
            Scalar loss using global token averaging with structural weighting.
        """
        B, L, V = logits.shape

        # Step 1: Flatten both tensors for efficient parallel computation.
        # Keeping them flat (B*L,) lets PyTorch use contiguous C++ memory
        # arrays for the element-wise multiply — no reshape overhead.
        pred_flat   = logits.view(-1, V)   # (B*L, V)
        target_flat = targets.view(-1)     # (B*L,)

        # Step 2: Compute per-token cross-entropy loss with NO reduction.
        # ignore_index=pad_id forces ce_loss to 0.0 at padding positions,
        # so they cannot contribute to either the numerator or denominator
        # even before we apply the explicit valid_mask below.
        # label_smoothing is applied per-token before reduction.
        ce_loss = F.cross_entropy(
            pred_flat,
            target_flat,
            ignore_index=self.pad_id,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )  # (B*L,)  — individual loss for every token position

        # Step 3: Look up the structural weight for every token in this batch.
        # self.token_weights is on the same device as logits (guaranteed by
        # register_buffer). Indexing with target_flat gives a (B*L,) tensor
        # where structural tokens have weight=structural_weight, others 1.0.
        current_weights = self.token_weights[target_flat]   # (B*L,)

        # Step 4: Build a binary validity mask.
        # 1.0 for real tokens, 0.0 for padding. This is our denominator mask.
        # Even though F.cross_entropy with ignore_index already zeros out
        # padding in ce_loss, we need this explicit mask to count valid tokens
        # accurately in the denominator.
        valid_mask = (target_flat != self.pad_id).float()   # (B*L,)

        # Step 5: Apply structural weights and silence any residual padding.
        # ce_loss is already 0.0 at pad positions (from ignore_index), but
        # multiplying by valid_mask makes the intent explicit and guards
        # against any future changes to the ignore_index behavior.
        weighted_loss = ce_loss * current_weights * valid_mask   # (B*L,)

        # Step 6: GLOBAL TOKEN AVERAGING — the mathematically correct reduction.
        #
        # Numerator:   sum of weighted loss over ALL valid tokens in the batch.
        # Denominator: total count of valid (non-padding) tokens in the batch.
        #
        # This means every token contributes exactly 1/N_valid to the gradient,
        # regardless of which sequence it came from. A token in a 5-token
        # equation and a token in a 50-token matrix are treated identically.
        # Structural tokens contribute structural_weight/N_valid.
        #
        # clamp(min=1.0): If an entirely-padded batch somehow reaches this
        # function (dataloader bug), we divide by 1.0 and return 0.0 loss
        # instead of NaN, preventing a training crash.
        total_valid_tokens = valid_mask.sum().clamp(min=1.0)

        final_loss = weighted_loss.sum() / total_valid_tokens   # scalar

        return final_loss