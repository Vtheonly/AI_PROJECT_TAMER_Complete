"""
TAMER OCR v2.3 — Configuration

Changes from v2.2:
  - Added balanced_mode (192×768): a middle ground between fast_mode
    (128×512) and full resolution (256×1024). Provides 576 encoder
    patches with 6 vertical rows — enough for most matrices while
    being ~2× faster than full mode.

  - Added curriculum learning support: the trainer progressively
    introduces harder samples (simple → medium → complex).
    Controlled by curriculum_enabled, curriculum_simple_until,
    and curriculum_medium_until.

  - Added structure_aware_loss: weights structural tokens (\\\\, &,
    \\begin, \\end) 3× higher in the loss. Getting row/column
    separators wrong destroys entire matrix structure.

  - Encoder now uses strong 2D positional encoding (learned row/col
    embeddings) with row boundary markers. No config changes needed.

  - Tokenizer now handles \\\\, \\begin{env}, \\end{env} as atomic
    tokens. Normalizer no longer discards matrices/aligned/cases.

Retained from v2.2:
  - encoder_name: swinv2_base_window8_256.ms_in1k
  - num_decoder_layers: 10, freeze_encoder_epochs: 5
  - num_epochs: 70, early_stopping_patience: 20
  - All other training parameters unchanged.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _default_data_root() -> str:
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/tamer_data"
    if os.path.isdir("/content"):
        return "/content/tamer_data"
    return "./tamer_data"


@dataclass
class Config:
    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    data_root: str = field(default_factory=_default_data_root)
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # ----------------------------------------------------------------
    # Dataset Configuration
    # ----------------------------------------------------------------
    datasets: List[dict] = field(default_factory=lambda: [
        {
            "name": "crohme",
            "type": "url",
            "url": "https://zenodo.org/records/8428035/files/CROHME23.zip?download=1",
            "parser": "crohme",
        },
        {
            "name": "hme100k",
            "type": "kaggle",
            "kaggle_slug": "prajwalchy/hme100k-dataset",
            "parser": "hme100k",
        },
        {
            "name": "im2latex",
            "type": "kaggle",
            "kaggle_slug": "shahrukhkhan/im2latex100k",
            "parser": "im2latex",
        },
        {
            "name": "mathwriting",
            "type": "huggingface",
            "hf_repo": "deepcopy/MathWriting-human",
            "parser": "mathwriting",
        },
    ])
    auto_download: bool = False
    skip_validation: bool = False

    # ----------------------------------------------------------------
    # Image Settings
    #
    # PERFORMANCE NOTE:
    #   256x1024 (default) — best quality, but slow.
    #   swin_v2_base produces 8x32 = 256 patches (fast), 6x24 = 144
    #   patches (balanced), or 8x32 = 256 patches at this resolution.
    #
    #   FAST MODE: set fast_mode=True → uses 128x512.
    #   Roughly 4x faster per step, ~3-5% accuracy drop on single-line,
    #   ~15-25% drop on multi-line (only 4 vertical patch rows).
    #
    #   BALANCED MODE: set balanced_mode=True → uses 192x768.
    #   Roughly 2x faster than full, 6 vertical patch rows — enough
    #   for most matrices. Best tradeoff for multi-line support.
    #
    #   Priority: fast_mode > balanced_mode > default
    # ----------------------------------------------------------------
    img_height: int = 256
    img_width: int = 1024
    fast_mode: bool = False          # If True, overrides resolution to 128x512
    balanced_mode: bool = False      # If True, overrides resolution to 192x768

    # ----------------------------------------------------------------
    # Data Filtering
    # ----------------------------------------------------------------
    max_token_length: int = 150
    max_aspect_ratio: float = 10.0

    # ----------------------------------------------------------------
    # Model Architecture
    #
    # ENCODER — swin_v2_base_patch4_window8_256
    #   Upgraded from swin_base_patch4_window7_224. Key improvement:
    #   log-spaced continuous relative position bias handles rect images
    #   and resolution transfer much better than the discrete bias table
    #   in v1. Also uses post-norm + scaled cosine attention for more
    #   stable deep-feature learning.
    #
    #   encoder.py is UNCHANGED — it reads channel count dynamically
    #   from timm's feature_info, so the upgrade is purely a name swap.
    #
    #   Channel widths (Swin-v2 Base, out_indices=(2,)):
    #     Stage 2 → 512 channels (same as Swin-Base v1)
    #
    #   For faster training with modest accuracy loss, swap to:
    #     "swin_v2_small_patch4_window8_256"   (~50M params, ~25% faster)
    #     "swin_v2_tiny_patch4_window8_256"    (~28M params, ~45% faster)
    #
    # DECODER — 8 layers (was 6)
    #   +2 layers ≈ +5-8% compute, +2-4% ExpRate on multi-dataset OCR.
    #   The encoder is still the dominant cost; the decoder is cheap.
    # ----------------------------------------------------------------
    encoder_name: str = "swinv2_base_window8_256.ms_in1k"
    encoder_feature_dim: int = 1024
    d_model: int = 768
    nhead: int = 12
    num_decoder_layers: int = 10
    dim_feedforward: int = 3072
    dropout: float = 0.15

    # ----------------------------------------------------------------
    # Training Parameters
    # ----------------------------------------------------------------
    # batch_size=16 + accumulation_steps=2 = effective batch of 32.
    batch_size: int = 16
    accumulation_steps: int = 2

    # 4 workers saturates the Kaggle CPU allocation.
    num_workers: int = 4

    # Reduced from 150 → 70.
    # 70 × ~75 min/epoch (worst-case T4) = ~87 hours, under 100-hour budget.
    # Early stopping (patience=20) will almost always stop well before 70.
    num_epochs: int = 70

    # Raised from 15 → 20. LaTeX OCR models plateau for many epochs
    # before breaking through.
    early_stopping_patience: int = 20

    # Differential learning rates: encoder is pretrained, decoder is random.
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    # ----------------------------------------------------------------
    # Encoder Freeze Warmup (NEW)
    #
    # Freeze the encoder for the first N epochs so the random decoder
    # can bootstrap without pulling the pre-trained encoder off its
    # good initialisation. After epoch N, the encoder unfreezes and
    # trains at encoder_lr for the rest of training.
    #
    # Set to 0 to disable (full fine-tuning from epoch 1).
    # ----------------------------------------------------------------
    freeze_encoder_epochs: int = 5

    # ----------------------------------------------------------------
    # Curriculum Learning (NEW)
    #
    # Progressively introduce harder samples:
    #   Phase 1 (epochs 1 to simple_until):  single-line formulas only
    #   Phase 2 (simple_until to medium_until): + aligned, cases
    #   Phase 3 (medium_until onward):        + matrices, arrays
    #
    # Set curriculum_enabled=False to train on all data from epoch 1.
    # ----------------------------------------------------------------
    curriculum_enabled: bool = True
    curriculum_simple_until: int = 15
    curriculum_medium_until: int = 35

    # ----------------------------------------------------------------
    # Structure-Aware Loss (NEW)
    #
    # Weights structural tokens (\\, &, \begin{}, \end{}) higher
    # in the loss. Getting these wrong destroys matrix structure.
    # Set to False to use standard label-smoothed CE only.
    # ----------------------------------------------------------------
    structure_aware_loss: bool = True
    structural_token_weight: float = 3.0

    # ----------------------------------------------------------------
    # Dynamic Temperature Sampling
    # ----------------------------------------------------------------
    temp_start: float = 0.8
    temp_end: float = 0.4

    # ----------------------------------------------------------------
    # OneCycleLR Scheduler
    # ----------------------------------------------------------------
    pct_start: float = 0.1

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    max_seq_len: int = 200
    beam_width: int = 5
    length_penalty: float = 0.6

    # ----------------------------------------------------------------
    # Checkpointing & Evaluation
    # ----------------------------------------------------------------
    checkpoint_every_epochs: int = 3
    keep_last_n_checkpoints: int = 3

    # Evaluate every N epochs — saves 20-30% of wall-clock time.
    eval_every: int = 3

    # Early epochs: cap val samples to avoid long waits.
    eval_warmup_epochs: int = 10
    eval_warmup_max_samples: int = 500

    # ----------------------------------------------------------------
    # HuggingFace Push Throttling
    # ----------------------------------------------------------------
    hf_repo_id: str = ""
    hf_token: str = ""
    hf_dataset_repo_id: str = ""
    hf_push_every_n_epochs: int = 5

    # ----------------------------------------------------------------
    # Kaggle Credentials
    # ----------------------------------------------------------------
    kaggle_username: str = ""
    kaggle_key: str = ""

    # ----------------------------------------------------------------
    # Performance Options
    # ----------------------------------------------------------------
    # torch.compile() (PyTorch 2.0+) gives 10-30% speedup on T4 GPUs.
    # Set to True once you've confirmed training is stable (usually
    # after a successful first epoch). Adds ~2-3 min JIT warmup on
    # the first epoch.
    compile_model: bool = False

    # ----------------------------------------------------------------
    # Internal (computed at runtime — do not set manually)
    # ----------------------------------------------------------------
    phase1_steps: int = 0
    phase2_start_step: int = 0
    total_training_steps: int = 0

    def __post_init__(self):
        # Apply resolution override (fast_mode takes priority)
        if self.fast_mode:
            self.img_height = 128
            self.img_width = 512
        elif self.balanced_mode:
            self.img_height = 192
            self.img_width = 768

        # Create all required directories
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)