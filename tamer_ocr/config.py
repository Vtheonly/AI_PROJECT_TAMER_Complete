"""
TAMER OCR v2.1 — Configuration

Key changes from original:
  - num_workers bumped to 4 (was 2; DataLoader was starving the GPU)
  - batch_size bumped to 16 (was 8; safe on 16GB T4 with AMP + 256x1024 images)
  - accumulation_steps reduced to 2 (effective batch stays at 32)
  - eval_every set to 3 (was 1; evaluating every epoch wasted 20-30% of training time)
  - hf_push_every_n_epochs added (HF push was blocking the training loop every epoch)
  - compile_model flag added (torch.compile gives 10-30% speedup on PyTorch 2.x)
  - early_stopping_patience raised to 20 (15 was too aggressive for a slow-starting OCR model)
  - Added fast_mode image resolution option (128x512 for quick experimentation)
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
    #   swin_base_patch4_window7_224 produces 64x256 = 16,384 patches at
    #   this resolution. Each forward pass is expensive.
    #
    #   FAST MODE: set fast_mode=True → uses 128x512 (4,096 patches).
    #   Roughly 4x faster per step, ~3-5% accuracy drop. Good for
    #   early experiments or if your epoch time is > 90 min.
    # ----------------------------------------------------------------
    img_height: int = 256
    img_width: int = 1024
    fast_mode: bool = False          # If True, overrides resolution to 128x512

    # ----------------------------------------------------------------
    # Data Filtering
    # ----------------------------------------------------------------
    max_token_length: int = 150
    max_aspect_ratio: float = 10.0

    # ----------------------------------------------------------------
    # Model Architecture
    #
    # PERFORMANCE NOTE:
    #   swin_base_patch4_window7_224 — strong but heavy (~87M params in encoder).
    #   For faster training with modest accuracy loss, swap to:
    #     "swin_small_patch4_window7_224"   (~50M params, ~25% faster)
    #     "swin_tiny_patch4_window7_224"    (~28M params, ~45% faster)
    #   encoder_feature_dim must match: base=1024, small=768, tiny=768
    # ----------------------------------------------------------------
    encoder_name: str = "swin_base_patch4_window7_224"
    encoder_feature_dim: int = 1024     # Must match encoder output channels
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # ----------------------------------------------------------------
    # Training Parameters
    # ----------------------------------------------------------------
    # batch_size=16 + accumulation_steps=2 = effective batch of 32.
    # Same effective batch as before (8x4), but fewer DataLoader iterations
    # per epoch means less Python/collation overhead.
    batch_size: int = 16
    accumulation_steps: int = 2

    # 4 workers saturates the Kaggle CPU allocation.
    # Do NOT go above 4 on Kaggle — you will get throttled.
    num_workers: int = 4

    num_epochs: int = 150

    # Raised from 15 → 20. Early LaTeX OCR training is noisy; the model
    # often plateaus for 10+ epochs before breaking through.
    early_stopping_patience: int = 20

    # Differential learning rates: encoder is pretrained, decoder is random.
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

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

    # Evaluate every N epochs.
    # Was 1 (every epoch), now 3. Saves ~20-30% of wall-clock time
    # since full greedy-decode evaluation is expensive.
    eval_every: int = 3

    # Early epochs: cap val samples to avoid long waits before the model
    # has learned anything meaningful.
    eval_warmup_epochs: int = 10       # Epochs to use reduced eval set
    eval_warmup_max_samples: int = 500  # Sample cap during warmup

    # ----------------------------------------------------------------
    # HuggingFace Push Throttling
    # ----------------------------------------------------------------
    hf_repo_id: str = ""
    hf_token: str = ""
    hf_dataset_repo_id: str = ""
    # Only push to HF every N epochs. Was pushing on EVERY best checkpoint,
    # which blocked the training loop with a network call mid-epoch.
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
    # Set to True once you've confirmed training is stable.
    # Note: compile adds ~2-3 minutes of JIT warmup on the first epoch.
    compile_model: bool = False

    # ----------------------------------------------------------------
    # Internal (computed at runtime — do not set manually)
    # ----------------------------------------------------------------
    phase1_steps: int = 0
    phase2_start_step: int = 0
    total_training_steps: int = 0

    def __post_init__(self):
        # Apply fast_mode resolution override
        if self.fast_mode:
            self.img_height = 128
            self.img_width = 512

        # Create all required directories
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)