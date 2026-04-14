import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Dataset Configuration
    datasets: List[str] = field(default_factory=lambda: [])
    auto_download: bool = False
    skip_validation: bool = False

    # Image Settings — FIXED: 256 height, 1024 width with aspect ratio preservation
    img_height: int = 256
    img_width: int = 1024

    # Data Filtering
    max_token_length: int = 150     # Discard samples with LaTeX longer than this
    max_aspect_ratio: float = 10.0  # Discard images where w/h or h/w exceeds this

    # Model Architecture — Swin-Base + Standard Transformer Decoder
    encoder_name: str = "swin_base_patch4_window7_224"
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    encoder_feature_dim: int = 1024  # Swin-Base output dimension

    # Training Parameters
    batch_size: int = 8
    accumulation_steps: int = 4  # Effective batch = 32
    num_workers: int = 2
    num_epochs: int = 150
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    # Dynamic Temperature Sampling
    temp_start: float = 0.8  # Start: upweight small datasets (CROHME)
    temp_end: float = 0.4    # End: more uniform sampling

    # OneCycleLR Scheduler
    pct_start: float = 0.1   # 10% warmup

    # Inference
    max_seq_len: int = 200
    beam_width: int = 5
    length_penalty: float = 0.6  # Penalize short predictions

    # Checkpointing — epoch-based, save every N epochs, auto-resume from latest
    checkpoint_every_epochs: int = 3
    keep_last_n_checkpoints: int = 3
    eval_every: int = 1

    # HuggingFace — Model Checkpoints
    hf_repo_id: str = ""
    hf_token: str = ""

    # HuggingFace — Processed Dataset Repository
    hf_dataset_repo_id: str = ""   # e.g. "username/tamer-preprocessed"
    # If empty, defaults to "{hf_username}/tamer-preprocessed"

    # Kaggle
    kaggle_username: str = "merselfares"
    kaggle_key: str = ""

    # 72-Hour Schedule (step counts are approximate)
    phase1_steps: int = 0        # Phase 1: Printed data only (Im2LaTeX + MathWriting)
    phase2_start_step: int = 0   # Phase 2: Full mixture training (all datasets)
    total_training_steps: int = 50000  # Rough estimate for 72 hours on T4

    def __post_init__(self):
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)
