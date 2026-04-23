"""
TAMER OCR v2.4 — Configuration

Zero-contradiction Kaggle offline edition.
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
    data_root: str = field(default_factory=_default_data_root)
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    local_backbone_path: str = ""
    sanitized_data_dir: str = ""

    datasets: List[dict] = field(default_factory=lambda: [
        {"name": "crohme",      "type": "url",         "url": "https://zenodo.org/records/8428035/files/CROHME23.zip?download=1", "parser": "crohme"},
        {"name": "hme100k",     "type": "kaggle",      "kaggle_slug": "prajwalchy/hme100k-dataset", "parser": "hme100k"},
        {"name": "im2latex",    "type": "kaggle",      "kaggle_slug": "shahrukhkhan/im2latex100k",  "parser": "im2latex"},
        {"name": "mathwriting", "type": "huggingface", "hf_repo": "deepcopy/MathWriting-human",       "parser": "mathwriting"},
    ])
    auto_download: bool = False
    skip_validation: bool = False

    img_height: int = 256
    img_width: int = 1024
    fast_mode: bool = False
    balanced_mode: bool = False

    max_token_length: int = 150
    max_aspect_ratio: float = 10.0

    encoder_name: str = "swinv2_base_window8_256.ms_in1k"
    encoder_feature_dim: int = 1024
    d_model: int = 768
    nhead: int = 12
    num_decoder_layers: int = 10
    dim_feedforward: int = 3072
    dropout: float = 0.15

    batch_size: int = 256
    accumulation_steps: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    num_epochs: int = 70
    early_stopping_patience: int = 20

    encoder_lr: float = 5e-6
    decoder_lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    freeze_encoder_epochs: int = 0

    curriculum_enabled: bool = True
    curriculum_simple_until: int = 15
    curriculum_medium_until: int = 30

    structure_aware_loss: bool = True
    structural_token_weight: float = 3.0

    temp_start: float = 0.8
    temp_end: float = 0.4
    pct_start: float = 0.1

    max_seq_len: int = 200
    beam_width: int = 5
    length_penalty: float = 0.6

    checkpoint_every_epochs: int = 1
    keep_last_n_checkpoints: int = 5
    eval_every: int = 2
    eval_warmup_epochs: int = 10
    eval_warmup_max_samples: int = 500

    hf_repo_id: str = ""
    hf_token: str = ""
    hf_dataset_repo_id: str = ""
    hf_push_every_n_epochs: int = 5

    kaggle_username: str = ""
    kaggle_key: str = ""

    compile_model: bool = True

    phase1_steps: int = 0
    phase2_start_step: int = 0
    total_training_steps: int = 0

    def __post_init__(self):
        if self.fast_mode:
            self.img_height = 128
            self.img_width = 512
        elif self.balanced_mode:
            self.img_height = 192
            self.img_width = 768

        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)

def kaggle_offline_config(
    sanitized_data_dir: str = "/kaggle/input/datasets/merselfares/tamer-sanitized-jsonl",
    data_dir: str = "/kaggle/input/datasets/merselfares/tamer-full-pipeline-v1/hf_data",
    local_backbone_path: str = "/kaggle/input/datasets/merselfares/swinv2-base-22k/swinv2_base_22k.safetensors",
) -> Config:
    """
    ULTIMATE MAXIMUM ACCURACY CONFIG
    Tuned for RTX 6000 Ada (96GB VRAM) - Offline Mode
    """
    cfg = Config()
    cfg.data_dir = data_dir
    cfg.sanitized_data_dir = sanitized_data_dir
    cfg.local_backbone_path = local_backbone_path
    cfg.output_dir = "/kaggle/working/outputs"
    cfg.checkpoint_dir = "/kaggle/working/checkpoints"
    cfg.log_dir = "/kaggle/working/logs"

    # --- MAXIMUM ACCURACY RESOLUTION ---
    # 384x1280 allows matrices and fractions to retain crisp structural details
    cfg.img_height = 384
    cfg.img_width = 1280

    # --- OOM-SAFE BATCHING FOR 96GB ---
    # 48 fits safely even with deep backpropagation graphs and compilation
    cfg.batch_size = 96 
    cfg.accumulation_steps = 2  # Effective batch size = 192
    
    cfg.num_workers = 10 # Utilize Kaggle CPU cores
    cfg.pin_memory = True
    cfg.prefetch_factor = 2
    cfg.persistent_workers = True
    cfg.compile_model = True

    # --- LEARNING RATES & REGULARIZATION ---
    cfg.encoder_lr = 2e-6     # Gentle on the pre-trained Swin features
    cfg.decoder_lr = 3e-4     
    cfg.weight_decay = 1e-4
    cfg.max_grad_norm = 1.0
    cfg.label_smoothing = 0.1
    
    # 15 epochs gives the decoder time to align with the massive encoder before unfreezing
    cfg.freeze_encoder_epochs = 0

    # --- TRAINING SCHEDULE ---
    cfg.num_epochs = 70
    cfg.early_stopping_patience = 20
    cfg.eval_every = 1        # Validate every epoch to catch spikes early
    cfg.checkpoint_every_epochs = 1
    cfg.keep_last_n_checkpoints = 3
    
    # --- CURRICULUM & LOSS ---
    cfg.curriculum_enabled = True
    cfg.curriculum_simple_until = 10
    cfg.curriculum_medium_until = 25
    cfg.structure_aware_loss = True
    cfg.structural_token_weight = 4.0 # Heavily penalize broken row/col separators

    # --- STRICT OFFLINE SETTINGS ---
    cfg.auto_download = False
    cfg.hf_token = ""
    cfg.hf_repo_id = ""

    import os
    for d in [cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir]:
        os.makedirs(d, exist_ok=True)

    return cfg  