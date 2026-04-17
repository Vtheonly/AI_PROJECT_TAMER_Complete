"""
TAMER OCR v2.4 — Configuration

Changes from v2.3:
  - RTX 6000 Ada Beast Mode defaults:
      batch_size=512, num_workers=24, compile_model=True
  - Learning rates scaled for large batch:
      encoder_lr=3.5e-5, decoder_lr=3.5e-4
  - Added local_backbone_path for offline Swin-v2 weight loading.
  - Added sanitized_data_dir: points trainer at the clean JSONL files
    written by the sanitization cell instead of the raw input folder.
  - All v2.3 features retained: curriculum, structure_aware_loss,
    balanced_mode, fast_mode, freeze_encoder_epochs, etc.
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

    
    
    sanitized_data_dir: str = "/kaggle/working/sanitized_processed"

    
    
    
    datasets: List[dict] = field(default_factory=lambda: [
        {
            "name": "crohme",
            "type": "url",
            "url": (
                "https://zenodo.org/records/8428035/files/"
                "CROHME23.zip?download=1"
            ),
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    batch_size: int = 512
    accumulation_steps: int = 1
    num_workers: int = 24

    num_epochs: int = 70
    early_stopping_patience: int = 20

    
    
    
    encoder_lr: float = 3.5e-5
    decoder_lr: float = 3.5e-4

    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    
    
    
    
    
    
    
    
    freeze_encoder_epochs: int = 3

    
    
    
    
    
    
    
    curriculum_enabled: bool = True
    curriculum_simple_until: int = 10
    curriculum_medium_until: int = 25

    
    
    
    structure_aware_loss: bool = True
    structural_token_weight: float = 3.0

    
    
    
    temp_start: float = 0.8
    temp_end: float = 0.4

    
    
    
    pct_start: float = 0.1

    
    
    
    max_seq_len: int = 200
    beam_width: int = 5
    length_penalty: float = 0.6

    
    
    
    checkpoint_every_epochs: int = 2
    keep_last_n_checkpoints: int = 3
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

        
        for path in [
            self.data_dir,
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir,
        ]:
            os.makedirs(path, exist_ok=True)