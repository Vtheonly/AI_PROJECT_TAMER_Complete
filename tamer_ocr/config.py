import os
import logging
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
    replay_datasets: List[str] = field(default_factory=lambda: [])
    replay_ratio: float = 0.05  # 5% of batch comes from replay
    auto_download: bool = False
    skip_validation: bool = False
    
    # Image Settings
    img_height: int = 128
    img_width: int = 512
    
    # Model Architecture
    encoder_name: str = "swinv2_tiny_window16_256"
    d_model: int = 256
    nhead: int = 8
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    coverage_dim: int = 64
    
    # Training Parameters
    batch_size: int = 16
    accumulation_steps: int = 2
    num_workers: int = 2
    num_epochs: int = 150
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    
    # SOTA Training Tricks
    text_only_pretrain: bool = False    # Phase 0
    freeze_encoder_epochs: int = 5      # Freeze Swin for N epochs
    decoder_lr_multiplier: float = 1.0  # Used in later stages
    encoder_lr_multiplier: float = 1.0  # Used in later stages
    
    # Scheduled Sampling (Exposure Bias Fix via Token Dropout)
    ss_start_epoch: int = 20
    ss_max_prob: float = 0.20           # Max 20% of tokens replaced
    
    # Loss Weights
    seq_loss_weight: float = 1.0
    pointer_loss_weight: float = 1.0
    coverage_loss_weight: float = 0.5
    
    # Inference
    max_seq_len: int = 200
    beam_width: int = 5
    use_grammar_constraints: bool = True
    
    # Checkpointing
    save_every: int = 1
    eval_every: int = 1
    hf_repo_id: str = "JJKK1212/tamer-math-ocr"
    
    def __post_init__(self):
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)