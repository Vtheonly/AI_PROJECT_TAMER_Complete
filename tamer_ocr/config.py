import os
import logging
from dataclasses import dataclass, field

@dataclass
class Config:
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
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
    num_workers: int = 4
    num_epochs: int = 150
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_epochs: int = 10
    label_smoothing: float = 0.1
    
    # Loss Weights
    seq_loss_weight: float = 1.0
    pointer_loss_weight: float = 1.0
    coverage_loss_weight: float = 0.5
    
    # Curriculum Learning
    curriculum_warmup_epochs: int = 15
    
    # Inference
    max_seq_len: int = 200
    beam_width: int = 5
    use_grammar_constraints: bool = True
    
    # Checkpointing
    save_every: int = 5
    eval_every: int = 5
    hf_repo_id: str = None  # e.g., "username/tamer-math-ocr"
    hf_token: str = None

    def __post_init__(self):
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)