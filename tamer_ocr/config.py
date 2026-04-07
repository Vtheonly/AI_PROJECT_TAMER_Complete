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
    datasets: List[str] = field(default_factory=lambda: [])  # List of dataset names to use
    auto_download: bool = False  # Auto-download missing datasets
    skip_validation: bool = False  # Skip pre-training validation (NOT RECOMMENDED)
    min_samples: int = 0  # Override minimum sample count for datasets
    train_split: float = 0.85
    val_split: float = 0.05
    test_split: float = 0.10
    
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
    
    # Authentication & Network Configuration
    # These can be set via config or environment variables (env takes precedence)
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    kaggle_api_token: str = field(default_factory=lambda: os.getenv("KAGGLE_API_TOKEN", ""))
    http_proxy: str = field(default_factory=lambda: os.getenv("HTTP_PROXY", ""))
    https_proxy: str = field(default_factory=lambda: os.getenv("HTTPS_PROXY", ""))

    def __post_init__(self):
        # Create directories
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)
        
        # Configure Proxies for underlying libraries
        if self.http_proxy:
            os.environ['HTTP_PROXY'] = self.http_proxy
        if self.https_proxy:
            os.environ['HTTPS_PROXY'] = self.https_proxy
            
        # Configure Kaggle Auth explicitly (supports both old and new auth methods)
        if self.kaggle_api_token:
            os.environ['KAGGLE_API_TOKEN'] = self.kaggle_api_token
        # Backward compatibility with old username/key method
        kaggle_username = os.getenv("KAGGLE_USERNAME", "")
        kaggle_key = os.getenv("KAGGLE_KEY", "")
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
