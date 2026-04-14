import os
from dataclasses import dataclass, field
from typing import List, Optional

_DEFAULT_DATA_ROOT = "/content/tamer_data" if os.path.isdir("/content") else "/kaggle/working/tamer_data"

@dataclass
class Config:
    # Paths
    data_root: str = _DEFAULT_DATA_ROOT
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Dataset Configuration
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

    # Image Settings
    img_height: int = 256
    img_width: int = 1024

    # Data Filtering
    max_token_length: int = 150     
    max_aspect_ratio: float = 10.0  

    # Model Architecture
    encoder_name: str = "swin_base_patch4_window7_224"
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    encoder_feature_dim: int = 1024 

    # Training Parameters
    batch_size: int = 8
    accumulation_steps: int = 4  
    num_workers: int = 2
    num_epochs: int = 150
    early_stopping_patience: int = 15  # Added to prevent overfitting
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    # Dynamic Temperature Sampling
    temp_start: float = 0.8  
    temp_end: float = 0.4    

    # OneCycleLR Scheduler
    pct_start: float = 0.1   

    # Inference
    max_seq_len: int = 200
    beam_width: int = 5
    length_penalty: float = 0.6  

    # Checkpointing
    checkpoint_every_epochs: int = 3
    keep_last_n_checkpoints: int = 3
    eval_every: int = 1

    # HuggingFace
    hf_repo_id: str = ""
    hf_token: str = ""
    hf_dataset_repo_id: str = ""   

    # Kaggle
    kaggle_username: str = ""  # Removed hardcoded username
    kaggle_key: str = ""

    phase1_steps: int = 0        
    phase2_start_step: int = 0   
    total_training_steps: int = 0  # Will be calculated dynamically in trainer

    def __post_init__(self):
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)