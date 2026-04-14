# TAMER OCR v2.0 — Handwritten Mathematical Expression Recognition

**Swin-Base Encoder + Standard Transformer Decoder**

## Architecture

| Component | Specification |
|-----------|--------------|
| **Encoder** | Swin-Base (`swin_base_patch4_window7_224`) with gradient checkpointing |
| **Decoder** | Standard 6-layer Transformer Decoder, 8 heads, d_model=512, dim_ff=2048 |
| **Image Size** | 256x1024 (aspect-ratio preserving, white padding) |
| **Positional Encoding** | 2D learned (encoder) + sinusoidal (decoder) |
| **Projection** | Linear(1024→512) + LayerNorm |

## What Was Removed (and Why)

| Removed | Reason |
|---------|--------|
| **TAM (Tree-Aware Module)** | O(T²) memory, slow convergence |
| **Pointer Network / Pointer Loss** | Teaches "linked list not tree" |
| **Grammar Constraints** | Too brittle, blocks valid math |
| **Zero-Visual Pretraining** | Taught model to ignore image |
| **Square Resizing** | Kills features of long equations |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| AMP | float16 with GradScaler |
| Encoder LR | 1e-5 |
| Decoder LR | 1e-4 |
| Scheduler | OneCycleLR (pct_start=0.1) |
| Label Smoothing | 0.1 |
| Batch Size | 8 × 4 accumulation = 32 effective |
| Gradient Clipping | max_norm=1.0 |
| Temperature Sampling | 0.8 → 0.4 (linear decay) |
| Beam Search | width=5, length_penalty=0.6 |

## Project Structure

```
tamer_ocr/
├── __init__.py
├── config.py                    # All hyperparameters in one place
├── logger.py                    # Logging setup
├── data/
│   ├── __init__.py
│   ├── latex_normalizer.py      # LaTeX cleaning & filtering
│   ├── tokenizer.py             # Global vocabulary builder
│   ├── dataset.py               # MathDataset with 256x1024 padding
│   ├── sampler.py               # Temperature-based sampling
│   ├── parser.py                # CROHME/Im2LaTeX/HME100K/MathWriting parsers
│   ├── data_manager.py          # Orchestrates download + parse + normalize
│   ├── downloader.py            # Legacy downloader
│   ├── advanced_downloader.py   # HF/Kaggle/Zenodo/GitHub downloader
│   ├── datasets_registry.py     # Dataset metadata registry
│   ├── augmentation.py          # Albumentations transforms
│   └── validator.py             # Pre-training validation
├── models/
│   ├── __init__.py
│   ├── attention.py             # PositionalEncoding1D + 2D
│   ├── encoder.py               # Swin-Base with gradient checkpointing
│   ├── decoder.py               # Standard Transformer Decoder
│   └── tamer.py                 # TAMERModel (encoder + decoder)
├── core/
│   ├── __init__.py
│   ├── trainer.py               # Main training pipeline
│   ├── engine.py                # Low-level train/eval step functions
│   ├── losses.py                # Label-smoothed CrossEntropy
│   └── inference.py             # Beam search + greedy decode
└── utils/
    ├── __init__.py
    ├── checkpoint.py            # Save/load/backup/HF push
    └── metrics.py               # ExpRate, Edit Distance, SER

train.py                         # CLI entry point
tamer_train_colab.ipynb          # Execution-only notebook
requirements.txt                 # Dependencies
```

## Quick Start

### CLI Training
```bash
pip install -r requirements.txt
python train.py
```

### Resume from Checkpoint
```bash
python train.py --resume checkpoints/step_10000.pt
```

### Evaluation Only
```bash
python train.py --eval-only checkpoints/best.pt --beam-eval
```

### Colab / Kaggle
Upload the codebase and run `tamer_train_colab.ipynb`. The notebook only:
1. Pulls the codebase
2. Calls `Trainer.run()`
3. Saves to HuggingFace

**No training logic in the notebook.**

## 72-Hour Training Schedule

| Phase | Hours | Data |
|-------|-------|------|
| 1 | 0–4 | Refactor & validate |
| 2 | 4–24 | Printed data (Im2LaTeX + MathWriting) |
| 3 | 24–72 | Full mixture (all datasets) |

## Key Design Decisions

1. **256x1024 images**: Preserves aspect ratio of long equations. Height fixed at 256, width padded to 1024 with white pixels.
2. **No tree structure**: Standard autoregressive decoding — simpler, faster, more reliable.
3. **Differential LR**: Encoder learns slowly (1e-5) to preserve pretrained features; decoder learns fast (1e-4).
4. **Dynamic Temperature Sampling**: Early training upweights small datasets (CROHME) with T=0.8; late training is more uniform with T=0.4.
5. **Step-based checkpointing**: Every 1000 steps, save model + optimizer + scaler + scheduler. Backup to Drive for Colab session hopping.
