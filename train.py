#!/usr/bin/env python3
"""
TAMER OCR v2.3 Training Entry Point.

Usage:
    python train.py                           # Full pipeline with defaults
    python train.py --resume checkpoint.pt    # Resume from specific checkpoint
    python train.py --eval-only best.pt       # Run evaluation only
    python train.py --epochs 50               # Override number of epochs
    python train.py --encoder-lr 2e-5         # Override encoder learning rate
    python train.py --fast-mode               # Use 128x512 resolution
    python train.py --balanced-mode           # Use 192x768 resolution
    python train.py --no-compile              # Disable torch.compile
"""

import argparse
import sys
import os
import torch

# ============================================================
# H100 GPU HARDWARE OPTIMIZATIONS
# ============================================================
# Must be set BEFORE any model or CUDA operations are initialized.
#
# cudnn.benchmark=True: cuDNN auto-tunes convolution algorithms for
#   the specific input sizes it sees. Adds a small warmup cost on
#   the first few batches, then picks the fastest kernel for the rest
#   of training. Always beneficial when input sizes are fixed.
#
# allow_tf32=True: Unlocks Tensor Cores on Ampere/Hopper GPUs (A100,
#   H100). TF32 uses FP32 range but 10-bit mantissa instead of 23-bit.
#   For neural network training this is indistinguishable from full
#   FP32 in practice, but runs ~2x faster on matmul operations.
#
# set_float32_matmul_precision('high'): PyTorch-level flag that
#   confirms we want the TF32 fast path for all torch.matmul calls.
#   Works in conjunction with the CUDA flag above.
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from tamer_ocr.config import Config
from tamer_ocr.core.trainer import Trainer
from tamer_ocr.utils.checkpoint import find_latest_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="TAMER OCR v2.3 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Directory for raw dataset storage',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory for model outputs and tokenizer',
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Directory for checkpoint storage',
    )

    # ── Resume / Eval ──────────────────────────────────────────────
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to a specific checkpoint to resume training from',
    )
    parser.add_argument(
        '--eval-only', type=str, default=None,
        help='Path to checkpoint — runs evaluation only, no training',
    )
    parser.add_argument(
        '--beam-eval', action='store_true',
        help='Use beam search during --eval-only (slower but more accurate)',
    )

    # ── Training Hyperparameters ───────────────────────────────────
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override config.num_epochs',
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Override config.batch_size',
    )
    parser.add_argument(
        '--encoder-lr', type=float, default=None,
        help='Override config.encoder_lr',
    )
    parser.add_argument(
        '--decoder-lr', type=float, default=None,
        help='Override config.decoder_lr',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Override config.num_workers',
    )

    # ── Resolution Modes ───────────────────────────────────────────
    parser.add_argument(
        '--fast-mode', action='store_true',
        help='Use 128x512 resolution (~4x faster, lower accuracy)',
    )
    parser.add_argument(
        '--balanced-mode', action='store_true',
        help='Use 192x768 resolution (~2x faster, good multi-line support)',
    )

    # ── Performance ────────────────────────────────────────────────
    parser.add_argument(
        '--no-compile', action='store_true',
        help='Disable torch.compile() (useful for debugging or older PyTorch)',
    )

    # ── HuggingFace ────────────────────────────────────────────────
    parser.add_argument(
        '--hf-repo', type=str, default=None,
        help='HuggingFace MODEL repo ID (e.g. username/tamer-ocr-model)',
    )
    parser.add_argument(
        '--hf-dataset-repo', type=str, default=None,
        help='HuggingFace DATASET repo ID (e.g. username/tamer-preprocessed)',
    )
    parser.add_argument(
        '--hf-token', type=str, default=None,
        help='HuggingFace API token (or set HF_TOKEN env var)',
    )

    # ── Misc ───────────────────────────────────────────────────────
    parser.add_argument(
        '--force-refresh', action='store_true',
        help='Force data re-download even if cached data exists',
    )

    return parser.parse_args()


def _print_banner():
    """Print a startup banner with system info."""
    print("=" * 70)
    print("TAMER OCR v2.3: Handwritten Mathematical Expression Recognition")
    print("Swin-v2-Base Encoder + 10-Layer Transformer Decoder")
    print("=" * 70)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"VRAM            : {vram_gb:.1f} GB")
        print(f"TF32 matmul     : {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN benchmark : {torch.backends.cudnn.benchmark}")
    print("=" * 70)


def _apply_args_to_config(config: Config, args: argparse.Namespace) -> Config:
    """
    Apply command-line argument overrides to the config object.
    Resolution mode flags are applied first so __post_init__ logic
    is reflected correctly in the final printed summary.
    """
    # Resolution modes (mutually exclusive; fast_mode wins if both set)
    if args.fast_mode:
        config.fast_mode = True
        config.balanced_mode = False
        # Re-apply __post_init__ resolution logic
        config.img_height = 128
        config.img_width = 512
    elif args.balanced_mode:
        config.balanced_mode = True
        config.img_height = 192
        config.img_width = 768

    # Paths
    if args.data_dir:
        config.data_dir = args.data_dir
        os.makedirs(config.data_dir, exist_ok=True)
    if args.output_dir:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training hyperparameters
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.encoder_lr is not None:
        config.encoder_lr = args.encoder_lr
    if args.decoder_lr is not None:
        config.decoder_lr = args.decoder_lr
    if args.workers is not None:
        config.num_workers = args.workers

    # Performance
    if args.no_compile:
        config.compile_model = False

    # HuggingFace
    if args.hf_repo:
        config.hf_repo_id = args.hf_repo
    if args.hf_dataset_repo:
        config.hf_dataset_repo_id = args.hf_dataset_repo
    if args.hf_token:
        config.hf_token = args.hf_token
    # Also pick up token from environment if not passed explicitly
    if not config.hf_token:
        config.hf_token = os.environ.get('HF_TOKEN', '')

    return config


def _print_config_summary(config: Config):
    """Print a concise summary of the active configuration."""
    eff_batch = config.batch_size * config.accumulation_steps
    print(f"Resolution      : {config.img_height}×{config.img_width}")
    print(f"Batch size      : {config.batch_size} × {config.accumulation_steps} accum = {eff_batch} effective")
    print(f"Num workers     : {config.num_workers}")
    print(f"Epochs          : {config.num_epochs}")
    print(f"Encoder LR      : {config.encoder_lr:.1e}")
    print(f"Decoder LR      : {config.decoder_lr:.1e}")
    print(f"torch.compile   : {config.compile_model}")
    print(f"Curriculum      : {config.curriculum_enabled} "
          f"(simple≤{config.curriculum_simple_until}, medium≤{config.curriculum_medium_until})")
    print(f"Structure loss  : {config.structure_aware_loss} "
          f"(weight={config.structural_token_weight})")
    print(f"Freeze encoder  : {config.freeze_encoder_epochs} epochs")
    print(f"HF model repo   : {config.hf_repo_id or '(not set)'}")
    print(f"HF dataset repo : {config.hf_dataset_repo_id or '(not set)'}")
    print("=" * 70)


def main():
    args = parse_args()

    _print_banner()

    config = Config()
    config = _apply_args_to_config(config, args)

    _print_config_summary(config)

    # ----------------------------------------------------------------
    # Eval-only mode
    # ----------------------------------------------------------------
    if args.eval_only:
        eval_checkpoint = args.eval_only

        if not os.path.exists(eval_checkpoint):
            print(f"ERROR: Checkpoint not found: {eval_checkpoint}")
            sys.exit(1)

        print(f"Eval-only mode | checkpoint: {eval_checkpoint}")
        print(f"Beam search: {args.beam_eval}")

        trainer = Trainer(config)
        trainer.preprocess_data()
        trainer.create_dataloaders()
        trainer.build_model()
        trainer.resume_from_checkpoint(eval_checkpoint)

        if args.beam_eval:
            metrics = trainer.evaluate_with_beam_search(max_samples=500)
        else:
            # Greedy evaluation over the full validation set
            metrics, _ = trainer._evaluate(
                use_beam_search=False,
                max_samples=None,
            )

        print("\nEvaluation Results:")
        print("-" * 40)
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k:<30s}: {v:.4f}")
            else:
                print(f"  {k:<30s}: {v}")
        return

    # ----------------------------------------------------------------
    # Full training pipeline
    # Auto-resumes from latest checkpoint if one exists.
    # ----------------------------------------------------------------
    trainer = Trainer(config)
    trainer.run(resume_from=args.resume)


if __name__ == '__main__':
    main()