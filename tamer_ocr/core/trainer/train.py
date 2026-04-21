#!/usr/bin/env python3
"""
TAMER OCR v2.4 — Training Entry Point.

Usage examples:
    python train.py                            # full pipeline with defaults
    python train.py --resume epoch_10.pt       # resume from specific checkpoint
    python train.py --eval-only best.pt        # greedy eval, no training
    python train.py --eval-only best.pt --beam-eval   # beam eval
    python train.py --epochs 50                # override number of epochs
    python train.py --encoder-lr 2e-5          # override encoder learning rate
    python train.py --fast-mode                # 128×512 resolution
    python train.py --balanced-mode            # 192×768 resolution
    python train.py --no-compile               # disable torch.compile
"""

import argparse
import os
import sys

import torch

# ──────────────────────────────────────────────────────────────────────
# Hardware optimisations — must be set BEFORE any CUDA initialisation.
#
#   cudnn.benchmark=True        cuDNN auto-tunes conv algorithms for the
#                               observed input shapes.  Zero cost after
#                               the first few warmup batches.
#
#   allow_tf32=True             Unlocks Tensor Cores on Ampere/Hopper.
#                               TF32 keeps FP32 range, reduces mantissa
#                               to 10 bits; effectively lossless for DNN
#                               training but ~2× faster on matmul.
#
#   set_float32_matmul_          PyTorch-level gate that confirms we want
#   precision('high')           the TF32 fast path for torch.matmul.
# ──────────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from tamer_ocr.config import Config
from tamer_ocr.core.trainer import Trainer


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TAMER OCR v2.4 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Paths ──────────────────────────────────────────────────────────
    path_grp = parser.add_argument_group("Paths")
    path_grp.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory for raw dataset storage.",
    )
    path_grp.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for model outputs and tokenizer.",
    )
    path_grp.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory for checkpoint storage.",
    )

    # ── Resume / Eval ──────────────────────────────────────────────────
    run_grp = parser.add_argument_group("Run mode")
    run_grp.add_argument(
        "--resume", type=str, default=None,
        help="Path to a specific checkpoint to resume training from.",
    )
    run_grp.add_argument(
        "--eval-only", type=str, default=None,
        metavar="CHECKPOINT",
        help="Run evaluation only (no training).",
    )
    run_grp.add_argument(
        "--beam-eval", action="store_true",
        help="Use beam search during --eval-only (slower, more accurate).",
    )

    # ── Training hyperparameters ───────────────────────────────────────
    train_grp = parser.add_argument_group("Training hyperparameters")
    train_grp.add_argument(
        "--epochs", type=int, default=None,
        help="Override config.num_epochs.",
    )
    train_grp.add_argument(
        "--batch-size", type=int, default=None,
        help="Override config.batch_size.",
    )
    train_grp.add_argument(
        "--encoder-lr", type=float, default=None,
        help="Override config.encoder_lr.",
    )
    train_grp.add_argument(
        "--decoder-lr", type=float, default=None,
        help="Override config.decoder_lr.",
    )
    train_grp.add_argument(
        "--workers", type=int, default=None,
        help="Override config.num_workers.",
    )

    # ── Resolution modes ───────────────────────────────────────────────
    res_grp = parser.add_argument_group("Resolution modes (mutually exclusive)")
    res_ex = res_grp.add_mutually_exclusive_group()
    res_ex.add_argument(
        "--fast-mode", action="store_true",
        help="128×512 resolution (~4× faster, lower accuracy).",
    )
    res_ex.add_argument(
        "--balanced-mode", action="store_true",
        help="192×768 resolution (~2× faster, good multi-line support).",
    )

    # ── Performance ────────────────────────────────────────────────────
    perf_grp = parser.add_argument_group("Performance")
    perf_grp.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile() (useful for debugging).",
    )
    perf_grp.add_argument(
        "--force-refresh", action="store_true",
        help="Force data re-download even if cached data exists.",
    )

    # ── HuggingFace ────────────────────────────────────────────────────
    hf_grp = parser.add_argument_group("HuggingFace Hub")
    hf_grp.add_argument(
        "--hf-repo", type=str, default=None,
        metavar="USER/REPO",
        help="HF model repo ID (e.g. username/tamer-ocr-model).",
    )
    hf_grp.add_argument(
        "--hf-dataset-repo", type=str, default=None,
        metavar="USER/REPO",
        help="HF dataset repo ID (e.g. username/tamer-preprocessed).",
    )
    hf_grp.add_argument(
        "--hf-token", type=str, default=None,
        help="HF API token (falls back to HF_TOKEN env var).",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────

def _apply_args_to_config(config: Config, args: argparse.Namespace) -> Config:
    """
    Apply CLI overrides to the config object.

    Resolution-mode flags are applied first so the final height/width
    values are reflected correctly in the printed summary.
    """
    # ── Resolution (mutually exclusive via argparse) ────────────────
    if args.fast_mode:
        config.fast_mode     = True
        config.balanced_mode = False
        config.img_height    = 128
        config.img_width     = 512
    elif args.balanced_mode:
        config.balanced_mode = True
        config.fast_mode     = False
        config.img_height    = 192
        config.img_width     = 768

    # ── Paths ────────────────────────────────────────────────────────
    if args.data_dir:
        config.data_dir = args.data_dir
        os.makedirs(config.data_dir, exist_ok=True)
    if args.output_dir:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Hyperparameters ──────────────────────────────────────────────
    if args.epochs      is not None: config.num_epochs  = args.epochs
    if args.batch_size  is not None: config.batch_size  = args.batch_size
    if args.encoder_lr  is not None: config.encoder_lr  = args.encoder_lr
    if args.decoder_lr  is not None: config.decoder_lr  = args.decoder_lr
    if args.workers     is not None: config.num_workers = args.workers

    # ── Performance ──────────────────────────────────────────────────
    if args.no_compile:
        config.compile_model = False
    if args.force_refresh:
        config.force_refresh = True

    # ── HuggingFace ──────────────────────────────────────────────────
    if args.hf_repo:
        config.hf_repo_id = args.hf_repo
    if args.hf_dataset_repo:
        config.hf_dataset_repo_id = args.hf_dataset_repo

    # Token: CLI flag > env var > existing config value
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if hf_token:
        config.hf_token = hf_token

    return config


def _print_banner() -> None:
    print("=" * 70)
    print("TAMER OCR v2.4 — Handwritten Mathematical Expression Recognition")
    print("Swin-v2-Base Encoder + 10-Layer Transformer Decoder")
    print("=" * 70)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        props   = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"VRAM            : {vram_gb:.1f} GB")
        print(f"TF32 matmul     : {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN benchmark : {torch.backends.cudnn.benchmark}")
    print("=" * 70)


def _print_config_summary(config: Config) -> None:
    eff_batch = config.batch_size * config.accumulation_steps
    print(f"Resolution      : {config.img_height}×{config.img_width}")
    print(
        f"Batch size      : {config.batch_size} × "
        f"{config.accumulation_steps} accum = {eff_batch} effective"
    )
    print(f"Num workers     : {config.num_workers}")
    print(f"Epochs          : {config.num_epochs}")
    print(f"Encoder LR      : {config.encoder_lr:.1e}")
    print(f"Decoder LR      : {config.decoder_lr:.1e}")
    print(f"torch.compile   : {config.compile_model}")
    print(
        f"Curriculum      : {config.curriculum_enabled} "
        f"(simple≤{config.curriculum_simple_until}, "
        f"medium≤{config.curriculum_medium_until})"
    )
    print(
        f"Structure loss  : {config.structure_aware_loss} "
        f"(weight={config.structural_token_weight})"
    )
    print(f"Freeze encoder  : {config.freeze_encoder_epochs} epochs")
    print(f"HF model repo   : {config.hf_repo_id or '(not set)'}")
    print(f"HF dataset repo : {config.hf_dataset_repo_id or '(not set)'}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    config = Config()
    config = _apply_args_to_config(config, args)

    _print_banner()
    _print_config_summary(config)

    # ── Eval-only mode ─────────────────────────────────────────────────
    if args.eval_only:
        eval_ckpt = args.eval_only
        if not os.path.exists(eval_ckpt):
            print(f"ERROR: Checkpoint not found: {eval_ckpt}", file=sys.stderr)
            sys.exit(1)

        print(f"Eval-only mode | checkpoint : {eval_ckpt}")
        print(f"Beam search    : {args.beam_eval}")

        trainer = Trainer(config)
        trainer.preprocess_data()
        trainer.create_dataloaders()
        trainer.build_model()
        trainer.resume_from_checkpoint(eval_ckpt)

        if args.beam_eval:
            metrics = trainer.evaluate_with_beam_search(max_samples=500)
        else:
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

    # ── Full training pipeline ─────────────────────────────────────────
    Trainer(config).run(resume_from=args.resume)


if __name__ == "__main__":
    main()