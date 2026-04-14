#!/usr/bin/env python3
"""
TAMER OCR v2.1 Training Entry Point.

Usage:
    python train.py                           # Full pipeline with defaults
    python train.py --resume checkpoint.pt    # Resume from specific checkpoint
    python train.py --eval-only best.pt       # Run evaluation only
    python train.py --epochs 50               # Override number of epochs
    python train.py --encoder-lr 2e-5         # Override encoder learning rate
"""

import argparse
import sys
import os
import torch

from tamer_ocr.config import Config
from tamer_ocr.core.trainer import Trainer
from tamer_ocr.utils.checkpoint import find_latest_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="TAMER OCR v2.1 Training")
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', type=str, default=None, help='Path to checkpoint for eval-only mode')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--encoder-lr', type=float, default=None)
    parser.add_argument('--decoder-lr', type=float, default=None)
    parser.add_argument('--beam-eval', action='store_true', help='Use beam search for eval-only mode')
    parser.add_argument('--hf-repo', type=str, default=None, help='HuggingFace MODEL repo ID')
    parser.add_argument('--hf-dataset-repo', type=str, default=None, help='HuggingFace DATASET repo ID')
    parser.add_argument('--hf-token', type=str, default=None)
    parser.add_argument('--force-refresh', action='store_true', help='Force data re-download')
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()

    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.encoder_lr:
        config.encoder_lr = args.encoder_lr
    if args.decoder_lr:
        config.decoder_lr = args.decoder_lr
    if args.hf_repo:
        config.hf_repo_id = args.hf_repo
    if args.hf_dataset_repo:
        config.hf_dataset_repo_id = args.hf_dataset_repo
    if args.hf_token:
        config.hf_token = args.hf_token

    print("=" * 70)
    print("TAMER OCR v2.1: Handwritten Mathematical Expression Recognition")
    print("Swin-Base Encoder + Standard Transformer Decoder")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Eval-only mode
    if args.eval_only:
        eval_checkpoint = args.eval_only
        if not os.path.exists(eval_checkpoint):
            print(f"ERROR: Checkpoint not found: {eval_checkpoint}")
            sys.exit(1)

        trainer = Trainer(config)
        trainer.preprocess_data()
        trainer.create_dataloaders()
        trainer.build_model()
        trainer.resume_from_checkpoint(eval_checkpoint)

        if args.beam_eval:
            metrics = trainer.evaluate_with_beam_search(max_samples=500)
        else:
            metrics = trainer._evaluate()

        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    # Full training pipeline (auto-resumes from latest checkpoint)
    trainer = Trainer(config)
    trainer.run(resume_from=args.resume)


if __name__ == '__main__':
    main()
