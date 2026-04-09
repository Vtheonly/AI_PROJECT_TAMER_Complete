import sys
import os
import argparse
import random
import logging
from pathlib import Path
from logger import setup_logger
from data import (
    LaTeXTokenizer, TreeMathDataset, get_collate_fn, 
    get_train_augmentation, get_val_augmentation,
    list_available_datasets, validate_before_training
)
from config import Config
from models.tamer import TAMERCore
from core.constraints import LaTeXGrammarConstraints
from core.losses import TreeGuidedLoss
from core.inference import constrained_beam_search
from utils.metrics import calculate_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint, push_checkpoint_to_hf
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import torch

def setup_optimizer(model, config, epoch, logger):
    """Handles Progressive Unfreezing and Differential Learning Rates"""
    if config.text_only_pretrain:
        # Phase 0: Freeze encoder, train decoder
        for param in model.encoder.parameters(): param.requires_grad = False
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        
    if epoch < config.freeze_encoder_epochs:
        # Early Stage: Freeze Encoder to protect ImageNet weights
        logger.info(f"Epoch {epoch}: Encoder is FROZEN. Training Decoder only.")
        for param in model.encoder.parameters(): param.requires_grad = False
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    else:
        # Unfrozen Stage: Apply differential learning rates
        logger.info(f"Epoch {epoch}: Encoder is UNFROZEN. Full Joint Training.")
        for param in model.encoder.parameters(): param.requires_grad = True
        
        return torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': config.lr * config.encoder_lr_multiplier},
            {'params': model.decoder.parameters(), 'lr': config.lr * config.decoder_lr_multiplier}
        ], weight_decay=config.weight_decay)

def apply_scheduled_sampling(ids, pad_id, unk_id, epoch, config):
    """Transformer equivalent of Scheduled Sampling (Word Dropout)."""
    if epoch < config.ss_start_epoch:
        return ids
    
    # Scale probability from 0 up to ss_max_prob over the remaining epochs
    progress = (epoch - config.ss_start_epoch) / max(1, (config.num_epochs - config.ss_start_epoch))
    prob = progress * config.ss_max_prob
    
    mask = torch.rand(ids.shape, device=ids.device) < prob
    # Don't replace special tokens (pad_id)
    mask = mask & (ids != pad_id)
    
    corrupted_ids = ids.clone()
    corrupted_ids[mask] = unk_id
    return corrupted_ids

def train_one_epoch(model, loader, criterion, optimizer, scaler, config, tokenizer, device, logger, epoch):
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(loader):
        if batch is None: continue
        
        images = batch['image'].to(device, non_blocking=True)
        ids = batch['ids'].to(device, non_blocking=True)
        parents = batch['parents'].to(device, non_blocking=True)
        
        # Apply Scheduled Sampling
        input_ids = apply_scheduled_sampling(ids, tokenizer.pad_id, tokenizer.unk_id, epoch, config)
        
        with torch.amp.autocast('cuda'):
            logits, pointers, coverage = model(images, input_ids, parents, text_only=config.text_only_pretrain)
            loss, seq_loss, ptr_loss, cov_loss = criterion(logits, pointers, ids, parents, coverage)
            scaled_loss = loss / config.accumulation_steps
            
        scaler.scale(scaled_loss).backward()
        
        if (batch_idx + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        epoch_loss += loss.item()
        
        if batch_idx % 50 == 0:
            logger.info(f"Epoch [{epoch}] Batch [{batch_idx}/{len(loader)}] - Loss: {loss.item():.4f}")
                        
    return epoch_loss / len(loader)

@torch.no_grad()
def validate_all_domains(model, val_loaders, tokenizer, grammar, config, device, logger):
    """Tests the model against ALL domains independently to prove no Catastrophic Forgetting"""
    model.eval()
    logger.info("="*50)
    logger.info("CROSS-DOMAIN VALIDATION REPORT")
    logger.info("="*50)
    
    all_domain_metrics = {}
    
    for domain_name, loader in val_loaders.items():
        metrics_sum = {'exact': 0, 'leq1': 0, 'bracket': 0, 'ser': 0, 'total': 0}
        
        for batch in loader:
            if batch is None: continue
            images = batch['image'].to(device)
            gt_latex = batch['latex']
            
            for i in range(images.size(0)):
                if config.text_only_pretrain:
                    # Skip inference during text-only pretrain as images are ignored
                    break
                    
                pred_tokens = constrained_beam_search(model, images[i:i+1], tokenizer, grammar, config)
                pred_latex = tokenizer.decode(pred_tokens)
                
                res = calculate_metrics(pred_latex, gt_latex[i])
                for k in ['exact', 'leq1', 'bracket', 'ser']:
                    metrics_sum[k] += res[k]
                metrics_sum['total'] += 1
                
            if metrics_sum['total'] > 100: # Fast evaluation subset
                break
                
        total = max(1, metrics_sum['total'])
        final_metrics = {
            'exact': (metrics_sum['exact'] / total) * 100,
            'ser': (metrics_sum['ser'] / total) * 100,
            'bracket': (metrics_sum['bracket'] / total) * 100
        }
        all_domain_metrics[domain_name] = final_metrics
        
        if not config.text_only_pretrain:
            logger.info(f"Domain: [{domain_name.upper()}] | ExpRate: {final_metrics['exact']:.1f}% | SER: {final_metrics['ser']:.1f}%")
        
    return all_domain_metrics

def prepare_data_and_loaders(config, tokenizer, logger):
    from data.validator import DatasetValidator
    from data.datasets_registry import get_registry
    import json
    from pathlib import Path
    
    validator = DatasetValidator(config)
    
    def fetch_samples(dataset_list):
        out = []
        for name in dataset_list:
            ds_dir = validator.data_dir / name
            annot_file = ds_dir / get_registry().get_config(name).annotations_file if get_registry().get_config(name) else ds_dir / "annotations.json"
            if annot_file.exists():
                with open(annot_file, 'r') as f:
                    out.extend(validator._extract_samples(json.load(f), ds_dir, get_registry().get_config(name)))
        return out

    # 1. Load Main & Replay Data
    main_samples = fetch_samples(config.datasets)
    replay_samples = fetch_samples(config.replay_datasets)
    
    # 2. Mix Replay into Main
    if replay_samples:
        random.shuffle(replay_samples)
        num_replay = int(len(main_samples) * config.replay_ratio)
        main_samples.extend(replay_samples[:num_replay])
        logger.info(f"Mixed {num_replay} replay samples into {len(main_samples)} main samples.")
    
    # 3. Create Multi-Domain Validation Loaders
    # This evaluates the model against EVERYTHING you have, regardless of the current training stage.
    all_possible_datasets = ['im2latex-100k', 'mathwriting', 'crohme', 'hme100k']
    val_loaders = {}
    
    for ds_name in all_possible_datasets:
        ds_samples = fetch_samples([ds_name])
        if ds_samples:
            # Take a small 200-sample slice for fast validation
            val_split = ds_samples[:200]
            val_ds = TreeMathDataset(val_split, config, tokenizer, transform=get_val_augmentation())
            val_loaders[ds_name] = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=get_collate_fn(tokenizer.pad_id))
    
    # 4. Main Training Loader
    train_ds = TreeMathDataset(main_samples, config, tokenizer, transform=get_train_augmentation(config.img_height, config.img_width))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=get_collate_fn(tokenizer.pad_id), num_workers=config.num_workers, pin_memory=True)
    
    # Pre-train tokenizer on all found data so vocab is complete
    all_text = [s['latex'] for s in main_samples]
    for vl in val_loaders.values():
        all_text.extend([s['latex'] for s in vl.dataset.samples])
    tokenizer.build_from_corpus(all_text)
    
    return train_loader, val_loaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--replay-datasets', nargs='+', default=[])
    parser.add_argument('--phase0', action='store_true', help="Text-Only Pretraining")
    parser.add_argument('--freeze-epochs', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None, help="Path to previous stage checkpoint")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--enc-lr-mult', type=float, default=1.0)
    parser.add_argument('--dec-lr-mult', type=float, default=1.0)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    config = Config()
    config.datasets = args.datasets
    config.replay_datasets = args.replay_datasets
    config.text_only_pretrain = args.phase0
    config.freeze_encoder_epochs = args.freeze_epochs if not args.phase0 else 999
    config.lr = args.lr
    config.encoder_lr_multiplier = args.enc_lr_mult
    config.decoder_lr_multiplier = args.dec_lr_mult
    config.num_epochs = args.num_epochs
    config.checkpoint_dir = args.output_dir
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    logger = setup_logger("TAMER", config.log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = LaTeXTokenizer()
    train_loader, val_loaders = prepare_data_and_loaders(config, tokenizer, logger)
    
    model = TAMERCore(len(tokenizer), config).to(device)
    grammar = LaTeXGrammarConstraints(tokenizer)
    criterion = TreeGuidedLoss(tokenizer.pad_id, config)
    scaler = GradScaler('cuda')
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        # When moving between stages, we DO NOT load the optimizer state.
        # We want a fresh optimizer with the new Learning Rates.
        start_epoch, _ = load_checkpoint(args.resume, model, None, None, device)
        start_epoch = 0 # Reset epoch counter for the new stage
        logger.info(f"Loaded weights from previous stage: {args.resume}")

    # Set up optimizer (handles freezing/unfreezing logic)
    optimizer = setup_optimizer(model, config, start_epoch, logger)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=config.min_lr)
    
    best_exprate = 0.0
    
    for epoch in range(start_epoch, config.num_epochs):
        # Dynamically unfreeze and adjust LRs if passing the freeze threshold
        if not config.text_only_pretrain and epoch == config.freeze_encoder_epochs:
            optimizer = setup_optimizer(model, config, epoch, logger)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=config.min_lr)
            
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, tokenizer, device, logger, epoch)
        scheduler.step()
        
        if (epoch + 1) % config.eval_every == 0:
            metrics_dict = validate_all_domains(model, val_loaders, tokenizer, grammar, config, device, logger)
            
            # Determine "Best" based on the MAIN dataset being trained currently
            main_domain = config.datasets[0] 
            current_exprate = metrics_dict.get(main_domain, {}).get('exact', 0.0)
            
            is_best = current_exprate > best_exprate
            if is_best:
                best_exprate = current_exprate
                save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics_dict, os.path.join(config.checkpoint_dir, 'best.pt'))
        
        save_checkpoint(model, optimizer, scheduler, epoch + 1, {}, os.path.join(config.checkpoint_dir, 'latest.pt'))

if __name__ == "__main__":
    main()