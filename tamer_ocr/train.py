import sys
import os
import argparse
import logging
from pathlib import Path
from logger import setup_logger
from data import (
    LaTeXTokenizer,
    TreeMathDataset,
    get_collate_fn,
    get_train_augmentation,
    get_val_augmentation,
    CurriculumSampler,
    list_available_datasets,
    validate_before_training,
    DatasetValidator,
    create_downloader,
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

def train_one_epoch(model, loader, criterion, optimizer, scaler, config, device, logger, epoch):
    model.train()
    epoch_loss, epoch_seq, epoch_ptr, epoch_cov = 0, 0, 0, 0
    
    for batch_idx, batch in enumerate(loader):
        if batch is None: continue
        
        images = batch['image'].to(device, non_blocking=True)
        ids = batch['ids'].to(device, non_blocking=True)
        parents = batch['parents'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            logits, pointers, coverage = model(images, ids, parents)
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
        epoch_seq += seq_loss.item()
        epoch_ptr += ptr_loss.item()
        epoch_cov += cov_loss.item()
        
        if batch_idx % 50 == 0:
            logger.info(f"Epoch [{epoch}] Batch [{batch_idx}/{len(loader)}] - "
                        f"Loss: {loss.item():.4f} (Seq: {seq_loss.item():.4f}, Ptr: {ptr_loss.item():.4f})")
                        
    return epoch_loss / len(loader)

@torch.no_grad()
def validate(model, loader, tokenizer, grammar, config, device, logger):
    logger.info("Starting Validation...")
    model.eval()
    metrics_sum = {'exact': 0, 'leq1': 0, 'bracket': 0, 'total': 0}
    
    for batch in loader:
        if batch is None: continue
        images = batch['image'].to(device)
        gt_latex = batch['latex']
        
        for i in range(images.size(0)):
            pred_tokens = constrained_beam_search(model, images[i:i+1], tokenizer, grammar, config)
            pred_latex = tokenizer.decode(pred_tokens)
            
            res = calculate_metrics(pred_latex, gt_latex[i])
            for k in ['exact', 'leq1', 'bracket']:
                metrics_sum[k] += res[k]
            metrics_sum['total'] += 1
            
        if metrics_sum['total'] > 100:
            break
            
    total = max(1, metrics_sum['total'])
    final_metrics = {k: (v / total) * 100 for k, v in metrics_sum.items() if k != 'total'}
    
    logger.info(f"Validation ExpRate: {final_metrics['exact']:.2f}% | Bracket Acc: {final_metrics['bracket']:.2f}%")
    return final_metrics

def load_datasets(config, logger):
    from data.validator import DatasetValidator
    from data.datasets_registry import get_registry
    
    all_samples = []
    dataset_info = {}
    validator = DatasetValidator(config)
    
    datasets = config.datasets if config.datasets else ['custom']
    logger.info(f"Configured datasets: {datasets}")
    
    for dataset_name in datasets:
        dataset_config = get_registry().get_config(dataset_name)
        dataset_dir = validator.data_dir / dataset_name if dataset_name != 'custom' else validator.data_dir
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_config:
            annot_file = dataset_dir / dataset_config.annotations_file
        else:
            annot_file = dataset_dir / "annotations.json"
        
        if not annot_file or not annot_file.exists():
            logger.warning(f"No annotation file found for dataset '{dataset_name}'. Skipping.")
            continue
            
        import json
        try:
            with open(annot_file, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load annotations for '{dataset_name}': {e}")
            continue
        
        samples = validator._extract_samples(annotations, dataset_dir, dataset_config)
        logger.info(f"  Loaded {len(samples)} samples from '{dataset_name}'")
        
        valid_samples = []
        from pathlib import Path
        from PIL import Image
        for sample in samples:
            img_path = Path(sample['image_path'])
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    valid_samples.append(sample)
                except Exception:
                    pass
        
        dataset_info[dataset_name] = {
            'total': len(samples),
            'valid': len(valid_samples),
            'dir': str(dataset_dir),
        }
        
        all_samples.extend(valid_samples)
        logger.info(f"  {len(valid_samples)}/{len(samples)} samples have valid images")
    
    return all_samples, dataset_info


def parse_args():
    parser = argparse.ArgumentParser(description='TAMER OCR Training')
    parser.add_argument('--datasets', nargs='+', default=None, help='List of datasets to use')
    parser.add_argument('--download', action='store_true', help='Auto-download missing datasets before training')
    parser.add_argument('--data-dir', type=str, default=None, help='Override data directory path')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--skip-validation', action='store_true', help='Skip dataset validation')
    parser.add_argument('--force', action='store_true', help='Force training even if validation has warnings')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list_datasets:
        from data.datasets_registry import get_registry, list_available_datasets
        available = list_available_datasets()
        print("\nAvailable Datasets:")
        for name in available:
            ds_config = get_registry().get_config(name)
            desc = ds_config.description if ds_config else "Custom dataset"
            print(f"  {name}: {desc}")
        return
    
    config = Config()
    
    if args.datasets: config.datasets = args.datasets
    if args.download: config.auto_download = True
    if args.data_dir: config.data_dir = args.data_dir
    if args.skip_validation: config.skip_validation = True
    if args.batch_size: config.batch_size = args.batch_size
    if args.num_epochs: config.num_epochs = args.num_epochs
    if args.lr: config.lr = args.lr
    
    logger = setup_logger("TAMER", config.log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Initialized TAMER Pipeline on device: {device}")
    
    if not config.skip_validation:
        logger.info("=" * 60)
        logger.info("PRE-TRAINING DATASET VALIDATION")
        logger.info("=" * 60)
        
        if config.auto_download:
            try:
                from data.validator import DatasetValidator
                validator = DatasetValidator(config)
                datasets = config.datasets if config.datasets else ['custom']
                
                for dataset_name in datasets:
                    if not validator.registry.validate_dataset_name(dataset_name):
                        continue
                    logger.info(f"Checking dataset: {dataset_name}")
                    validator.try_download_and_validate(dataset_name)
            except Exception as e:
                logger.error(f"Download error: {e}")
        
        try:
            result = validate_before_training(config)
            logger.info(f"Validation PASSED: {result.total_samples} samples ready")
        except RuntimeError as e:
            logger.error(f"Validation FAILED: {e}")
            sys.exit(1)
    
    all_samples, dataset_info = load_datasets(config, logger)
    if not all_samples:
        logger.error("No valid samples found! Cannot start training.")
        sys.exit(1)
    
    tokenizer = LaTeXTokenizer()
    tokenizer.build_from_corpus([s['latex'] for s in all_samples])
    
    train_ds = TreeMathDataset(all_samples, config, tokenizer, transform=get_train_augmentation(config.img_height, config.img_width))
    train_sampler = CurriculumSampler(train_ds, config.curriculum_warmup_epochs)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, collate_fn=get_collate_fn(tokenizer.pad_id), num_workers=config.num_workers, pin_memory=True)
    
    model = TAMERCore(len(tokenizer), config).to(device)
    grammar = LaTeXGrammarConstraints(tokenizer)
    criterion = TreeGuidedLoss(tokenizer.pad_id, config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=config.min_lr)
    scaler = GradScaler('cuda')
    
    start_epoch, best_metrics = load_checkpoint(os.path.join(config.checkpoint_dir, 'latest.pt'), model, optimizer, scheduler, device)
    best_exprate = best_metrics.get('exact', 0.0)
    
    for epoch in range(start_epoch, config.num_epochs):
        train_sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, device, logger, epoch)
        scheduler.step()
        
        logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % config.eval_every == 0:
            metrics = validate(model, train_loader, tokenizer, grammar, config, device, logger)
            is_best = metrics['exact'] > best_exprate
            if is_best:
                best_exprate = metrics['exact']
                best_path = os.path.join(config.checkpoint_dir, 'best.pt')
                save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, best_path)
                push_checkpoint_to_hf(best_path, config, epoch + 1, is_best=True)

        if (epoch + 1) % config.save_every == 0:
            latest_path = os.path.join(config.checkpoint_dir, 'latest.pt')
            metrics_dict = metrics if 'metrics' in locals() else {}
            save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics_dict, latest_path)
            push_checkpoint_to_hf(latest_path, config, epoch + 1, is_best=False)

if __name__ == "__main__":
    main()