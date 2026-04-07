from logger import setup_logger
from data.tokenizer import LaTeXTokenizer
from data.augmentation import get_train_augmentation, get_val_augmentation
from data.dataset import TreeMathDataset, get_collate_fn
from data.sampler import CurriculumSampler
from models.tamer import TAMERCore
from core.constraints import LaTeXGrammarConstraints
from core.losses import TreeGuidedLoss
from core.inference import constrained_beam_search
from utils.metrics import calculate_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint

def train_one_epoch(model, loader, criterion, optimizer, scaler, config, device, logger, epoch):
    model.train()
    epoch_loss, epoch_seq, epoch_ptr, epoch_cov = 0, 0, 0, 0
    
    for batch_idx, batch in enumerate(loader):
        if batch is None: continue
        
        images = batch['image'].to(device, non_blocking=True)
        ids = batch['ids'].to(device, non_blocking=True)
        parents = batch['parents'].to(device, non_blocking=True)
        
        with autocast():
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
            
        # Break early on validation just for tracing, remove if full val needed
        if metrics_sum['total'] > 100:
            break
            
    total = max(1, metrics_sum['total'])
    final_metrics = {k: (v / total) * 100 for k, v in metrics_sum.items() if k != 'total'}
    
    logger.info(f"Validation ExpRate: {final_metrics['exact']:.2f}% | Bracket Acc: {final_metrics['bracket']:.2f}%")
    return final_metrics

def main():
    config = Config()
    logger = setup_logger("TAMER", config.log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Initialized TAMER Pipeline on device: {device}")
    
    tokenizer = LaTeXTokenizer()
    # In production, load your dataset dicts here and call `tokenizer.build_from_corpus(all_latex)`
    dummy_samples = [{"image_path": "test.png", "latex": "\\frac{a}{b}"}] 
    tokenizer.build_from_corpus([s['latex'] for s in dummy_samples])
    
    train_ds = TreeMathDataset(dummy_samples, config, tokenizer, transform=get_train_augmentation(config.img_height, config.img_width))
    train_sampler = CurriculumSampler(train_ds, config.curriculum_warmup_epochs)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, collate_fn=get_collate_fn(tokenizer.pad_id))
    
    model = TAMERCore(len(tokenizer), config).to(device)
    grammar = LaTeXGrammarConstraints(tokenizer)
    criterion = TreeGuidedLoss(tokenizer.pad_id, config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=config.min_lr)
    scaler = GradScaler()
    
    start_epoch, best_metrics = load_checkpoint(os.path.join(config.checkpoint_dir, 'latest.pt'), model, optimizer, scheduler, device)
    best_exprate = best_metrics.get('exact', 0.0)

    for epoch in range(start_epoch, config.num_epochs):
        train_sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, device, logger, epoch)
        scheduler.step()
        
        logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % config.eval_every == 0:
            metrics = validate(model, train_loader, tokenizer, grammar, config, device, logger) # use val_loader normally
            
            is_best = metrics['exact'] > best_exprate
            if is_best:
                best_exprate = metrics['exact']
                save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, os.path.join(config.checkpoint_dir, 'best.pt'))
                
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics if 'metrics' in locals() else {}, os.path.join(config.checkpoint_dir, 'latest.pt'))

if __name__ == "__main__":
    main()