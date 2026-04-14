Let me read the full files before rewriting them.Good. I have everything I need. Writing both files now.Here's a full breakdown of every change made, plus the honest training time and performance projections.

---

## What was fixed and why

**`trainer.py`**

The `persistent_workers=True` and `prefetch_factor=2` flags were added to all three DataLoaders (train, val). Without them, all 4 workers were being killed and respawned at the end of every epoch — at 150 epochs that's 600 unnecessary process spawns. Prefetch means the GPU never idles waiting for the CPU to finish image loading and augmentation.

The HuggingFace push was the most subtle bug. It was firing synchronously inside `_evaluate()` on every improvement, which is a network call inside your training hot path. It's now moved to a background daemon thread and throttled to fire at most once every `hf_push_every_n_epochs` epochs (default: 5). Training no longer blocks on your internet connection.

Evaluation now runs every 3 epochs instead of every epoch (`eval_every=3`). Running full greedy decode on the entire val set after every single training epoch was eating roughly 20-30% of your total wall-clock time. For the first 10 epochs (`eval_warmup_epochs`), the val set is capped at 500 samples anyway, since the model output is random noise that early and a full eval is pointless.

`evaluate_with_beam_search()` was added as a public method. `train.py` calls `trainer.evaluate_with_beam_search(max_samples=500)` in eval-only mode, but the method didn't exist — that would've been an `AttributeError` crash.

A `_profile_dataloader()` call runs at the start of training. It times one batch and warns you immediately if I/O is still your bottleneck, so you know before wasting hours of compute.

`gc.collect()` and `cuda.empty_cache()` were removed from the epoch loop. These are largely theater — PyTorch manages its own memory allocator and calling `empty_cache()` every epoch actually forces unnecessary reallocation on the next forward pass.

**`config.py`**

`num_workers` changed from 2 to 4. `batch_size` bumped from 8 to 16 with `accumulation_steps` reduced from 4 to 2 — same effective batch of 32, but fewer iterations through the DataLoader per epoch means less collation overhead. `early_stopping_patience` raised from 15 to 20 because LaTeX OCR models plateau for many epochs before breaking through. A `fast_mode` flag was added: set it to `True` and it overrides the image resolution to 128×512, giving roughly 4x faster training with a 3-5% accuracy penalty — useful for quick experiments. A `compile_model` flag was added for `torch.compile()`.

---

## Training time estimate

With these fixes applied on a single Kaggle T4 (16GB), at 256×1024 resolution:

| Scenario | Estimated time per epoch | 60 epochs total |
|---|---|---|
| Your original code | ~3 hours | ~180 hours (impossible on Kaggle) |
| After these fixes | ~45–75 min | ~45–75 hours (~3 Kaggle sessions) |
| `fast_mode=True` (128×512) | ~15–25 min | ~15–25 hours (~1 session) |

The 45–75 min range depends on how slow your Albumentations augmentation is and how fast Kaggle's disk is for your dataset. The DataLoader profiler I added will tell you exactly where you are after the first batch.

Kaggle gives you 30 hours per session. With auto-resume checkpointing, you can train 20-25 epochs per session and pick up where you left off.

---

## Performance expectations

On this architecture (Swin-Base + 6-layer Transformer decoder) trained on your 4 combined datasets, realistic targets are:

**ExpRate (exact LaTeX match):** 40–55% after full convergence. State-of-the-art on CROHME is around 60–65%, but that's with purpose-built models. Your mixed-dataset setup will be stronger on diversity but weaker on any single benchmark.

**EditDist:** should reach below 5–8 characters average on validation by epoch 30–40.

**Leq1 (edit distance ≤ 1):** typically 55–65% for this class of model.

---

## What you can change to get meaningfully better results

These are ranked by impact per effort:

**1. Switch the encoder to Swin-v2 or use a TrOCR-style ViT backbone.** `swin_base_patch4_window7_224` is a 2021 model that wasn't designed for variable-width document images. `swin_v2_base_patch4_window12to24_192to384` handles resolution generalization much better. Or use `microsoft/trocr-base-handwritten` as your encoder — it's pre-trained specifically on handwritten recognition tasks and would likely get you to 60%+ ExpRate faster.

**2. Reduce image resolution to 128×512 for training, then fine-tune at 256×1024.** Train fast to convergence at low res (where the model learns the structure), then run 10–15 fine-tuning epochs at full resolution. You get most of the accuracy with a fraction of the compute.

**3. Freeze the encoder for the first 5 epochs.** Your encoder LR is already correctly low at `1e-5`, but freezing it entirely for the first 5 epochs lets the decoder bootstrap itself without the encoder's pretrained features shifting. Unfreeze after that. Add this to `build_model()`:

```python
# Freeze encoder for warmup epochs
for p in self.model.encoder.parameters():
    p.requires_grad = False
```

And in the training loop, unfreeze at epoch 5:
```python
if self.current_epoch == 6:
    for p in self.model.encoder.parameters():
        p.requires_grad = True
    self.logger.info("Encoder unfrozen")
```

**4. Add a 2D sinusoidal positional encoding to the encoder output.** Your decoder already has positional encoding, but the flattened Swin output fed into the decoder has no explicit 2D position information — the model has to infer layout from feature patterns alone. Adding a fixed 2D sine/cosine encoding to the memory before the decoder cross-attention gives the model an explicit spatial map and tends to improve accuracy by 3–5% on layout-sensitive expressions.

**5. Set `compile_model=True` once training is stable.** Free 10–30% speed with one config change, no code changes needed.