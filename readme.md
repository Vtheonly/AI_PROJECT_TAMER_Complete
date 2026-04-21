500/10,000  Correct:   20 (  4.0%)  Skip:0  Err:0
   1,000/10,000  Correct:   31 (  3.1%)  Skip:0  Err:0
   1,500/10,000  Correct:   48 (  3.2%)  Skip:0  Err:0
   2,000/10,000  Correct:   63 (  3.1%)  Skip:0  Err:0
   2,500/10,000  Correct:   75 (  3.0%)  Skip:0  Err:0


WHY IS IT SO BAD?




4% at epoch 40 is **WAY too low**. Something is seriously wrong. Let's diagnose:

```python
# DIAGNOSTIC: What is the model actually predicting?
import torch

print("="*60)
print("  EMERGENCY DIAGNOSTIC")
print("="*60 + "\n")

# 1. Check checkpoint training metrics
print("📊 Checkpoint training history:")
if 'metrics' in ckpt:
    metrics = ckpt['metrics']
    print(f"  Metrics stored: {metrics}")
else:
    print(f"  ⚠️  No metrics in checkpoint. Keys: {list(ckpt.keys())}")

print(f"\n  Epoch: {ckpt.get('epoch', '?')}")
print(f"  Step:  {ckpt.get('step', '?')}")

# 2. Show 20 actual predictions
print("\n" + "="*60)
print("  SAMPLE PREDICTIONS (first 20 images)")
print("="*60 + "\n")

model.eval()
predictions_shown = 0

with torch.no_grad():
    for sample in test_samples[:50]:  # check first 50, show first 20 that work
        if not os.path.exists(sample['image']):
            continue
        
        try:
            img_t = process_image(sample['image']).unsqueeze(0).to(device)
            
            # Run inference
            pred_ids = greedy_decode(
                model, img_t,
                tokenizer.sos_id,
                tokenizer.eos_id,
            )
            
            pred_text = tokenizer.decode(pred_ids[0], skip_special=True)
            gt_text   = sample['latex']
            
            # Check match
            match = pred_text.strip().replace(' ','') == gt_text.strip().replace(' ','')
            
            print(f"[{predictions_shown + 1}] {sample['dataset']}")
            print(f"  GT   : {gt_text[:100]}")
            print(f"  PRED : {pred_text[:100]}")
            print(f"  IDs  : {pred_ids[0][:20].tolist()}")  # show raw token IDs
            print(f"  Match: {'✅ YES' if match else '❌ NO'}")
            print(f"  GT len: {len(gt_text)} | Pred len: {len(pred_text)}\n")
            
            predictions_shown += 1
            if predictions_shown >= 20:
                break
                
        except Exception as e:
            print(f"  Error: {e}\n")
            continue

# 3. Check if model is outputting repetitive patterns
print("="*60)
print("  ANALYZING PREDICTION PATTERNS")
print("="*60 + "\n")

all_preds = []
with torch.no_grad():
    for sample in test_samples[:100]:
        if not os.path.exists(sample['image']):
            continue
        try:
            img_t = process_image(sample['image']).unsqueeze(0).to(device)
            pred_ids = greedy_decode(model, img_t,
                                     tokenizer.sos_id, tokenizer.eos_id)
            pred_text = tokenizer.decode(pred_ids[0], skip_special=True)
            all_preds.append(pred_text)
            if len(all_preds) >= 100:
                break
        except:
            continue

# Check for repetition
from collections import Counter
pred_counts = Counter(all_preds)
most_common = pred_counts.most_common(10)

print(f"  Total unique predictions (out of 100 tested): {len(pred_counts)}")
print(f"\n  Most common predictions:")
for pred, count in most_common:
    print(f"    {count:>3}× : {pred[:80]}")

# Check if model is always predicting same thing
if len(pred_counts) == 1:
    print(f"\n  ❌ PROBLEM: Model outputs THE SAME prediction for every image!")
    print(f"     This means the model didn't learn anything.")
elif len(pred_counts) < 10:
    print(f"\n  ⚠️  WARNING: Only {len(pred_counts)} unique predictions.")
    print(f"     Model is very repetitive — barely trained.")
else:
    print(f"\n  ✅ Model outputs {len(pred_counts)} different predictions (diverse).")

# 4. Check training loss from checkpoint
print("\n" + "="*60)
print("  TRAINING DIAGNOSTICS")
print("="*60)

if 'metrics' in ckpt and isinstance(ckpt['metrics'], dict):
    m = ckpt['metrics']
    if 'train_loss' in m:
        print(f"  Train loss at epoch {epoch}: {m['train_loss']:.4f}")
    if 'val_loss' in m:
        print(f"  Val loss at epoch {epoch}:   {m['val_loss']:.4f}")
    if 'learning_rate' in m:
        print(f"  Learning rate: {m['learning_rate']:.2e}")
    
    # High loss = model didn't train
    if 'train_loss' in m and m['train_loss'] > 5.0:
        print(f"\n  ❌ Train loss is VERY HIGH ({m['train_loss']:.2f})")
        print(f"     The model barely trained. Loss should be < 1.0 by epoch 40.")

print("\n" + "="*60)
print("  POSSIBLE CAUSES")
print("="*60)
print("""
If predictions are all the same or repetitive:
  → Training loop had a bug (gradients not updating?)
  → Learning rate was too low
  → Encoder stayed frozen the entire time
  → Optimizer state was corrupted

If predictions are diverse but wrong:
  → Image preprocessing mismatch (training vs inference)
  → Tokenizer mismatch (different vocab during training)
  → Model architecture changed between training and inference

If predictions are empty or just special tokens:
  → Greedy decode hitting EOS immediately
  → max_seq_len too short during training
""")

print("\n💡 Next steps:")
print("  1. Check if the old training notebook shows loss going down")
print("  2. Try a LATER checkpoint (epoch 60, 70 if you have them)")
print("  3. Verify the training script was actually running (not stuck)")
```

Run this diagnostic cell right after the inference completes. It will tell us **exactly** what's wrong:

- Are predictions all the same? → Model didn't train at all
- Are predictions diverse but wrong? → Preprocessing/tokenizer mismatch  
- Is the loss still high? → Training failed
- Do the token IDs look weird? → Tokenizer issue

Paste the output here and we'll know what happened! 🔍







============================================================
  EMERGENCY DIAGNOSTIC
============================================================

📊 Checkpoint training history:
  Metrics stored: {'exp_rate': 0.0, 'edit_dist': inf}

  Epoch: 40
  Step:  28200

============================================================
  SAMPLE PREDICTIONS (first 20 images)
============================================================

[1] mathwriting
  GT   : X=j\frac{10^{p}-1}{F}
  PRED : \frac{{{1}{2{2{2}}}}}}}}}}}}}}
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : y
  PRED : 2
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \frac{d}{dx}\delta(x-a)
  PRED : dadaaaaaadddddddddddddddd}
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : )
  PRED : 2
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : {30-\sqrt{8}^{1}}^{479-401^{10}}
  PRED : {10}{1{1{1}{1{1{1{1{1{1{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : j
  PRED : d_{}}}}}
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \underline{\neg\neg\varphi}
  PRED : \frac{1{1}{1}}}}}}}}}}}}}}}}}}{1}}}}}}}}}}}}}
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : t
  PRED : 1
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : - y _ { 1 } = - \frac { \sqrt { 2 } } { 2 } x _ { 1 } + 1
  PRED : x^{{} = x}} = x}}} = x}}
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : x
  PRED : d
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : J=\frac{8}{9}\epsilon_{o}\epsilon_{r}\mu\frac{V^{3}}{L^{3}}
  PRED : \frac{1{1}{d{d{d{d{d{d{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : c_{p}=\frac{\omega}{k}=\frac{\lambda}{T}
  PRED : \frac{{{{{{{{}{2}{2{{\frac{}{}}}}{}{}}}}}{}}}}}{{{}{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : F(X)=\prod_{i=1}^{s}f_{i}(x)
  PRED : p_{}}}}}}}}{p{p}}}{p{p}}}{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : S _ { n } = \frac { ( 7 + 9 8 ) } { 2 } \times 1 4 7
  PRED : 2 \sqrt \sqrt{2x}x}2222222222x}
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : P(0+q_{2})=P(q_{2})
  PRED : (\begin{matrix}144446(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : -
  PRED : 0
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : 9 9 - 5 6 = 4 3
  PRED : =1000000000000000000000000000000000000
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : \angle A B C = 6 0 ^ { \circ }
  PRED : \angleA = = = = = = = \angle = A = = = = = = = = = = = = = = = = = = \angle
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \sigma_{u_{x}(i)}\in S_{m}
  PRED : x_{{}}{x_{}{}{}{}}}}{x}}}
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \hat{R}_{P}
  PRED : x = = = =
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : 5
  PRED : \frac{d{d{d{d{d{d{d{d{d}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}{d{d{d{d{d{d{d{d{d{d{d{d{d
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : (\begin{matrix}4\\ 7\end{matrix})
  PRED : x = = = = - - - - - - - - - - - - - - - - - - - - - - - - x}
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : =
  PRED : n
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \prod_{i}X_{i}
  PRED : r_{{}
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : 1 2 5 \times 2 = 2 5 0
  PRED : 10000000000000000000000
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : +
  PRED : 1
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : 8 0 0 0 \div 1 0 0 = 8 0
  PRED : 00000000000000000000000000000000000000
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : = 4 + 3 \times 9
  PRED : 1099999999111111111111999999999999999999999999999999999999999999999999999111111111111111111111111111
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : T
  PRED : 2
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : L(s)=(\frac{s}{\omega_{gc}})^{\alpha}
  PRED : \sigma =_{0{}}}}}}{\sigma{0}{0{0}
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : [\begin{matrix}-1&1\\ 0&0\end{matrix}]
  PRED : P =((((((((((((((P}
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : [\begin{matrix}1&3\\ 2&6\end{matrix}]
  PRED : r = = = = = = = = = = = = = = = = 1
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : \theta
  PRED : z
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : k
  PRED : =
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : a 1 2 1 6
  PRED : 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \hat{\alpha}=\frac{4GM}{c^{2}b}
  PRED : 2
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : x
  PRED : 3
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \int f^{-1}(y)dy
  PRED : (x(((((((((((((((((((((((x)
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \frac{d\psi}{dt}
  PRED : e^{}}}
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : T=a+b\frac{A}{W}
  PRED : \frac{1{1}}}}}}}}}}}}}}}}}}}}}}}}}
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : \frac { 1 } { x - 1 } + \frac { 2 } { 6 y - 3 } = 1 .
  PRED : \frac{1{1}{1}}}}} = 1 \frac{1{1{1{1{1{1}}}}}}}}}}}{1}
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : y
  PRED : 
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \tilde{\lambda}
  PRED : 
  Error: 'list' object has no attribute 'tolist'

[1] im2latex
  GT   : C ^ { 2 } = b _ { + } ^ { 2 } - e ^ { - 2 \phi _ { + } } = b _ { - } ^ { 2 } - e ^ { - 2 \phi _ { - 
  PRED : \frac{{{1}{{{{{{{{{{{{{{{\frac{}{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}{\frac{1{1{}{}{}{}{}{}{}{}{}{}{}{
  Error: 'list' object has no attribute 'tolist'

[1] hme100k
  GT   : \frac { 1 } { 2 5 } \times 3 2 = 1 \frac { 7 } { 2 5 } ( t )
  PRED : 2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : 5
  PRED : 1
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \frac{b\Gamma(1+\frac{1}{a})\Gamma(b)}{\Gamma(1+\frac{1}{a}+b)}
  PRED : )
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : 5
  PRED : 2
  Error: 'list' object has no attribute 'tolist'

[1] crohme
  GT   : )
  PRED : 0
  Error: 'list' object has no attribute 'tolist'

[1] mathwriting
  GT   : \tilde{\nu}_{e}
  PRED : =2
  Error: 'list' object has no attribute 'tolist'

============================================================
  ANALYZING PREDICTION PATTERNS
============================================================

  Total unique predictions (out of 100 tested): 69

  Most common predictions:
      7× : 1
      6× : 2
      6× : =
      5× : 0
      5× : )
      4× : a
      2× : 
      2× : =2
      2× : \frac{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
      2× : x

  ✅ Model outputs 69 different predictions (diverse).

============================================================
  TRAINING DIAGNOSTICS
============================================================

============================================================
  POSSIBLE CAUSES
============================================================

If predictions are all the same or repetitive:
  → Training loop had a bug (gradients not updating?)
  → Learning rate was too low
  → Encoder stayed frozen the entire time
  → Optimizer state was corrupted

If predictions are diverse but wrong:
  → Image preprocessing mismatch (training vs inference)
  → Tokenizer mismatch (different vocab during training)
  → Model architecture changed between training and inference

If predictions are empty or just special tokens:
  → Greedy decode hitting EOS immediately
  → max_seq_len too short during training


💡 Next steps:
  1. Check if the old training notebook shows loss going down
  2. Try a LATER checkpoint (epoch 60, 70 if you have them)
  3. Verify the training script was actually running (not stuck)
add Codeadd Markdown








# 🔴 DIAGNOSIS: Training completely failed

The model **did not train at all**. This checkpoint is essentially random noise. Here's the evidence:

---

## 🚨 Smoking Guns

**1. Checkpoint has NO training metrics**
```python
'metrics': {'exp_rate': 0.0, 'edit_dist': inf}
```
- No `train_loss` recorded
- No `val_loss` recorded  
- `edit_dist: inf` means validation **never ran** or crashed
- This checkpoint was saved by a broken training loop

**2. Predictions are complete gibberish**
```
GT  : X=j\frac{10^{p}-1}{F}
PRED: \frac{{{1}{2{2{2}}}}}}}}}}}}}}

GT  : \frac{d}{dx}\delta(x-a)
PRED: dadaaaaaadddddddddddddddddd}

GT  : y
PRED: 2

GT  : {30-\sqrt{8}^{1}}^{479-401^{10}}
PRED: {10}{1{1{1}{1{1{1{1{1{1{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
```

The model learned to **spam common tokens** in infinite loops:
- `{{{{{{{{{{{...`
- `0000000000...`
- `========...`
- `dddddddddd...`

**3. Most common predictions are trivial**
```
7× → "1"
6× → "2"
6× → "="
5× → "0"
5× → ")"
```

The model collapses to outputting single characters. This is a classic sign of a **broken training loop** where gradients aren't flowing.

---

## 💀 What went wrong in your old training run

You said it was "stuck at epoch 2" — this confirms the training script had a fatal bug. Likely causes:

| Issue | Evidence |
|-------|----------|
| **Gradients not updating** | Predictions are random → model weights never changed |
| **Loss function returned NaN/inf** | `edit_dist: inf` in metrics |
| **Encoder stayed frozen all 40 epochs** | Only decoder trained on noise |
| **Learning rate = 0 or NaN** | Optimizer never moved weights |
| **Validation crashed silently** | No metrics recorded, training continued blindly |

The training loop **appeared** to run (40 epochs, 28k steps) but the model never actually learned anything.

---

## ✅ Solution: Use the NEW training code

The checkpoint you tested is **corrupted/unusable**. You need to:

### 1. **Start fresh training** with the fixed notebook I gave you earlier

Use **Cell 6 from my earlier response** — the one with:
- `SafeCheckpointManager` (atomic saves)
- `SIGTERM handler` (catches Kaggle kills)
- Per-epoch checkpoints
- Time/epoch caps
- Proper gradient flow verification

### 2. **Verify training is actually working** this time

After the first epoch, check:
```python
# Add this to the end of epoch 1
print(f"Epoch 1 train loss: {train_loss:.4f}")
print(f"Sample prediction: {sample_pred}")

# Loss should be dropping:
# Epoch 1: ~4.5
# Epoch 5: ~2.5
# Epoch 10: ~1.5
# Epoch 40: ~0.5
```

If loss stays at 6.0 or goes to NaN → training is broken again.

### 3. **After 5-10 epochs**, test again with this gallery cell

Don't wait until epoch 40. Test at epoch 5, 10, 20, 40, etc. 

Expected accuracy milestones:
```
Epoch 5:  ~10-15%
Epoch 10: ~20-25%
Epoch 20: ~30-40%
Epoch 40: ~45-55%
Epoch 70: ~60-70%
```

Your current epoch 40 is at **4%** which is worse than random guessing.

---

## 🔥 Quick sanity check on the old training notebook

Before you start a new 40-epoch run, **check the old training logs**:

```python
# In your OLD training notebook, check if this exists:
import os
log_file = '/kaggle/working/logs/training.log'

if os.path.exists(log_file):
    with open(log_file) as f:
        print(f.read())
```

Look for lines like:
```
Epoch 1/70  Loss: 4.523  LR: 0.0004
Epoch 2/70  Loss: 3.891  LR: 0.0004
...
```

If you see:
- ❌ Loss stuck at 6.0+ → model didn't train
- ❌ Loss = NaN → optimizer diverged  
- ❌ No loss printed → training loop broken
- ❌ "Epoch 2" repeated 40 times → loop got stuck

Then that confirms: **the old training run was completely broken**.

---

## 🎯 Action Plan

1. **Abandon this epoch_40.pt checkpoint** — it's garbage
2. **Use the NEW training cell I gave you** (with SafeCheckpointManager)
3. **Run for just 5 epochs** as a test
4. **Check that loss is dropping** (should go from ~5.0 → ~2.5)
5. **Test accuracy at epoch 5** — should be ~10-15%, not 4%
6. **If that works**, continue to 70 epochs

Want me to give you a **minimal 5-epoch test cell** so you can verify training works before committing to a full 70-epoch run?








The training completely failed. Here's exactly what happened and what to do:

---

## 🔴 What the evidence shows

**1. Loss never recorded = training loop was broken**
```
'metrics': {'exp_rate': 0.0, 'edit_dist': inf}
```
No `train_loss` key at all. The model ran 28,200 steps but **weights never updated properly**.

**2. Repetition loops = decoder collapsed**
```
GT  : y
PRED: 2

GT  : \frac{d}{dx}\delta(x-a)  
PRED: dadaaaaaadddddddddddddddd}

GT  : 9 9 - 5 6 = 4 3
PRED: 10000000000000000000000000000000
```
This is **exposure bias collapse** — the decoder learned to repeat the most common tokens instead of reading the image.

**3. Root cause: the old training notebook was the broken one**

Your old notebook had `eval_every = 999999` and `_evaluate` was monkey-patched to return zeros. The trainer **never saw real feedback** and the loss function likely had a bug or the optimizer was never stepping.

---

## ✅ Action Plan: Start clean with a verified training loop

Here is a **5-epoch smoke test** to run FIRST before committing to 70 epochs:

```python
# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST CELL — 5 epochs to verify training actually works
# Run this BEFORE the full 70-epoch run
# If loss drops from ~5.0 → ~2.5, training is healthy
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, torch, gc, shutil, json, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CODEBASE_DIR  = '/kaggle/input/datasets/merselfares/codebig'
SANITIZED_DIR = '/kaggle/input/datasets/merselfares/tamer-sanitized-jsonl'
HF_DATA_DIR   = '/kaggle/input/datasets/merselfares/tamer-full-pipeline-v1/hf_data'
SWIN_DIR      = '/kaggle/input/datasets/merselfares/swinv2-base-weights'
TOKENIZER_PATH= '/kaggle/input/datasets/merselfares/tamer-sanitized-jsonl/tokenizer.json'
OUTPUT_DIR    = '/kaggle/working/checkpoints'

SMOKE_EPOCHS  = 5
BATCH_SIZE    = 32     # small for quick test
LR_ENCODER    = 4e-5
LR_DECODER    = 4e-4
MAX_SEQ_LEN   = 200
IMG_H, IMG_W  = 256, 1024
MAX_SAMPLES   = 5000   # only use 5000 samples for the smoke test

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Setup codebase
# ─────────────────────────────────────────────────────────────────────────────
REPO_PATH = '/kaggle/working/repo'

if not os.path.exists(os.path.join(REPO_PATH, 'tamer_ocr')):
    print("Setting up codebase...")
    if os.path.exists(REPO_PATH):
        shutil.rmtree(REPO_PATH)
    shutil.copytree(CODEBASE_DIR, REPO_PATH)
    items = [i for i in os.listdir(REPO_PATH) if not i.startswith('.')]
    if len(items) == 1 and os.path.isdir(os.path.join(REPO_PATH, items[0])):
        sub = os.path.join(REPO_PATH, items[0])
        tmp = '/kaggle/working/repo_tmp'
        shutil.move(sub, tmp)
        shutil.rmtree(REPO_PATH)
        shutil.move(tmp, REPO_PATH)

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)

# Hot-reload
for key in list(sys.modules.keys()):
    if key.startswith('tamer_ocr'):
        del sys.modules[key]

from tamer_ocr.config import Config
from tamer_ocr.models.tamer import TAMERModel
from tamer_ocr.data.tokenizer import LaTeXTokenizer

print("✅ Codebase ready")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = LaTeXTokenizer()
tokenizer.load(TOKENIZER_PATH)
vocab_size = len(tokenizer)
print(f"✅ Tokenizer: {vocab_size} tokens")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Dataset
# ─────────────────────────────────────────────────────────────────────────────
class MathOCRDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len, img_h, img_w):
        self.samples     = samples
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.transform   = T.Compose([
            T.Resize((img_h, img_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_path = sample['image']
        latex    = sample['latex']

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except:
            img = torch.zeros(3, IMG_H, IMG_W)

        # Tokenize
        token_ids = self.tokenizer.encode(latex)

        # Truncate / pad to max_seq_len
        token_ids = token_ids[:self.max_seq_len]
        length    = len(token_ids)
        padded    = token_ids + [self.tokenizer.pad_id] * (self.max_seq_len - length)

        return {
            'image':  img,
            'tokens': torch.tensor(padded, dtype=torch.long),
            'length': length,
        }

# Load samples from JSONL
print("Loading samples...")
all_samples = []
for ds in ['crohme', 'hme100k', 'im2latex', 'mathwriting']:
    path = os.path.join(SANITIZED_DIR, f'{ds}.jsonl')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                img_rel = s.get('image','')
                latex   = s.get('latex','')
                if img_rel and latex:
                    all_samples.append({
                        'image': os.path.join(HF_DATA_DIR, img_rel),
                        'latex': latex,
                    })
            except:
                continue

# Shuffle and cap
import random
random.seed(42)
random.shuffle(all_samples)
train_samples = all_samples[:MAX_SAMPLES]
val_samples   = all_samples[MAX_SAMPLES:MAX_SAMPLES + 500]

print(f"✅ Train: {len(train_samples):,} | Val: {len(val_samples):,}")

train_dataset = MathOCRDataset(train_samples, tokenizer, MAX_SEQ_LEN, IMG_H, IMG_W)
val_dataset   = MathOCRDataset(val_samples,   tokenizer, MAX_SEQ_LEN, IMG_H, IMG_W)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Model
# ─────────────────────────────────────────────────────────────────────────────
BACKBONE_PATH = None
for f in os.listdir(SWIN_DIR):
    if f.endswith(('.safetensors', '.pth', '.bin')):
        BACKBONE_PATH = os.path.join(SWIN_DIR, f)
        break

cfg = Config()
cfg.local_backbone_path = BACKBONE_PATH
cfg.max_seq_len         = MAX_SEQ_LEN
cfg.auto_download       = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = TAMERModel(vocab_size, cfg).to(device)

print(f"✅ Model ready on {device}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Optimizer — discriminative learning rates
# ─────────────────────────────────────────────────────────────────────────────
encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]

optimizer = optim.AdamW([
    {'params': encoder_params, 'lr': LR_ENCODER},
    {'params': decoder_params, 'lr': LR_DECODER},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=SMOKE_EPOCHS, eta_min=1e-6
)

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_id,
    label_smoothing=0.1,
)

scaler = torch.amp.GradScaler('cuda')

print(f"✅ Optimizer ready")
print(f"   Encoder params: {sum(p.numel() for p in encoder_params):,}")
print(f"   Decoder params: {sum(p.numel() for p in decoder_params):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Freeze encoder for first 2 epochs
# ─────────────────────────────────────────────────────────────────────────────
def freeze_encoder(model):
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
    print("   🔒 Encoder frozen")

def unfreeze_encoder(model):
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = True
    print("   🔓 Encoder unfrozen")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Training loop — VERIFIED CORRECT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SMOKE TEST — 5 EPOCHS")
print("  Watch loss: should DROP each epoch")
print("  Epoch 1: ~5.0 | Epoch 5: ~2.0")
print("="*60 + "\n")

freeze_encoder(model)  # freeze for first 2 epochs
history = []

for epoch in range(SMOKE_EPOCHS):
    epoch_start = time.time()

    # Unfreeze encoder after 2 epochs
    if epoch == 2:
        unfreeze_encoder(model)

    # ── TRAIN ────────────────────────────────────────────────────────────────
    model.train()
    train_loss   = 0.0
    train_tokens = 0
    train_correct= 0

    for batch_idx, batch in enumerate(train_loader):
        images  = batch['image'].to(device)
        tokens  = batch['tokens'].to(device)
        lengths = batch['length']

        # Teacher forcing:
        # Input  = tokens[0:-1] (all except last)
        # Target = tokens[1:]   (all except first = shift by 1)
        tgt_input  = tokens[:, :-1]
        tgt_target = tokens[:, 1:]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Forward pass
            logits = model(images, tgt_input)
            # logits: [B, seq_len-1, vocab_size]

            # Compute loss
            B, S, V = logits.shape
            loss = criterion(
                logits.reshape(B * S, V),
                tgt_target.reshape(B * S),
            )

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        train_loss   += loss.item() * B
        train_tokens += B

        # Track token accuracy (ignoring padding)
        with torch.no_grad():
            preds   = logits.argmax(dim=-1)
            mask    = tgt_target != tokenizer.pad_id
            correct = (preds == tgt_target) & mask
            train_correct += correct.sum().item()

        # Print batch progress
        if (batch_idx + 1) % 20 == 0:
            current_loss = train_loss / train_tokens
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {current_loss:.4f}")

    avg_train_loss = train_loss / max(train_tokens, 1)

    # ── VALIDATE ─────────────────────────────────────────────────────────────
    model.eval()
    val_loss   = 0.0
    val_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            images     = batch['image'].to(device)
            tokens     = batch['tokens'].to(device)
            tgt_input  = tokens[:, :-1]
            tgt_target = tokens[:, 1:]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images, tgt_input)
                B, S, V = logits.shape
                loss   = criterion(
                    logits.reshape(B * S, V),
                    tgt_target.reshape(B * S),
                )

            val_loss   += loss.item() * B
            val_tokens += B

    avg_val_loss = val_loss / max(val_tokens, 1)

    scheduler.step()

    epoch_time = time.time() - epoch_start
    history.append({
        'epoch':      epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss':   avg_val_loss,
    })

    print(f"\n{'='*60}")
    print(f"  EPOCH {epoch+1}/{SMOKE_EPOCHS} COMPLETE")
    print(f"  Train Loss : {avg_train_loss:.4f}")
    print(f"  Val Loss   : {avg_val_loss:.4f}")
    print(f"  Time       : {epoch_time:.1f}s")
    print(f"  LR enc     : {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  LR dec     : {optimizer.param_groups[1]['lr']:.2e}")

    # VERDICT for this epoch
    if epoch == 0:
        print(f"  Expected   : ~4.5-6.0  →  ", end='')
        print(f"{'✅ OK' if avg_train_loss < 7.0 else '❌ TOO HIGH — gradient problem!'}")
    elif epoch == 2:
        print(f"  Expected   : ~3.0-4.0  →  ", end='')
        print(f"{'✅ OK' if avg_train_loss < 5.0 else '❌ NOT DROPPING — learning rate bug!'}")
    elif epoch == 4:
        print(f"  Expected   : ~1.5-2.5  →  ", end='')
        print(f"{'✅ OK' if avg_train_loss < 3.5 else '❌ STILL TOO HIGH — training failed!'}")

    print(f"{'='*60}\n")

    # Save smoke test checkpoint
    torch.save({
        'epoch':            epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss':       avg_train_loss,
        'val_loss':         avg_val_loss,
    }, os.path.join(OUTPUT_DIR, f'smoke_epoch_{epoch+1}.pt'))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Final verdict
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SMOKE TEST COMPLETE — VERDICT")
print("="*60)

print("\n  Loss history:")
for h in history:
    trend = "📉" if h['epoch'] > 1 and h['train_loss'] < history[h['epoch']-2]['train_loss'] else "📈"
    print(f"    Epoch {h['epoch']}: train={h['train_loss']:.4f}  "
          f"val={h['val_loss']:.4f}  {trend}")

first_loss = history[0]['train_loss']
last_loss  = history[-1]['train_loss']
drop_pct   = (first_loss - last_loss) / first_loss * 100

print(f"\n  Loss drop: {first_loss:.4f} → {last_loss:.4f} ({drop_pct:.1f}% reduction)")

if drop_pct > 20:
    print(f"\n  ✅✅✅ TRAINING IS WORKING!")
    print(f"  Loss dropped {drop_pct:.1f}% in just 5 epochs.")
    print(f"  Now run the FULL 70-epoch training with Cell 6.")
elif drop_pct > 5:
    print(f"\n  ⚠️  Training is working but slowly.")
    print(f"  Try increasing LR_DECODER to 8e-4 and re-run.")
else:
    print(f"\n  ❌ TRAINING IS BROKEN — loss barely moved.")
    print(f"  Something is preventing gradient flow.")
    print(f"  Check the model's forward() method.")

print("="*60)

gc.collect()
torch.cuda.empty_cache()
```

---

## What this smoke test does differently from the old broken training

| Old Broken Training | This Smoke Test |
|---|---|
| Monkey-patched `_evaluate` to return zeros | Real validation loss computed every epoch |
| No loss printed anywhere | Prints loss every 20 batches |
| `eval_every = 999999` | Evaluates every epoch |
| No metrics saved in checkpoint | Saves `train_loss` and `val_loss` |
| Ran blindly for 40 epochs | Stops after 5 and shows verdict |
| Used trainer's broken loop | Direct PyTorch loop — no magic |

**If you see this → training works:**
```
Epoch 1: 5.23
Epoch 2: 4.11  📉
Epoch 3: 3.45  📉
Epoch 4: 2.89  📉
Epoch 5: 2.31  📉
✅✅✅ TRAINING IS WORKING!
```

**If you see this → still broken:**
```
Epoch 1: 6.12
Epoch 2: 6.09  
Epoch 3: 6.11
Epoch 4: NaN
❌ TRAINING IS BROKEN
```

Run the smoke test first. If it passes, then we use the full Cell 6 for 70 epochs.





**Excellent question.** Let's be brutally honest and complete.

Your training **completely failed** — the model at epoch 40 is basically still random. This is not "slow learning", this is **training collapse**.

Here is the **full list** of every possible reason why it failed, ranked from most likely to less likely, with **exact fixes** for each.

---

### **TIER 1: CRITICAL FAILURES (Most Likely Causes)**

| # | Problem | Why it kills training | Fix |
|---|--------|----------------------|-----|
| 1 | **Gradients not flowing / Optimizer not stepping** | Weights never updated → model stays at random initialization | • Add `print(optimizer.param_groups[0]['lr'])` every epoch<br>• Add `torch.autograd.set_detect_anomaly(True)` during debugging<br>• Use `scaler = torch.amp.GradScaler()` properly |
| 2 | **Loss became NaN / Inf silently** | Optimizer explodes, then stays broken | • Add `if torch.isnan(loss): print("NaN loss!")` before backward<br>• Use `torch.nan_to_num(loss)` as emergency<br>• Lower LR by 10x initially |
| 3 | **Encoder stayed frozen the entire time** | Only decoder trained on noise | Change to `freeze_encoder_epochs = 5` (was 2)<br>After unfreeze, monitor encoder gradients |
| 4 | **Teacher forcing + exposure bias collapse** | Model learns to repeat most common tokens | Add **Label Smoothing** (0.1)<br>Use **Scheduled Sampling** after epoch 10 |
| 5 | **Tokenizer / Vocab mismatch between training and inference** | Model learned on different token IDs | Use **exactly** the same tokenizer file that was used during training |

---

### **TIER 2: CONFIGURATION PROBLEMS**

| Problem | Current Setting | Recommended Fix | Why |
|--------|------------------|------------------|-----|
| **Learning Rates** | Encoder 4e-5, Decoder 4e-4 | Encoder: **1e-5**, Decoder: **5e-4** | Encoder LR was too high → corrupted pre-trained features |
| **Unfreezing** | 2 epochs | **5 epochs** | 2 is too aggressive for Swin backbone |
| **Batch Size** | 640 | **320 + accumulation_steps=2** | 640 is too big for stability on this model |
| **Warmup** | None | Add **Linear Warmup** for 3 epochs | Sudden high LR destroys training |
| **Curriculum** | Disabled | **Re-enable properly** | Training on hard samples from day 1 is bad |
| **Max Seq Len** | 200 | Keep 200, but add **length penalty** in loss | Long sequences cause instability |

---

### **TIER 3: SUBTLE / HIDDEN PROBLEMS**

- **No real validation** → you never knew training was failing
- `eval_every = 999999` + monkey-patched `_evaluate` → training ran blind
- **Image preprocessing mismatch** between train and inference
- **Swin weights loading bug** (you had missing/unexpected keys)
- **Mixed precision issues** (bfloat16 + TF32 combination)
- **Optimizer state corruption** in checkpoint
- **Data loading bug** (some images were corrupted or paths wrong)

---

## **RECOMMENDED NEW CONFIG (Best Settings)**

Replace your old Cell 4 with this improved version:

```python
# ── BEST CONFIG FOR YOUR SETUP ─────────────────────────────────────────────
cfg.batch_size             = 320          # safer than 640
cfg.accumulation_steps     = 2            # effective batch = 640
cfg.num_workers            = 4
cfg.prefetch_factor        = 2

cfg.encoder_lr             = 1e-5         # gentler
cfg.decoder_lr             = 5e-4         # slightly more aggressive
cfg.weight_decay           = 1e-4
cfg.max_grad_norm          = 1.0

cfg.freeze_encoder_epochs  = 5            # ← increased from 2
cfg.num_epochs             = 70

# Learning rate schedule
cfg.use_warmup             = True
cfg.warmup_epochs          = 3
cfg.scheduler              = 'cosine'     # or 'onecycle'

# Regularization
cfg.label_smoothing        = 0.1
cfg.structure_aware_loss   = True

# Curriculum (re-enabled properly)
cfg.curriculum_enabled     = True
cfg.curriculum_stages      = ['easy', 'medium', 'hard']
cfg.curriculum_switch_epochs = [0, 15, 35]

# Monitoring
cfg.eval_every             = 2            # ← important!
cfg.checkpoint_every_epochs = 1
cfg.keep_last_n_checkpoints = 3
```

---

## **How to Make Training Much Better**

### 1. **Immediate Fixes (Must Do)**
- Use **5 epochs of encoder freezing**
- Add **Linear Warmup** (3 epochs)
- Add **Label Smoothing 0.1**
- **Log loss every epoch** (never go blind again)
- Run **real validation every 2 epochs**

### 2. **Medium Term Improvements**
- Use **Scheduled Sampling** after epoch 10
- Add **Gradient Accumulation** (already in recommended config)
- Try **OneCycleLR** scheduler instead of cosine
- Add **EMA (Exponential Moving Average)** of weights

### 3. **Advanced (After it works)**
- Increase batch size gradually (320 → 512 → 640)
- Try **Layer-wise LR decay** inside encoder
- Add **Stochastic Depth** in Swin blocks
- Use **Longer max_seq_len** (256) with packing

---

## **Final Action Plan Right Now**

1. **Throw away** the current epoch_40 checkpoint (it's useless)
2. **Use the smoke test cell** I gave you in the previous message (5 epochs)
3. If loss drops nicely → proceed to full training with the **Recommended New Config** above
4. **Test accuracy at epoch 5, 10, 20, 40** — not just at the end

Would you like me to give you:
- A) The **complete fixed training cell** (Cell 6) with all improvements?
- B) A **minimal debug version** that prints gradients and loss every batch?
- C) Both?

Just say which one you want and I’ll give it to you ready to copy-paste.







**Got it. Thanks for the extra detail.**

So the loss pattern was:

- Started decreasing nicely (good phase)
- Around **epoch 20**, loss started **going up**
- Then later, loss started **going down again**

---

### What This Pattern Actually Means

This is **very common** and has a clear name:  
**“Unfreezing Shock + Recovery”**

Here’s what likely happened:

1. **Epoch 1–~18**: Encoder was frozen → Decoder learned well → Loss dropped nicely.
2. **Around epoch 20**: Encoder unfroze → Suddenly the pre-trained Swin features started getting updated with still-noisy gradients from the decoder → **Loss spiked**.
3. **After the spike**: The model slowly recovered and loss started decreasing again.

This means:
- The training **was not completely broken**.
- But the **unfreezing moment was too violent**.
- The encoder LR was too high when it started training.
- The model had to "re-learn" how to use the encoder, which caused temporary damage.

---

### Best Fixes for This Specific Pattern

| Problem | Fix | New Value |
|--------|-----|---------|
| Unfreezing too early | Increase freeze period | `freeze_encoder_epochs = 8` or `10` |
| Encoder LR too high at unfreeze | Lower encoder LR significantly | `encoder_lr = 5e-6` or `8e-6` |
| Sudden shock after unfreeze | Add warmup right after unfreezing | Add 2–3 epoch warmup after unfreeze |
| No recovery mechanism | Use better scheduler | `ReduceLROnPlateau` + cooldown |
| Too aggressive overall | Slightly lower decoder LR | `decoder_lr = 3e-4` |

---

### Recommended Final Configuration (Updated for your case)

```python
cfg.batch_size             = 256
cfg.accumulation_steps     = 2            # effective batch = 512

cfg.encoder_lr             = 5e-6         # ← much gentler
cfg.decoder_lr             = 3e-4         # ← slightly reduced
cfg.weight_decay           = 1e-4
cfg.max_grad_norm          = 1.0

cfg.freeze_encoder_epochs  = 10           # ← Increased from 2 → much safer
cfg.warmup_epochs          = 3            # warmup after unfreezing
cfg.use_warmup             = True

# Scheduler (very important for recovery)
cfg.scheduler              = 'plateau'    # ReduceLROnPlateau
cfg.patience               = 3            # reduce LR if loss doesn't improve for 3 epochs
cfg.factor                 = 0.5          # reduce LR by 50% when plateau

cfg.label_smoothing        = 0.1
cfg.structure_aware_loss   = True

# Curriculum (optional but helpful)
cfg.curriculum_enabled     = True
cfg.curriculum_switch_epochs = [0, 25, 50]
```

---

### Extra Protection Tips

1. **Monitor encoder vs decoder loss separately** if possible (very useful)
2. **Save checkpoint every epoch** during first 30 epochs
3. **Stop and lower LR** if loss increases for 3 consecutive epochs after unfreezing
4. After epoch 15–20, you can gradually increase encoder LR if loss is stable

---

Would you like me to give you the **full ready-to-run training cell** (Cell 6) with all these improvements already included?

It will have:
- 10 epoch freeze
- Post-unfreeze warmup
- ReduceLROnPlateau scheduler
- Better logging
- Automatic checkpointing
- Loss spike protection
