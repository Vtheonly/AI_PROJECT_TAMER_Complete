import json
import os

def fix_notebook(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code": continue
        
        # Determine source string
        if isinstance(cell["source"], list):
            src = "".join(cell["source"])
        else:
            src = cell["source"]
            
        if "RESUME_PATH =" in src and "trainer.train()" in src and "RESUME_FROM_CHECKPOINT" not in src:
            # Replace the old logic
            new_src = src
            new_src = new_src.replace("# 4. POINT TO YOUR EPOCH 5 CHECKPOINT", "# 4. CHECKPOINT RESUME SETTINGS")
            new_src = new_src.replace(
                'RESUME_PATH = "/kaggle/input/datasets/merselfares/10theepoches/checkpoints/epoch_10.pt"', 
                'RESUME_FROM_CHECKPOINT = True  # Set to False to start from scratch\nRESUME_PATH = "/kaggle/input/datasets/merselfares/10theepoches/checkpoints/epoch_10.pt"'
            )
            
            old_logic = """if os.path.exists(RESUME_PATH):
    print(f"🔄 Resuming from Epoch 5...")
    trainer.resume_from_checkpoint(RESUME_PATH)
    
    # Force the trainer to start exactly at epoch 6
    # (trainer.resume_from_checkpoint sets this based on the file)
    print(f"✅ Ready. Next step will begin Epoch {trainer.current_epoch + 1}")
    
    # 5. GO!
    trainer.train()
else:
    print(f"❌ ERROR: Could not find checkpoint at {RESUME_PATH}")"""

            new_logic = """if RESUME_FROM_CHECKPOINT:
    if os.path.exists(RESUME_PATH):
        print(f"🔄 Resuming from checkpoint...")
        trainer.resume_from_checkpoint(RESUME_PATH)
        print(f"✅ Ready. Next step will begin Epoch {trainer.current_epoch + 1}")
    else:
        print(f"❌ ERROR: Could not find checkpoint at {RESUME_PATH}")
        raise FileNotFoundError(f"Checkpoint not found: {RESUME_PATH}")
else:
    print("🚀 Starting training from scratch!")

# 5. GO!
trainer.train()"""
            if old_logic in new_src:
                new_src = new_src.replace(old_logic, new_logic)
                modified = True
            else:
                print(f"Old logic not matched precisely in {filepath}")
            
            # Put it back in the original format
            if isinstance(cell["source"], list):
                lines = new_src.split("\n")
                # re-attach newlines correctly
                cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            else:
                cell["source"] = new_src

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f)
        print(f"Fixed {filepath}")

fix_notebook("tamer_ocr/kaggle2.txt")
fix_notebook("tamer_ocr/kaggle27.ipynb")
