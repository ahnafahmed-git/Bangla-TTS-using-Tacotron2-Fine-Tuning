#  Run AFTER all class definitions and DataLoaders
# ════════════════════════════════════════════════════════
import os
import torch

CHECKPOINT_DIR   = "/kaggle/working/checkpoints"
BEST_CHECKPOINT  = "/kaggle/working/checkpoints/best_model.pt"
TARGET_EPOCHS    = 10   # ← change this to however many total epochs you want
                        # Must be greater than previous epoch unless starting fresh

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Step 1: Copy checkpoint from your Kaggle dataset ────────────────────
# Replace Kaggle username
KAGGLE_USERNAME   = "ahmedahnaf513"   # update this
CHECKPOINT_DATASET = f"{KAGGLE_USERNAME}/bangla-tts-checkpoints"

print("Copying checkpoints from dataset...")
!kaggle datasets download -d {CHECKPOINT_DATASET} --unzip -p {CHECKPOINT_DIR}
print(f"Available checkpoints: {os.listdir(CHECKPOINT_DIR)}")

# ── Step 2: Find the latest checkpoint ──────────────────────────────────
import glob

all_checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/checkpoint_epoch*.pt"))
if all_checkpoints:
    latest = all_checkpoints[-1]
    print(f"\nLatest checkpoint: {latest}")
else:
    latest = BEST_CHECKPOINT
    print(f"\nUsing best model: {latest}")

# ── Step 3: Load checkpoint into model ──────────────────────────────────
ckpt = torch.load(latest, map_location=device)

# ── Restore vocab from checkpoint BEFORE loading state dict ─────────────
if "char2idx" in ckpt:
    preprocessor.char2idx = ckpt["char2idx"]
    preprocessor.idx2char = {v: k for k, v in ckpt["char2idx"].items()}
    vocab_size = len(preprocessor.char2idx)
    print(f"Vocab restored from checkpoint: {vocab_size} characters")

    # Re-adapt embedding layer to match checkpoint vocab size
    strategy.adapt_vocab(vocab_size)
    print(f"Embedding re-adapted to vocab_size={vocab_size}")
else:
    print("No vocab in checkpoint — using preprocessor vocab")


strategy.model.load_state_dict(ckpt["model_state"])
manager.optimizer.load_state_dict(ckpt["optim_state"])
manager.best_val_loss = ckpt["val_loss"]
start_epoch           = ckpt["epoch"]

print(f"\nResumed from epoch {start_epoch}")
print(f"Val loss at resume : {ckpt['val_loss']:.4f}")
print(f"Target epochs      : {TARGET_EPOCHS}")
print(f"Epochs remaining   : {TARGET_EPOCHS - start_epoch}")

# ── Step 4: Restore experiment log ──────────────────────────────────────
# Rebuild from checkpoint filenames so log is never lost
experiment_log = []
for ckpt_path in sorted(glob.glob(f"{CHECKPOINT_DIR}/checkpoint_epoch*.pt")):
    c = torch.load(ckpt_path, map_location='cpu')
    experiment_log.append({
        "id"               : c["epoch"],
        "model_name"       : c.get("strategy", "Tacotron2Strategy"),
        "hyperparameters"  : {
            "lr"          : c["config"]["learning_rate"],
            "batch_size"  : c["config"]["batch_size"],
            "weight_decay": c["config"]["weight_decay"],
        },
        "train_loss"       : c.get("train_loss", 0.0),   # restored from checkpoint
        "val_loss"         : c["val_loss"],
        "objective_metrics": {},
        "timestamp"        : "restored",
        "epoch_time_sec"   : 0.0,
    })

print(f"Restored {len(experiment_log)} epochs from checkpoints.")
for e in experiment_log:
    print(f"  Epoch {e['id']:03d} | train={e['train_loss']:.4f} | val={e['val_loss']:.4f}")

train_dataset = TTSDataset(splits["train"], preprocessor, CONFIG)
val_dataset   = TTSDataset(splits["val"],   preprocessor, CONFIG)
test_dataset  = TTSDataset(splits["test"],  preprocessor, CONFIG)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=tts_collate_fn,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=tts_collate_fn,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=tts_collate_fn,
                          num_workers=2, pin_memory=True)

# Update manager's loaders too
manager.train_loader = train_loader
manager.val_loader   = val_loader
print("DataLoaders rebuilt with restored vocab.")

# ── Step 5: Continue training ────────────────────────────────────────────
remaining = TARGET_EPOCHS - start_epoch
if remaining <= 0:
    print(f"Already at {start_epoch} epochs — target reached!")
else:
    print(f"\nStarting from epoch {start_epoch + 1}, running {remaining} epochs...\n")
    new_logs       = manager.train(remaining, start_epoch=start_epoch + 1)  # pass start
    experiment_log += new_logs
    print(f"\nTotal epochs completed: {start_epoch + remaining}")



'''
**Session 1 end:**

1. Run "Save checkpoint to Kaggle Dataset" cell
2. Run "Create dataset via API" cell once
3. Note your epoch count (currently 5)


**Session 2 start:**

1. Add your checkpoint dataset as input data in Kaggle UI
2. Run all class definition cells (Preprocessor, TTSDataset, etc.)
3. Run the Resume Training cell with TARGET_EPOCHS = 20
4. It auto-loads epoch 5 checkpoint and runs 15 more epochs

**Session 3+ start:**

Same as Session 2 — just update TARGET_EPOCHS = 30 etc.
'''
