# Run this to clear saved/stale chekcpoints
import os
import shutil
import glob

CHECKPOINT_DIR  = "/kaggle/working/checkpoints"
DATASET_DIR     = "/kaggle/working/checkpoint_dataset"
KAGGLE_USERNAME = "ahmedahnaf513"

# ── Step 1: Delete stale checkpoint locally ──────────────────────────────
stale = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch006.pt")
if os.path.exists(stale):
    os.remove(stale)
    print("Deleted stale checkpoint_epoch006.pt from working dir")

# ── Step 2: Rebuild dataset folder from clean checkpoints only ───────────
# Clear dataset folder completely first
for f in glob.glob(f"{DATASET_DIR}/*.pt"):
    os.remove(f)
    print(f"Cleared from dataset folder: {os.path.basename(f)}")

# Copy only the clean checkpoints
for f in glob.glob(f"{CHECKPOINT_DIR}/*.pt"):
    dst = os.path.join(DATASET_DIR, os.path.basename(f))
    shutil.copy(f, dst)
    print(f"Copied: {os.path.basename(f)}")

print(f"\nDataset folder now contains: {sorted(os.listdir(DATASET_DIR))}")

# ── Step 3: Push clean version to Kaggle ────────────────────────────────
!kaggle datasets version -p {DATASET_DIR} -m "removed_stale_epoch6_clean_epoch1_to_5"
print("Remote dataset updated — stale epoch 6 removed.")
