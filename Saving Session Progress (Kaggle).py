import os
import json
import shutil

CHECKPOINT_DIR  = "/kaggle/working/checkpoints"
DATASET_DIR     = "/kaggle/working/checkpoint_dataset"
KAGGLE_USERNAME = "ahmedahnaf513"   #Kaggle UserName

os.makedirs(DATASET_DIR, exist_ok=True)

# Copy all checkpoint files
for f in os.listdir(CHECKPOINT_DIR):
    if f.endswith(".pt"):
        src = os.path.join(CHECKPOINT_DIR, f)
        dst = os.path.join(DATASET_DIR, f)
        shutil.copy(src, dst)
        print(f"Copied: {f}")

# Write dataset metadata
meta = {
    "title"    : "bangla-tts-checkpoints",
    "id"       : f"{KAGGLE_USERNAME}/bangla-tts-checkpoints",
    "licenses" : [{"name": "CC0-1.0"}]
}
with open(os.path.join(DATASET_DIR, "dataset-metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDataset folder ready: {os.listdir(DATASET_DIR)}")

# ── Step 2: Push to Kaggle servers ───────────────────────────────────────
EPOCH = len(experiment_log)
!kaggle datasets version -p /kaggle/working/checkpoint_dataset/ -m "epoch_{EPOCH}_checkpoint"
print("Checkpoints saved to Kaggle dataset successfully.")
