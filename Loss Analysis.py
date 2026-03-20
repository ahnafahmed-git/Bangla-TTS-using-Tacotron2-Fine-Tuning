import matplotlib.pyplot as plt

# Deduplicate by epoch id — keep latest entry per epoch
seen = {}
for e in experiment_log:
    seen[e["id"]] = e
clean_log = [seen[k] for k in sorted(seen.keys())]

epochs       = [e["id"]          for e in clean_log]
train_losses = [e["train_loss"]  for e in clean_log]
val_losses   = [e["val_loss"]    for e in clean_log]
best_epoch   = epochs[val_losses.index(min(val_losses))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Plot 1: Continuous train vs val ──────────────────────────────────────
axes[0].plot(epochs, train_losses,
             marker='o', label='Train Loss', color='royalblue', linewidth=2)
axes[0].plot(epochs, val_losses,
             marker='s', label='Val Loss',   color='tomato',    linewidth=2)
axes[0].axvline(x=best_epoch, color='green', linestyle='--',
                alpha=0.7, label=f'Best epoch ({best_epoch})')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Train vs Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(epochs)

# ── Plot 2: Generalisation gap ────────────────────────────────────────────
delta  = [v - t for t, v in zip(train_losses, val_losses)]
colors = ['green' if d < 0.15 else 'orange' if d < 0.3 else 'tomato'
          for d in delta]
axes[1].bar(epochs, delta, color=colors)
axes[1].axhline(y=0, color='black', linewidth=0.8)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val Loss − Train Loss")
axes[1].set_title("Generalisation Gap (overfitting check)")
axes[1].set_xticks(epochs)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Tacotron2 Fine-tuning on Bangla TTS Dataset", fontsize=13)
plt.tight_layout()
plt.savefig("/kaggle/working/loss_curve.png", dpi=150)
plt.show()
print(f"Best val loss at epoch {best_epoch}: {min(val_losses):.4f}")
