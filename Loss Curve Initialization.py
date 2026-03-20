def plot_loss_curves(experiment_log: list, save_dir: str):
    import matplotlib.pyplot as plt

    epochs      = [e["id"]         for e in experiment_log]
    train_loss  = [e["train_loss"] for e in experiment_log]
    val_loss    = [e["val_loss"]   for e in experiment_log]
    best_epoch  = epochs[val_loss.index(min(val_loss))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot 1: Train vs Val loss ────────────────────────────────────────
    axes[0].plot(epochs, train_loss, marker='o', label='Train Loss', color='royalblue')
    axes[0].plot(epochs, val_loss,   marker='s', label='Val Loss',   color='tomato')
    axes[0].axvline(x=best_epoch, color='green', linestyle='--',
                    alpha=0.7, label=f'Best epoch ({best_epoch})')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Plot 2: Loss delta (overfitting check) ───────────────────────────
    delta = [v - t for t, v in zip(train_loss, val_loss)]
    axes[1].bar(epochs, delta, color=['green' if d < 0.1 else 'orange' for d in delta])
    axes[1].axhline(y=0, color='black', linewidth=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss − Train Loss")
    axes[1].set_title("Generalisation Gap (overfitting check)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Tacotron2 Fine-tuning on Bangla TTS Dataset", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=150)
    plt.show()
    print(f"Loss curve saved → {save_dir}/loss_curve.png")
