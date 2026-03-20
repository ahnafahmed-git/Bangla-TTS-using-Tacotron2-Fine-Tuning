# Using Tacotron2, can switch modularly in between different TTS models
strategy = Tacotron2Strategy() # Can be switched to VITSStrategy()
tacotron2 = strategy.load_model(device)
strategy.adapt_vocab(len(preprocessor.char2idx))

# Add training hyperparams to CONFIG
CONFIG.update({
    "learning_rate"  : 1e-4,
    "weight_decay"   : 1e-6,
    "batch_size"     : BATCH_SIZE,
    "checkpoint_dir" : "/kaggle/working/checkpoints",
    "num_epochs"     : 5,
})

# Instantiate manager and start training
manager = TTSManager(
    strategy     = strategy,
    train_loader = train_loader,
    val_loader   = val_loader,
    config       = CONFIG,
    device       = device,
)
print("Strategy and manager ready — no training started.")
