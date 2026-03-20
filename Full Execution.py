#  COMPLETE DEMO

BASE_DIR      = "/kaggle/working"
CHECKPOINT    = f"{BASE_DIR}/checkpoints/best_model.pt"
EXPERIMENT_ID = len(experiment_log)  # use last epoch as experiment id

# ── 1. Plot loss curves ──────────────────────────────────────────────────
plot_loss_curves(experiment_log, BASE_DIR)

# ── 2. Load inference pipeline ──────────────────────────────────────────
inference = TTSInference(
    strategy     = strategy,
    vocoder      = waveglow,
    preprocessor = preprocessor,
    config       = CONFIG,
    device       = device,
)
inference.load_checkpoint(CHECKPOINT)

# ── 3. Generate audio from custom Bangla text ────────────────────────────
print("\n" + "="*55)
print("  Generating audio from custom Bangla sentences")
print("="*55)
test_texts = [
    "বাংলাদেশ একটি সুন্দর দেশ।",
    "আমি বাংলায় কথা বলতে ভালোবাসি।",
    "কৃত্রিম বুদ্ধিমত্তা ভবিষ্যতের প্রযুক্তি।",
    "আজকের আবহাওয়া অনেক সুন্দর।",
    "শিক্ষা জাতির মেরুদণ্ড।",
]

custom_samples = []
for text in test_texts:
    try:
        row = inference.generate(
            text          = text,
            experiment_id = EXPERIMENT_ID,
            play          = True,
            save          = True,
        )
        custom_samples.append(row)
    except Exception as e:
        print(f"  [SKIP] {text[:30]}... → {e}")

# ── 4. Generate from test split (with reference wavs for metrics) ────────
print("\n" + "="*55)
print("  Generating from test split (for metric evaluation)")
print("="*55)
test_samples = inference.generate_test_samples(
    test_pairs    = splits["test"],
    n             = 10,
    experiment_id = EXPERIMENT_ID,
)

# ── 5. Plot attention alignment for one sample ───────────────────────────
print("\nPlotting attention alignment...")
inference.plot_alignment("বাংলাদেশ একটি সুন্দর দেশ।")

# ── 6. Run evaluation metrics on test samples ────────────────────────────
print("\n" + "="*55)
print("  Running objective evaluation metrics")
print("="*55)
evaluator      = Evaluator(CONFIG)
evaluated      = evaluator.evaluate_samples(test_samples)

# Aggregate metrics
valid = [s for s in evaluated if s.get("metrics", {}).get("mcd") is not None]
if valid:
    avg_mcd    = np.mean([s["metrics"]["mcd"]    for s in valid])
    avg_pesq   = np.mean([s["metrics"]["pesq"]   for s in valid if s["metrics"]["pesq"] > 0])
    avg_stoi   = np.mean([s["metrics"]["stoi"]   for s in valid])
    avg_mosnet = np.mean([s["metrics"]["mosnet"] for s in valid])

    print(f"\n  Average MCD    : {avg_mcd:.4f}  (lower is better)")
    print(f"  Average PESQ   : {avg_pesq:.4f}  (higher is better, max ~4.5)")
    print(f"  Average STOI   : {avg_stoi:.4f}  (higher is better, max 1.0)")
    print(f"  Average MOSNet : {avg_mosnet:.4f}  (higher is better, max 5.0)")

    # Attach avg metrics to last experiment log entry
    experiment_log[-1]["objective_metrics"] = {
        "avg_mcd"   : round(avg_mcd,    4),
        "avg_pesq"  : round(avg_pesq,   4),
        "avg_stoi"  : round(avg_stoi,   4),
        "avg_mosnet": round(avg_mosnet, 4),
    }

# ── 7. Display logging tables ────────────────────────────────────────────
logger      = ExperimentLogger()
all_samples = custom_samples + evaluated

exp_df      = logger.display_experiments(
    experiment_log,
    eval_metrics={EXPERIMENT_ID: experiment_log[-1]["objective_metrics"]}
)
samples_df  = logger.display_generated_samples(all_samples)

# ── 8. Save tables to Drive ──────────────────────────────────────────────
logger.save_tables_to_drive(exp_df, samples_df, BASE_DIR)

print("\n Full pipeline complete.")
print(f"   Generated audio  → {BASE_DIR}/generated_audio/")
print(f"   Checkpoints      → {BASE_DIR}/checkpoints/")
print(f"   Loss curve       → {BASE_DIR}/loss_curve.png")
print(f"   TTSExperiments   → {BASE_DIR}/TTSExperiments.csv")
print(f"   GeneratedSamples → {BASE_DIR}/GeneratedSamples.csv")
