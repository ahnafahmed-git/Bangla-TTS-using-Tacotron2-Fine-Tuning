# ════════════════════════════════════════════════════════
#  Uses real mels from test set instead of generated ones
# ════════════════════════════════════════════════════════

import random
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

# ── 1. Plot loss curves ──────────────────────────────────────────────────
plot_loss_curves(experiment_log, BASE_DIR)

# ── 2. Evaluator instance ────────────────────────────────────────────────
evaluator = Evaluator(CONFIG)

# ── 3. Reconstruct audio from real mels via WaveGlow ────────────────────
print("\n" + "="*55)
print("  Vocoder Reconstruction Evaluation")
print("  (real mel → WaveGlow → reconstructed wav)")
print("="*55)

test_samples_eval = []
chosen = random.sample(splits["test"], 10)

for i, (wav_path, text) in enumerate(chosen):
    try:
        # Load real mel
        real_mel = preprocessor.compute_mel(wav_path)             # (80, T)
        mel_tensor = torch.FloatTensor(real_mel).unsqueeze(0).to(device)  # (1, 80, T)

        # Reconstruct via WaveGlow
        with torch.no_grad():
            audio = waveglow.infer(mel_tensor, sigma=0.666)
        audio_np = audio.squeeze().cpu().numpy()
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)

        # Save reconstructed wav
        out_path = f"{BASE_DIR}/generated_audio/reconstructed_{i+1:03d}.wav"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, audio_np, CONFIG["sample_rate"])

        # Play in notebook
        print(f"\nSample {i+1}: {text[:60]}")
        display(Audio(audio_np, rate=CONFIG["sample_rate"]))

        # Log to GeneratedSamples
        test_samples_eval.append({
            "id"            : i + 1,
            "experiment_id" : len(experiment_log),
            "text_input"    : text,
            "audio_url"     : out_path,
            "reference_wav" : wav_path,
            "timestamp"     : datetime.now().isoformat(),
            "duration_sec"  : round(len(audio_np) / CONFIG["sample_rate"], 2),
            "mel_shape"     : list(real_mel.shape),
        })

    except Exception as e:
        print(f"  [SKIP] Sample {i+1}: {e}")

# ── 4. Run evaluation metrics ────────────────────────────────────────────
print("\n" + "="*55)
print("  Running Objective Metrics")
print("="*55)
evaluated = evaluator.evaluate_samples(test_samples_eval)

# Aggregate
valid = [s for s in evaluated if s.get("metrics", {}).get("mcd") is not None]
if valid:
    avg_mcd    = np.mean([s["metrics"]["mcd"]    for s in valid])
    avg_pesq   = np.mean([s["metrics"]["pesq"]   for s in valid if s["metrics"]["pesq"] > 0])
    avg_stoi   = np.mean([s["metrics"]["stoi"]   for s in valid])
    avg_mosnet = np.mean([s["metrics"]["mosnet"] for s in valid])

    print(f"\n  Average MCD    : {avg_mcd:.4f}  dB  (lower is better, good range: 4–8)")
    print(f"  Average PESQ   : {avg_pesq:.4f}      (higher is better, max ~4.5)")
    print(f"  Average STOI   : {avg_stoi:.4f}      (higher is better, max 1.0)")
    print(f"  Average MOSNet : {avg_mosnet:.4f}     (higher is better, max 5.0)")

    experiment_log[-1]["objective_metrics"] = {
        "avg_mcd"   : round(avg_mcd,    4),
        "avg_pesq"  : round(avg_pesq,   4),
        "avg_stoi"  : round(avg_stoi,   4),
        "avg_mosnet": round(avg_mosnet, 4),
        "note"      : "Vocoder reconstruction eval — model needs more epochs for end-to-end inference"
    }

# ── 5. Plot attention alignment (shows convergence state) ────────────────
print("\nPlotting attention alignment...")
try:
    inference.plot_alignment("বাংলাদেশ একটি সুন্দর দেশ।")
except Exception as e:
    print(f"  Alignment plot skipped: {e}")

# ── 6. Display tables ────────────────────────────────────────────────────
# Build eval_metrics dict properly for all epochs
# Attach vocoder eval metrics to the last epoch entry
avg_metrics = {
    "avg_mcd"   : round(avg_mcd,    4),
    "avg_pesq"  : round(avg_pesq,   4),
    "avg_stoi"  : round(avg_stoi,   4),
    "avg_mosnet": round(avg_mosnet, 4),
    "note"      : "Vocoder reconstruction eval",
}
experiment_log[-1]["objective_metrics"] = avg_metrics

# Pass correctly keyed dict to display
eval_metrics_display = {
    experiment_log[-1]["id"]: avg_metrics
}
logger     = ExperimentLogger()
exp_df     = logger.display_experiments(
    experiment_log,
    eval_metrics={len(experiment_log): experiment_log[-1]["objective_metrics"]}
)
samples_df = logger.display_generated_samples(evaluated)

# ── 7. Save to working dir ───────────────────────────────────────────────
logger.save_tables_to_drive(exp_df, samples_df, BASE_DIR)

print("\nEvaluation complete.")
print(f" Reconstructed audio → {BASE_DIR}/generated_audio/")
print(f" TTSExperiments      → {BASE_DIR}/TTSExperiments.csv")
print(f" GeneratedSamples    → {BASE_DIR}/GeneratedSamples.csv")
