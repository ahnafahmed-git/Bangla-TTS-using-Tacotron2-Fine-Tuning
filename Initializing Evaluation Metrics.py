from scipy.spatial.distance import euclidean
from pesq import pesq as pesq_score
from pystoi import stoi

class Evaluator:
    """
    Computes objective evaluation metrics for generated TTS audio:
      - MCD  : Mel Cepstral Distortion    (lower = better, 0 is perfect)
      - PESQ : Perceptual Eval Speech Quality (higher = better, max ~4.5)
      - STOI : Short-Time Objective Intelligibility (higher = better, max 1.0)
      - MOSNet: Predicted MOS score       (higher = better, max 5.0)

    Note: PESQ/STOI require a reference (ground truth) audio.
          MOSNet scores generated audio standalone.
    """

    def __init__(self, config):
        self.sr = config["sample_rate"]
        self.n_mfcc = 13  # standard for MCD

    # ── MCD ──────────────────────────────────────────────────────────────
    def compute_mcd(self, ref_wav: str, gen_wav: str) -> float:
        """
        Mel Cepstral Distortion between reference and generated audio.
        Lower is better. Typical good range: 4–8 dB.
        """
        ref, _ = librosa.load(ref_wav, sr=self.sr)
        gen, _ = librosa.load(gen_wav, sr=self.sr)

        # Compute MFCC (13 coefficients, skip C0)
        ref_mfcc = librosa.feature.mfcc(y=ref, sr=self.sr, n_mfcc=self.n_mfcc+1)[1:]
        gen_mfcc = librosa.feature.mfcc(y=gen, sr=self.sr, n_mfcc=self.n_mfcc+1)[1:]

        # Align lengths (trim to shorter)
        min_len  = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
        ref_mfcc = ref_mfcc[:, :min_len]
        gen_mfcc = gen_mfcc[:, :min_len]

        # MCD formula: (10√2 / ln10) * mean(||mfcc_ref - mfcc_gen||)
        diff    = ref_mfcc - gen_mfcc
        mcd     = (10.0 * np.sqrt(2) / np.log(10)) * np.mean(
                    np.sqrt(np.sum(diff**2, axis=0))
                  )
        return round(float(mcd), 4)

    # ── PESQ ─────────────────────────────────────────────────────────────
    def compute_pesq(self, ref_wav: str, gen_wav: str) -> float:
        """
        PESQ score. Higher is better (max ~4.5).
        Uses narrowband mode (nb) for 8kHz or wideband (wb) for 16kHz+.
        Note: PESQ requires both signals at 16kHz.
        """
        ref, _ = librosa.load(ref_wav, sr=16000)
        gen, _ = librosa.load(gen_wav, sr=16000)

        # Align lengths
        min_len = min(len(ref), len(gen))
        ref     = ref[:min_len]
        gen     = gen[:min_len]

        try:
            score = pesq_score(16000, ref, gen, 'wb')
        except Exception:
            score = -1.0  # PESQ fails on very short clips
        return round(float(score), 4)

    # ── STOI ─────────────────────────────────────────────────────────────
    def compute_stoi(self, ref_wav: str, gen_wav: str) -> float:
        """
        Short-Time Objective Intelligibility. Higher is better (max 1.0).
        Measures how intelligible the generated speech is.
        """
        ref, _ = librosa.load(ref_wav, sr=self.sr)
        gen, _ = librosa.load(gen_wav, sr=self.sr)

        min_len = min(len(ref), len(gen))
        ref     = ref[:min_len]
        gen     = gen[:min_len]

        score = stoi(ref, gen, self.sr, extended=False)
        return round(float(score), 4)

    # ── MOSNet (standalone — no reference needed) ─────────────────────────
    def compute_mosnet(self, gen_wav: str) -> float:
        """
        Predicts MOS (Mean Opinion Score) for generated audio.
        Uses a simple energy/spectral-based approximation when
        the full MOSNet model is unavailable on Colab.
        Range: 1.0–5.0, higher is better.
        """
        try:
            # Attempt full MOSNet if installed
            import mosnet
            model = mosnet.load_model()
            score = mosnet.predict(model, gen_wav)
            return round(float(score), 4)
        except ImportError:
            # Fallback: spectral flatness-based MOS approximation
            gen, _ = librosa.load(gen_wav, sr=self.sr)
            flatness   = np.mean(librosa.feature.spectral_flatness(y=gen))
            zcr        = np.mean(librosa.feature.zero_crossing_rate(gen))
            rms        = np.mean(librosa.feature.rms(y=gen))
            # Heuristic approximation (not a true MOS — label accordingly)
            approx_mos = float(np.clip(3.5 - flatness * 10 - zcr * 5 + rms * 2, 1.0, 5.0))
            return round(approx_mos, 4)

    # ── Full evaluation on a sample list ─────────────────────────────────
    def evaluate_samples(self, generated_samples: list) -> list:
        """
        Runs all metrics on each generated sample that has a reference wav.
        Updates the sample dict with metric scores in place.

        Args:
            generated_samples: list of GeneratedSamples dicts
                               (must have 'audio_url' and 'reference_wav')
        Returns:
            list of dicts with metrics filled in
        """
        results = []
        for i, sample in enumerate(generated_samples):
            gen_wav = sample.get("audio_url")
            ref_wav = sample.get("reference_wav")

            if not gen_wav or not Path(gen_wav).exists():
                print(f"  [SKIP] Sample {i+1}: generated audio not found")
                continue

            metrics = {"mosnet": self.compute_mosnet(gen_wav)}

            if ref_wav and Path(ref_wav).exists():
                metrics["mcd"]  = self.compute_mcd(ref_wav, gen_wav)
                metrics["pesq"] = self.compute_pesq(ref_wav, gen_wav)
                metrics["stoi"] = self.compute_stoi(ref_wav, gen_wav)
            else:
                metrics["mcd"]  = None
                metrics["pesq"] = None
                metrics["stoi"] = None
                print(f"  [WARN] Sample {i+1}: no reference wav, skipping MCD/PESQ/STOI")

            sample["metrics"] = metrics
            results.append(sample)

            print(
                f"  Sample {i+1:02d} | "
                f"MCD={metrics.get('mcd', 'N/A')} | "
                f"PESQ={metrics.get('pesq', 'N/A')} | "
                f"STOI={metrics.get('stoi', 'N/A')} | "
                f"MOSNet≈{metrics.get('mosnet', 'N/A')}"
            )

        return results
