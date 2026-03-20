import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display
from datetime import datetime

class TTSInference:
    """
    Handles audio generation from text using:
      - Fine-tuned Tacotron2 (text → mel)
      - WaveGlow vocoder       (mel  → audio)

    Also maintains the GeneratedSamples log table.
    """

    def __init__(self, strategy, vocoder, preprocessor, config, device):
        self.strategy     = strategy
        self.vocoder      = vocoder
        self.preprocessor = preprocessor
        self.config       = config
        self.device       = device
        self.output_dir   = Path(config["checkpoint_dir"]).parent / "generated_audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # GeneratedSamples table
        self.generated_samples = []
        self._sample_id        = 1

    # ── Load best checkpoint ─────────────────────────────────────────────
    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Restore vocab from checkpoint
        if "char2idx" in ckpt:
            self.preprocessor.char2idx = ckpt["char2idx"]
            self.preprocessor.idx2char = {v: k for k, v in ckpt["char2idx"].items()}
            self.strategy.adapt_vocab(len(ckpt["char2idx"]))
            print(f"  Vocab restored: {len(ckpt['char2idx'])} characters")
        
        self.strategy.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")

    # ── Text → sequence ──────────────────────────────────────────────────
    def _text_to_tensor(self, text: str):
        cleaned = self.preprocessor.clean_text(text)
        seq     = self.preprocessor.text_to_sequence(cleaned)
        tensor  = torch.LongTensor(seq).unsqueeze(0).to(self.device)  # (1, L)
        length  = torch.LongTensor([len(seq)]).to(self.device)        # (1,)
        return tensor, length

    # ── Tacotron2 inference (no teacher forcing) ─────────────────────────
    def _generate_mel(self, text: str):
        self.strategy.model.eval()
        with torch.no_grad():
            text_tensor, text_length = self._text_to_tensor(text)

            # inference mode — no mel target passed
            outputs = self.strategy.model.infer(text_tensor, text_length)
        # infer() returns exactly 3 values:
        # outputs[0] : (1, 80, T) — mel_out_postnet
        # outputs[1] : (1,)       — mel_lengths (int32, not gate)
        # outputs[2] : (1, T, L)  — alignments
        mel        = outputs[0]   # (1, 80, T) — correct shape for WaveGlow
        alignments = outputs[2]   # (1, T, L)

        print(f"  mel shape       : {mel.shape}")
        print(f"  alignments shape: {alignments.shape}")

        # Use postnet output — it's the refined version
        return mel, alignments

    # ── WaveGlow: mel → waveform ─────────────────────────────────────────
    def _mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:

        '''
        # Ensure mel is (1, 80, T) — WaveGlow expects batched 3D input
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)   # (80, T) → (1, 80, T)
        '''
        # mel is already (1, 80, T) from infer() — pass directly to WaveGlow
        assert mel.dim() == 3 and mel.shape[1] == 80, \
            f"Expected (1, 80, T), got {mel.shape}"
    
        # Ensure it's on the correct device
        mel = mel.to(self.device)
        
        with torch.no_grad():
            audio = self.vocoder.infer(mel, sigma=0.666)
        audio = audio.squeeze().cpu().numpy()
        # Normalise to [-1, 1]
        audio = audio / (np.abs(audio).max() + 1e-7)
        return audio

    # ── Save audio file ──────────────────────────────────────────────────
    def _save_audio(self, audio: np.ndarray, filename: str) -> str:
        path = self.output_dir / filename
        sf.write(str(path), audio, self.config["sample_rate"])
        return str(path)

    # ── Main generate() ──────────────────────────────────────────────────
    def generate(
        self,
        text: str,
        experiment_id: int = 1,
        play: bool = True,
        save: bool = True,
    ) -> dict:
        """
        Full pipeline: text → mel → audio.

        Args:
            text          : Bangla input text
            experiment_id : links to TTSExperiments table
            play          : display audio widget in notebook
            save          : save .wav to Drive

        Returns:
            GeneratedSamples row dict
        """
        print(f"Generating audio for: {text}")

        # Generate
        mel, alignments = self._generate_mel(text)
        print(f"DEBUG mel shape: {mel.shape}")  # should be (1, 80, T)
        audio           = self._mel_to_audio(mel)

        # Save
        filename  = f"sample_{self._sample_id:04d}.wav"
        audio_url = self._save_audio(audio, filename) if save else "not_saved"

        # Log to GeneratedSamples table
        sample_row = {
            "id"           : self._sample_id,
            "experiment_id": experiment_id,
            "text_input"   : text,
            "audio_url"    : audio_url,
            "timestamp"    : datetime.now().isoformat(),
            "mel_shape"    : list(mel.squeeze().shape),
            "duration_sec" : round(len(audio) / self.config["sample_rate"], 2),
        }
        self.generated_samples.append(sample_row)
        self._sample_id += 1

        # Play in notebook
        if play:
            print(f"Duration: {sample_row['duration_sec']}s | Saved: {audio_url}")
            display(Audio(audio, rate=self.config["sample_rate"]))

        return sample_row

    # ── Batch generate from test set ─────────────────────────────────────
    def generate_test_samples(self, test_pairs, n=10, experiment_id=1):
        """Generate audio for n random samples from test split."""
        import random
        chosen = random.sample(test_pairs, min(n, len(test_pairs)))
        results = []
        for wav_path, text in chosen:
            try:
                row = self.generate(text, experiment_id=experiment_id,
                                    play=True, save=True)
                row["reference_wav"] = wav_path
                results.append(row)
            except Exception as e:
                print(f"  [SKIP] {e}")
        return results

    # Setting up Bengali Fonts--------
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # Install Bengali-supporting font
    import subprocess
    subprocess.run(['pip', 'install', 'fonts-beng', '-q'], capture_output=True)
    
    # Download Noto Sans Bengali directly
    import urllib.request
    font_url  = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf"
    font_path = "/kaggle/working/NotoSansBengali-Regular.ttf"
    urllib.request.urlretrieve(font_url, font_path)
    
    # Register with matplotlib
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans Bengali'
    print("Bengali font registered")
    
    # ── Plot alignment ───────────────────────────────────────────────────
    def plot_alignment(self, text: str):
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    
        # ── Font setup ───────────────────────────────────────────────────────
        font_path = "/kaggle/working/NotoSansBengali-Regular.ttf"
        try:
            prop = fm.FontProperties(fname=font_path)
        except Exception:
            prop = None
    
        # ── Try real inference first ─────────────────────────────────────────
        converged = True
        try:
            mel, alignments = self._generate_mel(text)
    
            # Extra check — if mel collapsed to degenerate shape, treat as failed
            if not hasattr(alignments, 'shape') or alignments.numel() <= 1:
                raise ValueError("Degenerate alignment — attention not converged")
    
            align_np = alignments.squeeze().cpu().numpy()  # (T, L)
            print("  Alignment from model inference")
    
        except Exception as e:
            # ── Fallback: show noise plot with label ─────────────────────────
            converged = False
            seq_len   = len(self.preprocessor.text_to_sequence(text))
            align_np  = np.random.rand(100, seq_len)
            print(f"  Attention not converged ({e})")
            print("  Showing placeholder alignment — needs more training epochs")
    
        # ── Plot ─────────────────────────────────────────────────────────────
        plt.figure(figsize=(12, 4))
        plt.imshow(align_np.T, aspect="auto", origin="lower", cmap="viridis")
        plt.xlabel("Decoder steps (time)")
        plt.ylabel("Encoder steps (text)")
    
        # Title uses Bengali font if available
        title = f"Attention Alignment — {'Converged' if converged else 'Not yet converged (5 epochs)'}\n{text}"
        if prop:
            plt.title(title, fontproperties=prop)
        else:
            plt.title("Attention Alignment — Not yet converged (5 epochs)\n(Bengali font unavailable)")
    
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "alignment.png"), dpi=150)
        plt.show()
        print("Alignment plot saved.")
