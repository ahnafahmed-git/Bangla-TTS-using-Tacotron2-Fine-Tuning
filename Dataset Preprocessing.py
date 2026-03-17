# CONFIG (change hyperparams here only)
CONFIG = {
    # Paths
    "data_dir"      : "/kaggle/input/datasets/mobassir/comprehensive-bangla-tts/iitm_bangla_tts/comprehensive_bangla_tts",

    # Audio
    "sample_rate"   : 22050,
    "max_duration"  : 10.0,   # seconds — clips longer than this are dropped
    "min_duration"  : 1.0,    # seconds — clips shorter than this are dropped

    # Mel-spectrogram (must match Tacotron2 exactly)
    "n_fft"         : 1024,
    "hop_length"    : 256,
    "win_length"    : 1024,
    "n_mels"        : 80,
    "fmin"          : 0.0,
    "fmax"          : 8000.0,

    # Dataset split
    "train_ratio"   : 0.80,
    "val_ratio"     : 0.10,
    "test_ratio"    : 0.10,

    # Reproducibility
    "seed"          : 42,

    # Speaker selection: 'female', 'male', or 'both'
    "speaker"       : "both",
}

#  PREPROCESSOR CLASS
# ─────────────────────────────────────────────
class Preprocessor:
    """
    Handles all data preparation for Tacotron2 fine-tuning:
      - Parses metadata (LJSpeech pipe-delimited format)
      - Cleans and normalises Bangla text
      - Builds character vocabulary
      - Filters audio by duration
      - Splits into train / val / test
      - Computes and saves mel-spectrograms
    """

    # Bangla Unicode block: U+0980 – U+09FF
    BANGLA_RANGE = re.compile(r'[^\u0980-\u09FF\s,।\.\!\?]')

    def __init__(self, config: Dict):
        self.config = config
        self.data_dir   = Path(config["data_dir"])
        self.sr         = config["sample_rate"]

        random.seed(config["seed"])
        np.random.seed(config["seed"])

        # Created during build()
        self.samples    : List[Tuple[str, str]] = []  # (wav_path, text)
        self.char2idx   : Dict[str, int]        = {}
        self.idx2char   : Dict[int, str]        = {}

    # ── 1. Parse metadata ────────────────────────────────────────────────
    def _parse_metadata(self, speaker: str) -> List[Tuple[str, str]]:
        """
        Reads metadata_{speaker}.txt (format: filename|text)
        Returns list of (absolute_wav_path, raw_text).
        """
        meta_path = self.data_dir / speaker / "mono" / f"metadata_{speaker}.txt"
        wav_dir   = self.data_dir / speaker / "mono" / "wav"
        pairs = []

        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                filename, text = line.split("|", 1)
                wav_path = wav_dir / f"{filename}.wav"
                if wav_path.exists():
                    pairs.append((str(wav_path), text.strip()))
                else:
                    print(f"  [WARN] Missing wav: {wav_path.name}")

        print(f"  Loaded {len(pairs)} entries from {speaker} metadata.")
        return pairs

    # ── 2. Text cleaning ─────────────────────────────────────────────────
    def clean_text(self, text: str) -> str:
        """
        Normalises Bangla text for TTS:
          - Unicode NFC normalisation
          - Remove zero-width characters
          - Strip non-Bangla characters (keeps punctuation: , । . ! ?)
          - Collapse multiple spaces
        """
        # NFC normalisation — ensures consistent Unicode representation
        text = unicodedata.normalize("NFC", text)

        # Remove zero-width joiner / non-joiner artifacts
        text = text.replace("\u200c", "").replace("\u200d", "")

        # Remove anything outside Bangla Unicode block + allowed punctuation
        text = self.BANGLA_RANGE.sub("", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ── 3. Duration filter ───────────────────────────────────────────────
    def _get_duration(self, wav_path: str) -> float:
        info = sf.info(wav_path)
        return info.frames / info.samplerate

    def _filter_by_duration(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        min_d = self.config["min_duration"]
        max_d = self.config["max_duration"]
        filtered, dropped = [], 0

        for wav_path, text in pairs:
            dur = self._get_duration(wav_path)
            if min_d <= dur <= max_d:
                filtered.append((wav_path, text))
            else:
                dropped += 1

        print(f"  Duration filter: kept {len(filtered)}, dropped {dropped}")
        return filtered

    # ── 4. Vocabulary builder ────────────────────────────────────────────
    def _build_vocab(self, texts: List[str]) -> None:
        """
        Builds character-level vocabulary from all texts.
        Reserves index 0 for <PAD> and 1 for <EOS>.
        """
        special = ["<PAD>", "<EOS>"]
        chars   = sorted(set("".join(texts)))
        vocab   = special + chars

        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for i, c in enumerate(vocab)}
        print(f"  Vocabulary size: {len(vocab)} characters")

    def text_to_sequence(self, text: str) -> List[int]:
        """Converts cleaned text to list of character indices."""
        seq = [self.char2idx[c] for c in text if c in self.char2idx]
        seq.append(self.char2idx["<EOS>"])
        return seq

    # ── 5. Mel-spectrogram ───────────────────────────────────────────────
    def compute_mel(self, wav_path: str) -> np.ndarray:
        """
        Loads wav and computes 80-band mel-spectrogram
        matching Tacotron2's expected input format.
        Returns array of shape (n_mels, T).
        """
        y, _ = librosa.load(wav_path, sr=self.sr)

        mel = librosa.feature.melspectrogram(
            y          = y,
            sr         = self.sr,
            n_fft      = self.config["n_fft"],
            hop_length = self.config["hop_length"],
            win_length = self.config["win_length"],
            n_mels     = self.config["n_mels"],
            fmin       = self.config["fmin"],
            fmax       = self.config["fmax"],
        )

        # Convert to log scale (Tacotron2 works in log-mel space)
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        return mel  # shape: (80, T)

    # ── 6. Train / val / test split ──────────────────────────────────────
    def _split(
        self, pairs: List[Tuple[str, str]]
    ) -> Tuple[List, List, List]:
        random.shuffle(pairs)
        n       = len(pairs)
        n_train = int(n * self.config["train_ratio"])
        n_val   = int(n * self.config["val_ratio"])

        train = pairs[:n_train]
        val   = pairs[n_train : n_train + n_val]
        test  = pairs[n_train + n_val :]

        print(f"  Split → train: {len(train)}, val: {len(val)}, test: {len(test)}")
        return train, val, test

    # ── 7. Save metadata CSVs ────────────────────────────────────────────
    def _save_split_csv(
        self, pairs: List[Tuple[str, str]], split_name: str
    ) -> None:
        """Saves a split as a CSV: wav_path|cleaned_text|sequence"""
        import csv
        out_path = self.output_dir / f"{split_name}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["wav_path", "text", "sequence"])
            for wav_path, text in pairs:
                cleaned = self.clean_text(text)
                seq     = self.text_to_sequence(cleaned)
                writer.writerow([wav_path, cleaned, json.dumps(seq, ensure_ascii=False)])
        print(f"  Saved {split_name}.csv → {out_path}")

    # ── 8. Save vocabulary ───────────────────────────────────────────────
    def _save_vocab(self) -> None:
        vocab_path = self.output_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)
        print(f"  Saved vocab.json → {vocab_path}")

    # ── 9. Master build() ────────────────────────────────────────────────
    def build(self) -> Dict:
        """
        Full preprocessing pipeline. Call this once.
        Returns dict with train/val/test splits.
        """
        speaker = self.config["speaker"]

        print(f"\n{'='*50}")
        print(f"  Preprocessor starting | speaker: {speaker}")
        print(f"{'='*50}")

        # Step 1 — Parse
        if speaker == "both":
            pairs = self._parse_metadata("female") + self._parse_metadata("male")
        else:
            pairs = self._parse_metadata(speaker)

        # Step 2 — Filter by duration
        pairs = self._filter_by_duration(pairs)

        # Step 3 — Clean text
        print(f"  Cleaning text...")
        pairs = [(wav, self.clean_text(text)) for wav, text in pairs]

        # Step 4 — Drop empty texts after cleaning
        pairs = [(w, t) for w, t in pairs if len(t) > 0]
        print(f"  After text cleaning: {len(pairs)} samples remain")

        # Step 5 — Build vocabulary
        self._build_vocab([text for _, text in pairs])

        # Step 6 — Split
        train, val, test = self._split(pairs)

        print(f"\n Preprocessing complete.")

        return {"train": train, "val": val, "test": test}

# Execute preprocess
preprocessor = Preprocessor(CONFIG)
splits = preprocessor.build()

# Grab one sample from train
wav_path, text = splits["train"][0]

print(f"Text (cleaned) : {text}")
print(f"Sequence       : {preprocessor.text_to_sequence(text)[:15]}...")
print(f"Vocab size     : {len(preprocessor.char2idx)}")

# Plot it's mel-spectrogram
mel = preprocessor.compute_mel(wav_path)
plt.figure(figsize=(12, 4))
plt.imshow(mel, aspect="auto", origin="lower", cmap="magma")
plt.colorbar(label="Log Mel Energy")
plt.title(f"Mel-Spectrogram — {Path(wav_path).name}")
plt.xlabel("Time frames")
plt.ylabel("Mel bins (80)")
plt.tight_layout()
plt.show()

print(f"\nMel shape: {mel.shape}  →  (n_mels=80, T={mel.shape[1]} frames)")
