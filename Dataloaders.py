!pip install torch torchaudio numpy librosa matplotlib tqdm -q

class TTSDataset(Dataset):
    """
    PyTorch Dataset for Tacotron2 fine-tuning.
    Each item returns:
      - text_seq : character index tensor         (L,)
      - mel      : log mel-spectrogram tensor     (80, T)
      - text_len : scalar — actual text length
      - mel_len  : scalar — actual mel length
    """

    def __init__(self, pairs, preprocessor, config):
        """
        Args:
            pairs       : list of (wav_path, cleaned_text) from preprocessor
            preprocessor: fitted Preprocessor instance (has char2idx, compute_mel)
            config      : global CONFIG dict
        """
        self.pairs        = pairs
        self.preprocessor = preprocessor
        self.config       = config

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wav_path, text = self.pairs[idx]

        # ── Text → sequence of indices ──────────────────────────────────
        seq     = self.preprocessor.text_to_sequence(text)
        seq_len = len(seq)

        # ── Audio → mel-spectrogram ──────────────────────────────────────
        mel     = self.preprocessor.compute_mel(wav_path)  # (80, T)
        mel_len = mel.shape[1]

        return (
            torch.LongTensor(seq),           # (L,)
            torch.FloatTensor(mel),          # (80, T)
            torch.tensor(seq_len),           # scalar
            torch.tensor(mel_len),           # scalar
        )

def tts_collate_fn(batch):
    """
    Pads variable-length sequences and mels to the
    longest item in the batch so they can be stacked
    into tensors. Sorts by text length descending
    (required by Tacotron2's pack_padded_sequence).
    """
    seqs, mels, seq_lens, mel_lens = zip(*batch)

    # ── Filter out degenerate samples with mel_len < 2 ──────────────────
    valid = [(s, m, sl, ml) for s, m, sl, ml in 
             zip(seqs, mels, seq_lens, mel_lens) if ml.item() >= 2]
    
    if len(valid) == 0:
        # Extremely rare — return None and skip in training loop
        return None
    
    # ── Sort by text length descending ──────────────────────────────────
    sorted_indices = torch.argsort(torch.stack(seq_lens), descending=True)
    seqs     = [seqs[i]     for i in sorted_indices]
    mels     = [mels[i]     for i in sorted_indices]
    seq_lens = [seq_lens[i] for i in sorted_indices]
    mel_lens = [mel_lens[i] for i in sorted_indices]

    # Max lengths in this batch
    max_seq_len = max(seq_lens).item()
    max_mel_len = max(mel_lens).item()
    batch_size  = len(seqs)

    # Allocate padded tensors
    text_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mel_padded  = torch.zeros(batch_size, 80, max_mel_len, dtype=torch.float)

    # Fill in actual values
    for i, (seq, mel) in enumerate(zip(seqs, mels)):
        text_padded[i, :seq.size(0)]    = seq
        mel_padded [i, :, :mel.size(1)] = mel

    text_lens = torch.stack(seq_lens)
    mel_lens  = torch.stack(mel_lens)

    return text_padded, mel_padded, text_lens, mel_lens


# ── Instantiate datasets ─────────────────────────────────────────────────
train_dataset = TTSDataset(splits["train"], preprocessor, CONFIG)
val_dataset   = TTSDataset(splits["val"],   preprocessor, CONFIG)
test_dataset  = TTSDataset(splits["test"],  preprocessor, CONFIG)

# ── DataLoaders ──────────────────────────────────────────────────────────
BATCH_SIZE = 16  # reduce if runs out of GPU memory

train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    collate_fn  = tts_collate_fn,
    num_workers = 2,
    pin_memory  = True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    collate_fn  = tts_collate_fn,
    num_workers = 2,
    pin_memory  = True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    collate_fn  = tts_collate_fn,
    num_workers = 2,
    pin_memory  = True,
)

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")
print(f"Test batches  : {len(test_loader)}")

# Grab one batch and verify shapes
text_padded, mel_padded, text_lens, mel_lens = next(iter(train_loader))

print(f"text_padded : {text_padded.shape}   → (B, L_max)")
print(f"mel_padded  : {mel_padded.shape}  → (B, 80, T_max)")
print(f"text_lens   : {text_lens.shape}    → (B,)")
print(f"mel_lens    : {mel_lens.shape}    → (B,)")
print(f"\nSample text lengths in batch : {text_lens[:5].tolist()}")
print(f"Sample mel  lengths in batch : {mel_lens[:5].tolist()}")

all_text_lens = [len(preprocessor.text_to_sequence(text)) 
                 for _, text in splits["train"]]

print(f"Max text length  : {max(all_text_lens)}")
print(f"Min text length  : {min(all_text_lens)}")
print(f"Mean text length : {sum(all_text_lens)/len(all_text_lens):.1f}")
