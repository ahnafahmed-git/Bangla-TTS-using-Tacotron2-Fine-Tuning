import sys
sys.path.append('/content/tacotron2')

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load NVIDIA pretrained Tacotron2
tacotron2 = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_tacotron2',
    model_math='fp32',
    pretrained=True,
)
tacotron2 = tacotron2.to(device)
print("Tacotron2 loaded successfully!")
print(f"Parameters: {sum(p.numel() for p in tacotron2.parameters()):,}")

import torch.nn as nn

VOCAB_SIZE = len(preprocessor.char2idx)  # 73 characters
print(f"Bangla vocab size: {VOCAB_SIZE}")

# NVIDIA pretrained model has 148-char English vocab
# We replace the embedding layer with our Bangla vocab size
old_embedding = tacotron2.embedding
embedding_dim  = old_embedding.embedding_dim   # 512

tacotron2.embedding = nn.Embedding(
    num_embeddings = VOCAB_SIZE,
    embedding_dim  = embedding_dim,
    padding_idx    = 0,   # index 0 is <PAD>
).to(device)

# Initialise new embedding with small random weights
nn.init.xavier_uniform_(tacotron2.embedding.weight)

print(f"Embedding layer replaced: ({VOCAB_SIZE}, {embedding_dim})")
print(f"Total parameters after adaptation: {sum(p.numel() for p in tacotron2.parameters()):,}")

tacotron2.train()

# Grab one batch
text_padded, mel_padded, text_lens, mel_lens = next(iter(train_loader))

# Move to device
text_padded = text_padded.to(device)
mel_padded  = mel_padded.to(device)
text_lens   = text_lens.to(device)
mel_lens    = mel_lens.to(device)

# NVIDIA Tacotron2 expects: (text, text_lengths, mel_target, max_mel_len, mel_lengths)
max_mel_len = mel_padded.shape[2]  # T dimension

# NVIDIA's hub model expects inputs as a TUPLE
inputs = (text_padded, text_lens, mel_padded, max_mel_len, mel_lens)

with torch.no_grad():
    mel_out, mel_out_postnet, gate_out, alignments = tacotron2(inputs)

print(f"mel_out        : {mel_out.shape}        → (B, 80, T)")
print(f"mel_out_postnet: {mel_out_postnet.shape} → (B, 80, T)")
print(f"gate_out       : {gate_out.shape}       → (B, T) stop token")
print(f"alignments     : {alignments.shape}     → (B, T, L) attention")
print("\n Forward pass successful!")
