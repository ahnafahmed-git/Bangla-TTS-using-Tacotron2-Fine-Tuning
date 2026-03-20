class Tacotron2Loss(nn.Module):
    """
    Tacotron2 uses three losses:
    1. Mel loss       : MSE between predicted and target mel (before postnet)
    2. Postnet loss   : MSE between postnet output and target mel
    3. Gate loss      : BCE on stop token prediction
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, mel_out, mel_out_postnet, gate_out, 
                      mel_target, gate_target, mel_lens):
        """
        mel_out        : (B, 80, T) — decoder output
        mel_out_postnet: (B, 80, T) — postnet output
        gate_out       : (B, T)     — stop token logits
        mel_target     : (B, 80, T) — ground truth mel
        gate_target    : (B, T)     — ground truth stop token (1 at last frame)
        mel_lens       : (B,)       — actual mel lengths (for masking)
        """
        # ── Build padding mask ───────────────────────────────────────────
        # We don't want to compute loss on padded frames
        B, _, T = mel_target.shape
        mask = torch.arange(T, device=mel_lens.device).unsqueeze(0) < mel_lens.unsqueeze(1)
        # mask shape: (B, T)

        # Expand mask for mel: (B, 1, T) → (B, 80, T)
        mel_mask = mask.unsqueeze(1).expand_as(mel_target)

        # ── Mel losses (only on non-padded frames) ───────────────────────
        mel_loss     = self.mse(mel_out[mel_mask],         mel_target[mel_mask])
        postnet_loss = self.mse(mel_out_postnet[mel_mask], mel_target[mel_mask])

        # ── Gate loss ────────────────────────────────────────────────────
        gate_out    = gate_out[mask]
        gate_target = gate_target[mask]
        gate_loss   = self.bce(gate_out, gate_target)

        total_loss = mel_loss + postnet_loss + gate_loss
        return total_loss, mel_loss, postnet_loss, gate_loss


def build_gate_targets(mel_lens, max_len):
    """
    Creates stop token targets.
    0 everywhere except the last real frame which is 1.
    Shape: (B, T)
    """
    B = mel_lens.size(0)
    gate_targets = torch.zeros(B, max_len)
    for i, length in enumerate(mel_lens):
        gate_targets[i, length - 1] = 1.0
    return gate_targets

import os
import time
import json
from datetime import datetime
from tqdm import tqdm
from abc import ABC, abstractmethod

#  STRATEGY PATTERN — Abstract Base
# ─────────────────────────────────────────────
class TTSStrategy(ABC):
    """
    Abstract base class for TTS model strategies.
    Any new TTS model (FastSpeech2, VITS, etc.)
    must implement these three methods to plug into
    TTSManager without changing any other code.
    """

    @abstractmethod
    def load_model(self, device: torch.device):
        """Load and return the model, moved to device."""
        pass

    @abstractmethod
    def adapt_vocab(self, vocab_size: int):
        """Adapt the model's input layer to a new vocabulary size."""
        pass

    @abstractmethod
    def forward(self, batch: tuple):
        """
        Run one forward pass.
        Returns: (mel_out, mel_out_postnet, gate_out, alignments)
        """
        pass



#  STRATEGY — Tacotron2
class Tacotron2Strategy(TTSStrategy):
    """
    Concrete strategy for NVIDIA Tacotron2.
    Encapsulates all Tacotron2-specific logic so
    TTSManager stays model-agnostic.
    """

    def __init__(self):
        self.model = None

    def load_model(self, device: torch.device):
        print("Loading Tacotron2 from NVIDIA torch.hub...")
        self.model = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_tacotron2',
            model_math='fp32',
            pretrained=True,
            verbose=False,
        )
        self.model = self.model.to(device)
        print(f"Tacotron2 loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        return self.model

    def adapt_vocab(self, vocab_size: int):
        """Replace English embedding layer with Bangla vocab size."""
        assert self.model is not None, "Call load_model() first."
        old_emb       = self.model.embedding
        embedding_dim = old_emb.embedding_dim  # 512

        self.model.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0,
        ).to(next(self.model.parameters()).device)

        nn.init.xavier_uniform_(self.model.embedding.weight)
        print(f"Embedding adapted: vocab_size={vocab_size}, dim={embedding_dim}")

    def forward(self, batch: tuple):
        """
        batch: (text_padded, mel_padded, text_lens, mel_lens) — all on device
        Returns: (mel_out, mel_out_postnet, gate_out, alignments)
        """
        text_padded, mel_padded, text_lens, mel_lens = batch
        max_mel_len = mel_padded.shape[2]
        inputs      = (text_padded, text_lens, mel_padded, max_mel_len, mel_lens)
        return self.model(inputs)



#  STRATEGY — Placeholder for future models
class VITSStrategy(TTSStrategy):
    """
    Placeholder strategy for VITS.
    Swap this into TTSManager to switch models
    without touching any training/eval code.
    """

    def load_model(self, device: torch.device):
        raise NotImplementedError("VITSStrategy not implemented yet.")

    def adapt_vocab(self, vocab_size: int):
        raise NotImplementedError("VITSStrategy not implemented yet.")

    def forward(self, batch: tuple):
        raise NotImplementedError("VITSStrategy not implemented yet.")


class TTSManager:
    """
    Model-agnostic training manager.
    Accepts any TTSStrategy — swap the strategy to
    switch between Tacotron2, VITS, FastSpeech2, etc.
    """

    def __init__(self, strategy: TTSStrategy, train_loader, val_loader, config, device):
        self.strategy     = strategy
        self.model        = strategy.model   # direct ref for optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        self.criterion = Tacotron2Loss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = config["learning_rate"],
            weight_decay = config["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience = 3,
            factor   = 0.5,
        )

        self.experiment_log = []
        self.best_val_loss  = float("inf")
        self.checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, total_mel, total_post, total_gate = 0, 0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            if batch is None:   # Handles None Batch
               continue
            text_padded, mel_padded, text_lens, mel_lens = batch
            text_padded = text_padded.to(self.device)
            mel_padded  = mel_padded.to(self.device)
            text_lens   = text_lens.to(self.device)
            mel_lens    = mel_lens.to(self.device)

            max_mel_len  = mel_padded.shape[2]
            gate_targets = build_gate_targets(mel_lens, max_mel_len).to(self.device)

            # ← strategy handles forward, model-agnostically
            mel_out, mel_out_postnet, gate_out, _ = self.strategy.forward(
                (text_padded, mel_padded, text_lens, mel_lens)
            )

            loss, mel_loss, post_loss, gate_loss = self.criterion(
                mel_out, mel_out_postnet, gate_out,
                mel_padded, gate_targets, mel_lens
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_mel  += mel_loss.item()
            total_post += post_loss.item()
            total_gate += gate_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n = len(self.train_loader)
        return {
            "train_loss"     : total_loss / n,
            "train_mel_loss" : total_mel  / n,
            "train_post_loss": total_post / n,
            "train_gate_loss": total_gate / n,
        }

    def _val_epoch(self, epoch):
        self.model.eval()
        total_loss, total_mel, total_post, total_gate = 0, 0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for batch in pbar:
                if batch is None:   
                   continue
                text_padded, mel_padded, text_lens, mel_lens = batch
                text_padded = text_padded.to(self.device)
                mel_padded  = mel_padded.to(self.device)
                text_lens   = text_lens.to(self.device)
                mel_lens    = mel_lens.to(self.device)

                max_mel_len  = mel_padded.shape[2]
                gate_targets = build_gate_targets(mel_lens, max_mel_len).to(self.device)

                mel_out, mel_out_postnet, gate_out, _ = self.strategy.forward(
                    (text_padded, mel_padded, text_lens, mel_lens)
                )

                loss, mel_loss, post_loss, gate_loss = self.criterion(
                    mel_out, mel_out_postnet, gate_out,
                    mel_padded, gate_targets, mel_lens
                )

                total_loss += loss.item()
                total_mel  += mel_loss.item()
                total_post += post_loss.item()
                total_gate += gate_loss.item()

        n = len(self.val_loader)
        return {
            "val_loss"     : total_loss / n,
            "val_mel_loss" : total_mel  / n,
            "val_post_loss": total_post / n,
            "val_gate_loss": total_gate / n,
        }

    def _save_checkpoint(self, epoch, val_loss, train_loss, is_best=False):
        checkpoint = {
            "epoch"      : epoch,
            "val_loss"   : val_loss,
            "train_loss" : train_loss,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "char2idx"   : preprocessor.char2idx,
            "config"     : self.config,
            "strategy"   : self.strategy.__class__.__name__,
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pt")
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f" New best model saved (val_loss={val_loss:.4f})")

    def train(self, num_epochs, start_epoch=1):  # ← add start_epoch parameter
        print(f"\n{'='*55}")
        print(f"  Strategy     : {self.strategy.__class__.__name__}")
        print(f"  Epochs       : {num_epochs}")
        print(f"  Starting from: epoch {start_epoch}")
        print(f"  Device       : {self.device}")
        print(f"  Batch size   : {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"{'='*55}\n")
    
        for epoch in range(start_epoch, start_epoch + num_epochs):  # ← use start_epoch
            t0 = time.time()
    
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)
            elapsed       = time.time() - t0
    
            self.scheduler.step(val_metrics["val_loss"])
    
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
            self._save_checkpoint(
                epoch,
                val_metrics["val_loss"],
                train_metrics["train_loss"],
                is_best
            )
    
            log_entry = {
                "id"               : epoch,   # ← now correctly 6,7,8... instead of 1,2,3...
                "model_name"       : self.strategy.__class__.__name__,
                "hyperparameters"  : {
                    "lr"          : self.config["learning_rate"],
                    "batch_size"  : self.config["batch_size"],
                    "weight_decay": self.config["weight_decay"],
                },
                "train_loss"       : round(train_metrics["train_loss"], 4),
                "val_loss"         : round(val_metrics["val_loss"],     4),
                "objective_metrics": {},
                "timestamp"        : datetime.now().isoformat(),
                "epoch_time_sec"   : round(elapsed, 1),
            }
            self.experiment_log.append(log_entry)
    
            print(
                f"Epoch {epoch:03d} | "
                f"Train: {train_metrics['train_loss']:.4f} "
                f"(mel={train_metrics['train_mel_loss']:.4f} "
                f"post={train_metrics['train_post_loss']:.4f} "
                f"gate={train_metrics['train_gate_loss']:.4f}) | "
                f"Val: {val_metrics['val_loss']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
    
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        return self.experiment_log
