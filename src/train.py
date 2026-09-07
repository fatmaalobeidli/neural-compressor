"""
Train and evaluate the character-level GRU language model.

Uses non-overlapping chunks of the corpus per epoch (so training/validation
loss can be plotted against epoch number), tracks loss in both nats (native 
to CrossEntropyLoss) and bits/char (comparable to the gzip/bzip2 numbers 
from baseline.py).

Run from the project root:
    python src/train.py
"""

import math
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import CharGRU
from tokenizer import load_vocab, CORPUS_PATH

# hyperparameters
SEED = 42
SEQUENCE_LENGTH = 128
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
LEARNING_RATE = 3e-3
GRAD_CLIP = 1.0
EPOCHS = 10
VALIDATION_FRACTION = 0.1

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "char_gru.pt")


class CharacterDataset(Dataset):
    """Non-overlapping (input, target) chunks from a list of encoded token IDs."""

    def __init__(self, token_ids: list[int], sequence_length: int) -> None:
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.sequence_length = sequence_length
        self.num_sequences = (len(self.tokens) - 1) // sequence_length
        if self.num_sequences == 0:
            raise ValueError("Text is too short for the chosen sequence length.")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.sequence_length
        stop = start + self.sequence_length
        x = self.tokens[start:stop]
        y = self.tokens[start + 1:stop + 1]
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bits_per_char(loss_nats: float) -> float:
    """CrossEntropyLoss returns nats (natural log). Convert to bits."""
    return loss_nats / math.log(2)


def run_epoch(
    model: CharGRU,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,) -> float:
    """Run one pass over 'loader'. Trains if optimizer is given, else evaluates.
    Returns the loss averaged over all characters seen (not a naive per-batch mean,
    since the last batch of an epoch can be smaller than BATCH_SIZE)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_characters = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits, _ = model(x)  # hidden=None -> zeroed GRU state each batch
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()

        count = y.numel()
        total_loss += loss.item() * count
        total_characters += count

    return total_loss / total_characters


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    chars, stoi, _ = load_vocab()
    with open(CORPUS_PATH, encoding="utf-8") as f:
        text = f.read()
    encoded = [stoi[ch] for ch in text]

    split = int(len(encoded) * (1.0 - VALIDATION_FRACTION))
    train_ids, val_ids = encoded[:split], encoded[split:]

    train_dataset = CharacterDataset(train_ids, SEQUENCE_LENGTH)
    val_dataset = CharacterDataset(val_ids, SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = CharGRU(
        vocab_size=len(chars),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Corpus: {len(text):,} characters | Vocabulary: {len(chars)}")
    print(f"Training: {len(train_ids):,} | Validation: {len(val_ids):,}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train {train_loss:.4f} nats ({bits_per_char(train_loss):.3f} bits/char) | "
            f"val {val_loss:.4f} nats ({bits_per_char(val_loss):.3f} bits/char)" )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "chars": chars,
                    "config": {
                        "vocab_size": len(chars),
                        "embed_dim": EMBED_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                    },
                    "val_loss": val_loss,
                    "val_bits_per_char": bits_per_char(val_loss),
                    "epoch": epoch,
                },
                CHECKPOINT_PATH, )
            print(f"  Saved improved checkpoint to {CHECKPOINT_PATH}")

    print(f"\nBest validation: {bits_per_char(best_val_loss):.3f} bits/char")
    print("Compare against baseline.py: gzip 2.945 bits/char, bzip2 2.165 bits/char")


if __name__ == "__main__":
    main()