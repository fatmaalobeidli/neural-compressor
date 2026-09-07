"""
Train and evaluate the character-level GRU language model.

Uses non-overlapping chunks of the corpus per epoch (so training/validation
loss can be plotted against epoch number), tracks loss in both nats (native
to CrossEntropyLoss) and bits/char (comparable to the gzip/bzip2 numbers
from baseline.py).

Writes per-epoch metrics to results/training_log.csv, a loss curve plot to
results/plots/training_curve.png, and (if results/baseline_metrics.json
exists, i.e. baseline.py has already been run) a combined
results/benchmark_table.csv comparing gzip, bzip2, and the trained model.

Run from the project root:
    python src/train.py
"""

import csv
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "plots", "training_curve.png")
BENCHMARK_PATH = os.path.join(RESULTS_DIR, "benchmark_table.csv")
BASELINE_METRICS_PATH = os.path.join(RESULTS_DIR, "baseline_metrics.json")


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


def write_plot(history: list[dict]) -> None:
    """Plot train/val bits-per-char over epochs and save to PLOT_PATH."""
    epochs = [row["epoch"] for row in history]
    train_bpc = [row["train_bpc"] for row in history]
    val_bpc = [row["val_bpc"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_bpc, marker="o", label="train")
    ax.plot(epochs, val_bpc, marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("bits / character")
    ax.set_title("CharGRU training curve")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {PLOT_PATH}")


def write_benchmark_table(best_val_bpc: float) -> None:
    """Combine gzip/bzip2 numbers from baseline.py (if available) with the
    model's best validation BPC into results/benchmark_table.csv."""
    rows = [{"method": "neural_gru (val)", "bits_per_char": best_val_bpc}]

    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH, encoding="utf-8") as f:
            baseline = json.load(f)
        rows.insert(0, {"method": "bzip2", "bits_per_char": baseline["bzip2_bpc"]})
        rows.insert(0, {"method": "gzip", "bits_per_char": baseline["gzip_bpc"]})
    else:
        print(
            f"Warning: {BASELINE_METRICS_PATH} not found - run baseline.py first "
            "to include gzip/bzip2 in the benchmark table."
        )

    with open(BENCHMARK_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "bits_per_char"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark table to {BENCHMARK_PATH}")


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

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
    history = []

    print(f"Device: {device}")
    print(f"Corpus: {len(text):,} characters | Vocabulary: {len(chars)}")
    print(f"Training: {len(train_ids):,} | Validation: {len(val_ids):,}")

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=["epoch", "train_loss_nats", "train_bpc", "val_loss_nats", "val_bpc", "is_best"],
        )
        log_writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, device)
            is_best = val_loss < best_val_loss

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train {train_loss:.4f} nats ({bits_per_char(train_loss):.3f} bits/char) | "
                f"val {val_loss:.4f} nats ({bits_per_char(val_loss):.3f} bits/char)" )

            row = {
                "epoch": epoch,
                "train_loss_nats": train_loss,
                "train_bpc": bits_per_char(train_loss),
                "val_loss_nats": val_loss,
                "val_bpc": bits_per_char(val_loss),
                "is_best": is_best,
            }
            history.append(row)
            log_writer.writerow(row)
            log_file.flush()

            if is_best:
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
    print(f"Per-epoch log saved to {LOG_PATH}")

    write_plot(history)
    write_benchmark_table(bits_per_char(best_val_loss))


if __name__ == "__main__":
    main()
