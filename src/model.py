"""
Character-level language model for the neural compressor project.

Predicts a probability distribution over the next character given
a sequence of preceding characters. Trained with cross-entropy loss,
which is directly interpretable as bits-per-character (see train.py).
"""

import torch
from torch import nn


class CharGRU(nn.Module):
    """Predict the next character at every position in a sequence."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.num_layers = num_layers

        # Maps every char ID to a learned embed_dim-dimensional vector.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # batch_first=True so input/output tensors are shaped (batch, seq, features).
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """ x: Character IDs with shape (batch, sequence_length).
            hidden: Optional GRU state.

           Returns:
            logits: Scores with shape (batch, sequence_length, vocab_size).
            hidden: Final GRU state.
        """
        embedded = self.embedding(x)    # (batch, seq_len, embed_dim)
        gru_output, hidden = self.gru(embedded, hidden)
        logits = self.output(gru_output)
        return logits, hidden