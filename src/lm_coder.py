"""
Shared logic between compress.py and decompress.py: loading the trained
CharGRU checkpoint and converting its per-step softmax output into the
integer table the arithmetic coder needs (arithmetic_coder.py).

This lives in one place on purpose. compress.py and decompress.py MUST
compute the exact same frequency table at every step (both derive it from
the same model, given the same preceding characters), or the arithmetic
code becomes undecodable - importing one shared function instead of two
separately-typed copies is what guarantees that.
"""

import os

import torch
import torch.nn.functional as F

from model import CharGRU

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "char_gru.pt")
TOTAL = 1 << 16  # frequency precision (must stay under arithmetic_coder.MAX_TOTAL = 2**30)


def load_model(checkpoint_path: str = CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CharGRU(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    return model, chars, stoi


def quantize_probs(probs: list[float], total: int = TOTAL) -> list[int]:
    """Turn a probability distribution into integer frequencies summing to
    exactly 'total', with every symbol getting at least 1 count so nothing
    is ever unencodable, however unlikely the model thinks it
    is. Deterministic gives the same 'probs' - that determinism is what
    compress.py and decompress.py rely on to end up with the same table.
    """
    raw = [p * total for p in probs]
    freq = [max(1, int(r)) for r in raw]
    diff = total - sum(freq)

    if diff > 0:
        # Give the extra counts to the symbols whose floor() dropped the most 
        # probability mass, largest remainder first.
        order = sorted(range(len(freq)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in range(diff):
            freq[order[i % len(order)]] += 1
    elif diff < 0:
        # Only possible when many symbols got the "at least 1" floor
        # bumped above their true share (e.g. a huge vocab). Take counts
        #back from the currently-largest buckets first.
        order = sorted(range(len(freq)), key=lambda i: freq[i], reverse=True)
        i = 0
        while diff < 0:
            idx = order[i % len(order)]
            if freq[idx] > 1:
                freq[idx] -= 1
                diff += 1
            i += 1

    assert sum(freq) == total
    return freq


def cumulative(freq: list[int]) -> list[int]:
    cum = [0] * (len(freq) + 1)
    for i, f in enumerate(freq):
        cum[i + 1] = cum[i] + f
    return cum


def uniform_freq(vocab_size: int, total: int = TOTAL) -> list[int]:
    """The frequency table used for the very first character of a text,
    which has no preceding context for the model to condition on."""
    return quantize_probs([1.0 / vocab_size] * vocab_size, total)


class StepModel:
    """Produces one next-character distribution at a time, carrying the
    GRU's hidden state across calls. compress.py and decompress.py must
    each do this incrementally (rather than through batched forward pass)
    because decompress.py cannot see characters it hasn't decoded yet -
    running both sides through the identical step-by-step code path is
    what keeps their floating-point results (and their frequency tables) 
    identical.
    """

    def __init__(self, model: CharGRU, vocab_size: int, total: int = TOTAL) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.total = total
        self.hidden = None

    def freq_for_next(self, prev_char_id: int | None) -> list[int]:
        """Distribution for the character that comes after 'prev_char_id', 
        or the uniform distribution if 'prev_char_id' is None (predicting 
        the very first character of the text). Advances the hidden state 
        by feeding 'prev_char_id' through the model, so call this exactly 
        once per character position, in order.
        """
        if prev_char_id is None:
            return uniform_freq(self.vocab_size, self.total)
        with torch.no_grad():
            x = torch.tensor([[prev_char_id]])
            logits, self.hidden = self.model(x, self.hidden)
            probs = F.softmax(logits.squeeze(0).squeeze(0), dim=-1).tolist()
        return quantize_probs(probs, self.total)
