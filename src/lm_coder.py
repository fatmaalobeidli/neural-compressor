"""
Connects the trained CharGRU to the arithmetic coder.

The encoder and decoder have to build exactly the same frequency table at
every step, otherwise the output can't be decoded. That's why both sides
use the functions in this file instead of having their own copies.
"""

import bisect
import math
import os

import torch
import torch.nn.functional as F

from arithmetic_coder import ArithmeticDecoder, ArithmeticEncoder
from model import CharGRU

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "char_gru.pt")
TOTAL = 1 << 16  # frequency precision, has to stay below arithmetic_coder.MAX_TOTAL (2**30)


def load_model(checkpoint_path: str = CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CharGRU(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    return model, chars, stoi


def quantize_probs(probs: list[float], total: int = TOTAL) -> list[int]:
    """Turn the model's probabilities into integer frequencies that sum to
    exactly `total`. Every symbol gets at least 1, so nothing is ever
    impossible to encode. The result only depends on `probs`, so encoder
    and decoder get the same table.
    """
    if not probs or total < len(probs):
        raise ValueError("Frequency total must accommodate every symbol")
    if any(not math.isfinite(p) or p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
        raise ValueError("Expected a finite normalized probability distribution")
    raw = [p * total for p in probs]
    freq = [max(1, int(r)) for r in raw]
    diff = total - sum(freq)

    if diff > 0:
        # hand out the leftover counts to the largest rounding remainders
        order = sorted(range(len(freq)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in range(diff):
            freq[order[i % len(order)]] += 1
    elif diff < 0:
        # only happens when the minimum of 1 pushed us over the total,
        # take the extra counts back from the biggest entries
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
    """Table for the first character, where the model has no context yet."""
    return quantize_probs([1.0 / vocab_size] * vocab_size, total)


class StepModel:
    """Runs the GRU one character at a time and keeps the hidden state
    between calls. The decoder can only see characters it has already
    decoded, so the encoder has to work the same way.
    """

    def __init__(self, model: CharGRU, vocab_size: int, total: int = TOTAL) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.total = total
        self.hidden = None

    def freq_for_next(self, prev_char_id: int | None) -> list[int]:
        """Frequency table for the next character. Pass None for the first
        character. Call once per position, in order."""
        if prev_char_id is None:
            return uniform_freq(self.vocab_size, self.total)
        with torch.no_grad():
            x = torch.tensor([[prev_char_id]])
            logits, self.hidden = self.model(x, self.hidden)
            probs = F.softmax(logits.squeeze(0).squeeze(0), dim=-1).tolist()
        return quantize_probs(probs, self.total)


def encode_ids(char_ids: list[int], model: CharGRU, vocab_size: int) -> tuple[bytes, float]:
    """Arithmetic-code a list of character IDs.

    Returns the payload and the ideal code length in bits
    (sum of -log2 p over the quantized tables actually used).
    """
    step = StepModel(model, vocab_size)
    encoder = ArithmeticEncoder()
    prev = None
    ideal_bits = 0.0
    for char_id in char_ids:
        cum = cumulative(step.freq_for_next(prev))
        ideal_bits -= math.log2((cum[char_id + 1] - cum[char_id]) / cum[-1])
        encoder.encode_symbol(cum[char_id], cum[char_id + 1], cum[-1])
        prev = char_id
    return encoder.finish(), ideal_bits


def decode_ids(payload: bytes, count: int, model: CharGRU, vocab_size: int) -> list[int]:
    """Inverse of encode_ids."""
    step = StepModel(model, vocab_size)
    decoder = ArithmeticDecoder(payload)
    prev = None
    out = []
    for _ in range(count):
        cum = cumulative(step.freq_for_next(prev))
        target = decoder.get_cum_freq(cum[-1])
        char_id = bisect.bisect_right(cum, target) - 1
        decoder.decode_symbol(cum[char_id], cum[char_id + 1], cum[-1])
        out.append(char_id)
        prev = char_id
    return out
