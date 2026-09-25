"""
A from-scratch, bit-level arithmetic coder (the classic Witten-Neal-Cleary
algorithm, CACM 1987).

Both sides track an interval [low, high]. Encoding a symbol narrows it to
the fraction given by that symbol's (cum_low, cum_high, total) - integer
cumulative frequencies out of a fixed total. Decoding reverses this: find
where the encoded value falls inside [low, high], rescale it to
[0, total), and look up which symbol's range contains it.

Whenever low and high already share their top bit - or nearly do, when the
interval straddles the midpoint - that bit is emitted and both sides shift
left by one. This renormalization keeps enough precision no matter how
long the input is.

Works with any alphabet and any distribution: static (see
StaticFrequencyModel below, used for the self-tests) or one that changes
every symbol, like a language model's prediction (see lm_coder.py, where the model is
wired in).

Run this file directly for its self-tests:
    python src/arithmetic_coder.py
"""

from __future__ import annotations

import bisect

PRECISION = 32
FULL = 1 << PRECISION
HALF = FULL >> 1
QUARTER = FULL >> 2
THREE_QUARTER = 3 * QUARTER

# the frequency total has to stay well below FULL, otherwise a symbol can end
# up with a zero-width interval and can't be encoded. total <= QUARTER is the
# usual limit for this algorithm (Witten, Neal, Cleary).
MAX_TOTAL = QUARTER


class BitWriter:
    """Collects bits MSB-first and packs them into bytes (zero-padded)."""

    def __init__(self) -> None:
        self._bits: list[int] = []

    def write_bit(self, bit: int) -> None:
        self._bits.append(bit & 1)

    def to_bytes(self) -> bytes:
        bits = self._bits + [0] * ((-len(self._bits)) % 8)
        out = bytearray(len(bits) // 8)
        for i, bit in enumerate(bits):
            if bit:
                out[i // 8] |= 1 << (7 - i % 8)
        return bytes(out)


class BitReader:
    """Reads bits MSB-first from a byte string. Reading past the end
    returns 0 (equivalent to an infinite zero-padded tail), which is what
    the decoder's lookahead window needs while finishing the last symbols."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        byte_index = self._pos // 8
        self._pos += 1
        if byte_index >= len(self._data):
            return 0
        bit_index = 7 - ((self._pos - 1) % 8)
        return (self._data[byte_index] >> bit_index) & 1


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = FULL - 1
        self.pending_bits = 0
        self.writer = BitWriter()

    def _emit_bit(self, bit: int) -> None:
        self.writer.write_bit(bit)
        for _ in range(self.pending_bits):
            self.writer.write_bit(1 - bit)
        self.pending_bits = 0

    def encode_symbol(self, cum_low: int, cum_high: int, total: int) -> None:
        """Narrow [low, high] to the sub-interval for cumulative frequency
        range [cum_low, cum_high) out of `total`."""
        assert 0 <= cum_low < cum_high <= total <= MAX_TOTAL, (
            f"invalid frequency range ({cum_low}, {cum_high}, {total})"
        )
        span = self.high - self.low + 1
        self.high = self.low + (span * cum_high) // total - 1
        self.low = self.low + (span * cum_low) // total

        while True:
            if self.high < HALF:
                self._emit_bit(0)
            elif self.low >= HALF:
                self._emit_bit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending_bits += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        """Flush enough bits to disambiguate the final interval. Must be
        called exactly once, after the last encode_symbol call."""
        self.pending_bits += 1
        if self.low < QUARTER:
            self._emit_bit(0)
        else:
            self._emit_bit(1)
        return self.writer.to_bytes()


class ArithmeticDecoder:
    def __init__(self, data: bytes) -> None:
        self.low = 0
        self.high = FULL - 1
        self.reader = BitReader(data)
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self.reader.read_bit()

    def get_cum_freq(self, total: int) -> int:
        """Where `value` currently falls inside [low, high], rescaled to
        [0, total). Call this first to find out which symbol was encoded,
        then call decode_symbol with that symbol's range."""
        span = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * total - 1) // span
        return min(total - 1, scaled)

    def decode_symbol(self, cum_low: int, cum_high: int, total: int) -> None:
        """Advance the decoder state exactly as encode_symbol did on the
        encoder side for the same (cum_low, cum_high, total)."""
        span = self.high - self.low + 1
        self.high = self.low + (span * cum_high) // total - 1
        self.low = self.low + (span * cum_low) // total

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.reader.read_bit()


class StaticFrequencyModel:
    """A fixed (non-adaptive) integer frequency table over an alphabet.

    Used here to unit-test the coder itself - both with a uniform
    distribution and with a skewed one - independent of any language
    model. lm_coder.py uses a per-step table built from the trained
    CharGRU's predictions instead, with the same encoder/decoder.
    """

    def __init__(self, freqs: dict[str, int]) -> None:
        self.symbols = list(freqs.keys())
        self.freq = [freqs[s] for s in self.symbols]
        self.total = sum(self.freq)
        assert self.total <= MAX_TOTAL, "scale down frequencies: total exceeds MAX_TOTAL"
        self.cumulative = [0] * (len(self.freq) + 1)
        for i, f in enumerate(self.freq):
            self.cumulative[i + 1] = self.cumulative[i] + f
        self.index = {s: i for i, s in enumerate(self.symbols)}

    @classmethod
    def uniform(cls, alphabet: str) -> "StaticFrequencyModel":
        return cls({ch: 1 for ch in alphabet})

    def range_for_symbol(self, symbol: str) -> tuple[int, int, int]:
        i = self.index[symbol]
        return self.cumulative[i], self.cumulative[i + 1], self.total

    def symbol_for_cum_freq(self, cum_freq: int) -> str:
        i = bisect.bisect_right(self.cumulative, cum_freq) - 1
        return self.symbols[i]


def encode_text(text: str, model: StaticFrequencyModel) -> bytes:
    encoder = ArithmeticEncoder()
    for ch in text:
        cum_low, cum_high, total = model.range_for_symbol(ch)
        encoder.encode_symbol(cum_low, cum_high, total)
    return encoder.finish()


def decode_text(data: bytes, model: StaticFrequencyModel, length: int) -> str:
    decoder = ArithmeticDecoder(data)
    chars = []
    for _ in range(length):
        cf = decoder.get_cum_freq(model.total)
        symbol = model.symbol_for_cum_freq(cf)
        cum_low, cum_high, total = model.range_for_symbol(symbol)
        decoder.decode_symbol(cum_low, cum_high, total)
        chars.append(symbol)
    return "".join(chars)


if __name__ == "__main__":
    import random

    random.seed(0)

    print("--- Uniform-distribution round-trip tests ---")
    tests = [
        "a",
        "aaaaaaaaaa",
        "hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "".join(random.choice("abcdefghij") for _ in range(2000)),
        "".join(chr(random.randint(32, 126)) for _ in range(5000)),
    ]
    for text in tests:
        alphabet = "".join(sorted(set(text)))
        encoded = encode_text(text, StaticFrequencyModel.uniform(alphabet))
        decoded = decode_text(encoded, StaticFrequencyModel.uniform(alphabet), len(text))
        assert decoded == text, f"Round-trip failed for text starting {text[:30]!r}"
        print(
            f"  len={len(text):>6} | alphabet size={len(alphabet):>3} | "
            f"encoded={len(encoded):>6} bytes vs raw={len(text.encode('utf-8')):>6} bytes"
        )

    print("--- Skewed (non-uniform) static frequency table test ---")
    freqs = {"a": 10, "b": 8, "c": 6, "d": 4, "e": 2, "f": 1}
    text = "".join(random.choice("aaaaaaaaaabbbbbbbbccccccddddeef") for _ in range(3000))
    encoded = encode_text(text, StaticFrequencyModel(freqs))
    decoded = decode_text(encoded, StaticFrequencyModel(freqs), len(text))
    assert decoded == text
    print(f"  len={len(text)} chars -> {len(encoded)} bytes ({len(encoded) * 8 / len(text):.3f} bits/char)")

    print("--- Edge cases ---")
    single_symbol_model = StaticFrequencyModel({"x": 1})
    text = "x" * 500
    encoded = encode_text(text, single_symbol_model)
    decoded = decode_text(encoded, single_symbol_model, len(text))
    assert decoded == text
    print(f"  single-symbol alphabet OK ({len(encoded)} bytes for {len(text)} chars)")

    print("\nAll arithmetic_coder.py self-tests passed.")
