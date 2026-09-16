"""
Decode a file written by compress.py back into text, using the exact same
trained CharGRU + step-by-step frequency-table logic (see lm_coder.py).

Verifies against the original text by default, so you know immediately
whether the round trip was lossless, this must never silently fail:
    decode(encode(text)) == text, every time.

Run from the project root:
    python src/decompress.py [--in-path PATH] [--out PATH] [--verify-against PATH]
"""

import argparse
import bisect
import os
import struct
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arithmetic_coder import ArithmeticDecoder
from lm_coder import StepModel, cumulative, load_model

DEFAULT_IN = os.path.join(os.path.dirname(__file__), "..", "results", "compressed.bin")


def decompress(data: bytes, num_chars: int, model, chars: list, progress_every: int = 0) -> str:
    step_model = StepModel(model, len(chars))
    decoder = ArithmeticDecoder(data)
    prev_char_id = None
    decoded = []
    start = time.time()

    for i in range(num_chars):
        freq = step_model.freq_for_next(prev_char_id)
        cum = cumulative(freq)
        total = cum[-1]
        target = decoder.get_cum_freq(total)
        char_id = bisect.bisect_right(cum, target) - 1
        decoder.decode_symbol(cum[char_id], cum[char_id + 1], total)
        decoded.append(chars[char_id])
        prev_char_id = char_id

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  {i + 1:,}/{num_chars:,} chars ({elapsed:.1f}s, {rate:.0f} chars/s)")

    return "".join(decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default=DEFAULT_IN)
    parser.add_argument("--out", default=None, help="if set, write the decoded text here")
    parser.add_argument(
        "--verify-against", default=None,
        help="path to the original text file; compares its first N characters against the decoded output",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)

    with open(args.in_path, "rb") as f:
        raw = f.read()
    num_chars = struct.unpack(">I", raw[:4])[0]
    payload = raw[4:]

    model, chars, _ = load_model()
    print(f"Loaded model | vocab={len(chars)} | decoding {num_chars:,} characters")

    start = time.time()
    text = decompress(payload, num_chars, model, chars, progress_every=max(1, num_chars // 10))
    elapsed = time.time() - start
    print(f"Decoded {num_chars:,} characters in {elapsed:.1f}s")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote decoded text to {args.out}")

    if args.verify_against:
        with open(args.verify_against, encoding="utf-8") as f:
            original = f.read()[:num_chars]
        if original == text:
            print("Lossless check PASSED: decode(encode(text)) == text")
        else:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(original, text)) if a != b),
                min(len(original), len(text)),
            )
            print(f"Lossless check FAILED: first mismatch at character {first_diff}")
            print(f"  expected: {original[max(0, first_diff - 20):first_diff + 20]!r}")
            print(f"  got:      {text[max(0, first_diff - 20):first_diff + 20]!r}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
