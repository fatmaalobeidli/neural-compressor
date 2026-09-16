"""
End-to-end neural compression: encode a text file using the trained
CharGRU's per-character predictions as the arithmetic coder's probability
model (lm_coder.py).

The model processes one character at a time and carries its hidden state
forward, matching decompression exactly to prevent decoding errors. 

Keeping both sides on this identical step-by-step code path
(single-threaded, CPU only) is what makes the model's predicted
distribution - and so its quantized frequency table - bit-for-bit
identical during encoding and decoding.

This is slow, so use --chars to test on a shorter prefix.

Output format (also read by decompress.py):
    4 bytes  - number of characters encoded, big-endian uint32
    rest     - the arithmetic-coded bitstream

Run from the project root:
    python src/compress.py [--chars N] [--out PATH] [--in-path PATH]
"""

import argparse
import os
import struct
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arithmetic_coder import ArithmeticEncoder
from lm_coder import StepModel, cumulative, load_model
from tokenizer import CORPUS_PATH

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "results", "compressed.bin")


def compress(text: str, model, stoi: dict, progress_every: int = 0) -> bytes:
    step_model = StepModel(model, len(stoi))
    encoder = ArithmeticEncoder()
    prev_char_id = None
    start = time.time()

    for i, ch in enumerate(text):
        char_id = stoi[ch]
        freq = step_model.freq_for_next(prev_char_id)
        cum = cumulative(freq)
        encoder.encode_symbol(cum[char_id], cum[char_id + 1], cum[-1])
        prev_char_id = char_id

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  {i + 1:,}/{len(text):,} chars ({elapsed:.1f}s, {rate:.0f} chars/s)")

    return encoder.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chars", type=int, default=None,
        help="compress only the first N characters of the input (for quick testing)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--in-path", default=CORPUS_PATH)
    args = parser.parse_args()

    torch.set_num_threads(1)  # keep CPU float ops on a single deterministic path (see module docstring)

    with open(args.in_path, encoding="utf-8") as f:
        text = f.read()
    if args.chars is not None:
        text = text[: args.chars]

    model, chars, stoi = load_model()
    print(f"Loaded model | vocab={len(chars)} | compressing {len(text):,} characters")

    start = time.time()
    payload = compress(text, model, stoi, progress_every=max(1, len(text) // 10))
    elapsed = time.time() - start

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack(">I", len(text)))
        f.write(payload)

    bpc = len(payload) * 8 / len(text) if text else 0.0
    print(
        f"Wrote {args.out}: {len(payload):,} bytes for {len(text):,} chars "
        f"({bpc:.3f} bits/char) in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
