"""
Compress a UTF-8 text file with the trained CharGRU + arithmetic coder.

Slow (pure Python, one character at a time), so use --chars to try it on
a short prefix first.

Run from the project root:
    python src/compress.py [--in-path PATH] [--out PATH] [--chars N] [--checkpoint PATH]
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_format import encode_file, read_utf8
from lm_coder import CHECKPOINT_PATH
from tokenizer import CORPUS_PATH

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "results", "compressed.nlc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default=CORPUS_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--chars", type=int, default=None,
                        help="only compress the first N characters (for quick tests)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if args.chars is not None and args.chars < 0:
        parser.error("--chars must be nonnegative")
    raw = Path(args.in_path).read_bytes()
    if args.chars is not None:
        raw = read_utf8(raw)[:args.chars].encode("utf-8")

    start = time.time()
    packed, metrics = encode_file(raw, args.checkpoint)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(packed)

    print(f"Wrote {args.out}: {len(packed):,} bytes")
    if metrics["bits_per_char"] is not None:
        print(f"{metrics['bits_per_char']:.4f} bits/char, {metrics['escaped_chars']} unknown characters")
    print(f"Took {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
