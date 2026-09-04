"""
Computes gzip/bz2 baseline compression rates (bits per character)
for the corpus produced in tokenizer.py

To run: 
    python src/baseline.py
"""

import gzip 
import bz2
import os
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")

def load_corpus(path: str = CORPUS_PATH) -> str:
    """Read the pre-built corpus written by tokenizer.py."""
    with open(path, encoding="utf-8") as f:
        return f.read()

def bits_per_char(compressed: bytes, n_chars: int) -> float:
    return (len(compressed) * 8) / n_chars

def main():
    text = load_corpus()
    raw_bytes = text.encode("utf-8")
    n_chars = len(text)

    gzip_compressed = gzip.compress(raw_bytes, compresslevel=9)
    bz2_compressed = bz2.compress(raw_bytes, compresslevel=9)


    print(f"Corpus: {n_chars} chars, {len(raw_bytes)} raw bytes")
    print(f"gzip: {len(gzip_compressed)} bytes -> {bits_per_char(gzip_compressed, n_chars):.3f} bits/char")
    print(f"bzip2: {len(bz2_compressed)} bytes -> {bits_per_char(bz2_compressed, n_chars):.3f} bits/char")

if __name__ == "__main__":
    main()

