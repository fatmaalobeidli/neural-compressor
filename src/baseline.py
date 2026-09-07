"""
Computes gzip/bz2 baseline compression rates (bits per character)
for the corpus produced in tokenizer.py

Saves the results to results/baseline_metrics.json so train.py can pick
them up and include them in the combined benchmark table.

To run: 
    python src/baseline.py
"""

import gzip 
import bz2
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "baseline_metrics.json")

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

    gzip_bpc = bits_per_char(gzip_compressed, n_chars)
    bzip2_bpc = bits_per_char(bz2_compressed, n_chars)

    print(f"Corpus: {n_chars} chars, {len(raw_bytes)} raw bytes")
    print(f"gzip: {len(gzip_compressed)} bytes -> {gzip_bpc:.3f} bits/char")
    print(f"bzip2: {len(bz2_compressed)} bytes -> {bzip2_bpc:.3f} bits/char")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_chars": n_chars,
                "raw_bytes": len(raw_bytes),
                "gzip_bytes": len(gzip_compressed),
                "gzip_bpc": gzip_bpc,
                "bzip2_bytes": len(bz2_compressed),
                "bzip2_bpc": bzip2_bpc,
            },
            f,
            indent=2,
        )
    print(f"Saved baseline metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()
