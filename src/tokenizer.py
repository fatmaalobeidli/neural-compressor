"""
Character-level tokenizer for the neural compressor project. 

Reads all text files in data/, builds a vocabulary of unique characters,
and provides encode/decode functions between text and lists of integers.

To run:
    python src/tokenizer.py
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BOOK_FILES = ["pride_and_prejudice.txt", "frankenstein.txt", "alice_in_wonderland.txt"]
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

def load_corpus() -> str:
    """Read and concatenate all book files into one big string."""
    text = ""

    for filename in BOOK_FILES:
        path = os.path.join(DATA_DIR, filename)

        with open(path, encoding = "utf-8") as f:
            text += f.read()

    return text

def build_vocab(text: str):
    """Return (chars, stoi, itos) built from the unique characters in text."""
    chars = sorted(set(text))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return chars, stoi, itos

def encode(text: str, stoi: dict) -> list[int]:
    """Convert text into a list of integer character IDs."""
    return [stoi[ch] for ch in text]

def decode(ids: list[int], itos: dict) -> str:
    """Convert a list of integer character IDs into text."""
    return "".join(itos[i] for i in ids)

def save_vocab(chars: list, path: str = VOCAB_PATH) -> None:
    """Save the sorted character list - stoi.itos can be rebuilt from it."""
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)


def load_vocab(path: str = VOCAB_PATH):
    """Rebuild (chars, stoi, itos) from a saved vocab file."""
    with open(path, encoding = "utf-8") as f:
            chars = json.load(f)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def main():
    text = load_corpus()

    chars, stoi, itos = build_vocab(text)

    print(f"Corpus length: {len(text)} characters")
    print(f"Vocab size: {len(chars)}")
    print(f"Vocab: {chars}")

    # Sanity check - must hold for the whole corpus. 
    encoded = encode(text, stoi)
    decoded = decode(encoded, itos)
    
    assert decoded == text, "Sanity check failed! encode/decode do not match."
    print("Sanity check passed: decode(encode(text)) == text")

    save_vocab(chars)
    print(f"Saved vocab to {VOCAB_PATH}")

if __name__ == "__main__":
    main()