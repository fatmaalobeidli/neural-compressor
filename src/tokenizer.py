"""
Character-level tokenizer for the neural compressor project. 

Reads 3 text files in data/, builds a vocabulary of unique characters,
and provides encode/decode functions between text and lists of integers.

To run:
    python src/tokenizer.py
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BOOK_FILES = ["pride_and_prejudice.txt", "frankenstein.txt", "alice_in_wonderland.txt"]
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")

def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start = text.find(start_marker)
    assert start != -1, f"Start marker not found - check this book's format: {start_marker!r}"
    start = text.find("\n", start)
    text = text[start + 1:]

    end = text.find(end_marker)
    assert end != -1, f"End marker not found - check this book's format: {end_marker!r}"
    text = text[:end]

    return text.strip()


def load_corpus() -> str:
    """Read, clean, and concatenate all book files into one corpus."""
    texts = []

    for filename in BOOK_FILES:
        path = os.path.join(DATA_DIR, filename)

        with open(path, encoding = "utf-8") as f:
            text = f.read()

        text = clean_gutenberg_text(text)
        texts.append(text)

    return "\n\n".join(texts)

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
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(chars, f, ensure_ascii=False)

def save_corpus(text: str, path: str = CORPUS_PATH) -> None:
    """Save the concatenated corpus so downstream scripts don't re-implement load_corpus()."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

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
    save_corpus(text)

    print(f"Saved corpus to {CORPUS_PATH}")
    print(f"Saved vocab to {VOCAB_PATH}")

if __name__ == "__main__":
    main()