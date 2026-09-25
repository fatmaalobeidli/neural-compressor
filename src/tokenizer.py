"""
Character-level tokenizer for the neural compressor project. Strips the Gutenberg
boilerplate from each book and concatenates them into data/corpus.txt.

Also splits every book into train (first 90%) and validation (last 10%),
so each book shows up in both sets. Written to data/train.txt and data/val.txt.

Reads 3 text files in data/, builds a vocabulary of unique characters,
and provides encode/decode functions between text and lists of integers.

Saved to data/vocab.json

To run:
    python src/tokenizer.py
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BOOK_FILES = ["pride_and_prejudice.txt", "frankenstein.txt", "alice_in_wonderland.txt"]
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
VAL_PATH = os.path.join(DATA_DIR, "val.txt")
VALIDATION_FRACTION = 0.1

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


def load_books() -> list[str]:
    """Read and clean every book file."""
    texts = []

    for filename in BOOK_FILES:
        path = os.path.join(DATA_DIR, filename)

        with open(path, encoding = "utf-8") as f:
            text = f.read()

        text = clean_gutenberg_text(text)
        texts.append(text)

    return texts


def load_corpus() -> str:
    """All books concatenated into one corpus."""
    return "\n\n".join(load_books())


def split_books(texts: list[str], fraction: float = VALIDATION_FRACTION) -> tuple[str, str]:
    """Hold out the last `fraction` of every book for validation."""
    train_parts, val_parts = [], []
    for text in texts:
        cut = int(len(text) * (1.0 - fraction))
        train_parts.append(text[:cut])
        val_parts.append(text[cut:])
    return "\n\n".join(train_parts), "\n\n".join(val_parts)

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
    """Save a text file with LF line endings."""
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
    books = load_books()
    text = "\n\n".join(books)

    chars, stoi, itos = build_vocab(text)

    print(f"Corpus length: {len(text)} characters")
    print(f"Vocab size: {len(chars)}")
    print(f"Vocab: {chars}")

    # Sanity check - must hold for the whole corpus. 
    encoded = encode(text, stoi)
    decoded = decode(encoded, itos)
    
    assert decoded == text, "Sanity check failed! encode/decode do not match."
    print("Sanity check passed: decode(encode(text)) == text")

    train_text, val_text = split_books(books)
    save_vocab(chars)
    save_corpus(text)
    save_corpus(train_text, TRAIN_PATH)
    save_corpus(val_text, VAL_PATH)

    print(f"Saved corpus to {CORPUS_PATH}")
    print(f"Saved train split ({len(train_text):,} chars) to {TRAIN_PATH}")
    print(f"Saved validation split ({len(val_text):,} chars) to {VAL_PATH}")
    print(f"Saved vocab to {VOCAB_PATH}")

if __name__ == "__main__":
    main()