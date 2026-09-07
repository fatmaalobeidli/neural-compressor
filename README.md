# Neural Text Compressor

A small character-level language-model project that explores neural text
compression. It downloads three public-domain books, prepares them as a single
text corpus, measures traditional compression baselines, and trains a GRU to
predict the next character.

The model is evaluated using **bits per character (BPC)**. A lower BPC means
that the text can theoretically be represented using fewer bits.

> **Important:** This project currently estimates the neural model's
> theoretical compression rate. It does not yet create a compressed binary
> file. Producing a real compressed file would require connecting the model's
> probabilities to an entropy coder, such as arithmetic coding or range coding.

## Project overview

The project uses the following Project Gutenberg books:

- *Pride and Prejudice* by Jane Austen — Gutenberg ID 1342
- *Frankenstein* by Mary Shelley — Gutenberg ID 84
- *Alice's Adventures in Wonderland* by Lewis Carroll — Gutenberg ID 11

The processing pipeline is:

1. Download the books.
2. Remove the Project Gutenberg headers and footers.
3. Combine the cleaned books into one corpus.
4. Build a character-level vocabulary.
5. Measure gzip and bzip2 compression rates.
6. Train a GRU to predict each next character.
7. Compare the GRU's validation BPC with the traditional baselines.

## Project structure

```text
neural-compressor/
├── data/
│   ├── pride_and_prejudice.txt
│   ├── frankenstein.txt
│   ├── alice_in_wonderland.txt
│   ├── corpus.txt
│   └── vocab.json
├── checkpoints/
│   └── char_gru.pt          # best model so far, by validation loss
├── results/
│   ├── baseline_metrics.json   # written by baseline.py
│   ├── training_log.csv        # written by train.py, one row per epoch
│   ├── benchmark_table.csv     # written by train.py: gzip vs bzip2 vs model
│   └── plots/
│       └── training_curve.png  # written by train.py
├── experiments/              # reserved for the OOD generalization experiment (not started yet)
├── src/
│   ├── download.py
│   ├── tokenizer.py
│   ├── baseline.py
│   ├── model.py
│   └── train.py
└── README.md
```

Files inside `data/`, `checkpoints/`, and `results/` are generated as the
scripts are executed and are gitignored. The three book files are created by
`download.py`; `corpus.txt` and `vocab.json` are created by `tokenizer.py`;
`char_gru.pt` is created by `train.py`.

## What each Python file does

### `download.py`

Downloads the three UTF-8 book files from Project Gutenberg and stores them in
`data/`. Existing files are skipped, so rerunning the script does not download
them again.

### `tokenizer.py`

Reads and cleans the downloaded books, combines them into `corpus.txt`, and
builds a vocabulary containing every unique character in the corpus.

It also provides mappings between characters and integer IDs:

```text
character -> integer ID -> character
```

The ordered vocabulary is saved in `vocab.json` so the same mapping can be used
during training and when loading a saved model.

### `baseline.py`

Compresses `corpus.txt` with gzip and bzip2 and reports their sizes in bits per
character. These results provide traditional compression baselines for the GRU.
The numbers are also written to `results/baseline_metrics.json`, which
`train.py` reads to build the combined benchmark table.

### `model.py`

Defines the `CharGRU` neural network. It contains:

- An embedding layer that converts character IDs into learned vectors.
- A GRU that processes preceding characters and maintains a hidden state.
- A linear output layer that produces one next-character score for every
  character in the vocabulary.

The model returns logits and the final hidden state. Cross-entropy loss later
converts the logits into a measure of prediction error.

### `train.py`

Loads the corpus and vocabulary, converts the text into character IDs, and
divides it into training and validation sections. It creates shifted input and
target sequences such as:

```text
Input:   hell
Target:  ello
```

The script trains the GRU with cross-entropy loss, reports training and
validation BPC after every epoch, and saves the best checkpoint as
`checkpoints/char_gru.pt`.

It also writes, on every run:

- `results/training_log.csv` — one row per epoch (train/val loss in nats and
  bits/char, and whether that epoch improved the checkpoint).
- `results/plots/training_curve.png` — train vs. validation BPC over epochs.
- `results/benchmark_table.csv` — gzip, bzip2 (from
  `results/baseline_metrics.json`, if `baseline.py` has been run first) and
  the model's best validation BPC, side by side.

It automatically uses a CUDA GPU when one is available; otherwise, it trains on
the CPU.

## Requirements

- Python 3.10 or newer
- PyTorch
- matplotlib (for the training curve plot)
- An internet connection for the initial book download

No external packages are needed for downloading, tokenizing, or measuring the
traditional baselines because those scripts use Python's standard library.

## Installation

Create and activate a virtual environment from the project root.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch matplotlib
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch matplotlib
```

For a platform-specific GPU installation, use the installation command provided
by the official PyTorch installation selector.

## Usage

Run every command from the project root and in the following order.

### 1. Download the books

```bash
python src/download.py
```

### 2. Clean and tokenize the corpus

```bash
python src/tokenizer.py
```

This generates `data/corpus.txt` and `data/vocab.json`. The script also verifies
that decoding the encoded corpus reproduces the original corpus exactly.

### 3. Measure the traditional baselines

```bash
python src/baseline.py
```

Example output:

```text
Corpus: 1400000 chars, 1410000 raw bytes
gzip: 500000 bytes -> 2.857 bits/char
bzip2: 410000 bytes -> 2.343 bits/char
Saved baseline metrics to results/baseline_metrics.json
```

The exact values depend on the downloaded and cleaned text. Run this before
`train.py` so the benchmark table below includes gzip/bzip2.

### 4. Train the neural model

```bash
python src/train.py
```

Example training output:

```text
Epoch 01/10 | train loss 2.1034, BPC 3.035 | validation loss 1.8521, BPC 2.672
...
Best validation: 2.371 bits/char
Per-epoch log saved to results/training_log.csv
Saved training curve to results/plots/training_curve.png
Saved benchmark table to results/benchmark_table.csv
```

The checkpoint is updated whenever the validation loss improves. The most
recent run (epoch 9/10) reached **2.371 bits/char** on held-out text, beating
the gzip baseline (2.945) but not yet bzip2 (2.165).

## Understanding bits per character

For traditional compression, BPC is calculated as:

```text
BPC = compressed size in bits / number of text characters
```

For the neural model, cross-entropy measures how surprised the model is by each
correct next character. Because PyTorch reports cross-entropy in natural-log
units, it is converted to bits with:

```python
bpc = loss / math.log(2)
```

Lower values are better. For example, a result of `1.80 BPC` is better than
`2.40 BPC`.

The fairest neural result is the **validation BPC**, not the training BPC. The
training BPC measures performance on text the model has already learned from,
while validation BPC measures performance on held-out text.

## Training configuration

The main settings can be changed near the top of `train.py`:

```python
SEQUENCE_LENGTH = 128
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
LEARNING_RATE = 3e-3
GRAD_CLIP = 1.0
EPOCHS = 10
VALIDATION_FRACTION = 0.1
```

Larger models or longer training may improve predictions, but they also require
more memory and computation and may overfit a small corpus.

## Saved checkpoint

The saved `checkpoints/char_gru.pt` checkpoint contains:

- The model's learned parameters
- The character vocabulary
- The model configuration
- The best validation loss and BPC
- The epoch at which it was saved

Saving the vocabulary with the model is important because character IDs must
have exactly the same meanings when the model is loaded again.

## Current limitations

- The GRU estimates theoretical coding cost but is not yet connected to an
  entropy coder.
- The corpus is small and contains only three English books.
- The validation set comes from the end of the combined corpus, so results can
  depend on the order in which the books are combined.
- Model BPC and gzip/bzip2 BPC are useful comparisons, but they do not include
  exactly the same overhead. A deployable neural compressor would also need to
  account for its model and metadata sizes.

## Possible extensions

- Add arithmetic or range coding to produce real compressed files.
- Add decompression and verify that the original text is recovered exactly.
- Evaluate on a completely separate, unseen book.
- Compare GRU and LSTM architectures under the same training settings.
- Add text generation to inspect what the model has learned.
- Experiment with model size, sequence length, learning rate, and dropout.

## Data source

The books are downloaded from [Project Gutenberg](https://www.gutenberg.org/)
and are used as public-domain text data. Refer to each downloaded file and the
Project Gutenberg website for the applicable terms and notices.
