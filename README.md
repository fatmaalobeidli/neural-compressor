# Neural Text Compressor

Lossless text compression with a small character-level language model and an
arithmetic coder written from scratch. The model predicts the next character,
the arithmetic coder turns those predictions into bits. The better the
predictions, the smaller the file.

The project compares this against gzip and bzip2 and checks Shannon's source
coding theorem in practice: the compressed size should be almost exactly the
model's cross-entropy on the text.

## Results

Trained on three Project Gutenberg books (*Pride and Prejudice*,
*Frankenstein*, *Alice's Adventures in Wonderland*, 1,292,652 characters,
97 distinct characters). All three methods compress the same 129,266
characters from the validation split, and all three decompress to exactly
the original bytes.

| Method | Compressed size (bytes) | Bits per character |
|---|---:|---:|
| gzip (level 9) | 47,843 | 2.961 |
| bzip2 (level 9) | 38,505 | 2.383 |
| **Neural (GRU + arithmetic coding)** | **37,909** | **2.346** |

![Compression comparison](results/plots/compression_comparison.png)

![Training curve](results/plots/training_curve.png)

**Shannon's theorem in practice.** The information content of the sample under
the model (the sum of -log2 p over every character, using the exact
probabilities the coder sees) is 301,195.6 bits. The arithmetic coder
produced 301,200 bits, only 4.4 bits more for the whole sample.
The remaining 259 bytes of the file are the header and checksums.

**Caveats**

- The model (about 334k parameters, 1.3 MB) is not counted in the file size.
  That is the usual setup when both sides already share the model. Counting
  it would make a file this size much bigger than gzip's output.
- The validation split is the last 10% of the corpus, which is almost
  entirely *Alice*. The model mostly trained on the other two books, so this
  is closer to a cross-book test than a standard held-out split. That
  largely explains the gap between training (1.76 bits/char) and validation
  (2.37 bits/char).
- The same split was used to pick the best epoch, so it is not a fully
  untouched test set.
- Speed: about 8,000 characters per second in pure Python, compared to
  milliseconds for gzip and bzip2.

## How it works

1. **Model** (`src/model.py`): embedding (128), 1-layer GRU (256), linear
   layer, trained with cross-entropy on sequences of 128 characters
   (`src/train.py`). Cross-entropy in nats divided by ln 2 gives the model's
   estimate of bits per character.
2. **Arithmetic coder** (`src/arithmetic_coder.py`): 32-bit integer
   implementation following Witten, Neal and Cleary (1987), with its own
   self-tests on fixed distributions.
3. **Connecting the two** (`src/lm_coder.py`): at each step the GRU's
   softmax output is rounded to integer frequencies summing to 65,536
   (every character gets at least 1) and passed to the coder. The decoder
   runs the exact same steps, so it rebuilds the same table before
   decoding each character.
4. **File format** (`src/file_format.py`): small header plus SHA-256
   checksums, so a wrong model or a damaged file gives an error instead of
   wrong text. Details are in [docs/FORMAT.md](docs/FORMAT.md).

## Running it

Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Full pipeline:

```powershell
python src/download.py            # download the three books
python src/tokenizer.py           # clean them, build data/corpus.txt and data/vocab.json
python src/baseline.py            # gzip/bzip2 on the full corpus
python src/train.py               # train, writes checkpoints/char_gru.pt
python src/benchmark.py           # results table and plot above
```

Tests:

```powershell
python -m unittest discover -s tests -v
python src/arithmetic_coder.py
```

Compress and decompress a file:

```powershell
python src/compress.py --in-path data/corpus.txt --out results/corpus.nlc
python src/decompress.py --in-path results/corpus.nlc --out results/restored.txt --verify-against data/corpus.txt
```

For a quick try on a prefix:

```powershell
python src/compress.py --chars 20000 --out results/prefix.nlc
python src/decompress.py --in-path results/prefix.nlc --verify-against data/corpus.txt --verify-prefix
```

`benchmark.py` also works on any other UTF-8 file:

```powershell
python src/benchmark.py --in-path path/to/text.txt --out-dir results/other
```

A compressed file can only be decoded with the exact checkpoint that
created it. Retraining overwrites `checkpoints/char_gru.pt`, so keep a copy
if you need to read older files.

## Repository layout

```text
neural-compressor/
├── data/                  books, cleaned corpus, vocabulary
├── checkpoints/           trained model
├── docs/FORMAT.md         .nlc file format
├── experiments/           out-of-distribution experiment (in progress)
├── results/
│   ├── baseline_metrics.json
│   ├── training_log.csv
│   ├── benchmark_table.csv
│   ├── benchmark_metadata.json
│   └── plots/
├── src/
│   ├── download.py
│   ├── tokenizer.py
│   ├── baseline.py
│   ├── model.py
│   ├── train.py
│   ├── arithmetic_coder.py
│   ├── lm_coder.py
│   ├── file_format.py
│   ├── compress.py
│   ├── decompress.py
│   └── benchmark.py
├── tests/test_pipeline.py
└── requirements.txt
```

## Next steps

- Out-of-distribution experiment: compress text from other domains
  (another language, source code, modern news) with the same model and
  measure how much worse it gets.
- Hold out the end of each book instead of the end of the corpus, for a
  cleaner in-domain result.
- Full writeup of the theory and results.
