"""Compare actual compressed sizes on identical UTF-8 bytes and verify recovery."""

import argparse
import bz2
import csv
import gzip
import json
import platform
import time
from pathlib import Path

import torch

from file_format import encode_file, decode_file, read_utf8
from lm_coder import CHECKPOINT_PATH, load_model
from tokenizer import VAL_PATH

ROOT = Path(__file__).resolve().parents[1]


def benchmark(raw, checkpoint=CHECKPOINT_PATH):
    text = read_utf8(raw)
    if not text:
        raise ValueError("Benchmark input must be nonempty")
    loaded = load_model(checkpoint)
    start = time.perf_counter()
    packed, metrics = encode_file(raw, checkpoint, loaded)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    restored, _ = decode_file(packed, checkpoint, loaded)
    decode_seconds = time.perf_counter() - start
    assert restored == raw, "Neural byte round trip failed"
    rows = []
    for method, encode, decode in [
        ("gzip", lambda b: gzip.compress(b, compresslevel=9, mtime=0), gzip.decompress),
        ("bzip2", lambda b: bz2.compress(b, compresslevel=9), bz2.decompress),
    ]:
        start = time.perf_counter()
        compressed = encode(raw)
        seconds = time.perf_counter() - start
        start = time.perf_counter()
        assert decode(compressed) == raw
        decoding = time.perf_counter() - start
        rows.append(dict(method=method, compressed_bytes=len(compressed),
                         bits_per_char=len(compressed)*8/len(text),
                         encode_seconds=seconds, decode_seconds=decoding))
    rows.append(dict(method="neural_gru", compressed_bytes=len(packed),
                     bits_per_char=metrics["bits_per_char"],
                     encode_seconds=encode_seconds, decode_seconds=decode_seconds))
    metrics.update(checkpoint_bytes=Path(checkpoint).stat().st_size,
                   model_storage_included=False, byte_roundtrip_verified=True,
                   python_version=platform.python_version(), torch_version=str(torch.__version__),
                   neural_bits_per_char_with_checkpoint=(len(packed)+Path(checkpoint).stat().st_size)*8/len(text))
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", help="UTF-8 file to compress; default: data/val.txt")
    parser.add_argument("--chars", type=int, help="optional prefix length within the selected sample")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    if args.chars is not None and args.chars <= 0:
        parser.error("--chars must be positive")
    path = Path(args.in_path or VAL_PATH)
    selected = read_utf8(path.read_bytes())
    if args.chars is not None:
        selected = selected[:args.chars]
    rows, metrics = benchmark(selected.encode("utf-8"), args.checkpoint)
    metrics.update(source=path.name,
                   sample_kind="custom" if args.in_path else "validation (last 10% of each book, used for checkpoint selection)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "benchmark_metadata.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["method"] for r in rows], [r["bits_per_char"] for r in rows])
    ax.set_ylabel("Bits per character (complete files)")
    ax.set_title(f"Same {len(selected):,}-character sample; shared neural model")
    fig.tight_layout()
    (out / "plots").mkdir(exist_ok=True)
    fig.savefig(out / "plots" / "compression_comparison.png", dpi=150)
    plt.close(fig)
    for row in rows:
        print(f"{row['method']}: {row['compressed_bytes']:,} bytes, {row['bits_per_char']:.4f} BPC")
    print("All three byte round trips passed. Neural checkpoint storage is reported separately.")


if __name__ == "__main__":
    main()
