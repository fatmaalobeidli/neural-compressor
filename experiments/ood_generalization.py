"""
Out-of-distribution experiment: compress texts that are more and more
different from the training books with the same frozen model, and compare
against gzip and bzip2 on each one.

Targets (from closest to furthest from the training data):
    in-domain     held-out part of the training books (data/val.txt)
    austen        Sense and Sensibility, unseen book by the same author as Pride and Prejudice
    wikipedia     modern English (Wikipedia article "Large language model")
    german        Die Verwandlung by Kafka, in German
    python_code   argparse.py from the Python standard library

Texts are downloaded once into data/ood/. Every target is cut to the same
number of characters so the rows are comparable.

Saves the results to results/ood/ood_results.csv and the plot to
results/plots/ood_comparison.png.

Run from the project root:
    python experiments/ood_generalization.py [--chars N]
    python experiments/ood_generalization.py --plot-only    (only redraw the plot)
"""

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark import benchmark
from lm_coder import CHECKPOINT_PATH
from tokenizer import VAL_PATH, clean_gutenberg_text

OOD_DIR = ROOT / "data" / "ood"
RESULTS_DIR = ROOT / "results" / "ood"
PLOT_PATH = ROOT / "results" / "plots" / "ood_comparison.png"

GUTENBERG_URL = "https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
    "action": "query", "prop": "extracts", "explaintext": 1,
    "titles": "Large language model", "format": "json", "formatversion": 2,
})
# wikipedia refuses requests without a user agent
HEADERS = {"User-Agent": "neural-compressor (university project)"}


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def get_gutenberg(book_id: int) -> str:
    return clean_gutenberg_text(fetch(GUTENBERG_URL.format(book_id=book_id)))


def get_wikipedia() -> str:
    data = json.loads(fetch(WIKIPEDIA_URL))
    return data["query"]["pages"][0]["extract"]


def get_python_code() -> str:
    # a copy is saved in data/ood/, so the result doesn't change with the python version
    return Path(argparse.__file__).read_text(encoding="utf-8")


TARGETS = [
    ("in-domain", lambda: Path(VAL_PATH).read_text(encoding="utf-8")),
    ("austen", lambda: get_gutenberg(161)),
    ("wikipedia", get_wikipedia),
    ("german", lambda: get_gutenberg(22367)),
    ("python_code", get_python_code),
]


def load_target(name: str, source) -> str:
    """Load a target from data/ood/, downloading it the first time."""
    path = OOD_DIR / f"{name}.txt"
    if not path.exists():
        print(f"Getting {name} ...")
        text = source().replace("\r\n", "\n")
        path.write_text(text, encoding="utf-8", newline="\n")
    return path.read_text(encoding="utf-8")


def write_plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["target"] for row in rows]
    xs = range(len(rows))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([x - width for x in xs], [row["gzip_bpc"] for row in rows], width, label="gzip", color="tab:blue")
    ax.bar(list(xs), [row["bzip2_bpc"] for row in rows], width, label="bzip2", color="tab:orange")
    # neural bar in two parts: the model's predictions, and on top the cost of
    # characters that are not in the vocabulary (plus header and coder overhead)
    model = [row["model_bpc"] for row in rows]
    ax.bar([x + width for x in xs], model, width, label="neural: model predictions", color="tab:green")
    ax.bar([x + width for x in xs], [row["overhead_bpc"] for row in rows], width, bottom=model,
           label="neural: unknown characters", color="white", edgecolor="tab:green", hatch="//")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names)
    ax.set_ylabel("bits / character")
    ax.set_title(f"Same model on different texts ({int(rows[0]['chars']):,} characters each)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {PLOT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chars", type=int, default=50_000, help="characters per target")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--plot-only", action="store_true", help="redraw the plot from the saved csv")
    args = parser.parse_args()

    if args.plot_only:
        with (RESULTS_DIR / "ood_results.csv").open(encoding="utf-8") as f:
            rows = [{k: v if k == "target" else float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        write_plot(rows)
        return

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, source in TARGETS:
        text = load_target(name, source)
        if len(text) < args.chars:
            raise SystemExit(f"{name} only has {len(text):,} characters, use a smaller --chars")
        sample = text[:args.chars]

        print(f"Compressing {name} ...")
        results, metrics = benchmark(sample.encode("utf-8"), args.checkpoint)
        bpc = {row["method"]: row["bits_per_char"] for row in results}
        n = metrics["num_chars"]

        rows.append({
            "target": name,
            "chars": n,
            "gzip_bpc": bpc["gzip"],
            "bzip2_bpc": bpc["bzip2"],
            "neural_bpc": bpc["neural_gru"],
            # model cost only, without the stored unknown characters
            "model_bpc": metrics["quantized_model_bits"] / n,
            # everything else: header, checksums, unknown characters, coder overhead
            "overhead_bpc": bpc["neural_gru"] - metrics["quantized_model_bits"] / n,
            "unknown_chars": metrics["escaped_chars"],
            "unknown_pct": 100 * metrics["escaped_chars"] / n,
        })
        r = rows[-1]
        print(f"  gzip {r['gzip_bpc']:.3f} | bzip2 {r['bzip2_bpc']:.3f} | neural {r['neural_bpc']:.3f} "
              f"(model {r['model_bpc']:.3f}, {r['unknown_chars']:,} unknown chars)")

    out_path = RESULTS_DIR / "ood_results.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    print(f"Saved results to {out_path}")

    write_plot(rows)


if __name__ == "__main__":
    main()
