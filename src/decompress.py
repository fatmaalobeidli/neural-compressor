"""
Decompress a .nlc file written by compress.py.

The file stores a SHA-256 of the original bytes, so a bad decode always
raises an error instead of silently returning wrong text.
With --verify-against you can also compare against the original file.

Run from the project root:
    python src/decompress.py [--in-path PATH] [--out PATH] [--verify-against PATH [--verify-prefix]]
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_format import decode_file, read_utf8
from lm_coder import CHECKPOINT_PATH

DEFAULT_IN = os.path.join(os.path.dirname(__file__), "..", "results", "compressed.nlc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default=DEFAULT_IN)
    parser.add_argument("--out", default=None, help="write the decoded text here")
    parser.add_argument("--verify-against", default=None,
                        help="original file to compare with (whole file unless --verify-prefix)")
    parser.add_argument("--verify-prefix", action="store_true",
                        help="only compare the part that was compressed (used with --chars)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()
    if args.verify_prefix and not args.verify_against:
        parser.error("--verify-prefix requires --verify-against")

    start = time.time()
    raw, metadata = decode_file(Path(args.in_path).read_bytes(), args.checkpoint)

    if args.verify_against:
        original = Path(args.verify_against).read_bytes()
        if args.verify_prefix:
            original = read_utf8(original)[:metadata["num_chars"]].encode("utf-8")
        if original != raw:
            raise SystemExit("Lossless check FAILED: decoded bytes differ from the original")
        print("Lossless check passed: decoded bytes match the original")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(raw)
        print(f"Wrote {args.out}")
    print(f"Decoded {metadata['num_chars']:,} characters, checksum OK ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
