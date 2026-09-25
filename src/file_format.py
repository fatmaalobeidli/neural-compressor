"""
Reads and writes the .nlc file format (see docs/FORMAT.md).

Characters the model has never seen are saved as-is in the header and
replaced by a space for the model, so any UTF-8 text can be compressed
without retraining.
"""

import hashlib
import json
import struct
from pathlib import Path

import torch

from lm_coder import CHECKPOINT_PATH, decode_ids, encode_ids, load_model

MAGIC = b"NLC\x01"


def read_utf8(raw: bytes) -> str:
    """Strict UTF-8 decode. Line endings and BOMs are kept as they are."""
    return raw.decode("utf-8")


def checkpoint_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_file(raw: bytes, checkpoint=CHECKPOINT_PATH, loaded=None):
    torch.set_num_threads(1)
    text = read_utf8(raw)
    model, chars, stoi = loaded if loaded is not None else load_model(checkpoint)
    placeholder = " " if " " in stoi else chars[0]
    escaped = []
    char_ids = []
    for position, ch in enumerate(text):
        if ch not in stoi:
            escaped.append([position, ch])
            ch = placeholder
        char_ids.append(stoi[ch])
    payload, ideal_bits = encode_ids(char_ids, model, len(chars))
    metadata = {
        "num_chars": len(text), "raw_bytes": len(raw),
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "escaped": escaped,
    }
    header = json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    body = MAGIC + struct.pack(">I", len(header)) + header + payload
    packed = body + hashlib.sha256(body).digest()
    metrics = {
        "num_chars": len(text), "raw_bytes": len(raw),
        "compressed_bytes": len(packed), "payload_bytes": len(payload),
        "metadata_bytes": len(packed) - len(payload),
        "bits_per_char": len(packed) * 8 / len(text) if text else None,
        "quantized_model_bits": ideal_bits, "escaped_chars": len(escaped),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "input_sha256": metadata["raw_sha256"],
    }
    return packed, metrics


def decode_file(packed: bytes, checkpoint=CHECKPOINT_PATH, loaded=None):
    if packed[:4] != MAGIC:
        raise ValueError("Not an .nlc file (unknown header)")
    if len(packed) < 41:
        raise ValueError("Truncated compressed file")
    body, digest = packed[:-32], packed[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("Container checksum mismatch: file is damaged or truncated")
    header_size = struct.unpack(">I", body[4:8])[0]
    if header_size > len(body) - 9:
        raise ValueError("Invalid header length")
    metadata = json.loads(body[8:8 + header_size])
    if metadata["checkpoint_sha256"] != checkpoint_hash(checkpoint):
        raise ValueError("Checkpoint mismatch: supply the exact original checkpoint with --checkpoint")
    count = metadata["num_chars"]
    if not isinstance(count, int) or count < 0 or not count <= metadata["raw_bytes"] <= 4 * count:
        raise ValueError("Invalid character/byte count")
    torch.set_num_threads(1)
    model, chars, _ = loaded if loaded is not None else load_model(checkpoint)
    text = [chars[i] for i in decode_ids(body[8 + header_size:], count, model, len(chars))]
    previous = -1
    for position, ch in metadata["escaped"]:
        if not isinstance(position, int) or not previous < position < count or not isinstance(ch, str) or len(ch) != 1:
            raise ValueError("Invalid escaped character")
        text[position] = ch
        previous = position
    raw = "".join(text).encode("utf-8")
    if len(raw) != metadata["raw_bytes"] or hashlib.sha256(raw).hexdigest() != metadata["raw_sha256"]:
        raise ValueError("Decoded checksum mismatch: model inference or file contents differ")
    return raw, metadata
