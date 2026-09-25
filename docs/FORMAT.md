# .nlc file format (version 1)

```
"NLC" + version byte (0x01)
header length       4 bytes, big-endian
header              JSON (ASCII)
payload             arithmetic-coded bits, zero-padded to a full byte
SHA-256             32 bytes, over everything before it
```

The JSON header contains:

- `num_chars`, `raw_bytes`: length of the original text
- `checkpoint_sha256`: hash of the checkpoint used; decoding with a
  different one fails with an error
- `raw_sha256`: hash of the original bytes, checked after decoding
- `escaped`: list of `[position, character]` for characters not in the
  model's vocabulary

## Decoding checks

1. Magic bytes and version
2. Container checksum (catches damaged or truncated files)
3. Checkpoint hash
4. After decoding: length and SHA-256 of the original bytes

The output file is only written if all checks pass.

## Characters outside the vocabulary

The vocabulary is fixed when the model is trained (97 characters here).
Any other character is stored as-is in the header, and a space is fed to
the model and coded in its place. The decoder puts the original character
back afterwards. This means any UTF-8 text can be compressed without
retraining, but text with many unknown characters (for example another
script) gets expensive. This cost is included in the reported file size.

## Text handling

Input is read and written as raw bytes and must be valid UTF-8.
Line endings (LF, CRLF, CR, mixed), BOMs and empty files are preserved
exactly. No Unicode normalization is applied.

## Why the model runs one character at a time

During training, the hidden state is reset for every 128-character
sequence. During compression the hidden state carries through the whole
file, because that is the only way the decoder can reproduce it. So the
validation loss and the actual compressed size are close but not identical.

Encoding and decoding both use single-threaded CPU inference so the floating
point results match. Decoding on a different machine or PyTorch version is
not guaranteed to give the same probabilities; if it doesn't, the checksum
catches it.
