"""File-level regression tests using a small seeded model, without training."""
import hashlib
import json
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from file_format import encode_file, decode_file
from model import CharGRU


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.checkpoint = cls.folder / "model.pt"
        torch.manual_seed(7)
        chars = list(" abcdefghijklmnopqrstuvwxyz\n")
        config = dict(vocab_size=len(chars), embed_dim=8, hidden_dim=12, num_layers=1)
        model = CharGRU(**config)
        torch.save(dict(config=config, chars=chars, model_state_dict=model.state_dict()), cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_bytes(self):
        rng = random.Random(11)
        samples = [b"", b"a", b"a"*500, b"hello\nworld\n", b"hello\r\nworld\r\n",
                   b"a\rb\nc\r\nd\x00", "\ufeffcafe\u0301 \u00e9 \U0001f600 \u4e2d\u6587".encode(),
                   "".join(rng.choice("ab \n\r\U0001f600") for _ in range(500)).encode()]
        for raw in samples:
            with self.subTest(raw=raw[:30]):
                packed, metrics = encode_file(raw, self.checkpoint)
                decoded, _ = decode_file(packed, self.checkpoint)
                self.assertEqual(raw, decoded)
                self.assertEqual(metrics["compressed_bytes"], len(packed))

    def test_corruption_and_wrong_model(self):
        packed, _ = encode_file(b"hello world", self.checkpoint)
        for bad in (packed[:-1], packed[:20], packed + b"x", packed[:12]+bytes([packed[12]^1])+packed[13:], b"old format"):
            with self.assertRaises(ValueError):
                decode_file(bad, self.checkpoint)
        wrong = self.folder / "wrong.pt"
        wrong.write_bytes(self.checkpoint.read_bytes()+b"x")
        with self.assertRaisesRegex(ValueError, "Checkpoint mismatch"):
            decode_file(packed, wrong)

    def test_decoded_checksum(self):
        packed, _ = encode_file(b"hello world", self.checkpoint)
        size = struct.unpack(">I", packed[4:8])[0]
        metadata = json.loads(packed[8:8+size])
        metadata["raw_sha256"] = "0"*64
        header = json.dumps(metadata, separators=(",", ":")).encode()
        body = packed[:4]+struct.pack(">I",len(header))+header+packed[8+size:-32]
        with self.assertRaisesRegex(ValueError, "Decoded checksum mismatch"):
            decode_file(body+hashlib.sha256(body).digest(), self.checkpoint)

    def test_cli_paths_and_full_verification(self):
        original = self.folder / "input.txt"
        original.write_bytes(b"hello\nworld\r\n")
        def run(script, *args):
            return subprocess.run([sys.executable, str(ROOT/'src'/script), '--checkpoint',str(self.checkpoint),*args],
                                  cwd=self.folder, capture_output=True, text=True)
        result = run('compress.py','--in-path','input.txt','--out','packed.bin')
        self.assertEqual(result.returncode,0,result.stderr)
        result = run('decompress.py','--in-path','packed.bin','--out','nested/output.txt','--verify-against','input.txt')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(original.read_bytes(),(self.folder/'nested/output.txt').read_bytes())
        original.write_bytes(original.read_bytes()+b"extra")
        self.assertNotEqual(run('decompress.py','--in-path','packed.bin','--verify-against','input.txt').returncode,0)
        self.assertEqual(run('decompress.py','--in-path','packed.bin','--verify-against','input.txt','--verify-prefix').returncode,0)


if __name__ == "__main__":
    unittest.main()
