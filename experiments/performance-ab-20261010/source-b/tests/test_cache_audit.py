"""Tiny mocked decoder/cursor contracts; no real corpus, libraries or workers."""
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from slurm.cache_audit import audit_one, validate_cursor


class CacheAuditTests(unittest.TestCase):
    def test_decode_error_identifies_exact_file_and_manifest_hash(self):
        class Broken(io.BytesIO):
            def read(self, *args):
                raise ValueError('Data corruption detected')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'one.zst'
            path.write_bytes(b'bad')
            entry = {'path': str(path), 'repo_filename': 'data/source/one.jsonl.zst',
                     'sha256': hashlib.sha256(b'bad').hexdigest(), 'size': 3}
            decoder = SimpleNamespace(stream_reader=lambda *a, **k: Broken())
            library = SimpleNamespace(ZstdDecompressor=lambda: decoder)
            with patch.dict('sys.modules', {'zstandard': library}):
                result = audit_one((entry, float('inf')))
            self.assertEqual(result['status'], 'decode_or_io_error')
            self.assertTrue(result['manifest_hash_matches'])
            self.assertEqual(result['repo_filename'], entry['repo_filename'])
            self.assertEqual(result['error'], 'Data corruption detected')
            validate_cursor({'next_file': 1, 'records': [result]}, [entry])

    def test_completed_decode_hash_mismatch_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'one.zst'
            path.write_bytes(b'file')
            entry = {'path': str(path), 'repo_filename': 'one', 'sha256': 'different', 'size': 4}
            decoder = SimpleNamespace(stream_reader=lambda *a, **k: io.BytesIO(b'decoded'))
            with patch.dict('sys.modules', {'zstandard': SimpleNamespace(ZstdDecompressor=lambda: decoder)}):
                result = audit_one((entry, float('inf')))
                stopped = audit_one((entry, 0))
            self.assertEqual(result['status'], 'hash_mismatch')
            self.assertEqual(result['decoded_bytes'], 7)
            self.assertEqual(stopped['status'], 'interrupted')
            with self.assertRaises(ValueError):
                validate_cursor({'next_file': 1, 'records': [stopped]}, [entry])

    def test_cursor_cannot_skip_or_mix_file_order(self):
        source = {'path': 'a', 'repo_filename': 'a', 'sha256': 'hash', 'size': 5}
        record = {'path': 'a', 'repo_filename': 'a', 'expected_sha256': 'hash',
                  'expected_size': 5, 'status': 'ok'}
        with self.assertRaises(ValueError):
            validate_cursor({'next_file': 2, 'records': [record]}, [source])
        with self.assertRaises(ValueError):
            validate_cursor({'next_file': 1, 'records': [{**record, 'repo_filename': 'b'}]}, [source])
