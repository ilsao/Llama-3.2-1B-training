"""Metadata-only pinned replacement acceptance contracts."""
import unittest
from slurm.cache_repair import require_decoder_agreement, require_upstream_match


class ReplacementTests(unittest.TestCase):
    def test_pinned_hash_and_size_both_required(self):
        upstream = {'sha256': 'a' * 64, 'size': 84}
        require_upstream_match(dict(upstream), upstream)
        for replacement in [{'sha256': 'b' * 64, 'size': 84}, {'sha256': 'a' * 64, 'size': 83}]:
            with self.assertRaises(ValueError):
                require_upstream_match(replacement, upstream)

    def test_noncryptographic_hash_not_accepted(self):
        with self.assertRaises(ValueError):
            require_upstream_match({'sha256': 'short', 'size': 1}, {'sha256': 'short', 'size': 1})

    def test_complete_decoder_byte_hash_agreement(self):
        first = {'passed': True, 'decoded_bytes': 10, 'decoded_sha256': 'a' * 64}
        require_decoder_agreement(first, dict(first))
        for second in [{**first, 'passed': False}, {**first, 'decoded_bytes': 9},
                       {**first, 'decoded_sha256': 'b' * 64}]:
            with self.assertRaises(ValueError):
                require_decoder_agreement(first, second)

    def test_empty_decode_is_not_a_replacement(self):
        empty = {'passed': True, 'decoded_bytes': 0, 'decoded_sha256': 'a' * 64}
        with self.assertRaises(ValueError):
            require_decoder_agreement(empty, dict(empty))
