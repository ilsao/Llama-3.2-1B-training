"""Configuration checks only; no torch/model imports on the login node."""
import copy
import tempfile
from pathlib import Path
import unittest
from training_state import PROJECT, atomic_json, load_config


class OptimizerConfigurationTests(unittest.TestCase):
    def test_legacy_adamw_and_explicit_native_candidates(self):
        legacy = load_config(PROJECT / 'configs/train-pilot.json')
        self.assertNotIn('optimizer', legacy)
        for path in sorted((PROJECT / 'configs').glob('train-sweep-*.json')):
            cfg = load_config(path)
            self.assertIn(cfg['optimizer'], ('adamw', 'muon'))
            self.assertEqual(cfg['world_size'] * cfg['gradient_accumulation_steps'], 16)
            self.assertEqual(cfg['sequence_length'], 8192)
            self.assertEqual(cfg['max_h200_hours'], 2)

    def test_reject_invalid_or_unused_native_options(self):
        original = load_config(PROJECT / 'configs/train-sweep-muon-lr0.02.json')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for key, value in [('ns_steps', 0), ('ns_steps', True), ('momentum', 1),
                               ('epsilon', 0), ('learning_rate', float('nan')),
                               ('nesterov', 1), ('adjust_lr_fn', 'unknown'),
                               ('ns_coefficients', [3.4445, -4.775])]:
                cfg = copy.deepcopy(original)
                cfg['muon'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    atomic_json(path, cfg)
                    load_config(path)
            for optimizer in ('adamw', 'unknown'):
                cfg = {**original, 'optimizer': optimizer}
                atomic_json(path, cfg)
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
