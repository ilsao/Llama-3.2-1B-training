"""Pure configuration/stop estimates; no torch/model imports on login."""
import copy
from pathlib import Path
import tempfile
import unittest
from training_state import PROJECT, atomic_json, load_config, digest
from performance import TrainingPerformance


class PerformanceConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(PROJECT/'configs/be60p20-lrscaled-2b-20261010.json')
        self.settings = dict(model_compile=True, compile_mode='max-autotune',
                             muon_compile=True, async_transfer=True)

    def checked(self, settings):
        cfg = copy.deepcopy(self.cfg)
        cfg['performance'] = settings
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cfg.json'; atomic_json(path, cfg)
            return load_config(path)

    def test_old_config_digest_and_disabled_behavior(self):
        self.assertEqual(digest(self.cfg), 'c2ec561c2bae73988d292fa9fe1bc7093265a3c1a6f401930dceedea40da46bf')
        self.assertFalse(TrainingPerformance(self.cfg).enabled)

    def test_supported_settings(self):
        self.assertTrue(TrainingPerformance(self.checked(self.settings)).enabled)

    def test_bad_settings(self):
        for change in ({'model_compile':1}, {'compile_mode':'unknown'}, {'async_transfer':'yes'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.checked({**self.settings, **change})
        with self.assertRaises(ValueError): self.checked({'model_compile':True})

    def test_compile_outlier_expires_only_after_twenty_steps(self):
        perf = TrainingPerformance({'performance':self.settings})
        value = perf.observe(100, 0)
        for _ in range(19): value = perf.observe(1, value)
        self.assertEqual(value, 100)
        self.assertEqual(perf.observe(1, value), 1)

    def test_eager_maximum_is_preserved(self):
        self.assertEqual(TrainingPerformance({}).observe(1,100), 100)


if __name__ == '__main__': unittest.main()
