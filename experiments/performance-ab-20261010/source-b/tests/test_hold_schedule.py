"""Pure scheduler contracts: no model, dataset, GPU, or workers."""
import copy
from pathlib import Path
import tempfile
import unittest

from training_state import (PROJECT, atomic_json, hold_schedule_boundaries,
                            load_config, lr_multiplier, schedule_multiplier)

SETTINGS = dict(first_decay_ratio=.2, hold_ratio=.1,
                final_decay_ratio=.1, plateau_lr_ratio=.1)


class HoldScheduleTests(unittest.TestCase):
    def test_phase_boundaries_and_terminal_zero(self):
        def value(step):
            return lr_multiplier(step, 100, 10, 0, 'wsd_hold_zero', .4, SETTINGS)
        self.assertEqual(hold_schedule_boundaries(100, 10, SETTINGS), (60, 80, 90, 100))
        self.assertEqual(value(0), .1)
        for step in (9, 10, 59, 60):
            self.assertEqual(value(step), 1)
        self.assertAlmostEqual(value(79), .1)
        for step in range(80, 91):
            self.assertEqual(value(step), .1)
        self.assertEqual(value(99), 0)
        self.assertEqual(value(120), 0)
        values = [value(step) for step in range(10, 100)]
        self.assertTrue(all(a >= b - 1e-15 for a, b in zip(values, values[1:])))

    def test_rounding_and_reconstructed_scheduler(self):
        cfg = load_config(PROJECT / 'configs/training-diagnosis-20261010/e-hold-zero-seed42.json')
        self.assertEqual(hold_schedule_boundaries(3051, 170, SETTINGS), (1831, 2441, 2746, 3051))
        self.assertEqual(schedule_multiplier(3050, cfg), 0)
        for step in (170, 1831, 2440, 2441, 2745, 2746, 3049, 3050):
            self.assertEqual(schedule_multiplier(step, cfg), schedule_multiplier(step, copy.deepcopy(cfg)))

    def test_rejects_invalid_phases_and_ignored_settings(self):
        original = load_config(PROJECT / 'configs/training-diagnosis-20261010/e-hold-zero-seed42.json')
        cases = []
        for key, value in [('plateau_lr_ratio', 0), ('hold_ratio', float('nan')),
                           ('first_decay_ratio', True), ('hold_ratio', .00001),
                           ('final_decay_ratio', .8)]:
            cfg = copy.deepcopy(original);cfg['hold_schedule'][key] = value;cases.append(cfg)
        for key, value in [('min_lr_ratio', .1), ('fixed_horizon', False),
                           ('wsd_decay_ratio', .2), ('lr_scheduler', 'wsd')]:
            cfg = copy.deepcopy(original);cfg[key] = value;cases.append(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for index, cfg in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    atomic_json(path, cfg)
                    load_config(path)
        with self.assertRaises(ValueError):
            hold_schedule_boundaries(100, 70, SETTINGS)

    def test_existing_schedule_values_are_unchanged(self):
        self.assertEqual(lr_multiplier(60, 100, 10, .1, 'wsd', .4), 1)
        self.assertAlmostEqual(lr_multiplier(99, 100, 10, .1, 'wsd', .4), .1)
        self.assertAlmostEqual(lr_multiplier(99, 100, 10, .1, 'linear'), .1)
        self.assertAlmostEqual(lr_multiplier(99, 100, 10, .1, 'cosine'), .1)


if __name__ == '__main__':
    unittest.main()
