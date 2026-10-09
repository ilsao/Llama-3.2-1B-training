"""Dependency-free checks of the one-time continuation contract."""
import copy
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock
from slurm.continue_best_once import extended_config
from slurm.raw_continue_once_v2 import ManagedBatches


class ContinuationTests(TestCase):
    def test_extension_preserves_optimizer_geometry_and_wsd(self):
        path=Path(__file__).resolve().parents[1]/'.worktrees/muon-sweep/configs/baselines/muon-wsd-raw-20261009.json'
        original=json.loads(path.read_text());before=copy.deepcopy(original);cfg=extended_config(original)
        self.assertEqual(original,before)
        allowed={'max_steps','max_h200_hours','eval_steps','fixed_horizon','final_evaluation_reserve_seconds'}
        self.assertEqual({k for k in cfg if cfg[k]!=original[k]},allowed)
        self.assertEqual(cfg['scheduler_steps'],8500)
        self.assertEqual(cfg['world_size'],8)
        self.assertEqual(cfg['max_training_tokens'],6_000_000_000)
        self.assertLessEqual(cfg['eval_steps'],(cfg['max_steps']-original['max_steps'])//10)
        self.assertLessEqual(13.62+8*4200/3600,cfg['max_h200_hours'])

    def test_raw_workers_close_on_early_stop_and_exception(self):
        for fail in (False,True):
            batches=object.__new__(ManagedBatches);batches.close=Mock()
            try:
                with batches as active:
                    self.assertIs(active,batches)
                    if fail:raise ValueError('Stop before publishing a new prefix')
            except ValueError:pass
            batches.close.assert_called_once_with()
