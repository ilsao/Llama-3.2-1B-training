"""Exact fourth-run authorization and submission guards; no Slurm/model calls."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from slurm import submit_training as cli
from training_state import (BE60P20_2B_RUN_ID, PROJECT, ExperimentLedger,
                            be60p20_2b_authorized, load_config, read_json)


class BE60P20AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(PROJECT / 'configs/be60p20-lrscaled-2b-20261010.json')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = ExperimentLedger(self.root)

    def seed_three_formal_runs(self):
        historical = load_config(PROJECT / 'configs/train-pilot.json')
        historical.update(category='full', max_h200_hours=64)
        for index in range(3):
            self.ledger.reserve(f'old-{index}', f'old-full-{index}', historical, 1800, 4, False)

    def test_exact_fourth_run_keeps_history_and_cannot_admit_fifth_or_retry(self):
        self.seed_three_formal_runs()
        before = copy.deepcopy(read_json(self.ledger.path))
        self.ledger.reserve('new', BE60P20_2B_RUN_ID, self.cfg, 14400, 8, False)
        after = read_json(self.ledger.path)
        self.assertEqual(after['runs'][BE60P20_2B_RUN_ID]['category'], 'full')
        self.assertEqual(after['runs'][BE60P20_2B_RUN_ID]['formal_run_number'], 4)
        self.assertEqual(after['jobs']['new']['reserved_h200_hours'], 32)
        for key, value in before['jobs'].items():
            self.assertEqual(after['jobs'][key], value)
        for key, value in before['runs'].items():
            self.assertEqual(after['runs'][key], value)
        with self.assertRaises(ValueError):
            self.ledger.reserve('fifth', 'another-full-run', self.cfg, 1800, 8, False)
        with self.assertRaises(ValueError):
            self.ledger.reserve('retry', BE60P20_2B_RUN_ID, self.cfg, 14400, 8, False)

    def test_exception_rejects_other_id_changed_recipe_and_resume(self):
        self.assertTrue(be60p20_2b_authorized(BE60P20_2B_RUN_ID, self.cfg))
        self.assertFalse(be60p20_2b_authorized('another-id', self.cfg))
        self.assertFalse(be60p20_2b_authorized(BE60P20_2B_RUN_ID, self.cfg, resume=True))
        changed = copy.deepcopy(self.cfg)
        changed['muon']['learning_rate'] *= 2
        self.assertFalse(be60p20_2b_authorized(BE60P20_2B_RUN_ID, changed))
        for name, cfg, resume in [('another-id', self.cfg, False),
                                  (BE60P20_2B_RUN_ID, changed, False),
                                  (BE60P20_2B_RUN_ID, self.cfg, True)]:
            with self.subTest(name=name, resume=resume), self.assertRaises(ValueError):
                self.ledger.reserve('blocked', name, cfg, 14400, 8, resume)

    def test_exact_run_still_rejects_excess_time_and_gpu_count(self):
        for seconds, gpus in ((14401, 8), (14400, 9), (14400, 4)):
            with self.subTest(seconds=seconds, gpus=gpus), self.assertRaises(ValueError):
                self.ledger.reserve('blocked', BE60P20_2B_RUN_ID, self.cfg, seconds, gpus, False)

    def command(self, action='pilot', minutes=240, prep='dolma-raw-baseline-20261008'):
        seal = self.root / 'preprocessed' / prep / 'raw/ready.json'
        seal.parent.mkdir(parents=True, exist_ok=True)
        seal.write_text('{}')
        argv = ['submit_training.py', action, BE60P20_2B_RUN_ID,
                '--prep-id', prep, '--config', str(PROJECT / 'configs/be60p20-lrscaled-2b-20261010.json'),
                '--minutes', str(minutes), '--test-only']
        with patch.object(cli, 'storage_root', return_value=self.root), \
                patch.object(cli, 'ExperimentLedger') as ledger, \
                patch.object(cli.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as dispatch, \
                patch.object(cli.sys, 'argv', argv), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cli.main()
            ledger.return_value.reserve.assert_not_called()
            dispatch.assert_called_once()
            return dispatch.call_args.args[0]

    def test_submission_uses_exact_script_and_correct_argument_order(self):
        command = self.command()
        self.assertIn('--time=240', command)
        self.assertIn('--gpus-per-node=8', command)
        self.assertEqual(command[-4:], [str(PROJECT / 'slurm/be60p20-2b.sbatch'),
                                       BE60P20_2B_RUN_ID, 'dolma-raw-baseline-20261008',
                                       str(PROJECT / 'configs/be60p20-lrscaled-2b-20261010.json')])

    def test_submission_rejects_resume_wrong_preparation_and_longer_allocation(self):
        for settings in ({'action': 'resume'}, {'prep': 'filtered-prep'}, {'minutes': 241}):
            with self.subTest(settings=settings), self.assertRaises(SystemExit):
                self.command(**settings)


if __name__ == '__main__':
    unittest.main()
