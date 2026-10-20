"""Submission argument tests with mocked subprocess: never call Slurm."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training_state import PROJECT, load_config

spec = importlib.util.spec_from_file_location('submission_cli', PROJECT / 'slurm/submit_training.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class SubmissionCliTests(unittest.TestCase):
    def review(self, action, exclude=None):
        cfg = load_config(PROJECT / 'configs/train-pilot.json')
        if action == 'sweep-calibration':
            cfg['category'] = 'validation'
        argv = ['submit_training.py', action, 'cli-test', '--prep-id', 'test-prep',
                '--config', str(PROJECT / 'configs/train-pilot.json'), '--minutes', '10', '--test-only']
        if exclude is not None:
            argv += ['--exclude', exclude]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seal = root / 'preprocessed/test-prep/raw/ready.json'
            seal.parent.mkdir(parents=True)
            seal.write_text('{}')
            with patch.object(cli, 'storage_root', return_value=root), \
                    patch.object(cli, 'load_config', return_value=cfg), \
                    patch.object(cli, 'ExperimentLedger') as ledger, \
                    patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')) as run, \
                    patch.object(cli.sys, 'argv', argv), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cli.main()
                ledger.return_value.reserve.assert_not_called()
                run.assert_called_once()
                return run.call_args.args[0]

    def test_exclusion_preserves_validation_flag(self):
        command = self.review('resume', '25a-hgpn151')
        self.assertIn('--exclude=25a-hgpn151', command)
        self.assertEqual(command.count('--test-only'), 1)

    def test_calibration_retains_existing_excluded_node(self):
        command = self.review('sweep-calibration', '25a-hgpn151')
        self.assertIn('--exclude=25a-hgpn062,25a-hgpn151', command)

    def test_default_training_has_no_new_exclusions(self):
        command = self.review('pilot')
        self.assertFalse(any(arg.startswith('--exclude=') for arg in command))

    def test_rejects_malformed_node_names_before_subprocess(self):
        with self.assertRaises(SystemExit):
            self.review('resume', '25a-hgpn151;bad')


if __name__ == '__main__':
    unittest.main()
