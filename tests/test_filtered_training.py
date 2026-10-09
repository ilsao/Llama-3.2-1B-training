"""Token stopping, recipe compatibility and durable submission tests; no torch or workers."""
import copy
import contextlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training_state import (PROJECT, ExperimentLedger, TokenJournal, atomic_json,
                            load_config, read_json, schedule_multiplier, training_target_reached)
from slurm import submit_filtered_training as owner


class FilteredTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.cfg = load_config(PROJECT / 'configs/train-muon-wsd-filtered-2b.json')

    def test_confirmed_inputs_and_final_whole_batch(self):
        journal = TokenJournal(self.root / 'tokens.json', self.cfg['max_training_tokens'])
        first = journal.reserve(15258 * 131072)
        journal.confirm(first)
        self.assertFalse(training_target_reached(self.cfg, journal.data))
        final = journal.reserve(131072)
        self.assertEqual(journal.data['charged_tokens'], 2_000_027_648)
        self.assertFalse(training_target_reached(self.cfg, journal.data))
        journal.confirm(final)
        self.assertTrue(training_target_reached(self.cfg, journal.data))
        with self.assertRaises(ValueError): journal.reserve(131072)
        self.assertFalse(training_target_reached({'max_training_tokens': 6_000_000_000}, journal.data))

    def test_invalid_target_and_legacy_config(self):
        path = self.root / 'config.json'
        for value in (True, 0, -1, 2e9, 2_000_027_649):
            atomic_json(path, {**self.cfg, 'target_training_tokens': value})
            with self.subTest(value=value), self.assertRaises(ValueError): load_config(path)
        self.assertNotIn('target_training_tokens', load_config(PROJECT / 'configs/train-pilot.json'))

    def test_recipe_and_extended_wsd(self):
        best = read_json(PROJECT / 'configs/baselines/muon-wsd-raw-20261009.json')
        owner.validate_recipe(self.cfg, best)
        changed = copy.deepcopy(self.cfg); changed['muon']['learning_rate'] *= 2
        with self.assertRaises(ValueError): owner.validate_recipe(changed, best)
        self.assertEqual(self.cfg['max_steps'] * 16 * 8192, 2_000_027_648)
        self.assertAlmostEqual(schedule_multiplier(0, self.cfg), 1 / 170)
        self.assertEqual(schedule_multiplier(169, self.cfg), 1)
        self.assertEqual(schedule_multiplier(12207, self.cfg), 1)
        self.assertLess(schedule_multiplier(12208, self.cfg), 1)
        self.assertAlmostEqual(schedule_multiplier(15258, self.cfg), .1)
        self.assertLessEqual(self.cfg['eval_steps'], self.cfg['max_steps'] // 10)

    def summary(self):
        return dict(configuration=self.cfg, stop_reason='training_token_target',
                    training_token_target_reached=True, total_tokens_seen=2_000_027_648,
                    budget_tokens_charged=2_000_027_648,
                    initialization={'pretrained_weights_loaded': False},
                    data_provenance=dict(preparation_id=owner.PREP_ID,
                                         fingerprint=owner.PREPARATION_FINGERPRINT))

    def test_time_stop_or_wrong_provenance_never_satisfies_result(self):
        self.assertTrue(owner.training_result(self.summary()))
        for change in ({'stop_reason': 'wall_time_or_signal'}, {'total_tokens_seen': 1_999_896_576},
                       {'budget_tokens_charged': 2_000_027_649},
                       {'initialization': {'pretrained_weights_loaded': True}},
                       {'data_provenance': {'preparation_id': 'raw', 'fingerprint': owner.PREPARATION_FINGERPRINT}}):
            with self.subTest(change=change):
                self.assertFalse(owner.training_result({**self.summary(), **change}))

    def prepare_submission(self):
        best = {k: v for k, v in self.cfg.items() if k != 'target_training_tokens'}
        parent = self.root / 'runs/experiments/baseline-muon-wsd-raw-20261009'
        atomic_json(parent / 'summary.json', dict(export_global_step=13717, configuration=best))
        prep = self.root / 'preprocessed' / owner.PREP_ID
        for stage, details in [('select', {}), ('pack', {'training_tokens': 6_000_000_000}),
                               ('verify', dict(training={'input_tokens': 6_000_000_000},
                                               evaluation_documents=50000, subset_documents=10000))]:
            atomic_json(prep / stage / 'manifest.json', dict(fingerprint=owner.PREPARATION_FINGERPRINT, details=details))
        directory = self.root / 'runs/filtered' / owner.RUN_ID
        self.context.enter_context(patch.object(owner, 'DEFAULT_ROOT', self.root))
        self.context.enter_context(patch.object(owner, 'DIRECTORY', directory))
        self.context.enter_context(patch.object(owner, 'EXPERIMENT', self.root / 'runs/experiments' / owner.RUN_ID))
        self.context.enter_context(patch('slurm.filtering_resume.shared_upper', return_value=112))
        self.context.enter_context(patch('slurm.auto_optimizer_sweep.environment_identity', return_value={
            'packages_digest': self.cfg['sweep_environment_digest']}))
        self.context.enter_context(patch.object(owner, 'cron'))
        self.context.enter_context(patch.object(owner.subprocess, 'check_output', side_effect=lambda command, **kwargs:
            'main\n' if command[:2] == ['git', 'branch'] else 'abc\n'))
        return directory

    def test_one_submission_and_uncertain_outcome_retains_reservation(self):
        directory = self.prepare_submission()
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1 if command[0] == 'sbatch' else 0,
                                               stdout='', stderr='transport lost')
        with patch.object(owner.subprocess, 'run', side_effect=run) as invoked:
            with self.assertRaisesRegex(ValueError, 'Uncertain submission'): owner.submit()
            with self.assertRaisesRegex(ValueError, 'Existing submission'): owner.submit()
            self.assertEqual(sum(call.args[0][0] == 'sbatch' for call in invoked.call_args_list), 1)
        state = read_json(directory / 'launch.json')
        ledger = read_json(self.root / 'experiment-ledger.json')
        self.assertEqual(ledger['jobs'][state['reservation']]['status'], 'submission_uncertain')
        self.assertEqual(ledger['jobs'][state['reservation']]['reserved_h200_hours'], 64)

    def test_bind_eight_gpu_job_without_resume(self):
        directory = self.prepare_submission()
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout='123\n' if command[0] == 'sbatch' else '', stderr='')
        with patch.object(owner.subprocess, 'run', side_effect=run): owner.submit()
        state = read_json(directory / 'launch.json')
        self.assertEqual(state['job_id'], '123')
        self.assertTrue(state['random_initialization'])
        self.assertFalse(state['checkpoint_initialization'])
        self.assertEqual(ExperimentLedger(self.root).allocation('123', owner.RUN_ID)['previous_h200_hours'], 0)

    def test_64_hour_authorization_does_not_extend_other_runs_or_resumes(self):
        ledger = ExperimentLedger(self.root)
        with self.assertRaisesRegex(ValueError, '7200 seconds'):
            ledger.reserve('other', 'another-full-run', self.cfg, 28800, 8, False)
        with self.assertRaisesRegex(ValueError, '7200 seconds'):
            ledger.reserve('resume', owner.RUN_ID, self.cfg, 28800, 8, True)
        with self.assertRaises(ValueError):
            ledger.reserve('overshoot', owner.RUN_ID, self.cfg, 28801, 8, False)


if __name__ == '__main__': unittest.main()
