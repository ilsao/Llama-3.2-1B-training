"""Lightweight budget/cursor tests: stdlib only, no model or workers."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_state import (PROJECT, ExperimentLedger, TokenJournal, atomic_json,
                            digest, load_config, lr_multiplier, read_json, slurm_record, window_indices)
from train import checkpoint_state


class TrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(PROJECT / 'configs/train-pilot.json')

    def test_distributed_cursor_has_no_duplicates_or_automatic_padding(self):
        ranks = [window_indices(0, 33, 4, rank, 2, 4) for rank in range(4)]
        flattened = [index for windows in ranks for batch in windows for index in batch]
        self.assertEqual(sorted(flattened), list(range(32)))
        self.assertEqual(window_indices(32, 33, 4, 0, 2, 4), [])
        resumed = [window_indices(32, 64, 4, rank, 2, 4) for rank in range(4)]
        self.assertEqual(sorted(index for windows in resumed for batch in windows for index in batch), list(range(32, 64)))

    def test_pending_crash_reservation_never_rewinds(self):
        path = self.root / 'tokens.json'
        journal = TokenJournal(path, 100)
        first = journal.reserve(40)
        journal.confirm(first)
        journal.reserve(30)  # interrupted forward; deliberately remains charged
        resumed = TokenJournal(path, 100)
        self.assertEqual(resumed.data['confirmed_tokens'], 40)
        self.assertEqual(resumed.data['charged_tokens'], 70)
        with self.assertRaises(ValueError):
            resumed.reserve(31)
        resumed.confirm(resumed.reserve(30))
        self.assertEqual(resumed.data['confirmed_tokens'], 70)
        self.assertEqual(resumed.data['charged_tokens'], 100)

    def test_replayed_data_and_skipped_updates_still_count(self):
        journal = TokenJournal(self.root / 'tokens.json', 50)
        for _ in range(2):  # same inputs reprocessed after an older model checkpoint
            journal.confirm(journal.reserve(20))
        self.assertEqual(journal.data['confirmed_tokens'], 40)
        with self.assertRaises(ValueError):
            journal.reserve(20)
        with self.assertRaises(ValueError):
            TokenJournal(self.root / 'tokens.json', 60)

    def test_corrupt_token_journal_is_not_reset(self):
        path = self.root / 'tokens.json'
        journal = TokenJournal(path, 100)
        journal.reserve(10)
        altered = read_json(path)
        altered['charged_tokens'] = 5
        atomic_json(path, altered)
        with self.assertRaises(ValueError):
            TokenJournal(path, 100)

    def test_learning_rate_warmup_and_last_update_floor(self):
        self.assertAlmostEqual(lr_multiplier(0, 100, 10, 0.1), 0.1)
        self.assertAlmostEqual(lr_multiplier(9, 100, 10, 0.1), 1.0)
        self.assertAlmostEqual(lr_multiplier(10, 100, 10, 0.1), 1.0)
        self.assertAlmostEqual(lr_multiplier(99, 100, 10, 0.1), 0.1)

    def test_configuration_rejects_budget_and_evaluation_violations(self):
        for key, value in [('max_training_tokens', 6_000_000_001), ('world_size', 9),
                           ('max_h200_hours', 64.01), ('eval_steps', 11),
                           ('eval_documents', 10001), ('sequence_length', 4096),
                           ('wandb_mode', 'offline')]:
            cfg = {**self.config, key: value}
            path = self.root / 'config.json'
            atomic_json(path, cfg)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_config(path)

    def test_gpu_hour_reservation_resume_and_actual_reconciliation(self):
        ledger = ExperimentLedger(self.root)
        ledger.reserve('booking-1', 'pilot-1', self.config, 1800, 4, False)
        ledger.bind('booking-1', '111')
        with self.assertRaises(ValueError):
            ledger.reserve('booking-2', 'pilot-1', self.config, 900, 4, True)
        measured = {'job_id': '111', 'gpu_count': 4, 'h200_hours': 1.0,
                    'elapsed_seconds': 900, 'state': 'COMPLETED'}
        with patch('training_state.slurm_record', return_value=measured):
            ledger.reconcile('111')
            ledger.reconcile('111')  # idempotent accounting
        with self.assertRaises(ValueError):
            ledger.reserve('booking-2', 'pilot-1', self.config, 901, 4, True)
        ledger.reserve('booking-2', 'pilot-1', self.config, 900, 4, True)
        ledger.bind('booking-2', '222')
        allocation = ledger.allocation('222', 'pilot-1')
        self.assertEqual(allocation['previous_h200_hours'], 1)
        self.assertEqual(allocation['config_digest'], digest(self.config))
        data = read_json(ledger.path)
        self.assertEqual(data['total_h200_hours_measured'], 1)
        self.assertEqual(data['total_h200_hours_reserved_or_measured'], 2)

    def test_three_full_run_limit_is_shared(self):
        ledger = ExperimentLedger(self.root)
        cfg = {**self.config, 'category': 'full', 'max_h200_hours': 64}
        for index in range(3):
            ledger.reserve(f'b-{index}', f'full-{index}', cfg, 1800, 4, False)
        with self.assertRaises(ValueError):
            ledger.reserve('b-4', 'full-4', cfg, 1800, 4, False)

    def test_changed_config_and_run_collision_are_rejected(self):
        ledger = ExperimentLedger(self.root)
        ledger.reserve('b-1', 'pilot', self.config, 1800, 4, False)
        ledger.bind('b-1', '111')
        with patch('training_state.slurm_record', return_value={'job_id': '111', 'gpu_count': 4, 'h200_hours': 0.1}):
            ledger.reconcile('111')
        with self.assertRaises(ValueError):
            ledger.reserve('b-2', 'pilot', self.config, 900, 4, False)
        changed = copy.deepcopy(self.config)
        changed['learning_rate'] *= 2
        with self.assertRaises(ValueError):
            ledger.reserve('b-2', 'pilot', changed, 900, 4, True)

    def test_incomplete_checkpoint_is_refused(self):
        run = self.root / 'run'
        checkpoint = run / 'checkpoints/checkpoint-001'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'weights').write_bytes(b'abcd')
        state = {'global_step': 1}
        atomic_json(checkpoint / 'training-state.json', state)
        atomic_json(checkpoint / 'complete.json', {'state_digest': digest(state),
                    'files': [{'name': 'weights', 'size': 4}]})
        atomic_json(run / 'latest.json', {'checkpoint': checkpoint.name})
        self.assertEqual(checkpoint_state(run)[1], state)
        (checkpoint / 'weights').write_bytes(b'ab')
        with self.assertRaises(ValueError):
            checkpoint_state(run)

    def test_generic_slurm_tres_requires_allocated_hardware_evidence(self):
        raw = '111|COMPLETED|1800|cpu=12,gres/gpu=2|start|end|node1|\n'
        directory = self.root / 'logs/job-111'
        directory.mkdir(parents=True)
        with patch('training_state.subprocess.check_output', return_value=raw):
            with self.assertRaises(ValueError):
                slurm_record('111', self.root)
            (directory / 'gpus.txt').write_text('NVIDIA H200, GPU-1\nNVIDIA H200, GPU-2\n')
            measured = slurm_record('111', self.root)
            self.assertEqual(measured['h200_hours'], 1)
            self.assertIn('nvidia-smi', measured['gpu_type_evidence']['source'])
            (directory / 'gpus.txt').unlink()
            (directory / 'node.txt').write_text('NodeName=node1 Gres=gpu:H200:8(S:0-1)\n')
            self.assertEqual(slurm_record('111', self.root)['h200_hours'], 1)
            (directory / 'node.txt').write_text('NodeName=node1 Gres=gpu:H200:4,gpu:A100:4\n')
            with self.assertRaises(ValueError):
                slurm_record('111', self.root)

    def test_queue_cancellation_releases_unspent_gpu_reservation(self):
        ledger = ExperimentLedger(self.root)
        ledger.reserve('booking-1', 'pilot', self.config, 1800, 4, False)
        ledger.bind('booking-1', '111')
        raw = '111|CANCELLED|0||Unknown|end|None|\n'
        with patch('training_state.subprocess.check_output', return_value=raw):
            self.assertEqual(ledger.reconcile('111')['h200_hours'], 0)
        ledger.reserve('booking-2', 'pilot', self.config, 1800, 4, True)

    def test_reconciliation_finalizes_run_summary(self):
        ledger = ExperimentLedger(self.root)
        ledger.reserve('booking', 'pilot', self.config, 1800, 4, False)
        ledger.bind('booking', '111')
        summary = self.root / 'runs/experiments/pilot/summary.json'
        atomic_json(summary, {'h200_hours_final': 'pending'})
        measured = {'job_id': '111', 'gpu_count': 4, 'elapsed_seconds': 900, 'h200_hours': 1}
        with patch('training_state.slurm_record', return_value=measured):
            ledger.reconcile('111')
        self.assertEqual(read_json(summary)['h200_hours_final'], 1)


if __name__ == '__main__':
    unittest.main()
