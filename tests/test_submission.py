"""Mock Slurm responses; no allocations, models or data processing."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from slurm import submit_training
from training_state import ExperimentLedger, PROJECT, load_config, read_json


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = ExperimentLedger(self.root)
        self.cfg = load_config(PROJECT / 'configs/train-pilot.json')

    def submit(self, result):
        with patch('sys.argv', ['submit_training.py', 'runtime', 'runtime-test']), \
             patch.object(submit_training, 'storage_root', return_value=self.root), \
             patch.object(submit_training.subprocess, 'run', return_value=result) as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            submit_training.main()
        return launch

    def reservation(self):
        self.ledger.reserve('submission-fixture', 'fixture', self.cfg, 1800, 4, False)
        with self.ledger.transaction() as data:
            data['jobs']['submission-fixture'].update(submission_tag='submission-fixture',
                submission_started_utc='2026-10-07T16:00:00+00:00', status='submission_uncertain')

    def test_failed_and_malformed_response_keep_budget_and_block_retry(self):
        for response in [subprocess.CompletedProcess([], 1, '', 'connection lost'),
                         subprocess.CompletedProcess([], 0, 'unexpected response', '')]:
            with self.subTest(response=response), self.assertRaisesRegex(SystemExit, 'remains charged'):
                self.submit(response)
            data = read_json(self.ledger.path)
            item = next(iter(data['jobs'].values()))
            self.assertEqual(item['status'], 'submission_uncertain')
            self.assertIn('--comment=' + item['submission_tag'], item['submission_command'])
            self.assertGreater(data['total_h200_hours_reserved_or_measured'], 0)
            with self.assertRaises(ValueError):
                self.ledger.reserve('retry', 'runtime-test', {**self.cfg, 'world_size':4}, 600, 4, False)
            # Separate fixtures represent independent attempted submissions.
            self.ledger.path.unlink()

    def test_success_persists_tag_and_job_receipt(self):
        launch = self.submit(subprocess.CompletedProcess([], 0, '123;cluster\n', ''))
        data = read_json(self.ledger.path)
        self.assertEqual(list(data['jobs']), ['123'])
        self.assertEqual(data['jobs']['123']['returned_job_id'], '123')
        self.assertEqual(data['jobs']['123']['status'], 'submitted')
        self.assertIn('--comment=' + data['jobs']['123']['submission_tag'], launch.call_args.args[0])

    def test_recovery_deduplicates_queue_and_accounting_without_resubmitting(self):
        self.reservation()
        with patch.object(submit_training.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([],0,'123|submission-fixture\n',''),
            subprocess.CompletedProcess([],0,'123|submission-fixture|\n','')]) as query:
            self.assertEqual(submit_training.recover_submission(self.ledger,'submission-fixture'), '123')
        self.assertEqual([call.args[0][0] for call in query.call_args_list], ['squeue','sacct'])
        self.assertEqual(read_json(self.ledger.path)['jobs']['123']['reserved_h200_hours'], 2)

    def test_terminal_recovery_uses_startup_receipt_when_sacct_comments_absent(self):
        self.reservation()
        receipt = self.root / 'logs/job-123/allocation.txt'
        receipt.parent.mkdir(parents=True)
        receipt.write_text('JobId=123 Comment=submission-fixture NumCPUs=8\n')
        with patch.object(submit_training.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([],0,'',''),
            subprocess.CompletedProcess([],0,'123||\n','')]):
            self.assertEqual(submit_training.recover_submission(self.ledger,'submission-fixture'), '123')

    def test_absent_duplicate_or_failed_recovery_retains_reservation(self):
        self.reservation()
        cases=[(0,''),(0,'123|submission-fixture\n124|submission-fixture\n'),(1,'')]
        for code, output in cases:
            with self.subTest(code=code,output=output), \
                 patch.object(submit_training.subprocess,'run',return_value=subprocess.CompletedProcess([],code,output,'')), \
                 self.assertRaises((ValueError,RuntimeError)):
                submit_training.recover_submission(self.ledger,'submission-fixture')
            self.assertEqual(read_json(self.ledger.path)['total_h200_hours_reserved_or_measured'], 2)

    def test_receipt_recovers_binding_crash_without_releasing_budget(self):
        self.reservation()
        with self.ledger.transaction() as data:
            data['jobs']['submission-fixture']['returned_job_id']='123'
        with patch.object(submit_training.subprocess,'run') as query:
            self.assertEqual(submit_training.recover_submission(self.ledger,'submission-fixture'),'123')
            query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
