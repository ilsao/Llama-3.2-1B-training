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


class PreprocessingSubmissionTests(unittest.TestCase):
    def setUp(self):
        from training_state import atomic_json
        from slurm import submit_preprocessing
        self.helper = submit_preprocessing
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state_path = self.root / 'research-state.yaml'
        atomic_json(self.state_path, {'jobs': [], 'submission_intents': [],
            'accounting': {'shared_measured_plus_outstanding_reservations_upper_bound': 1.5}})

    def submit(self, response):
        with patch.object(self.helper.subprocess, 'run', return_value=response):
            return self.helper.submit('download', 'prep', ['--reuse-download-from', 'old'],
                                      self.root, self.state_path)

    def test_intent_is_durable_before_dispatch_and_success_binds(self):
        def dispatch(command, **kwargs):
            state = read_json(self.state_path)
            item = state['submission_intents'][0]
            self.assertEqual(item['status'], 'submission_uncertain')
            self.assertEqual(state['accounting']['shared_measured_plus_outstanding_reservations_upper_bound'], 25.5)
            self.assertIn('--comment=' + item['tag'], command)
            self.assertEqual(read_json(self.helper.receipt_path(self.root, item['tag'])), item)
            self.assertEqual(command[-2:], ['--reuse-download-from', 'old'])
            return subprocess.CompletedProcess([], 0, '123;cluster\n', '')
        with patch.object(self.helper.subprocess, 'run', side_effect=dispatch):
            self.assertEqual(self.helper.submit('download','prep',['--reuse-download-from','old'],
                                              self.root,self.state_path), '123')
        state = read_json(self.state_path)
        self.assertEqual(state['jobs'][0]['job_id'], '123')
        self.assertEqual(state['submission_intents'][0]['returned_job_id'], '123')
        with self.assertRaisesRegex(ValueError, 'recovered/reconciled'):
            self.submit(subprocess.CompletedProcess([],0,'124',''))

    def test_uncertain_reply_retains_reservation_and_blocks_other_stage(self):
        for response in [subprocess.CompletedProcess([],1,'','connection lost'),
                         subprocess.CompletedProcess([],0,'invalid','')]:
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, 'Do not resubmit'):
                self.submit(response)
            state = read_json(self.state_path)
            self.assertEqual(state['submission_intents'][0]['status'], 'submission_uncertain')
            self.assertEqual(state['accounting']['shared_measured_plus_outstanding_reservations_upper_bound'],25.5)
            with patch.object(self.helper.subprocess, 'run') as dispatch, self.assertRaises(ValueError):
                self.helper.submit('split','prep',[],self.root,self.state_path)
            dispatch.assert_not_called()
            from training_state import atomic_json
            atomic_json(self.state_path, {'jobs': [], 'submission_intents': [],
                'accounting': {'shared_measured_plus_outstanding_reservations_upper_bound': 1.5}})

    def test_returned_receipt_recovers_crash_without_query_or_resubmission(self):
        with patch.object(self.helper, 'bind', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                self.submit(subprocess.CompletedProcess([],0,'123',''))
        tag = read_json(self.state_path)['submission_intents'][0]['tag']
        with patch.object(self.helper.subprocess, 'run') as query:
            self.assertEqual(self.helper.recover(tag,self.root,self.state_path),'123')
            query.assert_not_called()
        self.assertEqual(read_json(self.state_path)['jobs'][0]['job_id'],'123')
        self.assertEqual(read_json(self.state_path)['accounting']['shared_measured_plus_outstanding_reservations_upper_bound'],25.5)

    def test_reconciled_intent_allows_next_stage_and_legacy_active_intent_blocks(self):
        from training_state import atomic_json
        self.submit(subprocess.CompletedProcess([],0,'123',''))
        atomic_json(self.root/'preprocessing-ledger.json', {'jobs': {'123': {'h200_hours':0.1}}})
        with patch.object(self.helper.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'124','')):
            self.assertEqual(self.helper.submit('split','prep',[],self.root,self.state_path),'124')
        state=read_json(self.state_path)
        state['submission_intents']=[{'prep_id':'other','job_id':'125','status':'submitted','tag':'legacy'}]
        atomic_json(self.state_path,state)
        with patch.object(self.helper.subprocess,'run') as dispatch, self.assertRaises(ValueError):
            self.helper.submit('download','other',[],self.root,self.state_path)
        dispatch.assert_not_called()

    def test_timeout_is_uncertain_but_exec_failure_releases_only_undispatched_intent(self):
        with patch.object(self.helper.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch',60)), \
             self.assertRaisesRegex(RuntimeError, 'retain intent'):
            self.helper.submit('download','prep',[],self.root,self.state_path)
        self.assertEqual(read_json(self.state_path)['accounting']['shared_measured_plus_outstanding_reservations_upper_bound'],25.5)
        with patch.object(self.helper.subprocess, 'run', side_effect=FileNotFoundError('sbatch')), \
             self.assertRaises(FileNotFoundError):
            self.helper.submit('download','another-prep',[],self.root,self.state_path)
        state=read_json(self.state_path)
        self.assertEqual(state['submission_intents'][-1]['status'],'not_dispatched')
        self.assertEqual(state['accounting']['shared_measured_plus_outstanding_reservations_upper_bound'],25.5)


if __name__ == '__main__':
    unittest.main()
