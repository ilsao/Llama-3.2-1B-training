"""Metadata controller safety tests, with no Slurm dispatch or native libraries."""
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace
from training_state import atomic_json
from slurm import filtering_resume as entry


class ControllerTests(TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);entry.directory(self.root).mkdir(parents=True)
        atomic_json(self.root/'preprocessing-ledger.json',{'jobs':{'1':{'h200_hours':1}},'total_h200_hours':1})

    def state(self,status='active'):
        return dict(status=status,jobs=[dict(job_id='1',stage='select',status='submitted')],pending=None)

    def test_active_allocation_never_submits_again(self):
        state=self.state()
        with patch.object(entry.subprocess,'run',return_value=SimpleNamespace(stdout='RUNNING|\n')),patch.object(entry,'submit') as submit:
            entry.tick(self.root,state)
        submit.assert_not_called();self.assertEqual(state['active_job']['job_id'],'1')

    def test_noncheckpoint_failure_blocks_without_retry(self):
        p=self.root/'logs/job-1';p.mkdir(parents=True);(p/'exit-code.txt').write_text('1')
        state=self.state()
        with patch.object(entry.subprocess,'run',return_value=SimpleNamespace(stdout='FAILED|\n')),patch.object(entry,'submit') as submit:
            entry.tick(self.root,state)
        submit.assert_not_called();self.assertEqual(state['status'],'blocked');self.assertEqual(state['jobs'][0]['status'],'reconciled')

    def test_disabled_reconciles_but_does_not_dispatch(self):
        p=self.root/'logs/job-1';p.mkdir(parents=True);(p/'exit-code.txt').write_text('75')
        state=self.state('disabled')
        with patch.object(entry.subprocess,'run',return_value=SimpleNamespace(stdout='FAILED|\n')),patch.object(entry,'submit') as submit:
            entry.tick(self.root,state)
        submit.assert_not_called();self.assertEqual(state['jobs'][0]['status'],'reconciled')

    def test_insufficient_budget_fails_before_pending_reservation(self):
        state=dict(status='active',jobs=[],pending=None)
        with patch.object(entry,'validate_seal'),patch.object(entry,'shared_upper',return_value=177),patch.object(entry,'submit') as submit:
            with self.assertRaisesRegex(ValueError,'budget'):entry.tick(self.root,state)
        submit.assert_not_called();self.assertIsNone(state['pending'])

    def test_success_promotes_only_next_stage(self):
        state=dict(status='active',jobs=[],pending=None)
        (self.root/'preprocessed'/entry.PREP/'select').mkdir();(self.root/'preprocessed'/entry.PREP/'select/manifest.json').write_text('{}')
        with patch.object(entry,'validate_seal'),patch.object(entry,'shared_upper',return_value=80),patch.object(entry,'submit',return_value='2') as submit:
            entry.tick(self.root,state)
        self.assertEqual(submit.call_args.args[0],'pack');self.assertEqual(state['jobs'][0]['job_id'],'2');self.assertIsNone(state['pending'])

    def test_train_override_releases_sixteen_cpus_before_evaluation(self):
        prep=self.root/'preprocessed'/entry.PREP
        (prep/'select').mkdir()
        atomic_json(prep/'configuration.json',{'config':{'max_training_tokens':100}})
        for tokens,cpus,extra in ((20,16,['--training-workers','15']),(100,8,[])):
            atomic_json(prep/'select/state.json',{'train_tokens':tokens})
            state=dict(status='active',jobs=[],pending=None,training_workers_override=15)
            with patch.object(entry,'validate_seal'),patch.object(entry,'shared_upper',return_value=80),patch.object(entry,'submit',return_value='2') as submit:
                entry.tick(self.root,state)
            self.assertEqual(submit.call_args.kwargs['cpus'],cpus)
            self.assertEqual(submit.call_args.args[2],extra)

    def test_evaluation_override_keeps_twelve_cpu_allocation(self):
        prep=self.root/'preprocessed'/entry.PREP;(prep/'select').mkdir()
        atomic_json(prep/'configuration.json',{'config':{'max_training_tokens':100}})
        for tokens,extra in ((20,['--training-workers','11','--evaluation-workers','11']),
                              (100,['--evaluation-workers','11'])):
            atomic_json(prep/'select/state.json',{'train_tokens':tokens})
            state=dict(status='active',jobs=[],pending=None,training_workers_override=11,evaluation_workers_override=11)
            with patch.object(entry,'validate_seal'),patch.object(entry,'shared_upper',return_value=80),patch.object(entry,'submit',return_value='2') as submit:
                entry.tick(self.root,state)
            self.assertEqual(submit.call_args.kwargs['cpus'],12)
            self.assertEqual(submit.call_args.args[2],extra)
