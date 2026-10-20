import importlib.util,json,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch,Mock
LIVE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('repair',LIVE/'controller_recovery.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class RecoveryTests(unittest.TestCase):
    def test_uncertain_C_does_not_block_D_reconciliation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r.base,'STATE',Path(temp)/'state.json'):
            state={'jobs':{'C':{'status':'submission_uncertain'},'D':{'status':'submitted','job_id':'123'}}}
            def one(state,key,item):
                if key=='C':raise ValueError('Recovery found 0 matching jobs')
                item.update(status='complete',accounting={'h200_hours':10})
            with patch.object(r,'reconcile_one',side_effect=one):r.reconcile_independently(state)
            self.assertIn('recovery_query_error',state['jobs']['C'])
            self.assertEqual(state['jobs']['D']['accounting']['h200_hours'],10)
            self.assertEqual(json.loads(r.base.STATE.read_text()),state)

    def test_actual_endpoint_failure_is_not_swallowed(self):
        with patch.object(r,'reconcile_one',side_effect=ValueError('Incomplete/mismatched endpoint')):
            with self.assertRaises(ValueError):r.reconcile_independently({'jobs':{'D':{'job_id':'123','status':'submitted'}}})

    def test_slow_sbatch_timeout_extended_without_retry(self):
        with patch.object(r,'ORIGINAL_RUN',side_effect=subprocess.TimeoutExpired('sbatch',180)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):r.bounded_run(['sbatch','--hold'],timeout=30)
            self.assertEqual(run.call_count,1);self.assertEqual(run.call_args.kwargs['timeout'],180)

    def test_discovery_requires_successful_scheduler_queries(self):
        item=dict(submission_started_utc='2026-10-17T06:54:19+00:00',submission_tag='exact-C')
        with patch.object(r,'checked_query',side_effect=RuntimeError('scheduler unavailable')):
            with self.assertRaises(RuntimeError):r.discover(item)

    def test_discovery_exact_tag_and_duplicate_detection(self):
        item=dict(submission_started_utc='2026-10-17T06:54:19+00:00',submission_tag='exact-C')
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'DEFAULT_ROOT',Path(temp)):
            with patch.object(r,'checked_query',side_effect=[{'stdout':'1|pf17-C-seed42|PENDING|exact-C\n2|pf17-C-seed42|PENDING|exact-C\n'},{'stdout':''}]):
                self.assertEqual(r.discover(item)['matches'],['1','2'])

    def test_new_owner_uses_its_own_stop_marker(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'LIVE',Path(temp)),patch.object(r.base,'STATE',Path(temp)/'new-state.json'),patch.object(r,'check'),patch.object(r,'reconcile_independently'):
            (Path(temp)/'disabled').write_text('legacy owner disabled')
            with patch.object(r.base,'submit') as submit:
                r.recovered_tick();submit.assert_called_once_with({'jobs':{},'status':'gate'},'gate','gate')
            (Path(temp)/'recovery-disabled').write_text('stop new owner')
            with patch.object(r.base,'submit') as submit:
                self.assertEqual(r.recovered_tick()['status'],'disabled');submit.assert_not_called()

if __name__=='__main__':unittest.main()
