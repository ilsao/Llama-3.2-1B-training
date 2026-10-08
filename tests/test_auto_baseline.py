"""Small metadata-only fault injection; never imports torch or submits real jobs."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from slurm import auto_baseline as auto
from training_state import atomic_json, digest, file_digest


class CampaignTests(unittest.TestCase):
    def job(self, seconds=900, gpus=8, **kwargs):
        return dict(kind='training', status='submitted', requested_seconds=seconds, gpu_count=gpus, **kwargs)

    def test_uncertain_and_failed_allocations_remain_charged(self):
        jobs=[self.job(status_override=1)]
        jobs[0]['status']='submission_uncertain'
        auto.check_budget(jobs,6300,8)
        with self.assertRaises(ValueError): auto.check_budget(jobs,6301,8)
        jobs[0].update(status='failed',elapsed_seconds=900,h200_hours=2)
        self.assertEqual(auto.budget(jobs),{'allocation_seconds':900,'h200_hours':2})

    def test_wall_time_limit_applies_even_with_fewer_gpus(self):
        jobs=[self.job(seconds=7200,gpus=4,elapsed_seconds=7200,h200_hours=8)]
        with self.assertRaises(ValueError): auto.check_budget(jobs,60,4)

    def test_requeue_geometry_cannot_buy_an_extra_campaign(self):
        jobs=[self.job(seconds=3600,elapsed_seconds=3600,h200_hours=8),self.job(seconds=3600)]
        with self.assertRaises(ValueError): auto.check_budget(jobs,1,1)

    def test_setup_separate_but_training_is_counted(self):
        jobs=[dict(kind='setup',status='submitted',requested_seconds=5400,gpu_count=1),self.job()]
        self.assertEqual(auto.budget(jobs)['h200_hours'],2)

    def test_never_dispatched_reservation_not_charged(self):
        job=self.job();job['status']='not_dispatched'
        self.assertEqual(auto.budget([job])['allocation_seconds'],0)

    def fixture(self, root):
        cfg=auto.load_campaign(auto.PROJECT/'configs/auto-baseline.json')
        cfg['root']=str(root)
        state={'status':'active','config_digest':digest(cfg),'config_path':'campaign.json','jobs':[]}
        return cfg,state

    def test_sbatch_timeout_retains_intent_and_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,state=self.fixture(Path(tmp));ledger=Mock()
            train=auto.base_config(cfg,'lr-low',8,15)
            ledger.transaction.return_value.__enter__=Mock(return_value={'jobs':{}})
            ledger.transaction.return_value.__exit__=Mock(return_value=False)
            # Use a real context manager to observe the durable tag.
            from contextlib import contextmanager
            @contextmanager
            def transaction():
                yield {'jobs':{state['jobs'][0]['tag']:{}}}
            ledger.transaction=transaction
            with patch.object(auto,'shared_cost',return_value=0),patch.object(auto,'ExperimentLedger',return_value=ledger),patch.object(auto.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',30)):
                with self.assertRaises(subprocess.TimeoutExpired): auto.submit_worker(cfg,state,'lr-low',train,15,8)
            saved=json.loads((auto.directory(cfg)/'state.json').read_text())
            self.assertEqual(saved['jobs'][0]['status'],'submission_uncertain')
            self.assertEqual(auto.budget(saved['jobs'])['h200_hours'],2)
            ledger.cancel_reservation.assert_not_called()

    def test_budget_rejection_precedes_ledger_or_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,state=self.fixture(Path(tmp));state['jobs']=[self.job(seconds=7200)]
            with patch.object(auto,'ExperimentLedger') as ledger,patch.object(auto.subprocess,'run') as run:
                with self.assertRaises(ValueError): auto.submit_worker(cfg,state,'lr-low',{},15,8)
                ledger.assert_not_called();run.assert_not_called()

    def test_manifest_requires_matching_parent_and_artifact_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,_=self.fixture(Path(tmp));prep=Path(tmp)/'preprocessed'/cfg['preparation_id']
            (prep/'split').mkdir(parents=True);(prep/'clean').mkdir()
            atomic_json(prep/'configuration.json',{'fingerprint':'pinned'})
            atomic_json(prep/'split/manifest.json',{'stage':'split'})
            artifact=prep/'clean/small.txt';artifact.write_text('ok')
            record={'fingerprint':'pinned','stage':'clean','parent_sha256':file_digest(prep/'split/manifest.json'),
                    'artifacts':[{'path':str(artifact),'size':2}]}
            atomic_json(prep/'clean/manifest.json',record)
            self.assertTrue(auto.manifest_ready(cfg,'clean'))
            artifact.write_text('changed')
            with self.assertRaises(ValueError): auto.manifest_ready(cfg,'clean')
            artifact.write_text('ok');atomic_json(prep/'split/manifest.json',{'stage':'different'})
            with self.assertRaises(ValueError): auto.manifest_ready(cfg,'clean')

    def test_live_data_job_prevents_duplicate_submission(self):
        cfg=auto.load_campaign(auto.PROJECT/'configs/auto-baseline.json')
        cfg.pop('raw_preparation_id',None)
        state={'jobs':[{'phase':'data-clean','status':'submitted'}]}
        with patch.object(auto,'manifest_ready') as ready, patch.object(auto,'save',side_effect=AssertionError('Live-state write attempted by test')):
            auto.data_advance(cfg,state)
            ready.assert_not_called()

    def test_running_job_does_not_release_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,state=self.fixture(Path(tmp));state['jobs']=[dict(phase='probe4',kind='training',tag='tag',job_id='123',status='submitted',requested_seconds=300,gpu_count=4)]
            with patch.object(auto,'read_json',return_value={'submission_intents':[]}),patch.object(auto,'slurm_record',side_effect=ValueError('running')),patch.object(auto,'ExperimentLedger') as ledger:
                auto.reconcile(cfg,state)
                ledger.return_value.reconcile.assert_not_called()
            self.assertEqual(state['jobs'][0]['status'],'submitted')

    def test_returned_id_recovery_binds_exact_existing_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,state=self.fixture(Path(tmp));state['jobs']=[dict(phase='probe4',kind='training',tag='tag',job_id=None,returned_job_id='123',status='submission_uncertain',requested_seconds=300,gpu_count=4)]
            ledger=Mock();ledger.path=Path(tmp)/'ledger.json'
            def read(path):
                return {'jobs':{'tag':{}}} if Path(path)==ledger.path else {'submission_intents':[]}
            with patch.object(auto,'read_json',side_effect=read),patch.object(auto,'find_submission_job',return_value='123'),patch.object(auto,'slurm_record',side_effect=ValueError('running')),patch.object(auto,'ExperimentLedger',return_value=ledger):
                auto.reconcile(cfg,state)
            ledger.bind.assert_called_once_with('tag','123')
            self.assertEqual(state['jobs'][0]['job_id'],'123')
            self.assertEqual(state['jobs'][0]['status'],'submitted')


if __name__=='__main__':
    unittest.main()
