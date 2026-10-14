"""Pure ownership/final recovery regression tests; no tensors or Slurm."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'slurm'))
import horizon_owner as owner
spec=importlib.util.spec_from_file_location('owner_recovery_controller',ROOT/'slurm/horizon_campaign_v4.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from training_state import ExperimentLedger, atomic_json, digest, load_config

class OwnerRecoveryTests(unittest.TestCase):
    def test_only_matching_script_hash_and_host_can_mutate(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);script=d/'controller.py';script.write_text('sealed owner')
            registry=d/'owner.json'
            atomic_json(registry,dict(script=str(script),sha256=hashlib.sha256(script.read_bytes()).hexdigest(),hostname='login4'))
            with patch.object(owner,'REGISTRY',registry),patch.object(owner.socket,'gethostname',return_value='login4'):
                self.assertTrue(owner.require_owner(script,'run-once'))
                other=d/'old.py';other.write_text('old')
                self.assertFalse(owner.require_owner(other,'run-once'))
                script.write_text('changed')
                self.assertFalse(owner.require_owner(script,'install'))
            script.write_text('sealed owner')
            with patch.object(owner,'REGISTRY',registry),patch.object(owner.socket,'gethostname',return_value='login2'):
                self.assertFalse(owner.require_owner(script,'run-once'))
                self.assertTrue(owner.require_owner(script,'verify-job'))

    def test_missing_owner_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(owner,'REGISTRY',Path(directory)/'missing'):
            self.assertFalse(owner.require_owner(ROOT/'slurm/horizon_campaign_v2.py','run-once'))

    def test_legacy_cli_cannot_submit_or_replace_cron(self):
        for name in ['horizon_campaign.py','horizon_campaign_v2.py','horizon_campaign_v3.py']:
            for action in ['run-once','install']:
                result=subprocess.run([sys.executable,str(ROOT/'slurm'/name),action],capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertRegex(result.stdout,'No active shared campaign owner|Superseded controller')
                self.assertNotIn('Submitted',result.stdout)

    def test_same_formal_identity_reserves_fresh_recovery_and_preserves_cost(self):
        cfg=load_config(ROOT/'configs/horizon-wsd-20261014/final-standard.json')
        run_id=c.CAMPAIGN+'-final-standard'
        with tempfile.TemporaryDirectory() as directory:
            ledger=ExperimentLedger(directory)
            ledger.reserve('failed',run_id,cfg,180*60,8,False)
            with ledger.transaction() as data:
                data['jobs']['failed'].update(h200_hours=15*8/3600,state='FAILED',status='reconciled')
            ledger.reserve('fresh-recovery',run_id,cfg,179*60,8,True)
            with ledger.transaction() as data:
                self.assertEqual(len(data['runs']),1)
                self.assertEqual(data['runs'][run_id]['formal_run_number'],5)
                self.assertEqual(data['jobs']['failed']['h200_hours'],15*8/3600)
                data['jobs']['fresh-recovery'].update(h200_hours=179*8/60,state='COMPLETED',status='reconciled')
            ledger.reserve('faithful-tail',run_id,cfg,60*60,8,True)
            with ledger.transaction() as data:
                self.assertAlmostEqual(sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values()),31.9)

    def test_manual_standard_recovery_is_idempotent_and_preserves_resplit(self):
        original=json.loads((ROOT/'experiments/horizon-wsd-20261014/controller-state.json').read_text())
        if 'final-standard-a-failed-original' in original['jobs']:
            original['jobs']['final-standard-a']=original['jobs'].pop('final-standard-a-failed-original')
        original.pop('standard_recovery_applied',None)
        original['stage']='final'
        failed=original['jobs']['final-standard-a'];cfg=load_config(failed['configuration'])
        run=failed['run_id']
        ledger=MagicMock()
        ledger.transaction.return_value.__enter__.return_value={
            'jobs':{'539819':dict(job_id='539819',run_id=run,state='FAILED',h200_hours=15*8/3600,reserved_h200_hours=24,allocation_kind='horizon_wsd_campaign')},
            'runs':{run:dict(config_digest=digest(cfg))}}
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);(d/'logs').mkdir()
            (d/'logs/hw-final-standard-a-539819.err').write_text('device=7, num_gpus=7')
            atomic_json(d/'state.json',original)
            atomic_json(d/'owner-recovery-protocol-commit.json',{'commit':'test'})
            with patch.object(c,'DIRECTORY',d),patch.object(c,'STATE',d/'state.json'),patch.object(c,'DEFAULT_ROOT',d), \
                 patch.object(c,'check'),patch.object(c,'reconcile'),patch.object(c,'report'), \
                 patch.object(c,'ExperimentLedger',return_value=ledger), \
                 patch.object(c,'plan',return_value={'allocation_cap_h200_hours':205.33333333333334}):
                c.recover_standard();first=(d/'state.json').read_text();c.recover_standard()
                self.assertEqual((d/'state.json').read_text(),first)
            state=json.loads(first)
            self.assertEqual(state['jobs']['final-resplit-a'],original['jobs']['final-resplit-a'])
            self.assertEqual(state['jobs']['final-standard-a-failed-original'],failed)
            self.assertAlmostEqual(state['standard_recovery_applied']['formal_h200_hour_upper_bound'],31.9)

    def test_final_stage_submits_fresh_standard_without_touching_active_resplit(self):
        original=json.loads((ROOT/'experiments/horizon-wsd-20261014/controller-state.json').read_text())
        original['stage']='final';original['jobs'].pop('final-standard-a',None)
        original['standard_recovery_applied']={'manual':True}
        original['jobs']['final-resplit-a'].update(status='submitted',slurm_state='RUNNING')
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);atomic_json(d/'state.json',original)
            submissions=[]
            def submit(s,key,name,path,cfg,minutes,mode='fresh',pause=0,parent='-',full_eval=False,test_only=False):
                if key in s['jobs']:return
                submissions.append((key,minutes,mode,pause))
                s['jobs'][key]={'status':'submitted','job_id':'test','slurm_state':'RUNNING'}
            with patch.object(c,'DIRECTORY',d),patch.object(c,'STATE',d/'state.json'),patch.object(c,'DISABLED',d/'disabled'), \
                 patch.object(c,'require_owner',return_value=True),patch.object(c,'reconcile'),patch.object(c,'check'),patch.object(c,'report'), \
                 patch.object(c,'config_for',return_value=(d/'cfg.json',{})),patch.object(c,'submit',side_effect=submit):
                c.run_once()
            self.assertEqual(submissions,[('final-standard-a',179,'fresh',12715)])

    def test_final_tails_resume_same_ids_and_request_full_evaluation(self):
        original=json.loads((ROOT/'experiments/horizon-wsd-20261014/controller-state.json').read_text())
        original['stage']='final';original['standard_recovery_applied']={'manual':True}
        for arm in ('standard','resplit'):
            original['jobs']['final-'+arm+'-a']={'status':'reconciled','slurm_state':'COMPLETED'}
            original['jobs'].pop('final-'+arm+'-b',None)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);atomic_json(d/'state.json',original)
            submitted=[]
            def submit(s,key,name,path,cfg,minutes,mode='fresh',pause=0,parent='-',full_eval=False,test_only=False):
                if key in s['jobs']:return
                submitted.append((key,name,minutes,mode,full_eval))
                s['jobs'][key]={'status':'submitted','job_id':'test','slurm_state':'RUNNING'}
            def completed(s,keys):
                return all(k in s['jobs'] and s['jobs'][k]['status']=='reconciled' for k in keys)
            with patch.object(c,'DIRECTORY',d),patch.object(c,'STATE',d/'state.json'),patch.object(c,'DISABLED',d/'disabled'), \
                 patch.object(c,'require_owner',return_value=True),patch.object(c,'reconcile'),patch.object(c,'check'),patch.object(c,'report'), \
                 patch.object(c,'config_for',return_value=(d/'cfg.json',{})),patch.object(c,'submit',side_effect=submit), \
                 patch.object(c,'completed',side_effect=completed):
                c.run_once()
            self.assertEqual(submitted,[('final-standard-b','final-standard',60,'resume',True),
                                        ('final-resplit-b','final-resplit',60,'resume',True)])

if __name__=='__main__':unittest.main()
