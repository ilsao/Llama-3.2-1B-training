"""Pure frozen-checkpoint LR sweep tests; no model/tensor loading."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'slurm'))
import resplit_lr_runtime as runtime
spec=importlib.util.spec_from_file_location('tail_sweep_controller',ROOT/'slurm/horizon_campaign_v5.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from training_state import atomic_json,digest,file_digest,load_config,schedule_multiplier
from horizon_runtime import seed_journal

class TailSweepTests(unittest.TestCase):
    def parent(self):return load_config(ROOT/'configs/horizon-wsd-20261014/final-resplit.json')
    def child(self,key):return load_config(ROOT/'configs/horizon-wsd-20261014'/(key+'.json'))

    def test_peak_scales_and_first_phase_are_exact(self):
        parent=self.parent()
        self.assertGreater(parent['phase_schedule'][1]['peak_ratio'],1)
        for key,scale in runtime.SCALES.items():
            child=self.child(key)
            self.assertEqual(runtime.validate_configs(parent,child,c.CAMPAIGN+'-'+key),scale)
            for i in [0,169,170,4000,12714]:
                self.assertEqual(schedule_multiplier(i,parent),schedule_multiplier(i,child))
            self.assertAlmostEqual(schedule_multiplier(12715,child),scale/170)
            self.assertLessEqual(schedule_multiplier(12884,child),scale)
            self.assertEqual(schedule_multiplier(15258,child),0)
            self.assertEqual(child['max_steps']*131072,2000027648)

    def test_only_requested_scale_and_numerical_change_allowed(self):
        parent=self.parent();child=self.child('tail-lr070')
        bad=copy.deepcopy(child);bad['phase_schedule'][1]['peak_ratio']=.8
        with self.assertRaises(ValueError):runtime.validate_configs(parent,bad,c.CAMPAIGN+'-tail-lr070')
        bad=copy.deepcopy(child);bad['weight_decay']=.2
        with self.assertRaises(ValueError):runtime.validate_configs(parent,bad,c.CAMPAIGN+'-tail-lr070')
        with self.assertRaises(ValueError):runtime.validate_configs(parent,child,c.CAMPAIGN+'-tail-lr080')

    def fixture(self,root):
        parent=root/'runs/experiments'/runtime.PARENT;checkpoint=parent/'checkpoints/frozen-phase1'
        checkpoint.mkdir(parents=True);(checkpoint/'optimizer.bin').write_bytes(b'test')
        original=self.parent()
        saved=dict(global_step=12715,attempted_updates=12715,cursor=203440,confirmed_tokens_at_checkpoint=1666580480,
                   charged_tokens_at_checkpoint=1666580480,config_digest=digest(original),code_digest='original-sealed-code',
                   environment_digest='original-env',preparation_fingerprint='same-data',wandb_id='parent-id')
        atomic_json(checkpoint/'training-state.json',saved)
        atomic_json(checkpoint/'complete.json',dict(state_digest=digest(saved),files=[dict(name='optimizer.bin',size=4)]))
        atomic_json(parent/'resolved-config.json',original)
        atomic_json(parent/'resource-accounting.json',dict(all_jobs_reconciled=True,budget_exceeded=False))
        atomic_json(parent/'token-journal.json',dict(confirmed_tokens=2000027648,charged_tokens=2000027648,pending={}))
        atomic_json(parent/'summary.json',dict(global_step=15259,initialization=dict(sample_signature='same-init')))
        atomic_json(parent/'data-provenance.json',dict(fingerprint='same-data'))
        atomic_json(parent/'latest.json',dict(checkpoint='WRONG-newer-step15259'))
        plan=dict(checkpoint=str(checkpoint),checkpoint_state_sha256=file_digest(checkpoint/'training-state.json'),
                  checkpoint_complete_sha256=file_digest(checkpoint/'complete.json'),overlay_hashes={'overlay':'sealed'})
        atomic_json(root/'resplit-lr-plan.json',plan)
        return saved,checkpoint

    def test_loads_explicit_phase1_ignoring_newer_latest_and_restores_momentum(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);saved,checkpoint=self.fixture(root)
            child_cfg=self.child('tail-lr070');run=root/'child';run.mkdir()
            state={**saved,'config_digest':digest(child_cfg),'wandb_id':'new-child-id'}
            accelerator=MagicMock();schedulers=[types.SimpleNamespace() for _ in range(2)]
            optimizers=[types.SimpleNamespace(param_groups=[{}]) for _ in range(2)]
            train=types.ModuleType('train');train.on_main=lambda accelerator,fn:fn()
            args=types.SimpleNamespace(root=root,fork_from_run=runtime.PARENT,run_id=c.CAMPAIGN+'-tail-lr070')
            with patch.object(runtime,'DIRECTORY',root),patch.dict(sys.modules,{'train':train}):
                loaded,child=runtime.fork_checkpoint(args,child_cfg,state,run,dict(fingerprint='same-data'),
                    dict(sample_signature='same-init'),accelerator,schedulers,optimizers)
            accelerator.load_state.assert_called_once_with(str(checkpoint))
            self.assertEqual(loaded,str(checkpoint));self.assertEqual(child['cursor'],203440)
            self.assertEqual(child['wandb_id'],'new-child-id')
            self.assertEqual(child['confirmed_tokens_at_checkpoint'],1666580480)
            self.assertAlmostEqual(optimizers[0].param_groups[0]['lr'],child_cfg['learning_rate']*.7/170)
            seed_journal(run,child_cfg,child)
            self.assertEqual(json.loads((run/'token-journal.json').read_text())['charged_tokens'],1666580480)
            overlay=json.loads((run/'execution-overlay.json').read_text())
            self.assertEqual(overlay['parent_current_journal_high_water'],2000027648)

    def test_source_mismatch_rejected_before_tensor_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);saved,checkpoint=self.fixture(root)
            cfg=self.child('tail-lr100');run=root/'child';run.mkdir()
            args=types.SimpleNamespace(root=root,fork_from_run=runtime.PARENT,run_id=c.CAMPAIGN+'-tail-lr100')
            state={**saved,'code_digest':'changed-code'};accelerator=MagicMock()
            train=types.ModuleType('train');train.on_main=lambda accelerator,fn:fn()
            with patch.object(runtime,'DIRECTORY',root),patch.dict(sys.modules,{'train':train}):
                with self.assertRaises(ValueError):runtime.fork_checkpoint(args,cfg,state,run,dict(fingerprint='same-data'),
                    dict(sample_signature='same-init'),accelerator,[],[])
            accelerator.load_state.assert_not_called()

    def test_controller_submits_exactly_four_distinct_requested_forks_once(self):
        original=json.loads((ROOT/'experiments/horizon-wsd-20261014/controller-state.json').read_text())
        original['stage']='final';original['resplit_lr_sweep_applied']=True
        original['jobs']['final-standard-a']['status']='submitted'
        for key in runtime.SCALES:original['jobs'].pop(key,None)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);atomic_json(d/'state.json',original);submitted=[]
            def submit(s,key,name,path,cfg,minutes,mode='fresh',pause=0,parent='-',full_eval=False,test_only=False):
                if key in s['jobs']:return
                submitted.append((key,minutes,mode,parent,full_eval))
                s['jobs'][key]=dict(status='submitted',slurm_state='RUNNING')
            with patch.object(c,'DIRECTORY',d),patch.object(c,'STATE',d/'state.json'),patch.object(c,'DISABLED',d/'disabled'), \
                 patch.object(c,'require_owner',return_value=True),patch.object(c,'check'),patch.object(c,'reconcile'), \
                 patch.object(c,'report'),patch.object(c,'submit',side_effect=submit):
                c.run_once();c.run_once()
            self.assertEqual(submitted,[(key,55,'fresh',runtime.PARENT,True) for key in runtime.SCALES])

if __name__=='__main__':unittest.main()
