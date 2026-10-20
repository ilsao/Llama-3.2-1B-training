"""Pure scheduler, recipe-difference and exact one-shot authorization tests."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training_state import load_config,schedule_multiplier,hold_schedule_boundaries,ExperimentLedger,read_json
from wsd20_contracts import authorized,RUN_ID

class Wsd20Tests(unittest.TestCase):
 def cfg(self):return load_config(ROOT.parent/'config.json')
 def test_dataset_entrypoint_dependencies_import_without_heavy_modules(self):
  import dataset
  import train
  self.assertTrue(callable(dataset.check_artifact));self.assertTrue(callable(train.main))
  for name in ('torch','datasets','pyarrow','fasttext'):
   self.assertNotIn(name,sys.modules)
 def test_only_decay_and_lineage_metadata_change_from_sota(self):
  cfg=self.cfg();old=read_json(ROOT.parents[1]/'raw3b-sota-20261015/config.json')
  for key in ('sweep_id','sweep_stage','sweep_purpose','parent_runs','wsd_decay_ratio','hold_schedule','phase_schedule'):
   cfg.pop(key);old.pop(key)
  self.assertEqual(cfg,old)
  cfg=self.cfg();old=read_json(ROOT.parents[1]/'raw3b-sota-20261015/config.json')
  for a,b in zip(cfg['phase_schedule'],old['phase_schedule']):
   self.assertEqual({k:v for k,v in a.items() if k!='hold_schedule'},{k:v for k,v in b.items() if k!='hold_schedule'})
   self.assertEqual(a['hold_schedule'],cfg['hold_schedule'])
  self.assertEqual(cfg['wsd_decay_ratio'],.2)
  self.assertEqual(cfg['hold_schedule']['plateau_lr_ratio'],.2)
  self.assertEqual(cfg['max_steps']*131072,cfg['target_training_tokens'])
 def test_phase_boundaries_plateau_warmup_zero_and_reheat(self):
  cfg=self.cfg()
  for phase,bounds in zip(cfg['phase_schedule'],[(15259,17802,18438,19074),(3052,3561,3688,3815)]):
   start=phase['start'];length=phase['end']-start;peak=phase['peak_ratio']
   self.assertEqual(hold_schedule_boundaries(length,170,phase['hold_schedule']),bounds)
   self.assertAlmostEqual(schedule_multiplier(start,cfg),peak/170)
   for i in (169,170,bounds[0]-1,bounds[0]):self.assertAlmostEqual(schedule_multiplier(start+i,cfg),peak)
   for i in (bounds[1]-1,bounds[1],bounds[2]-1,bounds[2]):self.assertAlmostEqual(schedule_multiplier(start+i,cfg),peak*.2)
   self.assertEqual(schedule_multiplier(phase['end']-1,cfg),0)
   values=[schedule_multiplier(start+i,cfg) for i in range(170,length)]
   self.assertTrue(all(a>=b-1e-15 for a,b in zip(values,values[1:])))
  self.assertAlmostEqual(schedule_multiplier(19074,cfg),.1/170)
 def test_exact_fresh_formal11_and_single_allocation_cap(self):
  cfg=self.cfg();self.assertTrue(authorized(RUN_ID,cfg));self.assertFalse(authorized(RUN_ID,cfg,True))
  for key in ('learning_rate','max_steps','wsd_decay_ratio'):
   bad=copy.deepcopy(cfg);bad[key]*=1.01;self.assertFalse(authorized(RUN_ID,bad))
  for run,seconds,gpus,resume in [('unrequested',21600,8,False),(RUN_ID,21601,8,False),(RUN_ID,21600,4,False),(RUN_ID,21600,8,True)]:
   with tempfile.TemporaryDirectory() as t:
    ledger=ExperimentLedger(t)
    with ledger.transaction() as d:d['runs'].update({str(i):dict(category='full') for i in range(10)})
    with self.assertRaises(ValueError):ledger.reserve('bad',run,cfg,seconds,gpus,resume)
  with tempfile.TemporaryDirectory() as t:
   ledger=ExperimentLedger(t)
   with ledger.transaction() as d:d['runs'].update({str(i):dict(category='full') for i in range(10)})
   ledger.reserve('once',RUN_ID,cfg,21600,8,False)
   self.assertEqual(read_json(ledger.path)['runs'][RUN_ID]['formal_run_number'],11)
   self.assertEqual(len(read_json(ledger.path)['runs']),11)
   with self.assertRaises(ValueError):ledger.reserve('twice',RUN_ID,cfg,21600,8,False)
 def controller(self):
  spec=importlib.util.spec_from_file_location('wsd20_controller',ROOT/'slurm/sota3b_controller.py')
  c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c
 def test_existing_submission_never_calls_slurm_again(self):
  c=self.controller()
  with patch.object(c,'owner',return_value=True),patch.object(c,'check'),patch.object(c.subprocess,'run',side_effect=AssertionError('must not resubmit')):
   c.submit({'job':{'status':'submission_uncertain'}})
 def test_uncertain_submission_keeps_reservation_and_defers_without_resubmit(self):
  from contextlib import nullcontext
  c=self.controller();state={'stage':'training','job':{'status':'submission_uncertain','reservation':'kept'}}
  with tempfile.TemporaryDirectory() as t:
   state_path=Path(t)/'state.json';state_path.write_text('{}')
   with patch.object(c,'STATE',state_path),patch.object(c,'owner',return_value=True),patch.object(c,'check'),patch.object(c,'locked',return_value=nullcontext()),patch.object(c,'read_json',return_value=state),patch.object(c,'report') as report,patch.object(c,'find_submission_job',side_effect=ValueError('not yet visible')),patch.object(c.subprocess,'run',side_effect=AssertionError('must not resubmit')):
    c.run_once()
   self.assertNotIn('blocked',state);self.assertEqual(state['job']['reservation'],'kept')
   self.assertEqual(state['job']['status'],'submission_uncertain');self.assertIn('recovery_deferred',state['job']);report.assert_called_once_with(state)
 def test_retirement_removes_only_own_cron_entry(self):
  from types import SimpleNamespace
  c=self.controller();other='* * * * * other-command # other-controller'
  mine='* * * * * my-command '+c.MARKER
  with patch.object(c.subprocess,'run',side_effect=[SimpleNamespace(returncode=0,stdout=other+'\n'+mine+'\n'),SimpleNamespace(returncode=0)]) as run:
   c.retire();self.assertEqual(run.call_args.kwargs['input'],other+'\n')

if __name__=='__main__':unittest.main()
