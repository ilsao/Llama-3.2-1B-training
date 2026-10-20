"""Lightweight plan/controller tests: no torch, models or Slurm submissions."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
LIVE=Path(__file__).resolve().parent
sys.path.insert(0,str(LIVE/'source-r2'))
from training_state import load_config,atomic_json,schedule_multiplier
from ppl_contracts import *
spec=importlib.util.spec_from_file_location('ppl_controller',LIVE/'source-r2/slurm/ppl_controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)

class Contracts(unittest.TestCase):
 def test_matched_configs_and_plan(self):
  configs=[load_config(p) for p in (LIVE/'configs').glob('*.json')]
  self.assertEqual(len(configs),6)
  stripped=[]
  for cfg in configs:
   c=copy.deepcopy(cfg);c.pop('ppl');stripped.append(c)
  self.assertTrue(all(c==stripped[0] for c in stripped))
  cfg=configs[0];rows=batch_plan(cfg)
  self.assertEqual(len(rows),4775);self.assertEqual(rows[-1]['end_tokens'],TARGET)
  self.assertEqual(sum(r['width']*8192 for r in rows),TARGET)
  self.assertEqual(rows[-1]['lr_multiplier'],0)
  endpoints={r['end_tokens']//Q for r in rows}
  self.assertTrue(set(boundaries())-{0}<=endpoints)
  self.assertTrue(set(evaluation_units())<=endpoints)
  self.assertEqual(len(set(observer_knots(cfg)['mean'])),5)
  self.assertLessEqual(cfg['eval_steps'],len(rows)//10)
  for i,r in enumerate(rows):
   self.assertEqual(r['start_tokens'],0 if i==0 else rows[i-1]['end_tokens'])
   self.assertLessEqual(r['width'],16 if r['start_tokens']<SWITCH*Q else 64)
   self.assertEqual(r['accumulation']*8,r['width'])

 def test_selection_complete_pair_only_and_raw(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(controller,'LIVE',Path(tmp)):
   jobs={controller.run_id(1,a):dict(status='complete') for a in ('dense','intradoc')}
   for arm,nll in [('dense',3.0),('intradoc',2.99)]:
    run=controller.run_id(1,arm)
    atomic_json(Path(tmp)/'receipts'/f'{run}.json',dict(run_id=run,mean_nll=nll,raw_perplexity=20.,initialization='same',provenance='same',source_digest='same',environment_digest='same'))
   incomplete=copy.deepcopy(jobs);incomplete[controller.run_id(1,'intradoc')]['status']='submitted'
   self.assertIsNone(controller.selection(dict(jobs=incomplete)))
   result=controller.selection(dict(jobs=jobs));self.assertEqual(result['mask'],'intradoc')
   self.assertFalse(result['checkpoint_training_load'])
   p=Path(tmp)/'receipts'/f"{controller.run_id(1,'intradoc')}.json"
   rec=json.loads(p.read_text());rec['mean_nll']=3.01;atomic_json(p,rec)
   with self.assertRaises(ValueError):controller.selection(dict(jobs=jobs))

 def test_tick_two_parallel_submissions_then_two_winner_configs(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(controller,'LIVE',Path(tmp)),patch.object(controller,'STATE',Path(tmp)/'state.json'),patch.object(controller,'reconcile'),patch.object(controller,'check'):
   atomic_json(Path(tmp)/'authorization.json',dict(approved=True))
   calls=[]
   with patch.object(controller,'submit',side_effect=lambda state,stage,arm,mask=None:calls.append((stage,arm,mask))),patch.object(controller,'selection',return_value=None):
    controller.tick()
   self.assertEqual(calls,[(1,'dense',None),(1,'intradoc',None)])
   calls.clear()
   with patch.object(controller,'submit',side_effect=lambda state,stage,arm,mask=None:calls.append((stage,arm,mask))),patch.object(controller,'selection',return_value={'mask':'intradoc'}):
    controller.tick()
   self.assertEqual(calls[-2:],[(2,'pe','intradoc'),(2,'nor','intradoc')])
   atomic_json(Path(tmp)/'state.json',dict(jobs={'failed-run':dict(status='failed')}))
   with patch.object(controller,'submit') as submit:
    result=controller.tick();submit.assert_not_called()
    self.assertEqual(result['status'],'stopped_on_failure')

if __name__=='__main__':unittest.main()
