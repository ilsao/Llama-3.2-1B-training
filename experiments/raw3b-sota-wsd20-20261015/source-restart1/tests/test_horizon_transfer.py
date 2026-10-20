"""Pure configuration, held-out separation and orchestration checks."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('horizon_transfer',ROOT/'slurm/horizon_transfer_controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from training_state import load_config,lr_multiplier

class HorizonTransferTests(unittest.TestCase):
 def test_targets_count_complete_batches_and_finish_cooldown(self):
  for tokens,minutes in [(200000000,30),(600000000,60)]:
   cfg=c.make_configuration(dict(kind='decay',tokens=tokens,minutes=minutes,muon_lr=.012))
   self.assertEqual(cfg['target_training_tokens'],cfg['max_steps']*131072)
   self.assertGreaterEqual(cfg['target_training_tokens'],tokens)
   self.assertLess(cfg['target_training_tokens']-tokens,131072)
   self.assertEqual(cfg['ema'],[]);self.assertNotIn('phase_schedule',cfg)
   self.assertEqual(cfg['warmup_steps'],170);self.assertEqual(cfg['eval_documents'],1000)
   self.assertLessEqual(cfg['eval_steps'],cfg['max_steps']//10)
   self.assertAlmostEqual(cfg['learning_rate']/cfg['muon']['learning_rate'],.03)
   self.assertEqual(lr_multiplier(cfg['max_steps']-1,cfg['max_steps'],170,0,'wsd_hold_zero',hold_schedule=cfg['hold_schedule']),0)
   with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'cfg.json';p.write_text(json.dumps(cfg));load_config(p)
 def test_high_lr_target_waits_for_screen_and_never_retries_failed_screen(self):
  case=dict(key='high',requires='screen',minutes=30)
  with patch.object(c.subprocess,'run',side_effect=AssertionError('No RPC or sbatch expected')):
   c.submit({'jobs':{}},case)
   with self.assertRaises(ValueError):c.submit({'jobs':{'screen':dict(status='reconciled',slurm_state='FAILED')}},case)
 def test_duplicate_target_is_not_submitted_twice(self):
  with patch.object(c.subprocess,'run',side_effect=AssertionError('No duplicate')):
   c.submit({'jobs':{'same':{}}},dict(key='same'))
 def test_direct_scaling_law_and_heldout_prediction(self):
  rows=[(t,.012*(t/400000000)**(-.4)) for t in [200000000,400000000,600000000]]
  fit=c.linear_law(rows)
  self.assertAlmostEqual(fit['beta'],-.4)
  self.assertAlmostEqual(c.transfer.predict(fit,800000000),.012*2**(-.4))
 def fixture(self,root,boundary=False):
  grid=[.004,.008,.016,.032,.064];cases=[];state={'jobs':{}};previous=[]
  for tokens in [200015872,400031744,600047616]:
   optimum=.012*(tokens/400031744)**(-.4) if not boundary else .2
   for i,lr in enumerate(grid):
    nll=3+.2*math.log(lr/optimum)**2
    if tokens==400031744:previous.append(dict(muon_lr=lr,mean_nll=nll,perplexity=math.exp(nll)));continue
    key=f'{tokens}-{i}';cases.append(dict(key=key,kind='decay',tokens=tokens,muon_lr=lr))
    state['jobs'][key]=dict(job_id=key,endpoint=dict(tokens=tokens,evaluation_identity='fixed',wandb_url='test',evaluation={'eval/nll':nll*1479302,'eval/prediction_tokens':1479302,'eval/perplexity':math.exp(nll)}))
  (root/'plan.json').write_text(json.dumps(dict(cases=cases,previous_wsd_points=previous,evaluation_identity='fixed')))
  return state
 def test_freezes_unvalidated_predictions_only_after_bracketing_each_horizon(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c,'LIVE',Path(t)):
   result=c.analyze(self.fixture(Path(t)))
   self.assertEqual(result['errors'],{});self.assertFalse(result['validated'])
   self.assertAlmostEqual(result['direct_wsd_law']['beta'],-.4)
   self.assertEqual([p['tokens'] for p in result['predictions']],[800000000,1000000000,2000000000,3000000000,4000000000,5000000000,6000000000])
   self.assertEqual(len(result['leave_one_horizon_out']),3)
 def test_boundary_minimum_never_produces_transfer_forecast(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c,'LIVE',Path(t)):
   result=c.analyze(self.fixture(Path(t),True))
   self.assertTrue(result['errors']);self.assertNotIn('predictions',result)

if __name__=='__main__':unittest.main()
