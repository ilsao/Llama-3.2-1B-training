import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('bracket',ROOT/'slurm/wsd_bracket_controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from training_state import load_config

class BracketTests(unittest.TestCase):
 def test_high_lr_configs_only_change_the_lr_pair(self):
  cfg=c.make_configuration('decay',.016570970826409352)
  self.assertEqual(cfg['target_training_tokens'],400031744)
  self.assertAlmostEqual(cfg['learning_rate']/cfg['muon']['learning_rate'],.03)
  self.assertEqual(cfg['ema'],[]);self.assertEqual(cfg['initialization'],'residual_half')
  self.assertEqual(cfg['eval_documents'],1000)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'cfg.json';p.write_text(json.dumps(cfg));load_config(p)
 def test_no_duplicate_submission(self):
  with patch.object(c.subprocess,'run',side_effect=AssertionError('No sbatch expected')):
   c.submit({'jobs':{'decay0':{}}},'decay0','decay',.012)
 def test_metadata_timeout_defers_without_blocking_or_submitting(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c,'LIVE',Path(t)),patch.object(c,'STATE',Path(t)/'state.json'),patch.object(c,'owner',return_value=True),patch.object(c,'check'),patch.object(c,'report'),patch.object(c.base,'reconcile',side_effect=subprocess.TimeoutExpired(['sacct'],30)),patch.object(c,'submit',side_effect=AssertionError('No submit expected')):
   c.run_once();state=json.loads((Path(t)/'state.json').read_text())
   self.assertFalse(state.get('blocked',False));self.assertIn('scheduler_deferred',state);self.assertEqual(state['jobs'],{})
 def fixture(self,t,optimum):
  root=Path(t);grid=[.004,.006,.008,.012,.016,.024]
  observations=[{'muon_lr':lr,'adamw_lr':lr*.03,'mean_nll':3+.2*math.log(lr/optimum)**2,'perplexity':math.exp(3+.2*math.log(lr/optimum)**2)} for lr in grid]
  (root/'plan.json').write_text(json.dumps({'previous_wsd_points':observations[:3],'evaluation_identity':'fixed'}))
  state={'jobs':{}}
  for i,row in enumerate(observations[3:]):
   state['jobs']['decay'+str(i)]={'multiplier':row['muon_lr']/c.transfer.ANCHOR_MUON,'job_id':str(i),'endpoint':{'tokens':400031744,'evaluation_identity':'fixed','wandb_url':'test','evaluation':{'eval/nll':row['mean_nll']*1479302,'eval/prediction_tokens':1479302,'eval/perplexity':row['perplexity']}}}
  return state
 def test_brackets_and_recovers_known_optimum(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c,'LIVE',Path(t)):
   result=c.bracket_result(self.fixture(t,.012))
   self.assertTrue(result['bracketed']);self.assertAlmostEqual(result['local_quadratic']['muon_lr'],.012)
 def test_boundary_minimum_is_not_reported_as_bracketed(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c,'LIVE',Path(t)):
   result=c.bracket_result(self.fixture(t,.05))
   self.assertFalse(result['bracketed']);self.assertNotIn('local_quadratic',result)
 def test_mismatched_evaluation_is_rejected(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c,'LIVE',Path(t)):
   state=self.fixture(t,.012);state['jobs']['decay0']['endpoint']['evaluation_identity']='wrong'
   with self.assertRaises(ValueError):c.bracket_result(state)

if __name__=='__main__':unittest.main()
