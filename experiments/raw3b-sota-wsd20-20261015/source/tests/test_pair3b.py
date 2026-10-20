"""Pure authorization and token-proportion tests. No corpus/model loading."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pair3b_contracts import authorized,RUNS
from training_state import load_config,ExperimentLedger
from mixture_reweight import choose_source,check_mix,TARGET

class Pair3BTests(unittest.TestCase):
 def config(self):return load_config(ROOT.parent/'configs/raw.json')
 def test_only_exact_requested_fresh_pair_is_authorized(self):
  cfg=self.config();old=load_config(ROOT.parent/'configs/filtered.json')
  for key in ['muon','learning_rate','hold_schedule','target_training_tokens','max_steps','world_size','seed']:self.assertEqual(cfg[key],old[key])
  for run in RUNS:
   self.assertTrue(authorized(run,cfg));self.assertFalse(authorized(run,cfg,True))
   changed=json.loads(json.dumps(cfg));changed['muon']['learning_rate']*=1.01
   self.assertFalse(authorized(run,changed))
  self.assertFalse(authorized('unrequested-3b',cfg))
  self.assertFalse(authorized('raw-filtered-3b-20261015-filtered',cfg))
 def test_formal_numbers_and_six_hour_caps_preserve_eight_earlier_runs(self):
  with tempfile.TemporaryDirectory() as t:
   ledger=ExperimentLedger(t)
   with ledger.transaction() as d:
    d['runs'].update({f'old{i}':dict(category='full') for i in range(8)})
   cfg=self.config()
   for i,run in enumerate(RUNS):ledger.reserve('tag'+str(i),run,cfg,21600,8,False)
   with ledger.transaction() as d:
    self.assertEqual(len(d['runs']),8+len(RUNS))
    for run,n in RUNS.items():self.assertEqual(d['runs'][run]['formal_run_number'],n)
   with self.assertRaises(ValueError):ledger.reserve('extra','unrequested-3b',cfg,21600,8,False)
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):ExperimentLedger(t).reserve('too-long',next(iter(RUNS)),cfg,21601,8,False)
 def test_actual_complete_batch_geometry_and_lr(self):
  cfg=self.config();self.assertEqual(cfg['max_steps']*8192*16,TARGET)
  self.assertEqual(cfg['target_training_tokens'],TARGET);self.assertEqual(cfg['ema'],[])
  self.assertNotIn('phase_schedule',cfg);self.assertEqual(cfg['warmup_steps'],170)
  self.assertAlmostEqual(cfg['muon']['learning_rate'],.005593716406714465)
  self.assertAlmostEqual(cfg['learning_rate']/cfg['muon']['learning_rate'],.03)
 def test_token_deficit_mixing_uses_lengths_not_document_counts(self):
  weights={'a':.8,'b':.2};counts={'a':0,'b':0};docs={'a':0,'b':0};lengths={'a':100,'b':1000}
  for _ in range(1000):
   source=choose_source(counts,weights);counts[source]+=lengths[source];docs[source]+=1
  self.assertLess(abs(counts['a']/sum(counts.values())-.8),.01)
  self.assertGreater(docs['a']/sum(docs.values()),.95)
 def test_final_check_counts_actual_processed_tokens_and_rejects_drift(self):
  raw={'a':TARGET//2,'b':TARGET-TARGET//2}
  self.assertEqual(max(check_mix(raw,raw).values()),0)
  with self.assertRaises(ValueError):check_mix({'a':TARGET},raw)
  with self.assertRaises(ValueError):check_mix({'a':10},raw)

class CancelledFilteredTests(unittest.TestCase):
 def test_cancelled_filtered_never_queries_slurm_or_submits(self):
  import importlib.util
  from unittest.mock import patch
  spec=importlib.util.spec_from_file_location('paircontroller',ROOT/'slurm/pair3b_controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
  with patch.object(c,'read_json',return_value={'filtered_enabled':False}),patch.object(c.subprocess,'run',side_effect=AssertionError('Cancelled arm cannot touch Slurm')):
   for key in ['mixprep','filtered']:c.submit({'jobs':{}},key,'preprocessing',960)

if __name__=='__main__':unittest.main()
