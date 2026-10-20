"""Login-safe numerical planning and exact authorization tests; no torch/model."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
LIVE=Path(__file__).resolve().parent
sys.path.insert(0,str(LIVE/'source'))
from training_state import load_config,ExperimentLedger,digest,atomic_json,read_json
from ppl_contracts import *
from fourb_authorization import authorized

class FourB(unittest.TestCase):
 def test_plan_and_lookup(self):
  cfg=load_config(LIVE/'config.json');rows=batch_plan(cfg)
  self.assertEqual(TARGET,4000055296);self.assertEqual(len(rows),19080)
  self.assertEqual(rows[-1]['end_tokens'],TARGET);self.assertEqual(rows[-1]['lr_multiplier'],0)
  self.assertEqual(sum(x['width']*8192 for x in rows),TARGET)
  endpoints={r['end_tokens']//Q for r in rows}
  self.assertTrue(set(boundaries())-{0}<=endpoints)
  self.assertTrue(set(evaluation_units())<=endpoints)
  self.assertLessEqual(cfg['eval_steps'],len(rows)//10)
  self.assertEqual(cfg['ema'],[])
  self.assertEqual(cfg['ppl'],dict(stage=3,arm='nor',mask='intradoc',matrix='nor'))
  for i,row in enumerate(rows):
   self.assertEqual(row['start_tokens'],0 if i==0 else rows[i-1]['end_tokens'])
   self.assertEqual(row['accumulation']*8,row['width'])
   self.assertLessEqual(row['width'],16 if row['start_tokens']<SWITCH*Q else 64)
  row=next(x for x in read_json(LIVE.parents[1]/'experiments/lr-transfer-horizons-20261015/calibration-result.json')['predictions'] if x['tokens']==4000000000)
  self.assertEqual(cfg['muon']['learning_rate'],row['muon_lr']);self.assertEqual(cfg['learning_rate'],2*row['adamw_lr'])

 def test_scoped_authorization(self):
  cfg=load_config(LIVE/'config.json')
  self.assertTrue(authorized(STUDY,cfg));self.assertFalse(authorized(STUDY,cfg,True))
  self.assertFalse(authorized(STUDY+'-other',cfg))
  changed=copy.deepcopy(cfg);changed['muon']['learning_rate']*=1.01
  self.assertFalse(authorized(STUDY,changed))
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'changed.json';atomic_json(p,changed)
   with self.assertRaises(ValueError):load_config(p)

 def test_ledger_preserves_history_and_single_56h_formal15(self):
  cfg=load_config(LIVE/'config.json')
  with tempfile.TemporaryDirectory() as tmp:
   ledger=ExperimentLedger(tmp)
   with ledger.transaction() as data:
    for i in range(14):data['runs'][f'old{i}']=dict(category='full',config_digest=str(i),max_h200_hours=64)
    data['jobs']['old-failure']=dict(run_id='old0',h200_hours=.1,reserved_h200_hours=1,state='FAILED')
   ledger.reserve('new-allocation',STUDY,cfg,25200,8,False)
   data=read_json(ledger.path)
   self.assertEqual(data['runs'][STUDY]['formal_run_number'],15)
   self.assertEqual(data['jobs']['new-allocation']['reserved_h200_hours'],56)
   self.assertEqual(data['jobs']['old-failure']['h200_hours'],.1)
   self.assertEqual(len(data['runs']),15)
   with self.assertRaises(ValueError):ledger.reserve('again',STUDY,cfg,25200,8,False)
   with self.assertRaises(ValueError):ledger.reserve('other',STUDY+'-other',cfg,25200,8,False)

if __name__=='__main__':unittest.main()
