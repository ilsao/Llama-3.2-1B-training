import copy,json,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
LIVE=Path(__file__).resolve().parent
sys.path.insert(0,str(LIVE/'source'))
from training_state import load_config,read_json,atomic_json,ExperimentLedger,hold_schedule_boundaries
from ppl_contracts import STUDY,Q,SPLIT,TARGET,batch_plan,token_prefix,multiplier_for_step
from batch3b_runtime import StepScheduler
from fourb_authorization import authorized

class Step30(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=load_config(LIVE/'config.json');cls.rows=batch_plan(cls.cfg)
  cls.old=read_json(LIVE.parent/'raw4b-normuon-adam2x-20261017/batch-plan.json')['rows']
 def test_matched_data_and_mask(self):
  self.assertEqual(len(self.rows),19080)
  for a,b in zip(self.rows,self.old):
   self.assertEqual({k:v for k,v in a.items() if k!='lr_multiplier'},{k:v for k,v in b.items() if k!='lr_multiplier'})
  self.assertEqual(token_prefix()[17804],3333292032);self.assertEqual(token_prefix()[-1],4000055296)
  self.assertEqual(self.cfg['ppl']['mask'],'intradoc');self.assertEqual(self.cfg['ppl']['matrix'],'nor')
  parent=LIVE.parent/'raw4b-normuon-adam2x-20261017/source'
  for name in ('attention_backend.py','ppl_optimizer.py'):
   self.assertEqual((LIVE/'source'/name).read_bytes(),(parent/name).read_bytes())
 def test_decay_in_actual_steps(self):
  # Independent expected boundary integers after round(20%,5%,5%) in each phase.
  for phase,expected in zip(self.cfg['phase_schedule'],[(12463,16024,16914,17804),(893,1148,1212,1276)]):
   n=phase['end']-phase['start'];peak=phase['peak_ratio'];base=phase['start'];start,hold,final,end=expected
   self.assertEqual(hold_schedule_boundaries(n,0,phase['hold_schedule']),expected)
   self.assertLessEqual(abs((n-start)/n-.3),1/n)
   for i in range(start,hold):
    self.assertAlmostEqual(self.rows[base+i]['lr_multiplier'],peak*(1-.8*(i-start)/(hold-start-1)))
   for i in range(hold,final):self.assertAlmostEqual(self.rows[base+i]['lr_multiplier'],peak*.2)
   for i in range(final,end):self.assertAlmostEqual(self.rows[base+i]['lr_multiplier'],peak*.2*(1-(i-final)/(end-final-1)))
   self.assertEqual(self.rows[base+end-1]['lr_multiplier'],0.)
  self.assertEqual(self.rows[12463]['start_tokens'],1633550336)
  self.assertEqual(self.rows[18697]['start_tokens'],3800432640)
  self.assertLess(self.rows[14000]['lr_multiplier'],self.old[14000]['lr_multiplier'])
 def test_preserve_token_warmup(self):
  counts=[0,0]
  for r,old in zip(self.rows,self.old):
   phase=0 if r['start_tokens']<SPLIT*Q else 1
   local=r['start_tokens']-(0 if phase==0 else SPLIT*Q)
   if local<170*Q:
    self.assertAlmostEqual(r['lr_multiplier'],old['lr_multiplier']);counts[phase]+=1
  self.assertEqual(counts,[170,43])
 def test_scheduler_consumes_actual_steps_and_roundtrips(self):
  opt=SimpleNamespace(param_groups=[dict(lr=.005,name='muon')]);s=StepScheduler(opt,self.cfg)
  for i in (0,169,170,12463,14000,15259,17803,17804,17846,17847,18697,18952,19016,19079,19080):
   s.set_step(i,token_prefix()[i]);self.assertAlmostEqual(opt.param_groups[0]['lr'],.005*multiplier_for_step(i,self.cfg))
   state=copy.deepcopy(s.state_dict());other=StepScheduler(SimpleNamespace(param_groups=[dict(lr=.005,name='muon')]),self.cfg)
   other.load_state_dict(state);self.assertEqual(other.state_dict(),state)
  with self.assertRaises(ValueError):s.set_step(18000,token_prefix()[17999])
  bad=copy.deepcopy(state);bad['warmup_clock']='optimizer_updates'
  with self.assertRaises(ValueError):s.load_state_dict(bad)
 def test_audit_rejects_previous_token_clock_log(self):
  from schedule_audit import audit_log
  path=Path('/work/asciibase64/Llama-3.2-1B-training/logs/raw4b-normuon-adam2x-563624.out')
  with self.assertRaisesRegex(ValueError,'Actual LR disagrees'):
   audit_log(path,self.cfg,require_complete=False)
 def test_exact_single_formal16_authorization(self):
  cfg=self.cfg;self.assertTrue(authorized(STUDY,cfg));self.assertFalse(authorized(STUDY,cfg,True))
  bad=copy.deepcopy(cfg);bad['ppl']['mask']='dense';self.assertFalse(authorized(STUDY,bad))
  with tempfile.TemporaryDirectory() as tmp:
   ledger=ExperimentLedger(tmp)
   with ledger.transaction() as d:
    for i in range(15):d['runs'][f'old{i}']=dict(category='full',config_digest=str(i),max_h200_hours=64)
    d['jobs']['prior-failure']=dict(run_id='old0',h200_hours=.1,reserved_h200_hours=1,state='FAILED')
   ledger.reserve('new',STUDY,cfg,25200,8,False);d=read_json(ledger.path)
   self.assertEqual(d['runs'][STUDY]['formal_run_number'],16)
   self.assertEqual(d['jobs']['new']['reserved_h200_hours'],56)
   self.assertEqual(d['jobs']['prior-failure']['h200_hours'],.1)
   with self.assertRaises(ValueError):ledger.reserve('again',STUDY,cfg,25200,8,False)
   with self.assertRaises(ValueError):ledger.reserve('over',STUDY,cfg,25201,8,False)
   with self.assertRaises(ValueError):ledger.reserve('other',STUDY+'-other',cfg,25200,8,False)

if __name__=='__main__':unittest.main()
