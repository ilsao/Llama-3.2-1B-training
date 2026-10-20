import copy,json,math,sys,tempfile,time,unittest
from pathlib import Path
LIVE=Path(__file__).resolve().parent
sys.path.insert(0,str(LIVE/'source'))
from training_state import load_config,ExperimentLedger,read_json,schedule_multiplier,hold_schedule_boundaries
from ppl_contracts import STUDY,TARGET,Q,END,SPLIT,SWITCH,batch_plan
from fourb_authorization import authorized

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg=load_config(LIVE/'config.json');cls.rows=batch_plan(cls.cfg)
    def test_nonpadding_budget_and_events(self):
        rows=self.rows;self.assertEqual((len(rows),rows[-1]['end_tokens']),(28619,6000082944))
        self.assertGreaterEqual(TARGET,6000000000);self.assertLess(TARGET,6000000000+Q)
        cursor=0
        for r in rows:
            self.assertEqual(r['start_tokens'],cursor);self.assertEqual(r['end_tokens']-cursor,r['width']*8192)
            self.assertEqual(r['accumulation']*8,r['width']);self.assertIn(r['width'],(16,32,48,64));cursor=r['end_tokens']
        self.assertEqual(sum(r['end_tokens']<=SPLIT*Q for r in rows),26708)
        self.assertTrue(all(r['width']==16 for r in rows if r['start_tokens']<SWITCH*Q))
        self.assertEqual(self.rows[-1]['lr_multiplier'],0.)
    def test_token_clock_and_peaks(self):
        c=self.cfg
        self.assertEqual(c['learning_rate'],.0001277883379390995*2)
        for phase in c['phase_schedule']:
            start,end=phase['start'],phase['end'];peak=phase['peak_ratio']
            bounds=hold_schedule_boundaries(end-start,170,phase['hold_schedule'])
            self.assertAlmostEqual(schedule_multiplier(start,c),peak/170)
            self.assertAlmostEqual(schedule_multiplier(start+169,c),peak)
            self.assertAlmostEqual(schedule_multiplier(start+bounds[1],c),peak*.2)
            self.assertEqual(schedule_multiplier(end-1,c),0.)
    def test_single_exact_allocation(self):
        c=self.cfg;self.assertTrue(authorized(STUDY,c));self.assertFalse(authorized(STUDY,c,True))
        for field,value in [('learning_rate',c['learning_rate']/2),('max_h200_hours',81),('max_training_tokens',TARGET+Q)]:
            bad=copy.deepcopy(c);bad[field]=value;self.assertFalse(authorized(STUDY,bad))
        with tempfile.TemporaryDirectory() as tmp:
            ledger=ExperimentLedger(tmp)
            with ledger.transaction() as d:
                for i in range(16):d['runs'][f'old{i}']=dict(category='full',config_digest=str(i),max_h200_hours=64)
                d['jobs']['failed-history']=dict(run_id='old0',h200_hours=.1,reserved_h200_hours=1,state='FAILED')
            ledger.reserve('new',STUDY,c,36000,8,False)
            d=read_json(ledger.path);self.assertEqual(d['runs'][STUDY]['formal_run_number'],17)
            self.assertEqual(d['jobs']['new']['reserved_h200_hours'],80)
            self.assertEqual(d['jobs']['failed-history']['h200_hours'],.1)
            for name,seconds,run,resume in [('twice',36000,STUDY,False),('too-long',36001,STUDY,False),('other',36000,STUDY+'-other',False),('resume',36000,STUDY,True)]:
                with self.assertRaises(ValueError):ledger.reserve(name,run,c,seconds,8,resume)
    def test_invalid_mask_rejected(self):
        c=copy.deepcopy(self.cfg);c['ppl']['mask']='dense'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):load_config(p)
    def test_tail_refill_preserves_document_boundaries(self):
        from dataset import TrainingPacker
        p=TrainingPacker(8,0);self.assertEqual(list(p.push(10,[1,2,3,4,5])),[])
        row=list(p.finish())[0];valid=sum(row['attention_mask'])
        filled=TrainingPacker(8,0,dict(tokens=row['input_ids'][:valid],labels=row['labels'][:valid],spans=copy.deepcopy(row['document_spans'])))
        new=list(filled.push(11,[6,7,8,9,10,11]))[0]
        self.assertEqual(new['input_ids'],[1,2,3,4,5,6,7,8]);self.assertEqual(new['attention_mask'],[1]*8)
        self.assertEqual(new['labels'][:valid],row['labels'][:valid]);self.assertEqual(new['labels'][5],-100)
        self.assertEqual(new['document_spans'][0],row['document_spans'][0])
        self.assertEqual(filled.state()['spans'][0]['document_start'],3)
    def test_wandb_timeout_keeps_receipt(self):
        from bounded_tracking import bounded
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';r=bounded(lambda:time.sleep(.3),p,seconds=.02)
            self.assertFalse(r['synced']);self.assertEqual(read_json(p)['error_type'],'TrackingTimeout')

if __name__=='__main__':unittest.main()
