"""No torch/model work: sealed factorial contracts and submission state graph."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
LIVE=Path(__file__).resolve().parent
sys.path.insert(0,str(LIVE/'source'))
from training_state import load_config,atomic_json
from ppl_contracts import batch_plan,TARGET,boundaries,evaluation_units
spec=importlib.util.spec_from_file_location('controller',LIVE/'source/slurm/factorial_controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Execution(unittest.TestCase):
    def test_only_two_factors_change(self):
        configs=[load_config(LIVE/'configs'/f'{a}-seed42.json') for a in 'ABCD']
        stripped=[]
        for cfg in configs:
            cfg=copy.deepcopy(cfg);cfg.pop('ppl');cfg.pop('learning_rate');stripped.append(cfg)
        self.assertTrue(all(x==stripped[0] for x in stripped))
        self.assertEqual(configs[1]['learning_rate'],2*configs[0]['learning_rate'])
        self.assertEqual([x['ppl']['cautious_decay'] for x in configs],[False,False,True,True])

    def test_exact_endpoint_and_events(self):
        for p in (LIVE/'configs').glob('*.json'):
            cfg=load_config(p);rows=batch_plan(cfg)
            self.assertEqual(len(rows),4775);self.assertEqual(rows[-1]['end_tokens'],TARGET)
            self.assertEqual(sum(r['width']*8192 for r in rows),TARGET)
            self.assertEqual(rows[-1]['lr_multiplier'],0)
            endpoints={r['end_tokens']//131072 for r in rows}
            self.assertTrue((set(boundaries())|set(evaluation_units()))-{0}<=endpoints)
            self.assertLessEqual(cfg['eval_steps'],len(rows)//10)

    def test_tampering_rejected(self):
        from ppl_contracts import validate
        cfg=load_config(LIVE/'configs/A-seed42.json')
        for key,value in [('seed',44),('learning_rate',1),('max_training_tokens',6000000001)]:
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):validate(bad)

    def test_wave_dependencies_and_failure_stop(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(c,'LIVE',Path(temp)),patch.object(c,'STATE',Path(temp)/'state.json'),patch.object(c,'check'),patch.object(c,'reconcile'):
            calls=[]
            def submit(state,key,kind='train'):
                if key not in state['jobs']:calls.append((key,kind));state['jobs'][key]=dict(status='submitted',kind=kind)
            with patch.object(c,'submit',side_effect=submit):
                c.tick();self.assertEqual(calls,[('gate','gate')])
                atomic_json(c.STATE,dict(jobs={'gate':dict(status='complete')}));calls.clear()
                c.tick();self.assertEqual(calls,[('A-seed42','train'),('B-seed42','train')])
                atomic_json(c.STATE,dict(jobs={k:dict(status='complete') for k in ('gate','A-seed42','B-seed42')}));calls.clear()
                c.tick();self.assertEqual(calls,[('C-seed42','train'),('D-seed42','train')])
                atomic_json(c.STATE,dict(jobs={'gate':dict(status='failed')}));calls.clear()
                self.assertEqual(c.tick()['status'],'stopped_on_failure');self.assertFalse(calls)

    def test_selection_practical_cost_and_identity_gates(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(c,'LIVE',Path(temp)):
            state=dict(jobs={})
            for arm,loss in zip('ABCD',[3,2.999,2.99,2.98]):
                atomic_json(Path(temp)/'receipts'/f'{arm}-seed42.json',dict(mean_nll=loss,initialization='same',provenance='same',source_digest='same',environment_digest='same'))
                state['jobs'][arm+'-seed42']=dict(accounting=dict(h200_hours=10))
            r=c.selection(state);self.assertEqual(r['winner'],'D');self.assertTrue(r['qualified'])
            self.assertAlmostEqual(r['interaction_nll'],-.009)
            state['jobs']['D-seed42']['accounting']['h200_hours']=11
            with self.assertRaises(ValueError):c.selection(state)

if __name__=='__main__':unittest.main()
