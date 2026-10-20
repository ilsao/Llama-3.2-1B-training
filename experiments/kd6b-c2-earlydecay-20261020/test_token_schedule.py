"""Pure arithmetic/regression checks; no torch/models/workers on login node."""
import copy,json,random,types,unittest
from pathlib import Path
import bootstrap
from contracts import CFG,TARGET,Q,event_units,eval_units
from critical_batch import window
from token_schedule import multiplier,Scheduler
from training_state import schedule_multiplier,digest
from launch import reserve_entries

class TokenScheduleTests(unittest.TestCase):
    def test_exact_decay_boundary_and_slope(self):
        d=2399928320
        self.assertEqual(multiplier(d-Q,CFG),1)
        self.assertEqual(multiplier(d,CFG),1)
        self.assertLess(multiplier(d+Q,CFG),1)
        for p in CFG['token_phases']:
            h,f,e=[p[k] for k in ('plateau_start_tokens','final_decay_start_tokens','end_tokens')]
            self.assertAlmostEqual(multiplier(h-Q,CFG),p['peak_ratio']*.2)
            self.assertAlmostEqual(multiplier(h,CFG),p['peak_ratio']*.2)
            self.assertAlmostEqual(multiplier(f,CFG),p['peak_ratio']*.2)
            self.assertAlmostEqual(multiplier(e-Q,CFG),0)
        self.assertEqual(multiplier(TARGET,CFG),0)
    def test_original_second_phase_exactly_preserved(self):
        old=json.loads((bootstrap.HERE.parent/'kd6b-r5-20261020/config.json').read_text())
        for u in range(38147,45777):
            self.assertAlmostEqual(multiplier(u*Q,CFG),schedule_multiplier(u,old),places=13)
    def test_lr_table_and_c2_exact(self):
        p=Path('experiments/lr-transfer-horizons-20261015/calibration-result.json')
        row=next(x for x in json.loads(p.read_text())['predictions'] if x['tokens']==6000000000)
        self.assertEqual(CFG['muon']['learning_rate'],row['muon_lr'])
        self.assertEqual(CFG['learning_rate'],2*row['adamw_lr'])
        old=json.loads((bootstrap.HERE.parent/'critical-batch-20261020/configs/c2-cbs-one.json').read_text())
        self.assertEqual(CFG['cbs'],old['cbs'])
    def test_dynamic_batch_cannot_move_token_events(self):
        events=[x*Q for x in event_units(CFG)];anchor=2399928320
        for policy in ('small','large','random'):
            seen=step=0;reached=False;next_eval=0;last_eval_step=0
            evals={x*Q for x in eval_units(CFG)};rng=random.Random(42)
            while seen<TARGET:
                desired={'small':1,'large':8}.get(policy) or rng.choice([1,2,4,8])
                r=window(seen,step,desired,events,TARGET)
                self.assertFalse(seen<anchor<r['end_tokens'])
                if seen==anchor:reached=True
                seen=r['end_tokens'];step=r['update']
                if seen in evals:
                    self.assertLessEqual(step-last_eval_step,570);last_eval_step=step
            self.assertTrue(reached);self.assertEqual(seen,TARGET)
    def test_scheduler_serialization_and_input_clock(self):
        o=types.SimpleNamespace(param_groups=[dict(lr=CFG['learning_rate'],name='adamw'),dict(lr=CFG['muon']['learning_rate'],name='muon')])
        s=Scheduler(o,CFG);s.set_tokens(2399928320+Q);state=s.state_dict()
        o2=types.SimpleNamespace(param_groups=[dict(lr=CFG['learning_rate'],name='adamw'),dict(lr=CFG['muon']['learning_rate'],name='muon')])
        t=Scheduler(o2,CFG);t.load_state_dict(state)
        self.assertEqual(s.state_dict(),t.state_dict());self.assertEqual(o.param_groups,o2.param_groups)
        with self.assertRaises(ValueError):s.set_tokens(1841)
        with self.assertRaises(ValueError):s.set_tokens(TARGET+Q)
    def test_ledger_preserves_history_and_refuses_second_allocation(self):
        # Build synthetic small records with the exact authorized run identities.
        scope=json.loads((bootstrap.HERE/'ledger-scope.json').read_text())
        d=dict(runs={k:dict(category='full' if k in scope['formal_ids'] else 'pilot') for k in scope['run_ids']},jobs={'old':dict(run_id='old',reserved_h200_hours=1,kept=True)})
        old=copy.deepcopy(d);reserve_entries(d,'new')
        self.assertEqual({k:v for k,v in d['runs'].items() if k in old['runs']},old['runs']);self.assertEqual(d['jobs']['old'],old['jobs']['old'])
        with self.assertRaises(AssertionError):reserve_entries(d,'second')
    def test_runner_uses_token_scheduler_and_fresh_paths(self):
        s=(bootstrap.HERE/'runner.py').read_text()
        self.assertIn('from token_schedule import Scheduler',s)
        self.assertIn('for scheduler in schedulers:scheduler.set_tokens(seen)',s)
        self.assertNotIn('startup-replacement',s)
        self.assertNotIn('==7080',s)
        self.assertIn('logical_charged_inputs_per_role=seen,',s)
        self.assertNotIn('job_type=\'pilot\'',s)

if __name__=='__main__':unittest.main()
