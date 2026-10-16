"""Pure contracts and submission checks;no model or torch import on login."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

LIVE = Path(__file__).resolve().parents[1]
SOURCE = LIVE / 'source'
sys.path.insert(0, str(SOURCE))
from batch3b_contracts import (STUDY, Q, END, TARGET, SWITCH, SPLIT, PLANNED_UPDATES,
    ADAM_LR, MUON_LR, events, evaluation_units, window, count_updates, beta2_for)
from batch3b_runtime import TokenScheduler, configure_update
from batch3b_authorization import authorized
from training_state import (atomic_json, read_json, load_config, schedule_multiplier,
                            hold_schedule_boundaries, ExperimentLedger)

spec = importlib.util.spec_from_file_location('new3b_controller', SOURCE / 'slurm/batch3b_controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class Study(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(LIVE / 'config.json')

    def test_exact_table_lrs(self):
        table = read_json(LIVE.parent / 'lr-transfer-horizons-20261015/calibration-result.json')
        row = next(r for r in table['predictions'] if r['tokens'] == 3000000000)
        self.assertEqual(row['muon_lr'], MUON_LR)
        self.assertEqual(row['adamw_lr'], ADAM_LR)
        self.assertEqual(row['status'], 'unvalidated_extrapolation')

    def test_exact_tail_boundaries(self):
        result = []
        for phase in self.cfg['phase_schedule']:
            boundaries = hold_schedule_boundaries(phase['end'] - phase['start'],170,phase['hold_schedule'])
            result.append(tuple(n + phase['start'] for n in boundaries))
        self.assertEqual(result, [(11444,16530,17802,19074),(21364,22381,22635,22889)])

    def test_window_never_crosses_events_or_endpoint(self):
        tokens,updates,seen = 0,0,set()
        while tokens < TARGET:
            width,accumulation = window(self.cfg,tokens)
            self.assertEqual(width,8 * accumulation)
            self.assertLessEqual(width,16 if tokens < SWITCH * Q else 64)
            next_tokens = tokens + width * 8192
            self.assertLessEqual(next_tokens,TARGET)
            self.assertFalse(any(tokens < e * Q < next_tokens for e in events(self.cfg)))
            tokens = next_tokens;updates += 1;seen.add(tokens // Q)
        self.assertEqual(tokens,3000107008)
        self.assertEqual(updates,PLANNED_UPDATES)
        self.assertEqual(count_updates(self.cfg),14312)
        self.assertTrue(set(events(self.cfg)) - {0} <= seen)
        self.assertEqual(window(self.cfg,TARGET),(0,0))
        self.assertEqual(window(self.cfg,SWITCH * Q)[0],64)

    def test_invalid_cursor_rejected(self):
        for tokens in (-Q,1,TARGET + Q):
            with self.assertRaises(ValueError):window(self.cfg,tokens)

    def test_warmup_resplit_and_last_zero_update(self):
        multiplier = lambda i:schedule_multiplier(i,self.cfg)
        self.assertAlmostEqual(multiplier(0),1 / 170)
        self.assertEqual(multiplier(169),1)
        self.assertEqual(multiplier(SWITCH),1)
        self.assertAlmostEqual(multiplier(16530),.2)
        self.assertEqual(multiplier(SPLIT - 1),0)
        self.assertAlmostEqual(multiplier(SPLIT),.1 / 170)
        self.assertAlmostEqual(multiplier(SPLIT + 169),.1)
        self.assertEqual(multiplier(END - 1),0)
        self.assertEqual(window(self.cfg,(SPLIT - 1) * Q),(16,2))
        self.assertEqual(window(self.cfg,(END - 1) * Q),(16,2))

    def test_beta2_and_moments_remain_fixed(self):
        class Optimizer:
            def __init__(self):
                self.param_groups=[dict(betas=(.9,.95))]
                self.state={'moments':object()}
        opts=[Optimizer(),Optimizer()];before=[o.state for o in opts]
        for width in (16,32,48,64):
            self.assertEqual(beta2_for(self.cfg,width,SWITCH * Q),.95)
            self.assertEqual(configure_update(opts,self.cfg,width,SWITCH * Q),.95)
            self.assertEqual(opts[0].param_groups[0]['betas'],(.9,.95))
            self.assertFalse(opts[0].param_groups[0]['variable_beta2'])
            for old,opt in zip(before,opts):self.assertIs(old,opt.state)

    def test_scheduler_does_not_end_at_old_1b_horizon(self):
        class Optimizer:
            def __init__(self):self.param_groups=[dict(name='muon',lr=MUON_LR)]
        opt=Optimizer();scheduler=TokenScheduler(opt,self.cfg)
        for unit in (0,7630,11444,16530,19073,19074,21364,22888):
            scheduler.set_tokens(unit * Q)
            self.assertEqual(opt.param_groups[0]['lr'],MUON_LR * schedule_multiplier(unit,self.cfg))
        state=scheduler.state_dict();scheduler.load_state_dict(state)
        self.assertEqual(scheduler.tokens,22888 * Q)

    def test_evaluation_cadence_uses_actual_updates(self):
        self.assertEqual(evaluation_units(),(2288,4577,6866,9155,11444,13733,16022,18311,20600))
        tokens=step=last=0;gaps=[]
        while tokens < TARGET:
            width,_=window(self.cfg,tokens);tokens+=width * 8192;step+=1
            if tokens // Q in evaluation_units() or step-last >= self.cfg['eval_steps'] or tokens==TARGET:
                gaps.append(step-last);last=step
        self.assertEqual(step,14312)
        self.assertLessEqual(max(gaps),step // 10)

    def test_native_update_math_unchanged(self):
        old=LIVE.parent / 'batch-normuon-1b-20261016/source-r4'
        for name in ('optimizers.py','recipe_runtime.py','raw_baseline.py','attention_backend.py'):
            self.assertEqual((SOURCE / name).read_bytes(),(old / name).read_bytes())
        def definition(path,name):
            tree=ast.parse(path.read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
        self.assertEqual(definition(SOURCE / 'batch3b_runtime.py','adamw_class'),
                         definition(old / 'batch1b_runtime.py','adamw_class'))

    def test_auth_exact_digest_and_fresh_only(self):
        self.assertTrue(authorized(STUDY,self.cfg))
        self.assertFalse(authorized(STUDY,self.cfg,True))
        self.assertFalse(authorized('different-run',self.cfg))
        changed=copy.deepcopy(self.cfg);changed['betas'][1]=.81450625
        self.assertFalse(authorized(STUDY,changed))

    def test_formal12_budget_and_single_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ledger=ExperimentLedger(root)
            atomic_json(ledger.path,dict(jobs={},runs={f'prior-{i}':dict(category='full') for i in range(11)}))
            ledger.reserve('reservation',STUDY,self.cfg,21540,8,False)
            data=read_json(ledger.path)
            self.assertEqual(data['runs'][STUDY]['formal_run_number'],12)
            self.assertEqual(len(data['runs']),12)
            self.assertAlmostEqual(data['total_h200_hours_reserved_or_measured'],47.86666666666667)
            with self.assertRaises(ValueError):ledger.reserve('another',STUDY,self.cfg,21540,8,False)
        for seconds,gpus,resume in ((21601,8,False),(21540,4,False),(21540,8,True)):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):ExperimentLedger(tmp).reserve('bad',STUDY,self.cfg,seconds,gpus,resume)

    def test_uncertain_submission_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp) / 'storage';root.mkdir();live=Path(tmp) / 'live';live.mkdir()
            atomic_json(root / 'experiment-ledger.json',dict(jobs={},runs={f'prior-{i}':dict(category='full') for i in range(11)}))
            atomic_json(live / 'plan.json',dict(excluded_nodes=[]))
            atomic_json(live / 'authorization.json',dict(approved=True))
            atomic_json(live / 'config.json',self.cfg)
            state={}
            with patch.multiple(controller,LIVE=live,DEFAULT_ROOT=root,STATE=live / 'state.json',CONFIG=live / 'config.json'),patch.object(controller,'owner',return_value=True),patch.object(controller,'check'):
                with patch.object(controller.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',30)) as runner:
                    controller.submit(state);controller.submit(state)
                    self.assertEqual(runner.call_count,1)
                self.assertEqual(state['job']['status'],'submission_uncertain')
                with patch.object(controller,'find_submission_job',return_value='777'),patch.object(controller.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'777|RUNNING|\n','')):
                    controller.reconcile(state)
                data=read_json(root / 'experiment-ledger.json')
                self.assertEqual(set(data['jobs']),{'777'})
                recovered={};controller.submit(recovered)
                self.assertEqual(recovered['job']['job_id'],'777')
                self.assertEqual(len(read_json(root / 'experiment-ledger.json')['jobs']),1)


if __name__ == '__main__':
    unittest.main()
