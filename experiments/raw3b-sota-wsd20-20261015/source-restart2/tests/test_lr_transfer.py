import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import lr_transfer as t
from training_state import load_config,schedule_multiplier
spec=importlib.util.spec_from_file_location('transfer_controller',ROOT/'slurm/lr_transfer_controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'configs/lr-transfer-base.json').read_text())
    def test_all_configs_preserve_model_geometry_and_budget(self):
        for kind in t.MINUTES:
            cfg=t.configuration(self.base,kind,.7)
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'cfg.json';p.write_text(json.dumps(cfg));load_config(p)
            self.assertEqual(cfg['target_training_tokens'],cfg['max_steps']*t.BATCH)
            self.assertAlmostEqual(cfg['learning_rate']/cfg['muon']['learning_rate'],t.RATIO)
            self.assertEqual(cfg['sequence_length'],8192);self.assertNotIn('phase_schedule',cfg)
            self.assertEqual(cfg['seed'],42);self.assertFalse(cfg['export_on_completion'])
            self.assertLessEqual(cfg['eval_steps'],max(1,cfg['max_steps']//10))
            self.assertEqual(cfg['eval_documents'],16 if kind=='validation' else 1000)
        self.assertLess(176.18666666666667+t.MAX_STUDY_HOURS,256)
    def test_constant_proxy_and_complete_wsd_have_distinct_endpoints(self):
        proxy=t.configuration(self.base,'proxy');wsd=t.configuration(self.base,'decay')
        self.assertAlmostEqual(schedule_multiplier(0,proxy),1/170)
        self.assertEqual(schedule_multiplier(169,proxy),1.)
        self.assertEqual(schedule_multiplier(proxy['max_steps']-1,proxy),1.)
        self.assertEqual(schedule_multiplier(wsd['max_steps']-1,wsd),0.)
        self.assertEqual(wsd['ema'],[]);self.assertEqual(proxy['ema'],[.6])
    def test_reject_proxy_with_resplit_or_disabled_ema(self):
        for changes in [{'ema':[]},{'lr_scheduler':'wsd_hold_zero'},{'phase_schedule':[]}]:
            cfg={**t.configuration(self.base,'proxy'),**changes}
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'cfg.json';p.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):load_config(p)
    def test_known_quadratic_optimum_recovered(self):
        optimum=.006;x=[.002,.004,.008,.016];curve=[(v,3+.3*math.log(v/optimum)**2) for v in x]
        result=t.parabola(curve)
        self.assertAlmostEqual(result['muon_lr'],optimum,places=12);self.assertAlmostEqual(result['r_squared'],1.)
    def test_projection_stops_slow_hardware_before_full_sweep(self):
        cfg=t.configuration(self.base,'proxy')
        self.assertTrue(t.projected_allocation(cfg,32,[.65]*20,5500)['fits'])
        self.assertFalse(t.projected_allocation(cfg,32,[2.2]*20,5500)['fits'])
        with self.assertRaises(ValueError):t.projected_allocation(cfg,32,[0.],5500)
    def test_boundary_and_nonconvex_optima_are_rejected(self):
        for curve in [[(.001,1.),(.002,2.),(.004,3.),(.008,4.)],[(.001,1.),(.002,2.),(.004,2.),(.008,1.)]]:
            with self.assertRaises(ValueError):t.parabola(curve)
    def make_cohorts(self):
        cohorts={m:{} for m in t.MULTIPLIERS}
        for step in list(range(1536,7297,384))+[7630]:
            tokens=step*t.BATCH;optimum=.005*(tokens/1e9)**(-.2)
            for m in t.MULTIPLIERS:
                lr=t.ANCHOR_MUON*m
                cohorts[m][step]={'tokens':tokens,'nll':3+.1*math.log(lr/optimum)**2,'evaluation_identity':'fixed'}
        return cohorts
    def test_known_power_law_forecasts_all_six_targets_and_holds_out_late_points(self):
        cohorts=self.make_cohorts();r=t.analyze(cohorts)
        self.assertTrue(r['proxy_validation_passed']);self.assertAlmostEqual(r['final_fit']['beta'],-.2)
        self.assertEqual(len(r['targets']),6)
        for row in r['targets']:
            self.assertAlmostEqual(row['muon_peak_lr'],.005*(row['tokens']/1e9)**(-.2))
            self.assertAlmostEqual(row['adamw_peak_lr'],row['muon_peak_lr']*t.RATIO)
        self.assertTrue(all(v[0]<=650e6 for v in r['early_fit']['points']))
        self.assertTrue(all(v['tokens']>=750e6 for v in r['held_out_validation']))
    def test_nontransferable_late_horizons_fail_validation(self):
        cohorts=self.make_cohorts()
        for m,rows in cohorts.items():
            for step,r in rows.items():
                if r['tokens']>=750e6:
                    optimum=.007
                    r['nll']=3+.1*math.log(t.ANCHOR_MUON*m/optimum)**2
        self.assertFalse(t.analyze(cohorts)['proxy_validation_passed'])
    def test_missing_lr_or_changed_eval_identity_is_rejected(self):
        cohorts=self.make_cohorts();missing=copy.deepcopy(cohorts);missing.pop(t.MULTIPLIERS[0])
        with self.assertRaises(ValueError):t.analyze(missing)
        cohorts[t.MULTIPLIERS[0]][1536]['evaluation_identity']='changed'
        with self.assertRaises(ValueError):t.analyze(cohorts)
    def test_publication_does_not_call_unchecked_results_validated(self):
        result=t.analyze(self.make_cohorts())
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'LIVE',Path(tmp)):
            c.publish(result);out=json.loads((Path(tmp)/'predictions.json').read_text())
            self.assertFalse(out['validated']);self.assertTrue(all(r['status']=='unvalidated_estimate' for r in out['targets']))
            c.publish(result,{'passed':True});self.assertTrue(json.loads((Path(tmp)/'predictions.json').read_text())['validated'])
    def test_existing_reservation_cannot_submit_again(self):
        state={'jobs':{'proxy0':{'status':'submitted','job_id':'123'}}}
        with patch.object(c.subprocess,'run',side_effect=AssertionError('must not run sbatch')):
            c.submit(state,'proxy0','proxy',.35)
    def test_failed_terminal_job_cannot_promote(self):
        state={'jobs':{'screen0':{'status':'reconciled','slurm_state':'FAILED'}}}
        with self.assertRaises(ValueError):c.completed(state,['screen0'])

if __name__=='__main__':unittest.main()
