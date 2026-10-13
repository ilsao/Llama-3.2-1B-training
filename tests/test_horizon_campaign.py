"""Pure campaign tests: no torch, datasets, models or workers on login."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/horizon-wsd-20261014/source'
sys.path.insert(0,str(SOURCE))
from horizon_contracts import (BATCH_TOKENS,SWEEP_STEPS,FINAL_STEPS,PREFIX_STEP,FINAL_SPLIT,
    check_fork_configs,scaled_lr,tail_settings,campaign_allocation_limits)
from training_state import (load_config,schedule_multiplier,hold_schedule_boundaries,
    atomic_json,TokenJournal,ExperimentLedger)


def base():
    cfg=json.loads((ROOT/'configs/cooldown-grid-20261010/be60p20-seed42.json').read_text())
    cfg.update(max_steps=SWEEP_STEPS,scheduler_steps=SWEEP_STEPS,max_training_tokens=SWEEP_STEPS*BATCH_TOKENS,
        target_training_tokens=SWEEP_STEPS*BATCH_TOKENS,eval_steps=534,max_h200_hours=16,
        hold_schedule=tail_settings(.6))
    return cfg


class CampaignContracts(unittest.TestCase):
    def test_rounded_horizons_and_scale(self):
        self.assertEqual(SWEEP_STEPS*BATCH_TOKENS,700055552)
        self.assertEqual(FINAL_STEPS*BATCH_TOKENS,2000027648)
        self.assertEqual(PREFIX_STEP,2136)
        self.assertEqual(FINAL_SPLIT,12715)
        eta=scaled_lr(.02,399900672,SWEEP_STEPS*BATCH_TOKENS)
        self.assertAlmostEqual(scaled_lr(eta,SWEEP_STEPS*BATCH_TOKENS,FINAL_STEPS*BATCH_TOKENS),scaled_lr(.02,399900672,FINAL_STEPS*BATCH_TOKENS))

    def test_clipping_off_and_negative_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.json';cfg=base();cfg['max_grad_norm']=0
            atomic_json(p,cfg);self.assertEqual(load_config(p)['max_grad_norm'],0)
            cfg['max_grad_norm']=-1;atomic_json(p,cfg)
            with self.assertRaises(ValueError):load_config(p)

    def test_shared_prefix_and_zero_tail(self):
        parent=base()
        for tail in (.6,.5,.4,.2,.1):
            cfg=copy.deepcopy(parent);cfg.update(hold_schedule=tail_settings(tail),wsd_decay_ratio=tail)
            check_fork_configs(parent,cfg)
            start=hold_schedule_boundaries(SWEEP_STEPS,170,cfg['hold_schedule'])[0]
            self.assertGreaterEqual(start,PREFIX_STEP)
            self.assertEqual(schedule_multiplier(SWEEP_STEPS-1,cfg),0)
        bad=copy.deepcopy(parent);bad['muon']['learning_rate']*=2
        with self.assertRaises(ValueError):check_fork_configs(parent,bad)

    def test_restart_matches_data_boundary(self):
        cfg=base();cfg.update(max_steps=FINAL_STEPS,scheduler_steps=FINAL_STEPS,max_training_tokens=FINAL_STEPS*BATCH_TOKENS,
             target_training_tokens=FINAL_STEPS*BATCH_TOKENS)
        ratio=((FINAL_STEPS-FINAL_SPLIT)/FINAL_SPLIT)**(-.32)
        cfg['phase_schedule']=[dict(start=0,end=FINAL_SPLIT,peak_ratio=1,warmup_steps=170,hold_schedule=tail_settings(.6)),
            dict(start=FINAL_SPLIT,end=FINAL_STEPS,peak_ratio=ratio,warmup_steps=170,hold_schedule=tail_settings(.6))]
        self.assertEqual(schedule_multiplier(FINAL_SPLIT-1,cfg),0)
        self.assertAlmostEqual(schedule_multiplier(FINAL_SPLIT,cfg),ratio/170)
        self.assertEqual(schedule_multiplier(FINAL_STEPS-1,cfg),0)
        self.assertEqual(FINAL_SPLIT*16+(FINAL_STEPS-FINAL_SPLIT)*16,FINAL_STEPS*16)

    def test_ledger_preserves_cost_and_final_exception_scope(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=ExperimentLedger(d)
            with ledger.transaction() as data:
                data['runs']={f'old-{i}':dict(category='full') for i in range(4)}
            cfg=base();cfg.update(category='full',sweep_id='horizon-wsd-20261014',max_steps=FINAL_STEPS,
                max_training_tokens=FINAL_STEPS*BATCH_TOKENS,max_h200_hours=32)
            ledger.reserve('new','horizon-wsd-20261014-final-standard',cfg,10800,8,False)
            with ledger.transaction() as data:
                self.assertEqual(len(data['runs']),5)
                self.assertEqual(data['jobs']['new']['reserved_h200_hours'],24)
            with self.assertRaises(ValueError):ledger.reserve('other','some-other-final',cfg,10800,8,False)
            with self.assertRaises(ValueError):ledger.reserve('retry','horizon-wsd-20261014-final-standard',cfg,3600,8,False)

    def test_fork_journal_counts_inherited_inputs(self):
        from horizon_runtime import seed_journal
        with tempfile.TemporaryDirectory() as d:
            cfg=base();state=dict(confirmed_tokens_at_checkpoint=PREFIX_STEP*BATCH_TOKENS,charged_tokens_at_checkpoint=PREFIX_STEP*BATCH_TOKENS)
            seed_journal(Path(d),cfg,state)
            journal=TokenJournal(Path(d)/'token-journal.json',cfg['max_training_tokens'])
            transaction=journal.reserve(BATCH_TOKENS);journal.confirm(transaction)
            self.assertEqual(journal.data['confirmed_tokens'],(PREFIX_STEP+1)*BATCH_TOKENS)
            with self.assertRaises(ValueError):seed_journal(Path(d),cfg,state)


if __name__=='__main__':unittest.main()
