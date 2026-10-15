"""Small pure contracts; no tensor/model imports on the login node."""
import copy
import json
from pathlib import Path
import sys
import unittest

LIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIVE/'source'))
from batch1b_contracts import (Q, END, TARGET, SPLIT, SWITCHES, beta2_for, build_configs,
                              check_fork, count_updates, validate, window)
from training_state import TokenJournal, hold_schedule_boundaries, schedule_multiplier
from batch1b_runtime import TokenScheduler, verify_checkpoint_files


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfgs = build_configs(json.loads((LIVE.parent/'raw3b-sota-wsd20-20261015/config.json').read_text()))

    def test_all_nine_configs(self):
        self.assertEqual(len(self.cfgs), 9)
        for cfg in self.cfgs.values():
            validate(cfg)
            self.assertEqual(cfg['target_training_tokens'], TARGET)
            self.assertEqual(cfg['world_size'], 8)

    def test_no_repeat_no_overshoot(self):
        for name,cfg in self.cfgs.items():
            start = cfg.get('fork_snapshot_tokens',0)
            tokens = start
            observed = []
            while tokens < TARGET:
                width,accumulation = window(cfg,tokens)
                self.assertEqual(width,accumulation*8)
                self.assertIn(width,(16,32,48,64))
                observed.append(tokens)
                tokens += width*8192
            self.assertEqual(tokens,TARGET)
            self.assertEqual(count_updates(cfg,start),len(observed))
            self.assertEqual(observed[-1],TARGET-Q)  # Final zero-LR update is present.
            self.assertIn((SPLIT-1)*Q,observed)
            self.assertIn(SPLIT*Q,observed)

    def test_forks_are_exact_and_only_authorized_changes(self):
        for key,unit in SWITCHES.items():
            fixed = self.cfgs['bs-'+key+'-fixed']
            scaled = self.cfgs['bs-'+key+'-scaled']
            check_fork(self.cfgs['control'],fixed,unit*Q)
            check_fork(self.cfgs['control'],scaled,unit*Q)
            self.assertEqual(window(fixed,unit*Q)[0],64)
            changed=copy.deepcopy(fixed);changed['learning_rate']*=2
            with self.assertRaises(ValueError):check_fork(self.cfgs['control'],changed,unit*Q)

    def test_beta2_and_boundary_batches(self):
        cfg=self.cfgs['bs-50-scaled']
        for width in (16,32,48,64):
            self.assertAlmostEqual(beta2_for(cfg,width,3815*Q),.95**(width/16))
        self.assertAlmostEqual(beta2_for(cfg,64,3815*Q),.81450625)
        self.assertEqual(beta2_for(self.cfgs['bs-50-fixed'],64,3815*Q),.95)
        self.assertEqual(beta2_for(cfg,16,3814*Q),.95)

    def test_token_schedule_and_warmup(self):
        cfg=self.cfgs['control']
        self.assertEqual(hold_schedule_boundaries(SPLIT,170,cfg['hold_schedule']),(5086,5934,6146,6358))
        self.assertEqual(hold_schedule_boundaries(END-SPLIT,170,cfg['hold_schedule']),(1018,1188,1230,1272))
        self.assertEqual(schedule_multiplier(SPLIT-1,cfg),0)
        self.assertAlmostEqual(schedule_multiplier(SPLIT,cfg),.1/170)
        self.assertEqual(schedule_multiplier(END-1,cfg),0)
        class Dummy:
            def __init__(self):self.param_groups=[{'name':'aux','lr':.001}]
        scheduler=TokenScheduler(Dummy(),cfg)
        scheduler.set_tokens(SPLIT*Q)
        state=scheduler.state_dict()
        other=TokenScheduler(Dummy(),cfg);other.load_state_dict(state)
        self.assertEqual(other.tokens,SPLIT*Q)
        self.assertAlmostEqual(other.optimizer.param_groups[0]['lr'],.001*.1/170)

    def test_evaluation_cadence(self):
        for cfg in self.cfgs.values():
            remaining=count_updates(cfg,cfg.get('fork_snapshot_tokens',0))
            self.assertLessEqual(cfg['eval_steps'],remaining//10)

    def test_adamw_lr_ratio_and_optimizer_isolation(self):
        for name,cfg in self.cfgs.items():
            self.assertAlmostEqual(cfg['learning_rate']/cfg['muon']['learning_rate'],.03)
            self.assertEqual(bool(cfg.get('matrix_optimizer')),name=='normuon')

    def test_inherited_journal_counts_replay(self):
        import tempfile
        from training_state import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tokens.json';start=3815*Q
            atomic_json(p,dict(limit=TARGET,confirmed_tokens=start,charged_tokens=start,next_transaction=0,pending={}))
            journal=TokenJournal(p,TARGET);transaction=journal.reserve(4*Q)
            self.assertEqual(journal.data['charged_tokens'],start+4*Q)
            journal.confirm(transaction)
            self.assertEqual(journal.data['confirmed_tokens'],start+4*Q)

    def test_missing_optimizer_state_or_rng_is_rejected(self):
        from tempfile import TemporaryDirectory
        from training_state import file_digest
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            names=['model.safetensors','optimizer.bin','optimizer_1.bin','training-state.json']
            names += [f'custom_checkpoint_{i}.pkl' for i in range(3)]
            names += [f'random_states_{i}.pkl' for i in range(8)]
            files=[]
            for name in names:
                path=root/name;path.write_bytes(name.encode())
                files.append(dict(name=name,size=path.stat().st_size,sha256=file_digest(path)))
            verify_checkpoint_files(root,dict(files=files))
            for missing in ('optimizer_1.bin','random_states_7.pkl','custom_checkpoint_1.pkl'):
                with self.assertRaises(ValueError):
                    verify_checkpoint_files(root,dict(files=[f for f in files if f['name']!=missing]))
            (root/'optimizer.bin').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify_checkpoint_files(root,dict(files=files))


if __name__=='__main__':unittest.main()
