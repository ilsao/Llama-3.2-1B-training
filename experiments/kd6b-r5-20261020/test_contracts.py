"""Small stdlib-only checks; no models, tensors, workers or Slurm allocations."""
import collections
import copy
import json
import runpy
import unittest
from bootstrap import HERE
from contracts import config,plan,event_units,TARGET,RUN,CONFIG_DIGEST
from training_state import digest,window_indices
from launch import reserve_entries


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=config();cls.rows=plan(cls.cfg)

    def test_exact_proposal(self):
        self.assertEqual(self.cfg,json.loads((HERE/'training-config.proposed.json').read_text()))
        self.assertEqual(digest(self.cfg),CONFIG_DIGEST)

    def test_original_6b_windows_and_lr(self):
        old=runpy.run_path(str(HERE.parent/'raw6b-normuon-adam2x-20261018/source-v3/ppl_contracts.py'))
        expected=old['batch_plan'](self.cfg)
        self.assertEqual(len(expected),len(self.rows))
        for got,ref in zip(self.rows,expected):
            ref={**ref,'accumulation':ref['accumulation']//2}
            self.assertEqual(got,ref)
        self.assertEqual(sum(r['width']*8192 for r in self.rows),TARGET)
        self.assertEqual(self.rows[-1]['lr_multiplier'],0.)
        self.assertEqual(dict(collections.Counter(r['width'] for r in self.rows)),{16:22897,32:3,48:2,64:5717})

    def test_distributed_boundary_windows(self):
        endpoints={n*131072 for n in event_units(self.cfg)}
        selected=[r for r in self.rows if r['start_tokens'] in endpoints or r['end_tokens'] in endpoints]
        for r in selected:
            cursor=r['start_tokens']//8192
            indices=[i for rank in range(8) for pair in window_indices(cursor,TARGET//8192,8,rank,2,r['accumulation']) for i in pair]
            self.assertEqual(sorted(indices),list(range(cursor,cursor+r['width'])))
            self.assertEqual(len(indices),len(set(indices)))
        self.assertLessEqual(self.cfg['eval_steps'],len(self.rows)//10)

    def ledger(self):
        return dict(runs={f'old-{i}':dict(category='full') for i in range(18)},jobs={'prior':dict(
            run_id='kd3b-hidden8-prior',h200_hours=176.1786111111111,reserved_h200_hours=176.1786111111111)})

    def test_scoped_reservation_preserves_history(self):
        d=self.ledger();before=copy.deepcopy(d);reserve_entries(d,self.cfg,'new')
        self.assertEqual({k:d['runs'][k] for k in before['runs']},before['runs'])
        self.assertEqual(d['jobs']['prior'],before['jobs']['prior'])
        self.assertEqual(d['runs'][RUN]['formal_run_number'],19)
        self.assertEqual(d['jobs']['new']['reserved_h200_hours'],72)
        with self.assertRaises(ValueError):reserve_entries(d,self.cfg,'again')

    def test_budget_and_config_fail_closed(self):
        d=self.ledger();d['jobs']['prior']['h200_hours']+=.01
        with self.assertRaises(ValueError):reserve_entries(d,self.cfg,'new')
        bad=copy.deepcopy(self.cfg);bad['max_training_tokens']+=1
        with self.assertRaises(ValueError):reserve_entries(self.ledger(),bad,'new')
        d=self.ledger();d['runs'].pop('old-0')
        with self.assertRaises(ValueError):reserve_entries(d,self.cfg,'new')


if __name__=='__main__':unittest.main()
