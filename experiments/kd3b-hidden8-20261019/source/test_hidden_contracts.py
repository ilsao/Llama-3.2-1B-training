"""Small stdlib-only regressions; safe on the login node."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parent
REPO=SOURCE.parents[2]
sys.path.insert(0,str(REPO/'slurm'))
from training_state import ExperimentLedger,atomic_json,read_json,window_indices,load_config
from hidden_contracts import RUN,SHAPE,BYTES,SLOTS,CAP,validate_peers
from ppl_contracts import batch_plan
from kd3b_hidden8 import reserve_pair


class Contracts(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(SOURCE.parent/'config.json')
        self.auth=read_json(SOURCE.parent/'authorization.json')

    def test_every_training_window_uses_the_producer_rows(self):
        plan=batch_plan(self.cfg)
        self.assertEqual(len(plan),14312)
        self.assertEqual(plan[-1]['end_tokens'],3000107008)
        key=0
        for step in plan:
            for rank in (0,7):
                student=window_indices(step['start_tokens']//8192,366224,8,rank,1,step['accumulation'])
                producer=[[8*(key+i)+rank] for i in range(step['accumulation'])]
                self.assertEqual(student,producer)
            key+=step['accumulation']
        self.assertEqual(key*8*8192,3000107008)

    def test_capacity_including_inflight_is_below_one_tb(self):
        self.assertEqual(SHAPE,(8191,3072))
        self.assertLess((SLOTS+1)*8*BYTES,CAP)
        self.assertGreater(SLOTS,8)

    def test_peer_identity_and_earliest_end(self):
        base=dict(run_id=RUN,config_digest='x',preparation_fingerprint='y',world_size=8)
        peers=dict(producer=dict(base,node='a',job_id='1',end_epoch=100),
                   consumer=dict(base,node='b',job_id='2',end_epoch=110))
        self.assertEqual(validate_peers(peers),100)
        for key,value in (('config_digest','changed'),('node','a'),('world_size',4)):
            wrong=copy.deepcopy(peers);wrong['consumer'][key]=value
            with self.assertRaises(ValueError):validate_peers(wrong)

    def ledger(self,root,cost=95.4186111111111):
        ledger=ExperimentLedger(root)
        original=dict(runs={f'formal{i}':dict(category='full',max_h200_hours=64,config_digest=str(i)) for i in range(17)},
            jobs={'old-failure':dict(run_id='methods-parallel-20261019-old',reserved_h200_hours=128,
                h200_hours=cost,state='FAILED')})
        atomic_json(ledger.path,original)
        return ledger,original

    def test_pair_reservation_preserves_history_and_rejects_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            ledger,old=self.ledger(root)
            reserve_pair(ledger,self.cfg,self.auth,dict(producer='p',consumer='c'))
            data=read_json(ledger.path)
            self.assertEqual(len(data['runs']),18)
            self.assertEqual(data['jobs']['old-failure'],old['jobs']['old-failure'])
            for k,v in old['runs'].items():self.assertEqual(data['runs'][k],v)
            self.assertEqual(data['runs'][RUN]['formal_run_number'],18)
            self.assertEqual(sum(data['jobs'][k]['reserved_h200_hours'] for k in ('p','c')),96)
            with self.assertRaises(ValueError):reserve_pair(ledger,self.cfg,self.auth,dict(producer='p2',consumer='c2'))
            self.assertEqual(read_json(ledger.path),data)

    def test_overbudget_pair_and_missing_role_do_not_mutate_ledger(self):
        with tempfile.TemporaryDirectory() as root:
            ledger,old=self.ledger(root,96)
            for tags in (dict(producer='p',consumer='c'),dict(producer='p')):
                with self.assertRaises(ValueError):reserve_pair(ledger,self.cfg,self.auth,tags)
                self.assertEqual(read_json(ledger.path),old)


if __name__=='__main__':unittest.main()
