"""Check scalar tensor-like LR storage without importing torch on the login node."""
from pathlib import Path
import sys
import unittest

LIVE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LIVE/'source-r3'))
from batch1b_runtime import TokenScheduler
from batch1b_contracts import build_configs,Q
import json


class Scalar:
    def __init__(self,value):self.value=value
    def __float__(self):return float(self.value)
    def fill_(self,value):self.value=value;return self


class ScalarLearningRate(unittest.TestCase):
    def test_preserves_storage_and_exact_configured_peak(self):
        cfg=build_configs(json.loads((LIVE.parent/'raw3b-sota-wsd20-20261015/config.json').read_text()))['normuon']
        class Optimizer:
            def __init__(self):
                self.scalar=Scalar(.0086)
                self.param_groups=[dict(name='muon',lr=self.scalar,initial_lr=cfg['muon']['learning_rate'])]
        opt=Optimizer();scheduler=TokenScheduler(opt,cfg)
        scheduler.set_tokens(170*Q)
        self.assertIs(opt.param_groups[0]['lr'],opt.scalar)
        self.assertEqual(scheduler.base_lrs[0],cfg['muon']['learning_rate'])
        self.assertEqual(float(opt.scalar),cfg['muon']['learning_rate'])
        json.dumps(scheduler.state_dict())
        loaded=Optimizer();other=TokenScheduler(loaded,cfg)
        other.load_state_dict(scheduler.state_dict())
        self.assertIs(loaded.scalar,loaded.param_groups[0]['lr'])
        self.assertEqual(other.tokens,170*Q)
        self.assertEqual(float(loaded.scalar),cfg['muon']['learning_rate'])


if __name__=='__main__':unittest.main()
