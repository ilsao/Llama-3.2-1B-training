"""Pure recovery tests; no torch/model/data workers or actual Slurm calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'experiments/horizon-wsd-20261014/source-v2'
sys.path.insert(0, str(SOURCE))
from horizon_runtime import fork_checkpoint
from horizon_contracts import CAMPAIGN, PREFIX_STEP, check_fork_configs
from training_state import atomic_json, load_config
spec = importlib.util.spec_from_file_location('branch_recovery_controller', ROOT/'slurm/horizon_campaign_v3.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class ReachedMetadata(Exception):
    pass

class RecoveryTests(unittest.TestCase):
    def test_generated_ids_pass_real_sealed_runtime_guard(self):
        train = types.ModuleType('train')
        train.on_main = lambda accelerator, function: function()
        train.checkpoint_state = MagicMock(side_effect=ReachedMetadata)
        with patch.dict(sys.modules, {'train': train}):
            for key in c.BRANCH_KEYS:
                args = types.SimpleNamespace(root='/tmp', fork_from_run=CAMPAIGN+'-stable-prefix',
                    run_id=CAMPAIGN+'-'+c.decay_run_name(key))
                with self.assertRaises(ReachedMetadata):
                    fork_checkpoint(args, {}, {}, Path('/tmp'), {}, {}, None, [], [])
        self.assertEqual(train.checkpoint_state.call_count, 5)

    def test_old_names_and_wrong_parent_remain_rejected(self):
        train = types.ModuleType('train')
        train.on_main = lambda accelerator, function: function()
        train.checkpoint_state = MagicMock()
        with patch.dict(sys.modules, {'train': train}):
            for parent, name in [(CAMPAIGN+'-stable-prefix', CAMPAIGN+'-decay60'),
                                 (CAMPAIGN+'-other', CAMPAIGN+'-decay-60-recovery1')]:
                args = types.SimpleNamespace(root='/tmp', fork_from_run=parent, run_id=name)
                with self.assertRaisesRegex(ValueError, 'restricted'):
                    fork_checkpoint(args, {}, {}, Path('/tmp'), {}, {}, None, [], [])
        train.checkpoint_state.assert_not_called()
        with self.assertRaises(ValueError): c.decay_run_name('other')

    def test_all_child_configs_preserve_real_parent_lr_prefix(self):
        parent = load_config(ROOT/'configs/horizon-wsd-20261014/stable-prefix.json')
        for key in c.BRANCH_KEYS:
            child = load_config(ROOT/'configs/horizon-wsd-20261014'/(key+'.json'))
            check_fork_configs(parent, child)
        bad = copy.deepcopy(child); bad['learning_rate'] *= 2
        with self.assertRaises(ValueError): check_fork_configs(parent, bad)

    def test_recovery_archives_costs_exactly_once(self):
        original = json.loads((ROOT/'experiments/horizon-wsd-20261014/controller-state.json').read_text())
        # Stable fixture independent of whether the actual recovery was applied.
        for key in c.BRANCH_KEYS:
            if key+'-failed-original' in original['jobs']:
                original['jobs'][key] = original['jobs'].pop(key+'-failed-original')
        original.pop('branch_recovery_applied', None)
        original['stage'] = 'decay'
        ledger = MagicMock()
        ledger.transaction.return_value.__enter__.return_value = {'jobs': {}}
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            atomic_json(d/'state.json', original)
            atomic_json(d/'branch-recovery-authorization.json', {'user_reply':'修復並繼續'})
            atomic_json(d/'branch-recovery-protocol-commit.json', {'commit':'test'})
            atomic_json(d/'branch-recovery-plan.json', {'failed_jobs':{k:original['jobs'][k]['job_id'] for k in c.BRANCH_KEYS}})
            with patch.object(c, 'DIRECTORY', d), patch.object(c, 'STATE', d/'state.json'), \
                 patch.object(c, 'check'), patch.object(c, 'reconcile'), patch.object(c, 'report'), \
                 patch.object(c, 'ExperimentLedger', return_value=ledger), \
                 patch.object(c, 'plan', return_value={'allocation_cap_h200_hours':205.33333333333334}):
                c.recover_decay()
                first = (d/'state.json').read_text()
                c.recover_decay()
                self.assertEqual((d/'state.json').read_text(), first)
            state = json.loads(first)
            self.assertTrue(state['branch_recovery_applied'])
            for key in c.BRANCH_KEYS:
                self.assertNotIn(key, state['jobs'])
                self.assertEqual(state['jobs'][key+'-failed-original'], original['jobs'][key])
            self.assertEqual(json.loads((d/'branch-recovery-original-state.json').read_text()), original)

    def test_failed_recovery_does_not_promote_or_retry(self):
        state = {'stage':'decay', 'jobs':{key:{'status':'reconciled','slurm_state':'FAILED'} for key in c.BRANCH_KEYS}}
        with self.assertRaisesRegex(ValueError, 'Stage failed'):
            c.completed(state, c.BRANCH_KEYS)

if __name__ == '__main__': unittest.main()
