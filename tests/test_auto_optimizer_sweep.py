"""Metadata-only campaign simulations; all Slurm/cron calls isolated and mocked."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from slurm import auto_optimizer_sweep as auto
from slurm import optimizer_sweep as stage
from training_state import ExperimentLedger, atomic_json, digest, load_config, read_json


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = self.root / 'environment'; (env / 'bin').mkdir(parents=True)
        (env / 'bin/python').touch()
        self.envpatch = patch.object(auto, 'ENVIRONMENT', str(env)); self.envpatch.start()
        self.addCleanup(self.envpatch.stop)
        identity = patch.object(auto, 'environment_identity', return_value={'packages_digest': 'fixed'})
        identity.start(); self.addCleanup(identity.stop)
        ready = self.root / 'preprocessed' / auto.PREPARATION_ID / 'raw/ready.json'
        atomic_json(ready, {'fixed': 'evaluation'})
        atomic_json(ready.parent / 'progress.json', {'train_files': [{'rows': 100000}]})
        self.state = auto.initialize(self.root)
        self.state.update(enabled=True, status='running'); auto.save(self.root, self.state)

    def complete_calibration_state(self):
        directory = stage.stage_path(self.root, 'calibration')
        manifest = read_json(directory / 'manifest.json'); state = read_json(directory / 'state.json')
        for i, c in enumerate(manifest['candidates']):
            state['jobs'][c['id']] = dict(status='terminal', job_id=str(i + 1),
                allocation={'state': 'COMPLETED'}, run_id=c['run_id'])
        atomic_json(directory / 'state.json', state)
        return directory, manifest, state

    def fake_prepare(self, root, name, calibration):
        directory = stage.stage_path(root, name)
        atomic_json(directory / 'state.json', {'enabled': False, 'status': 'prepared'})
        candidate = {'id': name + '-best', 'config': str(directory / 'config.json')}
        atomic_json(directory / 'config.json', {'stage': name})
        atomic_json(directory / 'manifest.json', {'candidates': [candidate]})

    def fake_stage_tick(self, root, name):
        self.calls.append(name)
        state = read_json(stage.stage_path(root, name) / 'state.json')
        if not state['enabled']:
            return state
        state.update(enabled=False, status='complete')
        atomic_json(stage.stage_path(root, name) / 'state.json', state)
        promotion = ({'survivors': ['survivor']} if name == 'initial' else
                     {'needs_medium': self.medium, 'finalists': ['a', 'm'], 'reason': 'test'})
        atomic_json(stage.stage_path(root, name) / 'report.json', dict(all_terminal=True,
            stage=name, results=[dict(id=name + '-best', status='complete', nll_100=3.0,
                                      run_id='run-' + name, wandb_url='link')], promotion=promotion))
        return state

    def simulate(self, medium):
        self.complete_calibration_state(); self.calls = []; self.medium = medium
        with patch.object(auto, 'check_calibration_receipt'), patch.object(stage, 'collect_calibration'), \
             patch.object(stage, 'prepare', side_effect=self.fake_prepare), \
             patch.object(stage, 'tick', side_effect=self.fake_stage_tick):
            for _ in range(4):
                auto.tick(self.root)
        return read_json(auto.path(self.root))

    def test_complete_automatic_ladder_with_conditional_medium(self):
        state = self.simulate(True)
        self.assertEqual(state['status'], 'complete')
        self.assertFalse(state['enabled']); self.assertEqual(state['phase'], 'done')
        self.assertEqual(self.calls, ['initial', 'short', 'short', 'medium', 'medium'])
        self.assertEqual(read_json(stage.base(self.root) / 'best-config.json'), {'stage': 'medium'})
        self.assertEqual(read_json(stage.base(self.root) / 'summary.json')['ranking_stage'], 'medium')

    def test_clear_ranking_stops_after_short_without_medium(self):
        self.simulate(False)
        self.assertFalse(stage.stage_path(self.root, 'medium').exists())
        self.assertEqual(read_json(stage.base(self.root) / 'summary.json')['ranking_stage'], 'short')

    def test_four_calibrations_sequential_geometry_environment_and_intent(self):
        submissions = []
        def submit(command, **kwargs):
            state = read_json(stage.stage_path(self.root, 'calibration') / 'state.json')
            entry = list(state['jobs'].values())[-1]
            ledger = read_json(ExperimentLedger(self.root).path)
            self.assertEqual(ledger['jobs'][entry['reservation']]['status'], 'submission_uncertain')
            op, gpus = (('adamw', 4), ('adamw', 8), ('muon', 4), ('muon', 8))[len(submissions)]
            self.assertIn(f'--gpus-per-node={gpus}', command)
            self.assertIn('--cpus-per-task=8', command)
            self.assertEqual(kwargs['env']['RUNTIME_OPTIMIZER'], op)
            self.assertEqual(command[-3], 'sweep-calibration')
            self.assertEqual(load_config(command[-1])['world_size'], gpus)
            submissions.append(command)
            return subprocess.CompletedProcess(command, 0, str(123 + len(submissions)), '')
        def record(job_id, root):
            index = int(job_id) - 124
            return dict(job_id=job_id, gpu_count=(4, 8, 4, 8)[index], state='COMPLETED',
                        elapsed_seconds=400, h200_hours=(4, 8, 4, 8)[index] / 9)
        with patch.object(stage.subprocess, 'run', side_effect=submit), \
             patch('training_state.slurm_record', side_effect=ValueError('running')):
            auto.tick(self.root); auto.tick(self.root)
        self.assertEqual(len(submissions), 1)
        with patch.object(stage.subprocess, 'run', side_effect=submit), \
             patch('training_state.slurm_record', side_effect=record), \
             patch.object(auto, 'check_calibration_receipt'):
            for _ in range(3):
                auto.tick(self.root)
        self.assertEqual(len(submissions), 4)
        jobs = read_json(ExperimentLedger(self.root).path)['jobs'].values()
        self.assertEqual(sum(j['status'] == 'submitted' for j in jobs), 1)

    def test_disable_reconciles_active_calibration_without_submission(self):
        with patch.object(stage.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123', '')):
            auto.tick(self.root)
        measured = dict(job_id='123', gpu_count=4, state='COMPLETED', elapsed_seconds=400, h200_hours=4/9)
        with patch('training_state.slurm_record', return_value=measured), \
             patch.object(auto, 'check_calibration_receipt'), patch.object(stage, 'dispatch') as submit:
            auto.disable(self.root)
        submit.assert_not_called()
        self.assertEqual(read_json(ExperimentLedger(self.root).path)['jobs']['123']['status'], 'reconciled')
        self.assertFalse(read_json(auto.path(self.root))['enabled'])

    def test_calibration_failure_stops_next_job(self):
        directory, manifest, state = self.complete_calibration_state()
        state['jobs'][manifest['candidates'][0]['id']]['allocation']['state'] = 'FAILED'
        atomic_json(directory / 'state.json', state)
        with patch.object(stage, 'dispatch') as submit:
            result = auto.tick(self.root)
        submit.assert_not_called(); self.assertEqual(result['status'], 'failed')

    def test_completed_but_invalid_calibration_stops_before_next(self):
        self.complete_calibration_state()
        with patch.object(auto, 'check_calibration_receipt', side_effect=ValueError('bad replay')), \
             patch.object(stage, 'collect_calibration') as collect:
            result = auto.tick(self.root)
        collect.assert_not_called(); self.assertIn('bad replay', result['stop_reason'])

    def test_capacity_failure_stops_after_calibration(self):
        self.complete_calibration_state()
        with patch.object(auto, 'check_calibration_receipt'), patch.object(stage, 'collect_calibration'), \
             patch.object(stage, 'prepare', side_effect=ValueError('fewer than 100 updates')):
            result = auto.tick(self.root)
        self.assertEqual(result['status'], 'failed'); self.assertFalse(result['enabled'])

    def test_source_change_blocks_submission_and_keeps_reconciliation(self):
        with patch.object(stage, 'snapshot', return_value={'changed': 'hash'}), \
             patch.object(stage, 'dispatch') as submit, \
             patch.object(auto, 'reconcile_calibration', wraps=auto.reconcile_calibration) as reconcile:
            result = auto.tick(self.root)
        submit.assert_not_called(); reconcile.assert_called_once()
        self.assertEqual(result['status'], 'blocked'); self.assertFalse(result['enabled'])

    def test_environment_change_blocks_submission(self):
        with patch.object(auto, 'environment_identity', return_value={'packages_digest': 'changed'}), \
             patch.object(stage, 'dispatch') as submit:
            result = auto.tick(self.root)
        submit.assert_not_called(); self.assertEqual(result['status'], 'blocked')

    def test_uncertain_submission_keeps_one_reservation_and_recovers(self):
        with patch.object(stage.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch', 60)):
            with self.assertRaises(subprocess.TimeoutExpired):
                auto.tick(self.root)
        before = read_json(ExperimentLedger(self.root).path)
        with patch.object(stage, 'recover_submission', side_effect=ValueError('no match')), \
             patch.object(stage, 'dispatch') as submit, self.assertRaises(ValueError):
            auto.tick(self.root)
        submit.assert_not_called(); self.assertEqual(read_json(ExperimentLedger(self.root).path), before)
        self.assertEqual(len(before['jobs']), 1)

    def test_cron_preserves_existing_entry_is_idempotent_and_has_one_owner(self):
        existing = '*/20 * * * * original # legacy\n'
        with patch.object(auto.subprocess, 'run', side_effect=[subprocess.CompletedProcess([], 0, existing, ''),
             subprocess.CompletedProcess([], 0, '', '')]) as call:
            line = auto.install_cron(self.root)
        self.assertEqual(call.call_args.kwargs['input'], existing + line + '\n')
        with patch.object(auto.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, existing + line + '\n', '')) as call:
            self.assertEqual(auto.install_cron(self.root), line)
            self.assertEqual(call.call_count, 1)
        with patch.object(auto.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
             '# llama-optimizer-sweep-' + auto.SWEEP_ID + '-initial\n', '')):
            with self.assertRaisesRegex(ValueError, 'already owns'):
                auto.install_cron(self.root)

    def test_launch_requires_cron_before_sbatch_and_cannot_reset_terminal(self):
        with patch.object(auto, 'install_cron', side_effect=RuntimeError('cron unavailable')), \
             patch.object(stage, 'dispatch') as submit, self.assertRaises(RuntimeError):
            auto.launch(self.root)
        submit.assert_not_called()
        state = read_json(auto.path(self.root)); state['status'] = 'complete'; auto.save(self.root, state)
        with patch.object(auto, 'install_cron') as cron, self.assertRaises(ValueError):
            auto.launch(self.root)
        cron.assert_not_called()

    def test_corrupt_manifest_cannot_dispatch(self):
        target = stage.stage_path(self.root, 'calibration') / 'manifest.json'
        manifest = read_json(target); manifest['allocation_seconds'] = 601; atomic_json(target, manifest)
        with patch.object(stage, 'dispatch') as submit, self.assertRaisesRegex(ValueError, 'manifest changed'):
            auto.tick(self.root)
        submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
