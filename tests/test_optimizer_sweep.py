"""Isolated stdlib sweep tests. No Slurm submission, torch import, model or corpus work."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from optimizer_sweep import (candidates, training_config, calibration_config, freeze_steps,
    prune_initial, select_finalists, result_from_run, TOKENS_PER_UPDATE)
from training_state import (atomic_json, digest, load_config, lr_multiplier,
                            schedule_multiplier, read_json, ExperimentLedger)
from slurm import optimizer_sweep as controller
from slurm import submit_training


class SweepContractsTests(unittest.TestCase):
    def load(self, cfg):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'; atomic_json(path, cfg)
            return load_config(path)

    def test_complete_matrix_and_controlled_muon_decay(self):
        matrix = candidates()
        self.assertEqual(len(matrix), 18)
        self.assertEqual(len({c['id'] for c in matrix}), 18)
        for c in matrix:
            cfg = self.load(training_config(c, 'initial', 150))
            self.assertEqual(cfg['max_steps'], cfg['scheduler_steps'])
            self.assertEqual(cfg['warmup_steps'], 50)
            self.assertEqual(cfg['eval_steps'], 15)
            self.assertEqual(cfg['max_training_tokens'], 150 * TOKENS_PER_UPDATE)
            self.assertFalse(cfg['export_on_completion'])
            self.assertTrue(cfg['retain_all_checkpoints'])
            if c['optimizer'] == 'muon':
                self.assertEqual(cfg['learning_rate'], .0003)
                self.assertAlmostEqual(cfg['muon']['learning_rate'] * cfg['muon']['weight_decay'], .002)
        for op in ('adamw', 'muon'):
            for gpus in (4, 8):
                cfg = self.load(calibration_config(op, gpus))
                self.assertEqual(gpus * cfg['gradient_accumulation_steps'], 16)
                self.assertEqual(cfg['category'], 'validation')

    def test_scheduler_endpoints_cooldown_and_legacy_cosine(self):
        cfg = training_config(candidates()[0], 'initial', 150)
        for scheduler in ('linear', 'cosine', 'wsd'):
            cfg['lr_scheduler'] = scheduler
            self.assertAlmostEqual(schedule_multiplier(0, cfg), 1 / 50)
            self.assertEqual(schedule_multiplier(49, cfg), 1)
            self.assertEqual(schedule_multiplier(50, cfg), 1)
            self.assertAlmostEqual(schedule_multiplier(149, cfg), .1)
            self.assertAlmostEqual(schedule_multiplier(200, cfg), .1)
            self.assertLess(schedule_multiplier(148, cfg), 1)
            # Restoring a scheduler cursor must produce the original future LR sequence.
            before = [schedule_multiplier(i, cfg) for i in range(120, 150)]
            restored = copy.deepcopy(cfg)
            self.assertEqual(before, [schedule_multiplier(i, restored) for i in range(120, 150)])
        cfg['lr_scheduler'] = 'wsd'
        self.assertEqual(schedule_multiplier(119, cfg), 1)
        self.assertEqual(schedule_multiplier(120, cfg), 1)
        self.assertLess(schedule_multiplier(121, cfg), 1)
        for i in range(200):
            cfg.pop('lr_scheduler', None)
            self.assertEqual(schedule_multiplier(i, cfg), lr_multiplier(i, 150, 50, .1))

    def test_reject_invalid_scheduler_contracts(self):
        cfg = training_config(candidates()[0], 'initial', 150)
        for change in ({'lr_scheduler': 'unknown'}, {'fixed_horizon': 1},
                       {'export_on_completion': 0}, {'checkpoint_policy': 'unknown'},
                       {'scheduler_steps': 149}, {'wsd_decay_ratio': 0},
                       {'lr_scheduler': 'wsd', 'warmup_steps': 140},
                       {'sweep_environment_digest': 'not-a-hash'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load({**cfg, **change})

    def test_frozen_common_horizon_uses_overhead_slowest_rank_and_buffer(self):
        calibration = dict(slowest_8gpu_update_seconds=.6, initialization_seconds=65,
                           checkpoint_seconds=16, evaluation_seconds=3.4)
        frozen = freeze_steps('initial', calibration, 100000)
        self.assertEqual(frozen['steps'], 150)
        self.assertEqual(frozen['stop_margin_seconds'], 60)
        self.assertEqual(freeze_steps('short', calibration, 1600)['steps'], 100)
        for changes in ({'initialization_seconds': 400}, {'slowest_8gpu_update_seconds': 0},
                        {'evaluation_seconds': float('nan')}):
            with self.assertRaises(ValueError):
                freeze_steps('initial', {**calibration, **changes}, 100000)
        with self.assertRaises(ValueError):
            freeze_steps('initial', calibration, 1599)

    def rows(self):
        return [dict(id=f'r-{s}', recipe='r', optimizer='adamw', status='complete',
                     nll_90=a, nll_100=b) for s, a, b in
                [('linear', 3.0, 2.9), ('cosine', 3.05, 2.95), ('wsd', 3.2, 3.1)]]

    def test_prune_only_with_both_matched_points_after_completion(self):
        rows = self.rows()
        self.assertEqual(prune_initial(rows)['eliminated'], ['r-wsd'])
        rows[-1]['nll_100'] = 2.85  # WSD late-decay gain: must survive.
        self.assertEqual(len(prune_initial(rows)['survivors']), 3)
        rows = self.rows(); rows[-1]['nll_90'] = 3.08
        self.assertEqual(len(prune_initial(rows)['survivors']), 3)
        rows[-1]['status'] = 'incomplete'
        report = prune_initial(rows)
        self.assertEqual(report['excluded'], ['r-wsd'])
        self.assertEqual(len(report['survivors']), 2)

    def test_pruning_is_local_to_each_lr_recipe(self):
        rows = self.rows(); other = [{**r, 'id': 'other-' + r['id'], 'recipe': 'other',
                                    'nll_90': r['nll_90'] + 20, 'nll_100': r['nll_100'] + 20} for r in rows]
        report = prune_initial(rows + other)
        self.assertEqual(len(report['survivors']), 4)
        self.assertEqual(len(report['eliminated']), 2)

    def test_medium_is_conditional_and_has_two_finalists(self):
        a = dict(id='a', optimizer='adamw', status='complete', nll_90=3.2, nll_100=3.0)
        m = dict(id='m', optimizer='muon', status='complete', nll_90=3.3, nll_100=3.1)
        self.assertFalse(select_finalists([a, m])['needs_medium'])
        self.assertTrue(select_finalists([a, {**m, 'nll_100': 3.02}])['needs_medium'])
        self.assertTrue(select_finalists([a, {**m, 'nll_100': 2.8}])['needs_medium'])
        self.assertEqual(select_finalists([a, m])['finalists'], ['a', 'm'])
        self.assertFalse(select_finalists([a])['needs_medium'])

    def test_per_sbatch_budget_does_not_deduct_other_runs_but_resume_accumulates(self):
        cfg = training_config(candidates()[0], 'initial', 100)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ExperimentLedger(tmp)
            for i in range(3):
                ledger.reserve(f'b{i}', f'pilot{i}', cfg, 7200, 8, False)
            self.assertEqual(read_json(ledger.path)['total_h200_hours_reserved_or_measured'], 48)
            with self.assertRaises(ValueError):
                ledger.reserve('too-long', 'bad', cfg, 7201, 8, False)
            ledger.bind('b0', '100')
            with patch('training_state.slurm_record', return_value=dict(job_id='100', gpu_count=8, h200_hours=16)):
                ledger.reconcile('100')
            ledger.reserve('resume', 'pilot0', cfg, 7200, 8, True)
            self.assertEqual(read_json(ledger.path)['total_h200_hours_reserved_or_measured'], 64)


class SweepResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.candidate = candidates()[0]
        self.cfg = training_config(self.candidate, 'initial', 100)
        self.path = self.root / 'runs/experiments/fixture'; self.path.mkdir(parents=True)
        self.summary = dict(configuration=self.cfg, global_step=100, stop_reason='max_steps',
            total_tokens_seen=100 * TOKENS_PER_UPDATE, budget_tokens_charged=100 * TOKENS_PER_UPDATE,
            initialization={'sample_signature': 'init'}, preparation_fingerprint='data', wandb_url='test-link')
        atomic_json(self.path / 'summary.json', self.summary)
        rows = [dict(**{'eval/nll': nll * 1000, 'eval/prediction_tokens': 1000,
            'eval/documents': 1000, 'eval/total_training_tokens_seen': step * TOKENS_PER_UPDATE})
            for step, nll in [(90, 3.0), (100, 2.9)]]
        (self.path / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.measured = dict(state='COMPLETED', h200_hours=.5)

    def result(self):
        return result_from_run(self.root, self.candidate, 'fixture', self.cfg, self.measured)

    def test_complete_results_use_global_nll_and_all_continuation_costs(self):
        accounting = dict(all_jobs_reconciled=True, budget_exceeded=False,
                          h200_hours_measured=1.25, jobs=[{'job_id': '1'}, {'job_id': '2'}])
        atomic_json(self.path / 'resource-accounting.json', accounting)
        r = self.result()
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['nll_90'], 3.0); self.assertEqual(r['nll_100'], 2.9)
        self.assertEqual(r['h200_hours'], 1.25)
        self.assertEqual(r['allocation_tokens_per_h200_hour'], 100 * TOKENS_PER_UPDATE / 1.25)
        accounting['all_jobs_reconciled'] = False
        atomic_json(self.path / 'resource-accounting.json', accounting)
        self.assertEqual(self.result()['status'], 'incomplete')

    def test_nonfinite_high_lr_is_excluded_with_processed_tokens_preserved(self):
        summary = {**self.summary, 'stop_reason': 'nonfinite_or_skipped_update',
                   'global_step': 1, 'total_tokens_seen': 2 * TOKENS_PER_UPDATE,
                   'budget_tokens_charged': 2 * TOKENS_PER_UPDATE}
        atomic_json(self.path / 'summary.json', summary)
        r = self.result()
        self.assertEqual(r['status'], 'unstable')
        self.assertEqual(r['tokens'], 2 * TOKENS_PER_UPDATE)
        self.assertEqual(prune_initial([r])['excluded'], [self.candidate['id']])

    def test_incomplete_cooldown_or_replay_charge_never_promotes(self):
        for change in ({'global_step': 99}, {'stop_reason': 'wall_time_or_signal'},
                       {'budget_tokens_charged': self.summary['budget_tokens_charged'] + TOKENS_PER_UPDATE}):
            atomic_json(self.path / 'summary.json', {**self.summary, **change})
            self.assertEqual(self.result()['status'], 'incomplete')
        atomic_json(self.path / 'summary.json', self.summary)
        (self.path / 'metrics.jsonl').write_text('')
        self.assertEqual(self.result()['status'], 'incomplete')


class SweepControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        prep = self.root / 'preprocessed' / controller.PREPARATION_ID / 'raw'
        prep.mkdir(parents=True)
        atomic_json(prep / 'ready.json', {'fixture': True})
        atomic_json(prep / 'progress.json', {'train_files': [{'rows': 100000}]})
        self.calibration = dict(passed=True, sweep_id=controller.SWEEP_ID, source_hashes=controller.snapshot(),
            environment=controller.ENVIRONMENT, environment_digest='a' * 64,
            initialization_signature='init', preparation_fingerprint='data',
            preparation_manifest_sha256=controller.file_digest(prep / 'ready.json'),
            slowest_8gpu_update_seconds=.6, initialization_seconds=65,
            checkpoint_seconds=16, evaluation_seconds=3.4)
        self.calibration_path = self.root / 'calibration.json'
        atomic_json(self.calibration_path, self.calibration)
        self.manifest = controller.prepare(self.root, 'initial', self.calibration_path)
        self.directory = controller.stage_path(self.root, 'initial')
        self.state_path = self.directory / 'state.json'

    def enable(self):
        state = read_json(self.state_path); state['enabled'] = True
        atomic_json(self.state_path, state)

    def test_prepare_is_disabled_and_templates_cannot_submit(self):
        self.assertFalse(read_json(self.state_path)['enabled'])
        self.assertEqual(len(self.manifest['candidates']), 18)
        with patch.object(controller.subprocess, 'run') as launch:
            controller.tick(self.root, 'initial')
            launch.assert_not_called()
        cfg = training_config(candidates()[0], 'initial', 100)
        path = self.root / 'template.json'; atomic_json(path, cfg)
        with patch('sys.argv', ['submit_training.py', 'pilot', 'fixture', '--config', str(path)]), \
             patch.object(submit_training, 'storage_root', return_value=self.root), \
             patch.object(submit_training.subprocess, 'run') as launch, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            submit_training.main()
        launch.assert_not_called()

    def test_durable_intent_before_submission_and_disabled_terminal_reconciliation(self):
        self.enable()
        def launch(command, **kwargs):
            state = read_json(self.state_path); entry = next(iter(state['jobs'].values()))
            jobs = read_json(ExperimentLedger(self.root).path)['jobs']
            self.assertEqual(entry['status'], 'dispatching')
            self.assertEqual(jobs[entry['reservation']]['status'], 'submission_uncertain')
            self.assertIn('--time=5', command)
            self.assertIn('--signal=B:USR1@60', command)
            self.assertNotIn('CAMPAIGN_END_EPOCH', kwargs['env'])
            return subprocess.CompletedProcess(command, 0, '123', '')
        with patch.object(controller.subprocess, 'run', side_effect=launch):
            controller.tick(self.root, 'initial')
        state = read_json(self.state_path); self.assertEqual(len(state['jobs']), 1)
        state['enabled'] = False; atomic_json(self.state_path, state)
        measured = dict(job_id='123', gpu_count=8, state='COMPLETED', h200_hours=.5, elapsed_seconds=225)
        with patch('training_state.slurm_record', return_value=measured), \
             patch.object(controller.subprocess, 'run') as launch:
            controller.tick(self.root, 'initial')
            launch.assert_not_called()
        report = read_json(self.directory / 'report.json')
        self.assertEqual(report['measured_h200_hours'], .5)
        self.assertEqual(report['results'][0]['status'], 'incomplete')
        self.assertFalse(report['all_terminal'])

    def test_uncertain_submission_keeps_reservation_and_never_retries(self):
        self.enable()
        with patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'lost reply')):
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                controller.tick(self.root, 'initial')
        before = read_json(ExperimentLedger(self.root).path)
        with patch.object(controller, 'recover_submission', side_effect=ValueError('no unique match')), \
             patch.object(controller.subprocess, 'run') as launch, self.assertRaises(ValueError):
            controller.tick(self.root, 'initial')
        launch.assert_not_called()
        self.assertEqual(read_json(ExperimentLedger(self.root).path), before)
        self.assertEqual(len(read_json(self.state_path)['jobs']), 1)

    def test_returned_job_receipt_recovers_binding_crash(self):
        self.enable()
        original_bind = ExperimentLedger.bind
        with patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123', '')), \
             patch.object(ExperimentLedger, 'bind', side_effect=RuntimeError('crash after receipt')):
            with self.assertRaises(RuntimeError):
                controller.tick(self.root, 'initial')
        with patch('training_state.slurm_record', side_effect=ValueError('running')), \
             patch.object(controller.subprocess, 'run') as launch:
            controller.tick(self.root, 'initial')
            launch.assert_not_called()
        self.assertEqual(next(iter(read_json(self.state_path)['jobs'].values()))['job_id'], '123')
        self.assertIn('123', read_json(ExperimentLedger(self.root).path)['jobs'])

    def test_complete_stage_runs_sequentially_ranks_and_stops_before_next_stage(self):
        self.enable()
        calls = []
        def launch(command, **kwargs):
            calls.append(command)
            run_id, config_path = command[-3], command[-1]
            cfg = load_config(config_path)
            path = self.root / 'runs/experiments' / run_id; path.mkdir(parents=True)
            atomic_json(path / 'summary.json', dict(configuration=cfg,
                global_step=cfg['max_steps'], stop_reason='max_steps',
                total_tokens_seen=cfg['max_training_tokens'], budget_tokens_charged=cfg['max_training_tokens'],
                initialization={'sample_signature': 'init'}, preparation_fingerprint='data', wandb_url='test-link'))
            quality = {'linear': 3.0, 'cosine': 3.05, 'wsd': 3.2}[cfg['lr_scheduler']]
            rows = [dict(**{'eval/nll': quality * 1000, 'eval/prediction_tokens': 1000,
                'eval/documents': 1000, 'eval/total_training_tokens_seen': cfg['max_training_tokens'] * percent // 100})
                for percent in (90, 100)]
            (path / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            return subprocess.CompletedProcess(command, 0, str(100 + len(calls)), '')
        def account(job_id, root=None):
            return dict(job_id=job_id, gpu_count=8, state='COMPLETED', h200_hours=.5, elapsed_seconds=225)
        with patch.object(controller.subprocess, 'run', side_effect=launch), \
             patch('training_state.slurm_record', side_effect=account):
            for _ in range(19):
                controller.tick(self.root, 'initial')
            controller.tick(self.root, 'initial')
        self.assertEqual(len(calls), 18)
        state = read_json(self.state_path); report = read_json(self.directory / 'report.json')
        self.assertEqual(state['status'], 'complete'); self.assertFalse(state['enabled'])
        self.assertTrue(report['all_terminal']); self.assertEqual(len(report['ranking']), 18)
        self.assertEqual(len(report['promotion']['survivors']), 12)
        self.assertEqual(report['measured_h200_hours'], 9)
        self.assertEqual(report['outstanding_reserved_h200_hours'], 0)
        self.assertFalse(controller.stage_path(self.root, 'short').exists())

    def test_frozen_source_and_stage_horizon_cannot_change(self):
        self.enable()
        with patch.object(controller, 'snapshot', return_value={'changed': 'hash'}), \
             patch.object(controller.subprocess, 'run') as launch, self.assertRaisesRegex(ValueError, 'Source changed'):
            controller.tick(self.root, 'initial')
        launch.assert_not_called()
        changed = {**self.calibration, 'slowest_8gpu_update_seconds': .3}
        atomic_json(self.calibration_path, changed)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            controller.prepare(self.root, 'initial', self.calibration_path)

    def test_promotion_requires_prior_terminal_evidence(self):
        atomic_json(self.directory / 'report.json', {'all_terminal': False})
        with self.assertRaisesRegex(ValueError, 'preceding stage'):
            controller.prepare(self.root, 'short', self.calibration_path)
        report = dict(all_terminal=True, promotion={'survivors': [candidates()[0]['id']]},
            results=[dict(id=candidates()[0]['id'], run_id='parent-run')])
        atomic_json(self.directory / 'report.json', report)
        manifest = controller.prepare(self.root, 'short', self.calibration_path)
        self.assertEqual(len(manifest['candidates']), 1)
        cfg = load_config(manifest['candidates'][0]['config'])
        self.assertEqual(cfg['parent_runs'], ['parent-run'])
        self.assertFalse(read_json(controller.stage_path(self.root, 'short') / 'state.json')['enabled'])


if __name__ == '__main__':
    unittest.main()
