"""Small synchronous fixtures only; real worker tests run through Slurm."""
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dataset as pipeline
from preparation_parallel import bounded_map, worker_count
from slurm.compare_scaling import compare, select_gpu_count


class FakeExecutor:
    def __init__(self):
        self.submitted = []
        self.futures = []
    def submit(self, function, item):
        self.submitted.append(item)
        future = Future()
        try:
            future.set_result(function(item))
        except Exception as exc:
            future.set_exception(exc)
        self.futures.append(future)
        return future


class ParallelTests(unittest.TestCase):
    def test_bounded_submission_order_and_failure(self):
        executor = FakeExecutor()
        generator = bounded_map(executor, lambda value: value * 2, range(100), 3)
        self.assertEqual(next(generator), 0)
        self.assertEqual(len(executor.submitted), 3)
        self.assertEqual(next(generator), 2)
        self.assertEqual(len(executor.submitted), 4)
        generator.close()
        executor = FakeExecutor()
        def fail(value):
            if value == 2:
                raise RuntimeError('worker failed')
            return value
        generator = bounded_map(executor, fail, range(100), 3)
        self.assertEqual(next(generator), 0)
        self.assertEqual(next(generator), 1)
        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
            next(generator)
        self.assertEqual(len(executor.submitted), 5)

    def test_cpu_budget_and_coordinator(self):
        cfg = {'preprocessing_workers': 7, 'score_workers': 7}
        for cpus, expected in [(1, 1), (2, 1), (4, 3), (8, 7)]:
            with patch.dict('os.environ', {'SLURM_CPUS_PER_TASK': str(cpus)}):
                self.assertEqual(worker_count(cfg), expected)
                self.assertEqual(worker_count(cfg, scoring=True), expected)
        for cpus in ('0', '9'):
            with patch.dict('os.environ', {'SLURM_CPUS_PER_TASK': cpus}), self.assertRaises(ValueError):
                worker_count(cfg)
        with self.assertRaises(ValueError):
            worker_count({'preprocessing_workers': 8})

    def test_scaling_requires_both_thresholds_and_enough_updates(self):
        self.assertEqual(select_gpu_count(1.8, 1.25, 20), 8)
        for speed, cost, count in [(1.79, 1, 20), (2, 1.26, 20), (2, 1, 19), (float('nan'), 1, 20)]:
            self.assertEqual(select_gpu_count(speed, cost, count), 4)

    def test_scaling_uses_reconciled_allocation_cost_and_matched_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = pipeline.read_json(pipeline.PROJECT / 'configs/train-scaling-4.json')
            for count, hours in [(4, 0.1), (8, 0.125)]:
                path = root / 'runs/experiments' / f'run{count}'
                path.mkdir(parents=True)
                pipeline.atomic_json(path / 'resolved-config.json',
                                     {**base, 'world_size': count, 'gradient_accumulation_steps': 16 // count})
                pipeline.atomic_json(path / 'summary.json',
                                     {'data_provenance': {'fingerprint': 'same'}, 'total_tokens_seen': 1000,
                                      'stop_reason': 'max_steps'})
                pipeline.atomic_json(path / 'execution-identity.json',
                                     {'source_hashes': {'train.py': 'same'}, 'environment': {'torch': 'same'}})
                pipeline.atomic_json(path / 'resource-accounting.json',
                                     {'all_jobs_reconciled': True, 'budget_exceeded': False,
                                      'h200_hours_measured': hours, 'jobs': [{'state': 'COMPLETED'}]})
                (path / 'metrics.jsonl').write_text(''.join(json.dumps(
                    {'train/global_step': step, 'train/step_seconds': 1 if count == 4 else 0.5,
                     'train/step_input_tokens': 50}) + '\n' for step in range(11, 31)))
            report = compare(root, 'run4', 'run8')
            self.assertEqual(report['selected_gpus'], 8)
            self.assertEqual(report['speedup'], 2)
            self.assertEqual(report['cost_ratio'], 1.25)
            account_path = root / 'runs/experiments/run8/resource-accounting.json'
            account = pipeline.read_json(account_path)
            pipeline.atomic_json(account_path, {**account, 'h200_hours_measured': 0.13})
            self.assertEqual(compare(root, 'run4', 'run8')['selected_gpus'], 4)
            pipeline.atomic_json(account_path, {**account, 'all_jobs_reconciled': False})
            with self.assertRaises(ValueError):
                compare(root, 'run4', 'run8')
            timed_summary = root / 'runs/experiments/run8/summary.json'
            summary = pipeline.read_json(timed_summary)
            pipeline.atomic_json(timed_summary, {**summary, 'stop_reason': 'wall_time_or_signal'})
            pipeline.atomic_json(account_path, {**account, 'jobs': [{'state': 'FAILED', 'job_id': '42'}]})
            with self.assertRaises(ValueError):
                compare(root, 'run4', 'run8')
            exit_path = root / 'logs/job-42/exit-code.txt'
            exit_path.parent.mkdir(parents=True)
            exit_path.write_text('75\n')
            self.assertEqual(compare(root, 'run4', 'run8')['selected_gpus'], 8)
            exit_path.write_text('1\n')
            with self.assertRaises(ValueError):
                compare(root, 'run4', 'run8')

    def test_reuse_requires_complete_matching_download_and_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'preprocessed/old'
            (source / 'download').mkdir(parents=True)
            cfg = {'dataset_id': 'fixture', 'dataset_revision': 'a' * 40}
            identity = {'config': cfg}
            key = pipeline.fingerprint(identity)
            pipeline.atomic_json(source / 'configuration.json', {'fingerprint': key, **identity})
            filename = 'data/stack_edu-test/shard_1.jsonl.zst'
            file = root / 'cached.zst'
            file.write_bytes(b'abc')
            item = {**pipeline.artifact(file), 'repo_filename': filename, 'source': 'stack_edu'}
            manifest = {'stage': 'download', 'fingerprint': key, 'parent_sha256': None,
                        'details': {'canonical_files': [filename], 'files': [item]},
                        'artifacts': [pipeline.artifact(file)]}
            captured = []
            prep = SimpleNamespace(root=root, path=root / 'preprocessed/new', config=cfg,
                                   reuse_download_from='old', finish=lambda *a, **kw: captured.append((a, kw)))
            with self.assertRaises(FileNotFoundError):
                pipeline.reuse_download(prep, [filename])
            pipeline.atomic_json(source / 'download/manifest.json', manifest)
            pipeline.reuse_download(prep, [filename])
            self.assertEqual(captured[0][0][2]['reused_download']['fingerprint'], key)
            with self.assertRaises(ValueError):
                pipeline.reuse_download(prep, ['data/stack_edu-test/shard_2.jsonl.zst'])
            prep.config = {**cfg, 'dataset_revision': 'b' * 40}
            with self.assertRaises(ValueError):
                pipeline.reuse_download(prep, [filename])
            prep.config = cfg
            file.write_bytes(b'xyz')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                pipeline.reuse_download(prep, [filename])
            file.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                pipeline.reuse_download(prep, [filename])


if __name__ == '__main__':
    unittest.main()
