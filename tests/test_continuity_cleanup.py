"""Small fixtures for scheduling and deletion safety; no models/workers/data scans."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dataset
from slurm import cleanup, heartbeat


class HeartbeatTests(unittest.TestCase):
    def test_install_twice_and_disable_preserve_other_cron_entries(self):
        original = '7 * * * * /another/project/monitor # other\n'
        installed = heartbeat.update_cron(original, True)
        self.assertEqual(heartbeat.update_cron(installed, True), installed)
        self.assertEqual(installed.count(heartbeat.MARKER), 1)
        self.assertEqual(heartbeat.update_cron(installed, False), original)
        self.assertIn('*/20 * * * * /usr/bin/python3', installed)

    def test_existing_controller_lock_skips_agent_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'runs/heartbeat'
            directory.mkdir(parents=True)
            with (directory / 'controller.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(heartbeat, 'ROOT', root), patch.object(heartbeat, 'STOP', root/'disabled'), \
                     patch('slurm.heartbeat.subprocess.Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(heartbeat.run_once(True), 0)
                    launch.assert_not_called()

    def test_disabled_skips_agent_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            flag = Path(temporary) / 'disabled'
            flag.touch()
            with patch.object(heartbeat, 'STOP', flag), patch('slurm.heartbeat.subprocess.Popen') as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(heartbeat.run_once(), 0)
                launch.assert_not_called()


class CleanupTests(unittest.TestCase):
    def fixture(self, root):
        prep = root / 'preprocessed/example'
        def save(relative, value):
            path = prep / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.atomic_json(path, value)
            return path
        def asset(relative):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            return dataset.artifact(path)
        corpus = asset('caches/hub/corpus.zst')
        raw = asset('caches/datasets/raw.arrow')
        train_docs = asset('preprocessed/example/select/train-doc.arrow')
        eval_docs = asset('preprocessed/example/select/eval-doc.arrow')
        eval_raw = asset('preprocessed/example/split/evaluation-raw/original.arrow')
        tokenizer = asset('preprocessed/example/select/tokenizer/tokenizer.json')
        packed_asset = asset('preprocessed/example/pack/train.arrow')
        fingerprint = 'fixed'
        save('configuration.json', {'fingerprint':fingerprint,'config':{'evaluation_documents':50000}})
        pack = {'fingerprint':fingerprint, 'artifacts':[packed_asset], 'details':{}}
        path = save('pack/manifest.json', pack)
        save('verify/manifest.json', {'fingerprint':fingerprint, 'artifacts':[eval_raw],
             'parent_sha256':dataset.sha256(path), 'details':{'pack_manifest_sha256':dataset.sha256(path),
             'evaluation_raw_documents':50000, 'evaluation_raw_files':[eval_raw]}})
        save('split/manifest.json', {'fingerprint':fingerprint, 'details':{'raw_cache_files':[raw],
             'evaluation_raw_files':[eval_raw], 'evaluation_raw_documents':50000}})
        save('select/manifest.json', {'fingerprint':fingerprint,'artifacts':[train_docs,eval_docs,tokenizer],
             'details':{'train_files':[train_docs], 'eval_files':[eval_docs],
             'tokenizer_directory':str(Path(tokenizer['path']).parent)}})
        save('download/manifest.json', {'fingerprint':fingerprint,'artifacts':[corpus], 'details':{}})
        return prep, corpus, raw, train_docs, eval_raw, eval_docs, tokenizer, packed_asset

    def test_plan_preserves_other_preparation_and_all_durable_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prep, corpus, raw, train, *retained = self.fixture(root)
            other = root / 'preprocessed/unfinished/download/state.json'
            other.parent.mkdir(parents=True)
            dataset.atomic_json(other, {'files':[corpus]})
            plan = cleanup.cleanup_plan(root, 'example')
            self.assertEqual({i['path'] for i in plan['candidates']}, {raw['path'],train['path']})
            self.assertEqual(plan['protected_candidate_count'], 1)
            unrelated = root.parent / (root.name + '-other-project')
            unrelated.write_text('keep')
            try:
                cleanup.apply_plan(root, plan)
                for item in [corpus, *retained]:
                    dataset.check_artifact(item, True)
                self.assertEqual(unrelated.read_text(), 'keep')
                self.assertFalse(Path(raw['path']).exists())
                self.assertFalse(Path(train['path']).exists())
                # Retried retirement preserves receipts and does not require deleted inputs.
                self.assertEqual(cleanup.apply_plan(root, plan), 0)
                self.assertEqual(dataset.read_json(prep/'retention.json')['state'], 'retired')
            finally:
                unrelated.unlink()

    def test_outside_symlink_and_hardlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); root = base / 'project'; root.mkdir()
            outside = base/'another-project'; outside.write_text('keep')
            link=root/'link';link.symlink_to(outside)
            for path in [outside,link]:
                with self.assertRaises(ValueError): cleanup.contained_file(root,path)
            inside=root/'inside';inside.write_text('keep')
            internal_link=root/'internal';internal_link.symlink_to(inside)
            with self.assertRaises(ValueError):cleanup.contained_file(root,internal_link)
            import os
            os.link(inside,root/'hardlink')
            with self.assertRaises(ValueError):cleanup.contained_file(root,inside)

    def test_missing_eval_archive_or_changed_seal_blocks_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prep,*_=self.fixture(root)
            manifest=prep/'verify/manifest.json'
            value=dataset.read_json(manifest)
            value['details']['evaluation_raw_documents']=49999
            dataset.atomic_json(manifest,value)
            with self.assertRaises(ValueError):cleanup.cleanup_plan(root,'example')

    def test_historical_manifest_mutation_blocks_retirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); prep, *_ = self.fixture(root)
            sealed = dataset.read_json(prep / 'verify/manifest.json')
            sealed['details']['manifest_sha256'] = {'select': dataset.sha256(prep / 'select/manifest.json')}
            dataset.atomic_json(prep / 'verify/manifest.json', sealed)
            self.assertTrue(cleanup.cleanup_plan(root, 'example')['candidates'])
            selected = dataset.read_json(prep / 'select/manifest.json')
            selected['details']['train_files'] = []
            dataset.atomic_json(prep / 'select/manifest.json', selected)
            with self.assertRaisesRegex(ValueError, 'Historical manifest changed'):
                cleanup.cleanup_plan(root, 'example')

    def test_active_project_job_blocks_cleanup(self):
        from types import SimpleNamespace
        with patch('slurm.cleanup.subprocess.run',return_value=SimpleNamespace(stdout=f'123|{cleanup.PROJECT}\n')), \
             patch.dict('os.environ',{'SLURM_JOB_ID':'456'}):
            with self.assertRaises(ValueError):cleanup.assert_no_other_project_jobs()

    def test_changed_candidate_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prep,corpus,*_=self.fixture(root)
            plan=cleanup.cleanup_plan(root,'example')
            Path(corpus['path']).write_text('tampered')
            with self.assertRaises(ValueError):cleanup.apply_plan(root,plan)
            self.assertFalse((prep/'retention.json').exists())
            self.assertTrue(Path(corpus['path']).exists())

if __name__ == '__main__':
    unittest.main()
