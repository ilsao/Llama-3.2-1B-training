"""Tiny metadata and file import contracts; no corpus, models or workers."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import dataset


class RepairImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repair_root = self.root / 'runs/cache-repair/repair'
        self.config = {'dataset_id': 'dataset', 'dataset_revision': 'a' * 40}
        self.files = [self.item('stack_edu', 'one', 'old1'),
                      self.item('common_crawl', 'two', 'old2'),
                      self.item('finemath', 'three', 'old3')]
        replacement = {**self.files[1], 'path': str(self.repair_root / 'hub/new'), 'sha256': 'b' * 64}
        decoded = {'passed': True, 'decoded_bytes': 100, 'decoded_sha256': 'c' * 64}
        self.result = {'passed': True, 'approved_for_new_preparation_import': True,
                       'old_artifacts_modified': False, 'document_count': 10,
                       'identity': {**self.config, 'source_preparation': 'old',
                                    'source_fingerprint': 'fingerprint', 'source_manifest_sha256': 'manifest'},
                       'original': dict(self.files[1]), 'replacement': replacement,
                       'upstream': {'sha256': 'b' * 64, 'size': 20},
                       'replacement_decoders': [{**decoded, 'backend': 'python-zstandard'},
                                                {**decoded, 'backend': 'arrow-zstd'}]}

    def item(self, source, name, filename):
        return {'repo_filename': f'data/{source}-family/{name}.jsonl.zst',
                'source': source, 'path': str(self.root / filename), 'size': 20, 'sha256': 'a' * 64}

    def validate(self, result=None, files=None):
        return dataset.validated_repair_files(self.files if files is None else files,
                 self.result if result is None else result, 'old', 'fingerprint', 'manifest',
                 self.config, self.repair_root)

    def test_only_authenticated_artifact_changes_and_order_is_preserved(self):
        original = copy.deepcopy(self.files)
        new = self.validate()
        self.assertEqual(self.files, original)
        self.assertEqual([x['repo_filename'] for x in new], [x['repo_filename'] for x in original])
        self.assertEqual(new[0], original[0]); self.assertEqual(new[2], original[2])
        self.assertEqual(new[1], self.result['replacement'])

    def test_failed_partial_or_different_source_is_rejected(self):
        cases = [('passed', False), ('approved_for_new_preparation_import', False),
                 ('old_artifacts_modified', True), ('document_count', 0)]
        for key, value in cases:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate({**self.result, key: value})
        for key in ('dataset_id', 'dataset_revision', 'source_preparation',
                    'source_fingerprint', 'source_manifest_sha256'):
            bad = copy.deepcopy(self.result); bad['identity'][key] = 'other'
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(bad)

    def test_original_membership_and_bytes_must_match(self):
        for files in [self.files[:1], self.files + [self.files[1]],
                      [self.files[0], {**self.files[1], 'sha256': 'b' * 64}, self.files[2]]]:
            with self.assertRaises(ValueError): self.validate(files=files)

    def test_upstream_filename_source_size_and_safe_path_are_required(self):
        for key, value in [('sha256', 'd' * 64), ('size', 19), ('source', 'stack_edu'),
                           ('repo_filename', self.files[0]['repo_filename']),
                           ('path', str(self.root / 'other-project/new')),
                           ('path', self.files[1]['path'])]:
            bad = copy.deepcopy(self.result); bad['replacement'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.validate(bad)

    def test_independent_complete_decoders_are_required(self):
        for key, value in [('passed', False), ('decoded_bytes', 0),
                           ('decoded_sha256', 'e' * 64), ('backend', 'python-zstandard')]:
            bad = copy.deepcopy(self.result); bad['replacement_decoders'][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(bad)

    def test_compute_import_hashes_replacement_and_preserves_old_manifest(self):
        source = self.root / 'preprocessed/old'; (source / 'download').mkdir(parents=True)
        (source / '.lock').touch(); self.repair_root.mkdir(parents=True)
        (self.repair_root / '.lock').touch()
        for item in self.files + [self.result['replacement']]:
            p = Path(item['path']); p.parent.mkdir(parents=True, exist_ok=True);p.write_text(p.name)
            item.update(dataset.artifact(p))
        self.result['original'] = dict(self.files[1])
        new = self.result['replacement'];self.result['upstream'] = {k:new[k] for k in ('sha256','size')}
        identity = {'config': self.config}; fingerprint = dataset.fingerprint(identity)
        dataset.atomic_json(source / 'configuration.json', {**identity, 'fingerprint': fingerprint})
        names = [x['repo_filename'] for x in self.files]
        manifest_path = source / 'download/manifest.json'
        dataset.atomic_json(manifest_path, {'stage':'download','parent_sha256':None,
            'fingerprint':fingerprint,'details':{'files':self.files,'canonical_files':names},
            'artifacts':[{k:x[k] for k in ('path','size','sha256')} for x in self.files]})
        original_manifest = manifest_path.read_bytes()
        self.result['identity'].update(source_fingerprint=fingerprint,
            source_manifest_sha256=dataset.sha256(manifest_path),
            code_sha256=dataset.sha256(dataset.PROJECT / 'slurm/cache_repair.py'))
        self.result.update(job_id='1', jobs=['1'])
        result_path = self.repair_root / 'result.json';dataset.atomic_json(result_path,self.result)
        dataset.atomic_json(self.root / 'preprocessing-ledger.json', {'jobs':{'1':{
            'state':'COMPLETED','stage':'cache-repair','preparation-id':'repair'}}})
        new_directory = self.root / 'preprocessed/new/download';new_directory.mkdir(parents=True)
        outputs = []
        prep = SimpleNamespace(root=self.root,path=new_directory.parent,config={**self.config,'workers':1},
            reuse_download_from='old',repair_download_from='repair',
            repair_result_sha256=dataset.sha256(result_path),directory=lambda stage:new_directory,
            finish=lambda *args,**kwargs:outputs.append((args,kwargs)))
        dataset.reuse_download(prep,names)
        args,kwargs = outputs[0]
        self.assertEqual(args[2]['files'], [self.files[0], new, self.files[2]])
        self.assertEqual(kwargs['records'][1]['sha256'],new['sha256'])
        self.assertEqual(manifest_path.read_bytes(), original_manifest)
        Path(new['path']).write_text('tampered')
        with self.assertRaises(ValueError): dataset.reuse_download(prep,names)
        self.assertEqual(len(outputs),1)
        prep.repair_result_sha256 = 'f' * 64
        with self.assertRaises(ValueError): dataset.reuse_download(prep,names)
