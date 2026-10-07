#!/usr/bin/env python3
"""Slurm-only 1/7-worker equivalence, native ordering and interrupted-resume checks."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as pipeline
from preparation_parallel import configure_processes, configure_threads, ordered_executor


class Fixture:
    def __init__(self, path, cfg, split, evaluation, inverse):
        import numpy as np
        self.path, self.config, self.manifests = path, cfg, {'split': {'details': split}}
        self.path.mkdir(parents=True)
        self.directory('split')
        np.save(self.path / 'split/inverse.npy', inverse)
        np.save(self.path / 'split/evaluation.npy', evaluation)
    def directory(self, stage):
        path = self.path / stage
        path.mkdir(exist_ok=True)
        return path
    def manifest(self, stage):
        return self.manifests[stage]
    def finish(self, stage, paths, details):
        self.manifests[stage] = {'details': details}
        pipeline.atomic_json(self.directory(stage) / 'manifest.json', {'details': details})


def stop_after_progress(prep, stage):
    if stage in {'clean', 'score'}:
        path = prep.path / stage / ('documents.sqlite' if stage == 'clean' else 'scores.sqlite')
        if path.exists():
            with sqlite3.connect(path) as db:
                key = 'next_row' if stage == 'clean' else 'last_row'
                if pipeline.progress(db, key, -1) >= 0:
                    raise pipeline.StopRequested('fixture interrupt after a committed batch')
    else:
        path = prep.path / stage / 'state.json'
        if path.exists() and pipeline.read_json(path)['train_files']:
            raise pipeline.StopRequested('fixture interrupt after a committed shard')


def run_fixture(prep, resume):
    for stage in ('clean', 'score', 'select', 'pack'):
        function = getattr(pipeline, 'stage_' + stage)
        if resume:
            with patch.object(pipeline, 'stop_if_requested', side_effect=lambda: stop_after_progress(prep, stage)):
                try:
                    function(prep)
                except pipeline.StopRequested:
                    pass
                else:
                    raise AssertionError(f'{stage} fixture did not reach its interruption boundary')
        function(prep)
    with sqlite3.connect(prep.path / 'clean/documents.sqlite') as db:
        winners = db.execute('SELECT * FROM winners ORDER BY position').fetchall()
        empty = db.execute('SELECT * FROM empty ORDER BY row_index').fetchall()
    with sqlite3.connect(prep.path / 'score/scores.sqlite') as db:
        scores = db.execute('SELECT * FROM scores ORDER BY row_index').fetchall()
    with sqlite3.connect(prep.path / 'select/candidates.sqlite') as db:
        candidates = db.execute('SELECT * FROM candidates ORDER BY position').fetchall()
    selection, packed = [prep.manifest(stage)['details'] for stage in ('select', 'pack')]
    records = {'winners': winners, 'empty': empty, 'scores': scores, 'candidates': candidates}
    for name, files in [('train_documents', selection['train_files']), ('eval_documents', selection['eval_files']),
                        ('train_sequences', packed['train_files']), ('eval_sequences', packed['eval_files'])]:
        records[name] = [row for row, _ in pipeline.iter_arrow(files)]
    records['counters'] = {key: value for key, value in packed.items() if key.endswith('tokens') or key.endswith('count')}
    # Arrow bytes should be identical too, independent of worker scheduling.
    records['arrow_hashes'] = [[item['sha256'] for item in details[key]]
                              for details in (selection, packed) for key in ('train_files', 'eval_files')]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require_compute()
    configure_threads()
    process_start_methods = configure_processes()
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    root = pipeline.setup_storage(args.root)
    if not 1 <= len(args.run_id) <= 96 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in args.run_id) or args.run_id in {'.', '..'}:
        parser.error('Use a valid unique run ID')
    path = root / 'runs/preparation-validation' / args.run_id
    path.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import pyarrow as pa
    from datasets import Dataset, load_dataset, load_from_disk
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    rows = []
    for i in range(128):
        text = ('This is a detailed explanation of an educational topic. ' + str(i))
        if i % 8 == 0:
            text = 'long document ' * 6000 + str(i)
        elif i % 13 == 0:
            text = '   '
        elif i % 11 == 0:
            text = 'A repeated document for exact deduplication.'
        row = {'id': str(i), 'text': text}
        if i % 4 == 0:
            row.update(metadata={'lang': {'en': 0.99}, 'dolma2_qc': {'0': 0.2, '1': 0.8},
                                 'sa_remove_ranges': [], 'mixed': [None, {'rule.2': [1, 2]}]},
                       attributes={'different': True}, unknown={'unicode': '臺灣', 'integer': 2**60})
        elif i % 4 == 1:
            row.update(metadata='string metadata', attributes=[1, 2])
        elif i % 4 == 2:
            row.update(metadata=None, attributes=None)
        rows.append(row)
    shard_paths, projection_paths = [], []
    import zstandard as zstd
    for shard in range(8):
        target = path / f'fixture-{shard}.jsonl.zst'
        payload = ''.join(json.dumps(row) + '\n' for row in rows[shard * 16:(shard + 1) * 16]).encode()
        target.write_bytes(zstd.ZstdCompressor().compress(payload))
        shard_paths.append(str(target))
        projection = path / f'projection-{shard}.jsonl.zst'
        projected = [{'id': row['id'], 'text': row['text']} for row in rows[shard * 16:(shard + 1) * 16]]
        projection.write_bytes(zstd.ZstdCompressor().compress(
            ''.join(json.dumps(row) + '\n' for row in projected).encode()))
        projection_paths.append(str(projection))
    # Exercise native loading after the same kind of threaded hash prelude as
    # completed-download reuse. Every subsequent child must start fresh.
    with ordered_executor(7, threads=True) as ordered:
        assert len(list(ordered(pipeline.artifact, map(Path, shard_paths)))) == 8
    print('native-loader validation: compressed shards and threaded prelude ready', flush=True)
    native, lossless = [], []
    for workers in (1, 7):
        print(f'native-loader validation: loading with {workers} workers', flush=True)
        native.append(load_dataset('json', data_files={'train': projection_paths}, split='train',
                                   num_proc=workers, cache_dir=str(path / f'native-{workers}')))
        lossless.append(pipeline.load_lossless_raw(shard_paths, workers, path / f'lossless-{workers}'))
    assert native[0].to_list() == [{'id': row['id'], 'text': row['text']} for row in rows]
    assert native[0].to_dict() == native[1].to_dict(), 'Native parallel loader changed canonical order'
    permutations = [np.asarray(ds.shuffle(seed=42)._indices.column(0), dtype=np.uint64) for ds in native]
    assert np.array_equal(*permutations), 'Parallel construction changed canonical shuffle indices'
    assert lossless[0].to_dict() == lossless[1].to_dict(), 'Adapter worker count changed original JSON/order'
    for adapted in lossless:
        assert [pipeline.original_document(row) for row in adapted] == rows
        pipeline.compare_raw_chunk(adapted, 0, rows)
        assert np.array_equal(np.asarray(adapted.shuffle(seed=42)._indices.column(0), dtype=np.uint64),
                              permutations[0]), 'Adapter changed canonical seed-42 indices'
    # Bounded real-corpus coverage on the compute node, no corpus sampling on login.
    from itertools import islice
    source_manifest = pipeline.read_json(root / 'preprocessed/dolma-dclm10-001/download/manifest.json')
    sampled_sources, real_rows, real_files, real_projections = set(), [], [], []
    for item in source_manifest['details']['files']:
        if item['source'] in sampled_sources:
            continue
        sampled_sources.add(item['source'])
        records = list(islice(pipeline.raw_json_records(item['path']), 2))
        assert records, 'Empty real source sample'
        real_rows.extend(document for _, document in records)
        sample_path = path / f'real-{len(real_files)}.jsonl.zst'
        sample_path.write_bytes(zstd.ZstdCompressor().compress(''.join(line for line, _ in records).encode()))
        real_files.append(str(sample_path))
        projection = path / f'real-projection-{len(real_projections)}.jsonl.zst'
        projected = [{'id': doc.get('id'), 'text': doc['text']} for _, doc in records]
        projection.write_bytes(zstd.ZstdCompressor().compress(
            ''.join(json.dumps(doc) + '\n' for doc in projected).encode()))
        real_projections.append(str(projection))
    assert sampled_sources == {item['source'] for item in source_manifest['details']['files']}
    real_reference = load_dataset('json', data_files={'train': real_projections}, split='train',
                                  cache_dir=str(path / 'real-reference'))
    real_perm = np.asarray(real_reference.shuffle(seed=42)._indices.column(0), dtype=np.uint64)
    for workers in (1, 7):
        real_raw = pipeline.load_lossless_raw(real_files, workers, path / f'real-adapter-{workers}')
        pipeline.compare_raw_chunk(real_raw, 0, real_rows)
        assert np.array_equal(np.asarray(real_raw.shuffle(seed=42)._indices.column(0), dtype=np.uint64), real_perm)
        assert [pipeline.original_document(row) for row in real_raw] == real_rows
    print(f'lossless loader: original fields and canonical indices checked on {len(real_rows)} real documents', flush=True)
    raw_path = path / 'raw'
    lossless[0].save_to_disk(str(raw_path))
    raw = load_from_disk(str(raw_path))
    perm = permutations[0]
    inverse = np.empty(len(raw), dtype=np.uint64)
    inverse[perm] = np.arange(len(raw))
    evaluation = perm[-5:]
    assert not set(map(int, evaluation)) & set(map(int, perm[:-5]))
    split = {'row_count': len(raw), 'train_document_count': len(raw) - 5,
             'raw_cache_files': [pipeline.artifact(Path(item['filename'])) for item in raw.cache_files],
             'ranges': [{'start': 0, 'end': 64, 'source': 'common_crawl'},
                        {'start': 64, 'end': 128, 'source': 'stack_edu'}]}
    cfg = pipeline.read_json(pipeline.PROJECT / 'configs/dataset.json')
    cfg.update(max_training_tokens=30000, document_batch_size=4, document_shard_tokens=4096,
               sequence_shard_rows=2, evaluation_documents=5, evaluation_subset_documents=2)
    # Exercise the production split entrypoint as well as its component contracts.
    from types import SimpleNamespace
    canonical_names = [f'data/common_crawl-fixture-0000/shard_{i}.jsonl.zst' for i in range(8)]
    builder = SimpleNamespace(config=SimpleNamespace(data_files={
        'train': ['hf://datasets/fixture@pinned/' + name for name in canonical_names]}))
    for workers in (1, 7):
        split_root = path / f'full-split-{workers}'
        full_prep = Fixture(split_root / 'preprocessed/fixture', {**cfg, 'preprocessing_workers': workers},
                            split, evaluation, inverse)
        full_prep.root = split_root
        full_prep.manifests['download'] = {'details': {'canonical_files': canonical_names, 'files': [
            {'path': filename, 'repo_filename': name, 'source': 'common_crawl'}
            for filename, name in zip(shard_paths, canonical_names)]}}
        with patch('datasets.load_dataset_builder', return_value=builder):
            pipeline.stage_split(full_prep)
        completed = full_prep.manifest('split')['details']
        assert completed['row_count'] == len(rows)
        assert [item['end'] for item in completed['ranges']] == list(range(16, 129, 16))
        assert np.array_equal(np.load(full_prep.path / 'split/permutation.npy'), permutations[0])
        assert np.array_equal(np.load(full_prep.path / 'split/evaluation_subset.npy'), evaluation[:2])
        assert [pipeline.original_document(row) for row, _ in
                pipeline.iter_arrow(completed['evaluation_raw_files'])] == [rows[int(i)] for i in evaluation]
    archives = []
    for interrupted in (False, True):
        archive_prep = Fixture(path / f'archive-resume-{interrupted}', cfg, split, evaluation, inverse)
        if interrupted:
            counter = {'calls': 0}
            def interrupt_archive():
                counter['calls'] += 1
                if counter['calls'] == 2:
                    raise pipeline.StopRequested('interrupt archive after first shard')
            with patch.object(pipeline, 'stop_if_requested', side_effect=interrupt_archive):
                try:
                    pipeline.archive_evaluation(archive_prep, raw, evaluation, split['ranges'], inverse)
                except pipeline.StopRequested:
                    pass
                else:
                    raise AssertionError('archive did not reach interruption boundary')
        files = pipeline.archive_evaluation(archive_prep, raw, evaluation, split['ranges'], inverse)
        restored = [row for row, _ in pipeline.iter_arrow(files)]
        assert [row['row_index'] for row in restored] == evaluation.tolist()
        assert [pipeline.original_document(row) for row in restored] == [rows[int(i)] for i in evaluation]
        assert [row['document_json'] for row in restored] == [raw[int(i)]['document_json'] for i in evaluation]
        assert [row['evaluation_slot'] for row in restored] == list(range(len(evaluation)))
        archives.append({'rows': restored, 'hashes': [item['sha256'] for item in files]})
    assert archives[0] == archives[1], 'Evaluation original archive resume changed content/order'

    expected = None
    for workers, resume in ((1, False), (7, False), (7, True)):
        recipe = {**cfg, 'preprocessing_workers': workers, 'score_workers': workers}
        prep = Fixture(path / f'workers-{workers}-resume-{resume}', recipe, split, evaluation, inverse)
        actual = run_fixture(prep, resume)
        if expected is None:
            expected = actual
        else:
            assert actual == expected, f'Worker/resume result changed: {workers}/{resume}'
    # Isolated synthetic sealed fixture: prove offline load still works after
    # manifest-owned intermediate retirement. This is not a real-corpus verify seal.
    from slurm.cleanup import cleanup_plan, apply_plan
    offline_root = path / 'offline-cleanup'
    copied_raw_path = offline_root / 'caches/datasets/raw'
    raw.save_to_disk(str(copied_raw_path))
    copied_raw = load_from_disk(str(copied_raw_path))
    offline_split = {**split, 'raw_cache_files': [pipeline.artifact(Path(i['filename']))
                                                for i in copied_raw.cache_files]}
    offline = Fixture(offline_root / 'preprocessed/fixture', cfg, offline_split, evaluation, inverse)
    with patch.object(pipeline, 'load_raw', return_value=copied_raw):
        run_fixture(offline, False)
    originals = pipeline.archive_evaluation(offline, copied_raw, evaluation, split['ranges'], inverse)
    seal_id = 'synthetic-fixture-only'
    pipeline.atomic_json(offline.path / 'configuration.json', {'fingerprint': seal_id, 'config': cfg})
    corpus = offline_root / 'caches/hub/fixture.jsonl.zst'
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_bytes(b'synthetic download ownership fixture')
    manifests = {
        'download': {'details': {}, 'artifacts': [pipeline.artifact(corpus)]},
        'split': {'details': {**offline_split, 'evaluation_raw_files': originals,
                             'evaluation_raw_documents': len(evaluation)},
                  'artifacts': offline_split['raw_cache_files'] + originals},
        'select': {'details': offline.manifest('select')['details']},
        'pack': {'details': offline.manifest('pack')['details']},
    }
    selected_offline = manifests['select']['details']
    manifests['select']['artifacts'] = (selected_offline['train_files'] + selected_offline['eval_files'] +
        [pipeline.artifact(p) for p in Path(selected_offline['tokenizer_directory']).iterdir() if p.is_file()])
    manifests['pack']['artifacts'] = manifests['pack']['details']['train_files'] + manifests['pack']['details']['eval_files']
    for stage, value in manifests.items():
        value.update(fingerprint=seal_id, stage=stage)
        pipeline.atomic_json(offline.directory(stage) / 'manifest.json', value)
    packed_digest = pipeline.sha256(offline.path / 'pack/manifest.json')
    report = offline.directory('verify') / 'fixture-report.json'
    pipeline.atomic_json(report, {'scope': 'synthetic offline retirement contract only'})
    pipeline.atomic_json(offline.path / 'verify/manifest.json',
        {'fingerprint': seal_id, 'stage': 'verify', 'parent_sha256': packed_digest,
         'details': {'pack_manifest_sha256': packed_digest, 'evaluation_raw_documents': len(evaluation),
                     'evaluation_raw_files': originals},
         'artifacts': [pipeline.artifact(report)] + originals})
    before = {key: pipeline.load_prepared(offline.path, key).to_dict()
              for key in ('train', 'evaluation', 'evaluation_subset')}
    retirement = cleanup_plan(offline_root, 'fixture')
    assert retirement['candidates'], 'Offline cleanup fixture selected no intermediates'
    apply_plan(offline_root, retirement)
    for key, expected_rows in before.items():
        assert pipeline.load_prepared(offline.path, key).to_dict() == expected_rows, \
            f'Offline loader changed after retirement: {key}'
    assert all(not Path(item['path']).exists() for item in retirement['candidates'])
    assert all(Path(item['path']).exists() for item in originals + selected_offline['eval_files'])

    # A worker failure in the first physical batch must commit no cursor or manifest.
    bad = Fixture(path / 'worker-failure', {**cfg, 'preprocessing_workers': 7}, split, evaluation, inverse)
    invalid = [{'id': row['id'], 'text': row['text']} for row in rows]
    first_training = next(row for row in range(4) if int(inverse[row]) < split['train_document_count'])
    invalid[first_training] = {**invalid[first_training], 'text': None}
    with patch.object(pipeline, 'load_raw', return_value=Dataset.from_list(invalid)):
        try:
            pipeline.stage_clean(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid worker input was accepted')
    with sqlite3.connect(bad.path / 'clean/documents.sqlite') as db:
        assert pipeline.progress(db, 'next_row', 0) == 0
    assert not (bad.path / 'clean/manifest.json').exists()
    result = {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'], 'allocated_cpus': int(os.environ['SLURM_CPUS_PER_TASK']),
              'process_start_methods': process_start_methods,
              'compressed_native_input': True, 'threaded_prelude': True,
              'lossless_schema_adapter': True, 'real_sample_documents': len(real_rows),
              'real_sample_sources': sorted(sampled_sources),
              'workers': [1, 7], 'checks': ['native ordering', 'canonical shuffle', 'split disjointness',
              'dedup winners', 'official scores', 'candidate order', 'token IDs and budget',
              'packing and masks', 'Arrow checksums', 'committed resume', 'worker failure cursor',
              'full original evaluation archive and interrupted resume',
              'offline training/full-eval/subset reload after intermediate retirement']}
    pipeline.atomic_json(path / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
