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
from preparation_parallel import configure_threads, ordered_executor


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
        rows.append({'id': str(i), 'text': text})
    shard_paths = []
    for shard in range(8):
        target = path / f'fixture-{shard}.jsonl'
        target.write_text(''.join(json.dumps(row) + '\n' for row in rows[shard * 16:(shard + 1) * 16]))
        shard_paths.append(str(target))
    native = []
    for workers in (1, 7):
        native.append(load_dataset('json', data_files={'train': shard_paths}, split='train',
                                   num_proc=workers, cache_dir=str(path / f'native-{workers}')))
    assert native[0].to_dict() == native[1].to_dict(), 'Native parallel loader changed canonical order'
    permutations = [np.asarray(ds.shuffle(seed=42)._indices.column(0), dtype=np.uint64) for ds in native]
    assert np.array_equal(*permutations), 'Parallel construction changed canonical shuffle indices'
    raw_path = path / 'raw'
    native[0].save_to_disk(str(raw_path))
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
        assert [row['document'] for row in restored] == [raw[int(i)] for i in evaluation]
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
    # A worker failure in the first physical batch must commit no cursor or manifest.
    bad = Fixture(path / 'worker-failure', {**cfg, 'preprocessing_workers': 7}, split, evaluation, inverse)
    invalid = list(rows)
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
              'workers': [1, 7], 'checks': ['native ordering', 'canonical shuffle', 'split disjointness',
              'dedup winners', 'official scores', 'candidate order', 'token IDs and budget',
              'packing and masks', 'Arrow checksums', 'committed resume', 'worker failure cursor',
              'full original evaluation archive and interrupted resume']}
    pipeline.atomic_json(path / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
