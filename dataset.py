#!/usr/bin/env python3
"""Reproducible document-first Dolma preparation. Heavy entrypoints require Slurm.

Only standard-library modules are imported at module load, so logic tests and
--help do not load a classifier, tokenizer, torch, Arrow, or the corpus.
"""
from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Iterator

from third_party.dclm_fasttext import classify_fasttext_hq_prob

PROJECT = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/work/asciibase64/Llama-3.2-1B-training')
STAGES = ('download', 'split', 'clean', 'score', 'select', 'pack', 'verify')
SOURCES = {'common_crawl', 'dolma1_7', 'finemath', 'olmocr_science_pdfs', 'rpj', 'stack_edu'}
STOP = False


class StopRequested(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def artifact(path: Path) -> dict:
    return {'path': str(path.resolve()), 'size': path.stat().st_size, 'sha256': sha256(path)}


def check_artifact(item: dict, checksums=False) -> None:
    path = Path(item['path'])
    if not path.is_file() or path.stat().st_size != item['size']:
        raise ValueError(f'Missing or changed artifact: {path}')
    if checksums and sha256(path) != item['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')


def source_for_path(filename: str) -> str:
    parts = Path(filename).parts
    if len(parts) != 3 or parts[0] != 'data' or not filename.endswith('.jsonl.zst'):
        raise ValueError(f'Unrecognized dataset shard: {filename}')
    source = parts[1].split('-', 1)[0]
    if source not in SOURCES:
        raise ValueError(f'Unrecognized source: {filename}')
    return source


def repo_filename(url: str) -> str:
    from urllib.parse import unquote
    url = unquote(str(url))
    if '/data/' not in url:
        raise ValueError(f'Unexpected canonical data URL: {url}')
    filename = 'data/' + url.split('/data/', 1)[1]
    source_for_path(filename)
    return filename


def top_count(count: int, fraction: float) -> int:
    # Integer arithmetic for the agreed 10%, avoiding float boundary rounding.
    if fraction != 0.1:
        raise ValueError('This recipe requires exactly the top 10% of documents')
    return count // 10


def fits_document(token_count: int, remaining: int) -> bool:
    return token_count >= 2 and token_count <= remaining


def evaluation_windows(tokens: list[int], length: int) -> Iterator[tuple[int, list[int]]]:
    if length < 2:
        raise ValueError('Context length must be at least two')
    start = 0
    while start < len(tokens):
        window = tokens[start:start + length]
        if len(window) < 2:
            break
        yield start, window
        if start + length >= len(tokens):
            break
        start += length - 1


def padded_sequence(tokens, labels, spans, length, pad_id, evaluation_slot=-1):
    if not 0 < len(tokens) <= length or len(labels) != len(tokens):
        raise ValueError('Invalid sequence')
    valid = len(tokens)
    labels = list(labels)
    labels[0] = -100  # Causal shift never predicts the first input position.
    return {'input_ids': list(tokens) + [pad_id] * (length - valid),
            'attention_mask': [1] * valid + [0] * (length - valid),
            'labels': labels + [-100] * (length - valid),
            'document_spans': spans, 'evaluation_slot': evaluation_slot}


class TrainingPacker:
    """Serializable carry, including structural BOS masks and document offsets."""
    def __init__(self, length, pad_id, state=None):
        self.length, self.pad_id = length, pad_id
        state = state or {}
        self.tokens = state.get('tokens', [])
        self.labels = state.get('labels', [])
        self.spans = state.get('spans', [])

    def state(self):
        return {'tokens': self.tokens, 'labels': self.labels, 'spans': self.spans}

    def push(self, row_index: int, tokens: list[int]):
        offset = len(self.tokens)
        self.spans.append({'row_index': row_index, 'document_start': 0,
                           'sequence_start': offset, 'length': len(tokens)})
        self.tokens.extend(tokens)
        self.labels.extend([-100] + tokens[1:])
        while len(self.tokens) >= self.length:
            spans, remainder = [], []
            for span in self.spans:
                start, size = span['sequence_start'], span['length']
                taken = min(size, max(0, self.length - start))
                if taken:
                    spans.append({**span, 'length': taken})
                if size > taken:
                    remainder.append({**span, 'sequence_start': max(0, start - self.length),
                                      'document_start': span['document_start'] + taken,
                                      'length': size - taken})
            result = padded_sequence(self.tokens[:self.length], self.labels[:self.length],
                                     spans, self.length, self.pad_id)
            self.tokens = self.tokens[self.length:]
            self.labels = self.labels[self.length:]
            self.spans = remainder
            yield result

    def finish(self):
        if self.tokens:
            yield padded_sequence(self.tokens, self.labels, self.spans, self.length, self.pad_id)
            self.tokens, self.labels, self.spans = [], [], []


def pack_evaluation_document(doc, length, pad_id):
    for start, tokens in evaluation_windows(doc['tokens'], length):
        spans = [{'row_index': doc['row_index'], 'document_start': start,
                  'sequence_start': 0, 'length': len(tokens)}]
        yield padded_sequence(tokens, [-100] + tokens[1:], spans, length, pad_id,
                              doc['evaluation_slot'])


def stop_if_requested():
    deadline = int(os.environ.get('PREP_STOP_EPOCH', '0'))
    if STOP or (deadline and time.time() >= deadline):
        raise StopRequested('Checkpointed; stopping before the allocation wall-time limit')


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Heavy preparation must execute on a Slurm compute node; use slurm scripts')


def setup_storage(root):
    root = root.resolve()
    auth_home = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    os.environ.setdefault('HF_TOKEN_PATH', str(auth_home / 'token'))
    if not root.is_relative_to(Path('/work/asciibase64')):
        raise ValueError('Large artifacts must live under /work/asciibase64/')
    for name in ('datasets', 'caches/hub', 'caches/datasets', 'caches/torch',
                 'caches/xdg', 'tmp', 'logs', 'preprocessed', 'runs', 'environments'):
        (root / name).mkdir(parents=True, exist_ok=True)
    for key, value in {'HF_HOME': root / 'caches', 'HF_HUB_CACHE': root / 'caches/hub',
                       'HF_DATASETS_CACHE': root / 'caches/datasets',
                       'TORCH_HOME': root / 'caches/torch', 'XDG_CACHE_HOME': root / 'caches/xdg',
                       'TMPDIR': root / 'tmp', 'SQLITE_TMPDIR': root / 'tmp',
                       'WANDB_DIR': root / 'runs/wandb', 'WANDB_CACHE_DIR': root / 'caches/wandb'}.items():
        os.environ[key] = str(value)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import tempfile
    tempfile.tempdir = str(root / 'tmp')
    probe = root / 'tmp' / f'access-{os.getpid()}'
    probe.write_text('compute-node storage probe')
    probe.unlink()
    return root


class Preparation:
    def __init__(self, root, prep_id, config):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', prep_id):
            raise ValueError('Use a simple, unique preparation ID without path separators')
        self.root = root
        self.path = root / 'preprocessed' / prep_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.config = config
        versions = {name: importlib.metadata.version(name) for name in
                    ('datasets', 'numpy', 'pyarrow', 'transformers', 'tokenizers',
                     'huggingface-hub', 'fasttext', 'zstandard', 'torch')}
        expected_versions = dict(line.strip().split('==') for line in
                                 (PROJECT / 'requirements-data.txt').read_text().splitlines()
                                 if line.strip() and not line.startswith('#'))
        for name, expected in expected_versions.items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f'Use the pinned environment: {name}=={expected}')
        distribution_versions = sorted((d.metadata['Name'], d.version)
                                       for d in importlib.metadata.distributions())
        self.identity = {'config': config, 'versions': versions, 'python': sys.version,
                         'all_distribution_versions': distribution_versions,
                         'source_sha256': {p: sha256(PROJECT / p) for p in
                            ('dataset.py', 'third_party/dclm_fasttext.py', 'requirements-data.txt')}}
        self.key = fingerprint(self.identity)
        existing = self.path / 'configuration.json'
        if existing.exists():
            if read_json(existing)['fingerprint'] != self.key:
                raise ValueError('Configuration/code/environment changed; use a new preparation ID')
        else:
            atomic_json(existing, {'fingerprint': self.key, **self.identity,
                                   'code_commit': subprocess.check_output(
                                       ['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                                   'git_status': subprocess.check_output(
                                       ['git', 'status', '--short'], cwd=PROJECT, text=True)})

    def directory(self, stage):
        path = self.path / stage
        path.mkdir(exist_ok=True)
        return path

    def manifest(self, stage, full=False):
        path = self.path / stage / 'manifest.json'
        value = read_json(path)
        if value['fingerprint'] != self.key or value['stage'] != stage:
            raise ValueError(f'Mixed preparation artifacts: {path}')
        if stage != 'download':
            parent = STAGES[STAGES.index(stage) - 1]
            if value['parent_sha256'] != sha256(self.path / parent / 'manifest.json'):
                raise ValueError(f'Parent manifest changed: {stage}')
        for item in value['artifacts']:
            check_artifact(item, full)
        return value

    def done(self, stage):
        if (self.path / stage / 'manifest.json').exists():
            self.manifest(stage)
            print(f'{stage}: already complete', flush=True)
            return True
        return False

    def finish(self, stage, paths, details):
        parent = STAGES[STAGES.index(stage) - 1] if stage != 'download' else None
        atomic_json(self.directory(stage) / 'manifest.json',
                    {'stage': stage, 'fingerprint': self.key,
                     'parent_sha256': sha256(self.path / parent / 'manifest.json') if parent else None,
                     'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     'artifacts': [artifact(Path(p)) for p in paths], 'details': details})

    def event(self, stage, status):
        with (self.path / 'jobs.jsonl').open('a') as stream:
            stream.write(json.dumps({'stage': stage, 'status': status, 'timestamp': time.time(),
                                     'job_id': os.environ.get('SLURM_JOB_ID'),
                                     'node': os.environ.get('SLURMD_NODENAME'),
                                     'fingerprint': self.key}) + '\n')


def database(path):
    db = sqlite3.connect(path, timeout=120, uri=True)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    return db


def progress(db, key, default=None):
    row = db.execute('SELECT value FROM progress WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_progress(db, key, value):
    db.execute('INSERT OR REPLACE INTO progress VALUES (?, ?)', (key, json.dumps(value)))


def readonly(path):
    return sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)


def iter_arrow(files, cursor=(0, 0)):
    import pyarrow as pa
    first_shard, first_row = cursor
    for shard in range(first_shard, len(files)):
        with pa.memory_map(files[shard]['path'], 'r') as source:
            table = pa.ipc.open_stream(source).read_all()
            offset = first_row if shard == first_shard else 0
            # Only document shards are converted to Python rows in bounded batches.
            for batch in table.slice(offset).to_batches(max_chunksize=32):
                for row in batch.to_pylist():
                    offset += 1
                    after = (shard + 1, 0) if offset == len(table) else (shard, offset)
                    yield row, after


def document_schema():
    import pyarrow as pa
    return pa.schema([('row_index', pa.uint64()), ('shuffle_position', pa.uint64()),
                      ('id', pa.string()), ('source', pa.string()),
                      ('text_sha256', pa.binary(32)), ('tokens', pa.list_(pa.uint32())),
                      ('evaluation_slot', pa.int32())])


def sequence_schema(length):
    import pyarrow as pa
    span = pa.struct([('row_index', pa.uint64()), ('document_start', pa.uint32()),
                      ('sequence_start', pa.uint32()), ('length', pa.uint32())])
    return pa.schema([('input_ids', pa.list_(pa.uint32(), length)),
                      ('attention_mask', pa.list_(pa.uint8(), length)),
                      ('labels', pa.list_(pa.int32(), length)),
                      ('document_spans', pa.list_(span)), ('evaluation_slot', pa.int32())])


def write_arrow(path, rows, schema):
    import pyarrow as pa
    temporary = path.with_suffix('.partial')
    table = pa.Table.from_pylist(rows, schema=schema)
    with pa.OSFile(str(temporary), 'wb') as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(table)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {**artifact(path), 'rows': len(rows)}


def state_for(path, initial):
    return read_json(path) if path.exists() else initial


def load_raw(prep):
    import datasets
    details = prep.manifest('split')['details']
    pieces = [datasets.Dataset.from_file(item['path']) for item in details['raw_cache_files']]
    raw = datasets.concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
    if len(raw) != details['row_count']:
        raise ValueError('Canonical dataset row count changed')
    return raw


def stage_download(prep):
    from datasets import load_dataset_builder
    from huggingface_hub import hf_hub_download
    cfg = prep.config
    builder = load_dataset_builder(cfg['dataset_id'], revision=cfg['dataset_revision'])
    splits = builder.config.data_files
    if list(splits) != ['train']:
        raise ValueError(f'Unexpected dataset configuration: {list(splits)}')
    names = [repo_filename(url) for url in splits['train']]
    if len(names) != len(set(names)) or not names:
        raise ValueError('Duplicate or missing canonical shard names')
    directory = prep.directory('download')
    state_path = directory / 'state.json'
    state = state_for(state_path, {'canonical_files': names, 'files': []})
    if state['canonical_files'] != names:
        raise ValueError('Canonical dataset file order changed')
    for item in state['files']:
        check_artifact(item, True)
    for name in names[len(state['files']):]:
        stop_if_requested()
        path = Path(hf_hub_download(cfg['dataset_id'], name, repo_type='dataset',
                                   revision=cfg['dataset_revision']))
        state['files'].append({**artifact(path), 'repo_filename': name,
                               'source': source_for_path(name)})
        atomic_json(state_path, state)
        print(f'download: {len(state["files"])}/{len(names)}', flush=True)
    prep.finish('download', [item['path'] for item in state['files']],
                {'canonical_files': names, 'files': state['files'],
                 'builder': type(builder).__name__, 'builder_config': builder.config.name})


def raw_json_documents(path):
    import zstandard as zstd
    import io
    with open(path, 'rb') as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed, read_across_frames=True) as reader:
            with io.TextIOWrapper(reader, encoding='utf-8') as lines:
                for line in lines:
                    if line.strip():
                        yield json.loads(line)


def stage_split(prep):
    import numpy as np
    from datasets import load_dataset, load_dataset_builder
    cfg = prep.config
    download = prep.manifest('download')['details']
    builder = load_dataset_builder(cfg['dataset_id'], revision=cfg['dataset_revision'])
    if [repo_filename(url) for url in builder.config.data_files['train']] != download['canonical_files']:
        raise ValueError('Native builder canonical order differs from download manifest')
    raw = load_dataset(cfg['dataset_id'], revision=cfg['dataset_revision'], split='train')
    n = len(raw)
    if n <= cfg['evaluation_documents'] or raw._indices is not None:
        raise ValueError('Expected complete, unindexed train document split')
    directory = prep.directory('split')
    state_path = directory / 'state.json'
    state = state_for(state_path, {'row_count': n, 'ranges': []})
    if state['row_count'] != n:
        raise ValueError('Dataset row count changed on resume')
    # Authenticate source lineage against the actual native HF ordering, not an
    # assumed filename sort or the incomplete web viewer's row count.
    start = state['ranges'][-1]['end'] if state['ranges'] else 0
    for item in download['files'][len(state['ranges']):]:
        stop_if_requested()
        count = 0
        chunk = []
        for doc in raw_json_documents(item['path']):
            if not isinstance(doc.get('text'), str):
                raise ValueError(f'Non-string text in {item["repo_filename"]}')
            chunk.append(doc)
            if len(chunk) == cfg['document_batch_size']:
                compare_raw_chunk(raw, start + count, chunk)
                count += len(chunk)
                chunk = []
                stop_if_requested()
        if chunk:
            compare_raw_chunk(raw, start + count, chunk)
            count += len(chunk)
        state['ranges'].append({'start': start, 'end': start + count,
                                'repo_filename': item['repo_filename'], 'source': item['source']})
        start += count
        atomic_json(state_path, state)
        print(f'split lineage: {len(state["ranges"])}/{len(download["files"])} shards', flush=True)
    if start != n:
        raise ValueError('Source lineage does not cover the complete native train split')
    stop_if_requested()
    shuffled = raw.shuffle(seed=cfg['shuffle_seed'])  # canonical lab semantics
    if shuffled._indices is None:
        raise ValueError('Native shuffle did not produce document indices')
    permutation_path, inverse_path = directory / 'permutation.npy', directory / 'inverse.npy'
    permutation = np.lib.format.open_memmap(permutation_path, mode='w+', dtype='uint64', shape=(n,))
    inverse = np.lib.format.open_memmap(inverse_path, mode='w+', dtype='uint64', shape=(n,))
    for offset in range(0, n, 1_000_000):
        stop_if_requested()
        size = min(1_000_000, n - offset)
        indices = np.asarray(shuffled._indices.column(0).slice(offset, size), dtype=np.uint64)
        permutation[offset:offset + size] = indices
        inverse[indices] = np.arange(offset, offset + size, dtype=np.uint64)
    permutation.flush()
    inverse.flush()
    eval_path, subset_path = directory / 'evaluation.npy', directory / 'evaluation_subset.npy'
    np.save(eval_path, permutation[-cfg['evaluation_documents']:])
    np.save(subset_path, permutation[-cfg['evaluation_documents']:
                                     -cfg['evaluation_documents'] + cfg['evaluation_subset_documents']])
    cache_files = [artifact(Path(item['filename'])) for item in raw.cache_files]
    if not cache_files:
        raise ValueError('Expected disk-backed native Arrow cache files')
    prep.finish('split', [permutation_path, inverse_path, eval_path, subset_path] +
                [item['path'] for item in cache_files],
                {'row_count': n, 'train_document_count': n - cfg['evaluation_documents'],
                 'ranges': state['ranges'], 'raw_cache_files': cache_files,
                 'raw_fingerprint': raw._fingerprint, 'features': raw.features.to_dict(),
                 'shuffle': 'datasets.Dataset.shuffle(seed=42), then index selection',
                 'seed': cfg['shuffle_seed'], 'evaluation_subset': 'first 10000 of reserved ordered documents'})


def compare_raw_chunk(raw, offset, docs):
    actual = raw[offset:offset + len(docs)]
    if len(actual['text']) != len(docs):
        raise ValueError('Raw shard exceeds native dataset length')
    for i, doc in enumerate(docs):
        if actual['text'][i] != doc['text'] or actual['id'][i] != doc.get('id'):
            raise ValueError(f'Canonical ordering mismatch at original row {offset + i}')


def source_at(ranges, ends, index):
    position = bisect.bisect_right(ends, index)
    if position >= len(ranges) or index < ranges[position]['start']:
        raise ValueError(f'No source lineage for row {index}')
    return ranges[position]['source']


def stage_clean(prep):
    import numpy as np
    raw = load_raw(prep)
    cfg = prep.config
    details = prep.manifest('split')['details']
    ranges = details['ranges']
    ends = [item['end'] for item in ranges]
    inverse = np.load(prep.path / 'split/inverse.npy', mmap_mode='r')
    train_count = details['train_document_count']
    path = prep.directory('clean') / 'documents.sqlite'
    with database(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS winners '
                   '(digest BLOB PRIMARY KEY, row_index INTEGER NOT NULL, '
                   'position INTEGER NOT NULL, source TEXT NOT NULL) WITHOUT ROWID')
        db.execute('CREATE TABLE IF NOT EXISTS empty (row_index INTEGER PRIMARY KEY)')
        start = progress(db, 'next_row', 0)
        for batch in raw.select(range(start, len(raw))).iter(batch_size=cfg['document_batch_size']):
            stop_if_requested()
            with db:
                for i, text in enumerate(batch['text']):
                    row = start + i
                    pos = int(inverse[row])
                    if pos >= train_count:
                        continue  # evaluation never enters cleaning or hashing
                    if not isinstance(text, str):
                        raise ValueError(f'Non-string training text at row {row}')
                    if not text.strip():
                        db.execute('INSERT OR IGNORE INTO empty VALUES (?)', (row,))
                        continue
                    digest = hashlib.sha256(text.encode('utf-8')).digest()
                    source = source_at(ranges, ends, row)
                    db.execute('INSERT INTO winners VALUES (?, ?, ?, ?) '
                               'ON CONFLICT(digest) DO UPDATE SET row_index=excluded.row_index, '
                               'position=excluded.position, source=excluded.source '
                               'WHERE excluded.position < winners.position', (digest, row, pos, source))
                start += len(batch['text'])
                set_progress(db, 'next_row', start)
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS winner_rows ON winners(row_index)')
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS winner_positions ON winners(position)')
        db.execute('CREATE INDEX IF NOT EXISTS winner_sources ON winners(source, row_index)')
        empty_count = db.execute('SELECT COUNT(*) FROM empty').fetchone()[0]
        eligible = db.execute('SELECT COUNT(*) FROM winners').fetchone()[0]
        source_counts = dict(db.execute('SELECT source, COUNT(*) FROM winners GROUP BY source'))
    prep.finish('clean', [path], {'eligible_documents': eligible, 'empty_documents': empty_count,
                 'duplicate_documents': train_count - empty_count - eligible, 'source_counts': source_counts,
                 'reason_encoding': 'empty table: empty; winners: eligible; remaining train rows: exact duplicate',
                 'deduplication': 'SHA256 of unchanged UTF-8 text, retain lowest shuffle position; training only'})


_WORKER_MODEL = None


def initialize_scorer(model_path):
    global _WORKER_MODEL
    import fasttext
    _WORKER_MODEL = fasttext.load_model(model_path)
    if set(_WORKER_MODEL.get_labels()) != {'__label__hq', '__label__cc'}:
        raise ValueError('Unexpected DCLM classifier label space')


def score_texts(texts):
    result = [float(classify_fasttext_hq_prob(_WORKER_MODEL, text)) for text in texts]
    if any(not math.isfinite(value) or not -1e-4 <= value <= 1.0001 for value in result):
        raise ValueError('Invalid classifier scores')
    return result  # Preserve official probabilities, including fastText epsilon.


def stage_score(prep):
    from huggingface_hub import hf_hub_download
    import multiprocessing
    cfg = prep.config
    raw = load_raw(prep)
    model_path = hf_hub_download(cfg['classifier_id'], cfg['classifier_filename'],
                                 revision=cfg['classifier_revision'])
    path = prep.directory('score') / 'scores.sqlite'
    with contextlib.closing(readonly(prep.path / 'clean/documents.sqlite')) as clean, database(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS scores '
                   '(row_index INTEGER PRIMARY KEY, position INTEGER NOT NULL, score REAL NOT NULL)')
        last = progress(db, 'last_row', -1)
        query = clean.execute("SELECT row_index, position FROM winners WHERE source='common_crawl' "
                              'AND row_index>? ORDER BY row_index', (last,))
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=cfg['score_workers'], mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_scorer, initargs=(model_path,)) as pool:
            while True:
                stop_if_requested()
                rows = query.fetchmany(cfg['document_batch_size'] * cfg['score_workers'])
                if not rows:
                    break
                texts = raw.select([row[0] for row in rows])['text']
                size = cfg['document_batch_size']
                batches = [texts[i:i + size] for i in range(0, len(texts), size)]
                scores = [score for batch in pool.map(score_texts, batches) for score in batch]
                if len(scores) != len(rows):
                    raise ValueError('Missing fastText scores')
                with db:
                    db.executemany('INSERT INTO scores VALUES (?, ?, ?)',
                                   [(row, pos, value) for (row, pos), value in zip(rows, scores)])
                    set_progress(db, 'last_row', rows[-1][0])
        db.execute('CREATE INDEX IF NOT EXISTS score_ranking ON scores(score DESC, position ASC)')
        count = db.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
        expected = clean.execute("SELECT COUNT(*) FROM winners WHERE source='common_crawl'").fetchone()[0]
        if count != expected:
            raise ValueError('Not every eligible Common Crawl document was scored')
    prep.finish('score', [path, model_path], {'scored_documents': count,
                'workers': cfg['score_workers'], 'input': 'full text; official strip and splitlines join',
                'classifier_sha256': sha256(Path(model_path)), 'device': 'CPU',
                'score': 'official DCLM high-quality probability; no rounding or clamping'})


def build_candidates(prep):
    """Materialize global top-k once, before any Llama tokenization."""
    directory = prep.directory('select')
    path = directory / 'candidates.sqlite'
    keep = top_count(prep.manifest('score')['details']['scored_documents'],
                     prep.config['common_crawl_keep_fraction'])
    with database(path) as db:
        if progress(db, 'complete', False):
            return path
        db.execute('ATTACH DATABASE ? AS clean', (f'file:{prep.path / "clean/documents.sqlite"}?mode=ro',))
        db.execute('ATTACH DATABASE ? AS scored', (f'file:{prep.path / "score/scores.sqlite"}?mode=ro',))
        # SQLite disk sorting avoids holding the full score population in RAM.
        db.set_progress_handler(lambda: int(STOP or (int(os.environ.get('PREP_STOP_EPOCH', '0')) > 0
                and time.time() >= int(os.environ['PREP_STOP_EPOCH']))), 100000)
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS candidates '
                       '(row_index INTEGER PRIMARY KEY, position INTEGER UNIQUE NOT NULL, source TEXT NOT NULL)')
            db.execute('DELETE FROM candidates')
            db.execute("INSERT INTO candidates SELECT row_index, position, source FROM clean.winners "
                       "WHERE source != 'common_crawl'")
            db.execute("INSERT INTO candidates SELECT row_index, position, 'common_crawl' "
                       'FROM scored.scores ORDER BY score DESC, position ASC LIMIT ?', (keep,))
            set_progress(db, 'common_crawl_kept', keep)
            boundary = db.execute('SELECT score, position FROM scored.scores '
                                  'ORDER BY score DESC, position ASC LIMIT 1 OFFSET ?', (keep - 1,)).fetchone() if keep else None
            set_progress(db, 'boundary', boundary)
            set_progress(db, 'complete', True)
    return path


def load_tokenizer(prep):
    from transformers import AutoTokenizer
    cfg = prep.config
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_id'], revision=cfg['tokenizer_revision'])
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Llama tokenizer must supply BOS and EOS IDs')
    # Padding uses EOS but is distinguished by attention_mask and ignored labels.
    return tokenizer


def make_document(raw_doc, row, position, source, tokenizer, evaluation_slot=-1):
    text = raw_doc['text']
    if not isinstance(text, str):
        raise ValueError(f'Non-string evaluation/training document: {row}')
    tokens = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    return {'row_index': row, 'shuffle_position': position, 'id': raw_doc['id'],
            'source': source, 'text_sha256': hashlib.sha256(text.encode('utf-8')).digest(),
            'tokens': tokens, 'evaluation_slot': evaluation_slot}


def stage_select(prep):
    import numpy as np
    cfg = prep.config
    candidates_path = build_candidates(prep)
    raw = load_raw(prep)
    tokenizer = load_tokenizer(prep)
    directory = prep.directory('select')
    state_path = directory / 'state.json'
    state = state_for(state_path, {'train_files': [], 'eval_files': [], 'last_position': -1,
                                 'train_tokens': 0, 'train_documents': 0, 'budget_skipped_documents': 0,
                                 'next_evaluation_slot': 0, 'evaluation_tokens': 0})
    for item in state['train_files'] + state['eval_files']:
        check_artifact(item, True)
    schema = document_schema()
    with contextlib.closing(readonly(candidates_path)) as db:
        cursor = db.execute('SELECT row_index, position, source FROM candidates WHERE position>? '
                            'ORDER BY position', (state['last_position'],))
        rows_buffer, tokens_buffer = [], 0
        while cfg['max_training_tokens'] - state['train_tokens'] >= 2:
            stop_if_requested()
            rows = cursor.fetchmany(32)
            if not rows:
                break
            documents = raw.select([row[0] for row in rows])
            for (row, position, source), original in zip(rows, documents):
                doc = make_document(original, row, position, source, tokenizer)
                count = len(doc['tokens'])
                if fits_document(count, cfg['max_training_tokens'] - state['train_tokens']):
                    rows_buffer.append(doc)
                    tokens_buffer += count
                    state['train_tokens'] += count
                    state['train_documents'] += 1
                else:
                    state['budget_skipped_documents'] += 1
                state['last_position'] = position
                if tokens_buffer >= cfg['document_shard_tokens']:
                    path = directory / f'train-documents-{len(state["train_files"]):06d}.arrow'
                    state['train_files'].append(write_arrow(path, rows_buffer, schema))
                    rows_buffer, tokens_buffer = [], 0
                    atomic_json(state_path, state)
                if state['train_tokens'] == cfg['max_training_tokens']:
                    break
            # Never checkpoint a cursor beyond documents still in memory.
            if not rows_buffer:
                atomic_json(state_path, state)
        if rows_buffer:
            path = directory / f'train-documents-{len(state["train_files"]):06d}.arrow'
            state['train_files'].append(write_arrow(path, rows_buffer, schema))
        atomic_json(state_path, state)
    if state['train_documents'] == 0:
        raise ValueError('No complete training document fits the token budget')
    split = prep.manifest('split')['details']
    ranges = split['ranges']
    ends = [item['end'] for item in ranges]
    evaluation = np.load(prep.path / 'split/evaluation.npy', mmap_mode='r')
    for offset in range(state['next_evaluation_slot'], len(evaluation), 32):
        stop_if_requested()
        indices = [int(row) for row in evaluation[offset:offset + 32]]
        rows_buffer = []
        for i, (row, original) in enumerate(zip(indices, raw.select(indices))):
            slot = offset + i
            doc = make_document(original, row, split['train_document_count'] + slot,
                                source_at(ranges, ends, row), tokenizer, slot)
            rows_buffer.append(doc)
            state['evaluation_tokens'] += len(doc['tokens'])
        path = directory / f'eval-documents-{len(state["eval_files"]):06d}.arrow'
        state['eval_files'].append(write_arrow(path, rows_buffer, schema))
        state['next_evaluation_slot'] = offset + len(indices)
        atomic_json(state_path, state)
    with contextlib.closing(readonly(candidates_path)) as db:
        selection = {'candidate_documents': db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
                     'common_crawl_kept': progress(db, 'common_crawl_kept'),
                     'score_boundary': progress(db, 'boundary')}
    tokenizer_dir = directory / 'tokenizer'
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer_files = sorted(path for path in tokenizer_dir.iterdir() if path.is_file())
    prep.finish('select', [candidates_path] + tokenizer_files +
                [item['path'] for item in state['train_files'] + state['eval_files']],
                {**state, **selection, 'bos_token_id': tokenizer.bos_token_id,
                 'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.eos_token_id,
                 'tokenizer_directory': str(tokenizer_dir),
                 'tokenization': 'add_special_tokens=False; prepend one BOS; append one EOS; no text truncation',
                 'selection_order': 'original canonical shuffle position, skip whole documents exceeding remaining tokens'})


def stage_pack(prep):
    cfg = prep.config
    selected = prep.manifest('select')['details']
    directory = prep.directory('pack')
    state_path = directory / 'state.json'
    state = state_for(state_path, {'train_files': [], 'eval_files': [], 'train_cursor': [0, 0],
                                 'eval_cursor': [0, 0], 'carry': {}, 'train_complete': False,
                                 'training_tokens': 0, 'evaluation_input_tokens': 0,
                                 'evaluation_prediction_tokens': 0, 'subset_prediction_tokens': 0})
    for item in state['train_files'] + state['eval_files']:
        check_artifact(item, True)
    length, pad_id = cfg['sequence_length'], selected['pad_token_id']
    schema = sequence_schema(length)
    if not state['train_complete']:
        packer = TrainingPacker(length, pad_id, state['carry'])
        buffer = []
        for doc, cursor in iter_arrow(selected['train_files'], state['train_cursor']):
            stop_if_requested()
            buffer.extend(packer.push(doc['row_index'], doc['tokens']))
            state['train_cursor'] = cursor
            if len(buffer) >= cfg['sequence_shard_rows']:
                state['training_tokens'] += sum(sum(row['attention_mask']) for row in buffer)
                path = directory / f'train-{len(state["train_files"]):06d}.arrow'
                state['train_files'].append(write_arrow(path, buffer, schema))
                buffer = []
                state['carry'] = packer.state()
                atomic_json(state_path, state)
        buffer.extend(packer.finish())
        if buffer:
            state['training_tokens'] += sum(sum(row['attention_mask']) for row in buffer)
            path = directory / f'train-{len(state["train_files"]):06d}.arrow'
            state['train_files'].append(write_arrow(path, buffer, schema))
        state['carry'] = packer.state()
        state['train_complete'] = True
        atomic_json(state_path, state)
    if state['training_tokens'] != selected['train_tokens']:
        raise ValueError('Packing changed the non-padding training token count')
    buffer = []
    for doc, cursor in iter_arrow(selected['eval_files'], state['eval_cursor']):
        stop_if_requested()
        sequences = list(pack_evaluation_document(doc, length, pad_id))
        buffer.extend(sequences)
        state['evaluation_input_tokens'] += sum(sum(row['attention_mask']) for row in sequences)
        predictions = sum(sum(label != -100 for label in row['labels'][1:]) for row in sequences)
        state['evaluation_prediction_tokens'] += predictions
        if doc['evaluation_slot'] < cfg['evaluation_subset_documents']:
            state['subset_prediction_tokens'] += predictions
        state['eval_cursor'] = cursor
        if len(buffer) >= cfg['sequence_shard_rows']:
            path = directory / f'eval-{len(state["eval_files"]):06d}.arrow'
            state['eval_files'].append(write_arrow(path, buffer, schema))
            buffer = []
            atomic_json(state_path, state)
    if buffer:
        path = directory / f'eval-{len(state["eval_files"]):06d}.arrow'
        state['eval_files'].append(write_arrow(path, buffer, schema))
    atomic_json(state_path, state)
    prep.finish('pack', [item['path'] for item in state['train_files'] + state['eval_files']],
                {**state, 'sequence_length': length, 'pad_token_id': pad_id,
                 'training_sequence_count': sum(item['rows'] for item in state['train_files']),
                 'evaluation_sequence_count': sum(item['rows'] for item in state['eval_files']),
                 'training_attention': 'ordinary causal attention across EOS-separated training documents',
                 'training_loss': 'ignore structural BOS, first input position of each sequence, and padding',
                 'evaluation_loss': 'independent documents, stride 8191; each within-document target once'})


def expected_sequences(documents, length, pad_id, evaluation=False):
    if evaluation:
        for doc, _ in documents:
            yield from pack_evaluation_document(doc, length, pad_id)
    else:
        packer = TrainingPacker(length, pad_id)
        for doc, _ in documents:
            yield from packer.push(doc['row_index'], doc['tokens'])
        yield from packer.finish()


def validate_sequences(actual, expected, length):
    import itertools
    total, targets, count = 0, 0, 0
    for pair in itertools.zip_longest(actual, expected):
        item, reference = pair
        if item is None or reference is None:
            raise ValueError('Packed sequence count mismatch')
        row = item[0]
        if row != reference:
            raise ValueError(f'Packed tokens/masks/provenance differ at sequence {count}')
        if any(len(row[key]) != length for key in ('input_ids', 'attention_mask', 'labels')):
            raise ValueError('Context length mismatch')
        total += sum(row['attention_mask'])
        targets += sum(label != -100 for label in row['labels'][1:])
        count += 1
        if count % 128 == 0:
            stop_if_requested()
    return {'input_tokens': total, 'prediction_tokens': targets, 'sequences': count}


def stage_verify(prep):
    import numpy as np
    cfg = prep.config
    # Verification completion is a seal over immutable manifests and artifacts.
    for stage in STAGES[:-1]:
        stop_if_requested()
        prep.manifest(stage, full=True)
    split = prep.manifest('split')['details']
    n = split['row_count']
    raw = load_raw(prep)
    canonical = raw.shuffle(seed=cfg['shuffle_seed'], load_from_cache_file=False,
                            indices_cache_file_name=str(prep.directory('verify') / 'canonical-shuffle.arrow'))
    perm = np.load(prep.path / 'split/permutation.npy', mmap_mode='r')
    inverse = np.load(prep.path / 'split/inverse.npy', mmap_mode='r')
    if perm.shape != (n,) or inverse.shape != (n,):
        raise ValueError('Split index length mismatch')
    # Inverse equality proves uniqueness and full coverage without global sorting.
    for offset in range(0, n, 1_000_000):
        stop_if_requested()
        values = perm[offset:offset + 1_000_000]
        reference = np.asarray(canonical._indices.column(0).slice(offset, len(values)), dtype=np.uint64)
        if not np.array_equal(values, reference):
            raise ValueError('Split differs from the canonical native seed-42 shuffle')
        if np.any(values >= n) or not np.array_equal(
                inverse[values], np.arange(offset, offset + len(values), dtype=np.uint64)):
            raise ValueError('Split permutation is not a bijection')
    evaluation = np.load(prep.path / 'split/evaluation.npy')
    subset = np.load(prep.path / 'split/evaluation_subset.npy')
    if not np.array_equal(evaluation, perm[-cfg['evaluation_documents']:]):
        raise ValueError('Evaluation is not the last 50000 shuffled documents')
    if not np.array_equal(subset, evaluation[:cfg['evaluation_subset_documents']]):
        raise ValueError('Evaluation subset membership changed')
    selected = prep.manifest('select')['details']
    packed = prep.manifest('pack')['details']
    raw = load_raw(prep)
    ranges, ends = split['ranges'], [item['end'] for item in split['ranges']]
    seen = np.zeros(n, dtype=np.bool_)
    tokens, docs, previous = 0, 0, -1
    with contextlib.closing(readonly(prep.path / 'select/candidates.sqlite')) as candidates:
        for doc, _ in iter_arrow(selected['train_files']):
            stop_if_requested()
            row, pos = doc['row_index'], doc['shuffle_position']
            if not 0 <= row < n or seen[row] or int(inverse[row]) != pos or pos >= split['train_document_count']:
                raise ValueError('Duplicate, evaluation, or incorrect training document membership')
            if pos <= previous or not candidates.execute('SELECT 1 FROM candidates WHERE row_index=? '
                                                         'AND position=? AND source=?',
                                                         (row, pos, doc['source'])).fetchone():
                raise ValueError('Training document order or eligibility changed')
            original = raw[row]
            if (original['id'] != doc['id'] or hashlib.sha256(original['text'].encode()).digest() != doc['text_sha256']
                    or source_at(ranges, ends, row) != doc['source']):
                raise ValueError('Training document provenance mismatch')
            if doc['evaluation_slot'] != -1 or doc['tokens'][0] != selected['bos_token_id'] or doc['tokens'][-1] != selected['eos_token_id']:
                raise ValueError('Training document token boundaries changed')
            seen[row] = True
            previous = pos
            tokens += len(doc['tokens'])
            docs += 1
    if tokens != selected['train_tokens'] or tokens > cfg['max_training_tokens'] or docs != selected['train_documents']:
        raise ValueError('Training artifact exceeds budget or counters disagree')
    eval_tokens, eval_docs = 0, 0
    for doc, _ in iter_arrow(selected['eval_files']):
        stop_if_requested()
        row = doc['row_index']
        if eval_docs >= len(evaluation) or row != int(evaluation[eval_docs]) or seen[row] or doc['evaluation_slot'] != eval_docs:
            raise ValueError('Evaluation membership, order, or disjointness mismatch')
        original = raw[row]
        if (original['id'] != doc['id'] or hashlib.sha256(original['text'].encode()).digest() != doc['text_sha256']
                or int(inverse[row]) != doc['shuffle_position'] or source_at(ranges, ends, row) != doc['source']):
            raise ValueError('Evaluation text/provenance changed')
        if doc['tokens'][0] != selected['bos_token_id'] or doc['tokens'][-1] != selected['eos_token_id']:
            raise ValueError('Evaluation document token boundaries changed')
        seen[row] = True
        eval_tokens += len(doc['tokens'])
        eval_docs += 1
    if eval_docs != cfg['evaluation_documents'] or eval_tokens != selected['evaluation_tokens']:
        raise ValueError('Incomplete evaluation documents')
    length, pad_id = cfg['sequence_length'], selected['pad_token_id']
    train = validate_sequences(iter_arrow(packed['train_files']),
              expected_sequences(iter_arrow(selected['train_files']), length, pad_id), length)
    evaluation_stats = validate_sequences(iter_arrow(packed['eval_files']),
              expected_sequences(iter_arrow(selected['eval_files']), length, pad_id, True), length)
    if train['input_tokens'] != tokens or train['input_tokens'] != packed['training_tokens']:
        raise ValueError('Packed training tokens disagree')
    if (evaluation_stats['prediction_tokens'] != eval_tokens - eval_docs or
            evaluation_stats['prediction_tokens'] != packed['evaluation_prediction_tokens']):
        raise ValueError('Evaluation windows lost or duplicated prediction targets')
    clean_details = prep.manifest('clean')['details']
    with contextlib.closing(readonly(prep.path / 'score/scores.sqlite')) as scores, \
            contextlib.closing(readonly(prep.path / 'clean/documents.sqlite')) as clean, \
            contextlib.closing(readonly(prep.path / 'select/candidates.sqlite')) as candidates:
        score_count = scores.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
        expected_cc = clean_details['source_counts'].get('common_crawl', 0)
        if score_count != expected_cc:
            raise ValueError('Missing eligible Common Crawl scores')
        ranked = scores.execute('SELECT row_index, position FROM scores ORDER BY score DESC, position ASC LIMIT ?',
                                (top_count(score_count, cfg['common_crawl_keep_fraction']),))
        kept = candidates.execute("SELECT COUNT(*) FROM candidates WHERE source='common_crawl'").fetchone()[0]
        expected_kept = top_count(score_count, cfg['common_crawl_keep_fraction'])
        if kept != expected_kept:
            raise ValueError('Common Crawl keep count is not exactly floor(N/10)')
        for row, pos in ranked:
            if not candidates.execute("SELECT 1 FROM candidates WHERE row_index=? AND position=? "
                                      "AND source='common_crawl'", (row, pos)).fetchone():
                raise ValueError('Global top-10% membership mismatch')
        others = candidates.execute("SELECT COUNT(*) FROM candidates WHERE source!='common_crawl'").fetchone()[0]
        if others != clean_details['eligible_documents'] - expected_cc:
            raise ValueError('Non-Common-Crawl candidate count changed')
        for row, pos, source in clean.execute("SELECT row_index, position, source FROM winners WHERE source!='common_crawl'"):
            if not candidates.execute('SELECT 1 FROM candidates WHERE row_index=? AND position=? AND source=?',
                                      (row, pos, source)).fetchone():
                raise ValueError('Non-Common-Crawl winner missing')
    report = prep.directory('verify') / 'report.json'
    atomic_json(report, {'fingerprint': prep.key, 'pack_manifest_sha256': sha256(prep.path / 'pack/manifest.json'),
                        'training': train, 'evaluation': evaluation_stats,
                        'evaluation_documents': eval_docs, 'subset_documents': len(subset),
                        'training_limit': cfg['max_training_tokens'],
                        'scope': 'artifact checksums, split bijection, provenance, global quality ranking, '
                                 'budget, full sequence reconstruction and loss masks'})
    prep.finish('verify', [report, prep.path / 'verify/canonical-shuffle.arrow'], read_json(report))
    print(json.dumps(read_json(report), indent=2), flush=True)


def load_prepared(preparation_path, split='train', verify_checksums=True):
    """Load only sealed artifacts. Run on compute nodes; preserves provenance columns.

    split is train, evaluation, or evaluation_subset. Trainer must still enforce
    the processed-token budget across repeated inputs, all ranks and resumes.
    """
    require_compute()
    path = Path(preparation_path).resolve()
    setup_storage(path.parents[1])
    import datasets
    configuration = read_json(path / 'configuration.json')
    verified = read_json(path / 'verify/manifest.json')
    packed_manifest = path / 'pack/manifest.json'
    packed = read_json(packed_manifest)
    if (verified['fingerprint'] != configuration['fingerprint'] or
            packed['fingerprint'] != configuration['fingerprint'] or
            verified['details']['pack_manifest_sha256'] != sha256(packed_manifest)):
        raise ValueError('Artifact is not sealed by verification')
    for item in verified['artifacts']:
        check_artifact(item, True)
    if split not in ('train', 'evaluation', 'evaluation_subset'):
        raise ValueError('split must be train, evaluation, or evaluation_subset')
    files = packed['details']['train_files' if split == 'train' else 'eval_files']
    for item in files:
        check_artifact(item, verify_checksums)
    pieces = [datasets.Dataset.from_file(item['path']) for item in files]
    if not pieces:
        raise ValueError('No packed sequences')
    result = datasets.concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
    if split == 'evaluation_subset':
        result = result.filter(lambda slot: slot < configuration['config']['evaluation_subset_documents'],
                               input_columns=['evaluation_slot'])
    return result


def smoke(prep):
    """Small real-library/model check, deliberately runnable only in Slurm."""
    import numpy as np
    import datasets
    from huggingface_hub import hf_hub_download
    from multiprocessing import get_context
    cfg = prep.config
    fixture = datasets.Dataset.from_dict({'id': [str(i) for i in range(100)],
                                         'text': [f'Document {i}' for i in range(100)]})
    expected = np.random.default_rng(42).permutation(100)
    actual = np.asarray(fixture.shuffle(seed=42)._indices.column(0))
    if not np.array_equal(actual, expected):
        raise ValueError('Pinned native Dataset shuffle semantics changed')
    model_path = hf_hub_download(cfg['classifier_id'], cfg['classifier_filename'], revision=cfg['classifier_revision'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
              initializer=initialize_scorer, initargs=(model_path,)) as pool:
        scores = list(pool.map(score_texts, [['A lesson on photosynthesis.\nPlants convert light into energy.']]))[0]
    tokenizer = load_tokenizer(prep)
    doc = make_document({'id': 'smoke', 'text': 'A lesson on photosynthesis.'}, 0, 0, 'common_crawl', tokenizer)
    directory = prep.path / 'smoke'
    directory.mkdir(exist_ok=True)
    doc_file = write_arrow(directory / 'documents.arrow', [doc], document_schema())
    length = cfg['sequence_length']
    rows = list(expected_sequences(iter_arrow([doc_file]), length, tokenizer.eos_token_id))
    seq_file = write_arrow(directory / 'sequences.arrow', rows, sequence_schema(length))
    restored = datasets.Dataset.from_file(seq_file['path'])
    if restored[0]['input_ids'] != rows[0]['input_ids']:
        raise ValueError('Arrow/Hugging Face round trip failed')
    report = {'native_shuffle': True, 'official_classifier_scores': scores, 'tokenizer': True,
              'packed_sequence_length': len(restored[0]['input_ids']), 'training_weights_loaded': False}
    atomic_json(directory / 'report.json', report)
    print(json.dumps(report, indent=2))


def validate_config(config):
    required = read_json(PROJECT / 'configs/dataset.json')
    if set(config) != set(required):
        raise ValueError('Configuration keys must match configs/dataset.json')
    fixed = ('shuffle_seed', 'evaluation_documents', 'evaluation_subset_documents', 'sequence_length',
             'common_crawl_keep_fraction', 'dataset_id', 'classifier_id', 'classifier_filename',
             'dclm_source_commit', 'tokenizer_id')
    for name in fixed:
        if config[name] != required[name]:
            raise ValueError(f'This recipe requires {name}={required[name]}')
    if not isinstance(config['max_training_tokens'], int) or not 2 <= config['max_training_tokens'] <= 6_000_000_000:
        raise ValueError('max_training_tokens must be between 2 and 6000000000')
    for name in ('score_workers', 'document_batch_size', 'document_shard_tokens', 'sequence_shard_rows'):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('dataset_revision', 'classifier_revision', 'tokenizer_revision'):
        if not re.fullmatch(r'[0-9a-f]{40}', config[name]):
            raise ValueError(f'{name} must be an immutable commit SHA')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=(*STAGES, 'smoke'))
    parser.add_argument('--prep-id', required=True)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/dataset.json')
    args = parser.parse_args()
    require_compute()
    config = read_json(args.config)
    validate_config(config)
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use the documented Python 3.11 data environment')
    root = setup_storage(args.root)
    prep = Preparation(root, args.prep_id, config)
    def handler(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGUSR1, handler)
    signal.signal(signal.SIGTERM, handler)
    with (prep.path / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage != 'smoke' and prep.done(args.stage):
            return 0
        if args.stage in STAGES and args.stage != 'download':
            prep.manifest(STAGES[STAGES.index(args.stage) - 1])
        prep.event(args.stage, 'started')
        print(json.dumps({'configuration': config, 'fingerprint': prep.key,
                          'job_id': os.environ.get('SLURM_JOB_ID')}, indent=2), flush=True)
        try:
            (smoke if args.stage == 'smoke' else globals()['stage_' + args.stage])(prep)
        except StopRequested as exc:
            prep.event(args.stage, 'checkpointed')
            print(str(exc), file=sys.stderr)
            return 75
        except sqlite3.OperationalError as exc:
            if 'interrupted' in str(exc).lower():
                prep.event(args.stage, 'checkpointed')
                return 75
            prep.event(args.stage, 'failed')
            raise
        except Exception:
            prep.event(args.stage, 'failed')
            raise
        prep.event(args.stage, 'completed')
        return 0


if __name__ == '__main__':
    sys.exit(main())
