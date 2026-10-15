"""Seekable IPC record-batch access, preserving immutable canonical raw identity.

No native libraries are imported until a compute-only reader is opened.
"""
import bisect
from collections import OrderedDict
import hashlib
import time
from pathlib import Path
import dataset as pipeline
from filtering_reader import IndexedRaw


def validate_sidecar(value, original, identity):
    expected = dict(format_version=1, original=original, identity=identity)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Batch index identity mismatch')
    content = {k: v for k, v in value.items() if k != 'content_sha256'}
    if value.get('content_sha256') != pipeline.fingerprint(content):
        raise ValueError('Batch index checksum mismatch')
    row, end = 0, 0
    for item in value['batches']:
        if (item['start_row'] != row or item['rows'] < 0 or
                item['offset'] < end or item['end_offset'] <= item['offset'] or
                item['end_offset'] > original['size']):
            raise ValueError('Invalid batch index range')
        row += item['rows']; end = item['end_offset']
    if row != original['rows']:
        raise ValueError('Batch index row count mismatch')
    return value


class BatchRaw(IndexedRaw):
    def __init__(self, index, directory, identity):
        pipeline.require_compute()
        super().__init__(index)
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.stats = dict(shard_opens=0, batch_reads=0, indexes_built=0,
                          index_seconds=0., fetch_seconds=0., fetch_documents=0,
                          maximum_cached_shards=0)
        self.indexes = {}

    def _index(self, shard, source):
        import pyarrow as pa
        original = self.files[shard]
        target = self.directory / f'shard-{shard:06d}.json'
        if target.exists():
            return validate_sidecar(pipeline.read_json(target), original, self.identity)
        started = time.monotonic(); source.seek(0)
        schema = pa.ipc.read_schema(pa.ipc.read_message(source))
        batches, row = [], 0
        while source.tell() < original['size']:
            pipeline.stop_if_requested(); offset = source.tell()
            try:
                message = pa.ipc.read_message(source)
            except EOFError:
                # A stream terminator must end exactly at the recorded file size.
                if source.tell() != original['size']:
                    raise ValueError('Trailing bytes after IPC terminator')
                break
            if message.type != 'record batch':
                raise ValueError('Unsupported IPC message (including dictionaries)')
            batch = pa.ipc.read_record_batch(message, schema)
            batches.append(dict(offset=offset, end_offset=source.tell(),
                                start_row=row, rows=batch.num_rows))
            row += batch.num_rows
            del batch, message
        if row != original['rows']:
            raise ValueError('IPC stream row count differs from canonical shard')
        value = dict(format_version=1, original=original, identity=self.identity,
                     schema_sha256=hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
                     batches=batches)
        value['content_sha256'] = pipeline.fingerprint(value)
        validate_sidecar(value, original, self.identity)
        pipeline.atomic_json(target, value)
        self.stats['indexes_built'] += 1
        self.stats['index_seconds'] += time.monotonic() - started
        return value

    def open(self, shard):
        import pyarrow as pa
        if shard in self.cache:
            self.cache.move_to_end(shard); return self.cache[shard]
        if len(self.cache) >= self.max_open:
            _, old = self.cache.popitem(last=False)
            old['source'].close()
        pipeline.check_artifact(self.files[shard])
        source = pa.memory_map(self.files[shard]['path'], 'r')
        try:
            sidecar = self.indexes.get(shard)
            if sidecar is None:
                sidecar = self._index(shard, source); self.indexes[shard] = sidecar
            source.seek(0); schema = pa.ipc.read_schema(pa.ipc.read_message(source))
            if hashlib.sha256(schema.serialize().to_pybytes()).hexdigest() != sidecar['schema_sha256']:
                raise ValueError('Raw schema changed')
            value = dict(source=source, schema=schema, batches=sidecar['batches'],
                         ends=[b['start_row'] + b['rows'] for b in sidecar['batches']])
        except BaseException:
            source.close(); raise
        self.cache[shard] = value; self.stats['shard_opens'] += 1
        self.maximum_open = max(self.maximum_open, len(self.cache))
        self.stats['maximum_cached_shards'] = self.maximum_open
        return value

    def fetch(self, indices, columns=None):
        import pyarrow as pa
        if len(indices) > 2048:
            raise ValueError('Raw selection exceeds bounded document batch')
        started = time.monotonic(); groups = OrderedDict(); output = [None] * len(indices)
        for slot, row in enumerate(indices):
            shard, local = self.locate(row)
            groups.setdefault(shard, []).append((slot, local))
        for shard, group in groups.items():
            pipeline.stop_if_requested(); value = self.open(shard); batch_groups = OrderedDict()
            for slot, local in group:
                number = bisect.bisect_right(value['ends'], local)
                batch_groups.setdefault(number, []).append((slot, local))
            for number, requested in batch_groups.items():
                pipeline.stop_if_requested(); descriptor = value['batches'][number]
                source = value['source']; source.seek(descriptor['offset'])
                message = pa.ipc.read_message(source)
                if message.type != 'record batch' or source.tell() != descriptor['end_offset']:
                    raise ValueError('Indexed IPC message changed')
                batch = pa.ipc.read_record_batch(message, value['schema'])
                if batch.num_rows != descriptor['rows']:
                    raise ValueError('Indexed batch row count changed')
                if columns is not None:
                    batch = batch.select(columns)
                selected = batch.take(pa.array([local - descriptor['start_row'] for _, local in requested], type=pa.int64()))
                for (slot, _), record in zip(requested, selected.to_pylist()):
                    output[slot] = record
                del selected, batch, message
                self.stats['batch_reads'] += 1
        if any(row is None for row in output):
            raise ValueError('Missing indexed document')
        self.stats['fetch_documents'] += len(indices)
        self.stats['fetch_seconds'] += time.monotonic() - started
        return output

    def shuffle(self, **kwargs):
        # Native HF shuffle depends on row count and seed, not document content.
        # A zero-buffer null column avoids corpus-wide table concatenation while
        # stage_verify still compares every generated index with the saved split.
        import datasets
        import pyarrow as pa
        return datasets.Dataset.from_dict({'__canonical_row__': pa.nulls(len(self))}).shuffle(**kwargs)

    def close(self):
        while self.cache:
            _, value = self.cache.popitem(last=False); value['source'].close()
