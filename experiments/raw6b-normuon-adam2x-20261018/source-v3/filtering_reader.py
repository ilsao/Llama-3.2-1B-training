"""Bounded canonical raw-shard access; core preparation identity stays immutable.

No Arrow/datasets imports at module load. Build/use only in allocated jobs.
"""
import bisect
from collections import OrderedDict
import gc
from pathlib import Path
import dataset as pipeline


class Selection:
    def __init__(self, reader, indices):
        self.reader, self.indices = reader, list(indices)
        if len(self.indices) > 2048:
            raise ValueError('Raw selection exceeds bounded document batch')

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.reader.fetch(self.indices))

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.reader.fetch(self.indices, [key])]
        if isinstance(key, slice):
            rows = self.reader.fetch(self.indices[key])
            return {k: [row[k] for row in rows] for k in (rows[0] if rows else [])}
        return self.reader[self.indices[key]]


def build_index(prep):
    """One shard/batch at a time. Restartable, immutable source order and row counts."""
    import pyarrow as pa
    details = prep.manifest('split')['details']
    files = details['raw_cache_files']
    directory = prep.path / 'filtering-access'; directory.mkdir(exist_ok=True)
    identity = dict(preparation_fingerprint=prep.key,
                    split_sha256=pipeline.sha256(prep.path / 'split/manifest.json'))
    target = directory / 'index.json'; pending = directory / 'index-progress.json'
    saved = pipeline.read_json(target if target.exists() else pending) if target.exists() or pending.exists() else dict(**identity, files=[])
    if any(saved.get(k) != v for k, v in identity.items()):
        raise ValueError('Raw index belongs to different canonical split')
    if len(saved['files']) > len(files):
        raise ValueError('Raw index has extra shards')
    for old, original in zip(saved['files'], files):
        if {k: old[k] for k in original} != original:
            raise ValueError('Raw indexed prefix changed')
        pipeline.check_artifact(original)
    if target.exists():
        if len(saved['files']) != len(files) or sum(f['rows'] for f in saved['files']) != details['row_count']:
            raise ValueError('Raw index is incomplete')
        return saved
    for number in range(len(saved['files']), len(files)):
        pipeline.stop_if_requested(); item = files[number]; pipeline.check_artifact(item)
        rows = 0
        with pa.memory_map(item['path'], 'r') as source:
            with pa.ipc.open_stream(source) as stream:
                for batch in stream:
                    pipeline.stop_if_requested(); rows += batch.num_rows
                    del batch
        saved['files'].append({**item, 'rows': rows})
        if number % 32 == 31 or number == len(files)-1:
            pipeline.atomic_json(pending, saved)
            print(f'raw-index: {number+1}/{len(files)} shards', flush=True)
        gc.collect()
    if sum(f['rows'] for f in saved['files']) != details['row_count']:
        raise ValueError('Indexed row count differs from the complete canonical dataset')
    saved['row_count'] = details['row_count']
    pipeline.atomic_json(target, saved)
    return saved


class IndexedRaw:
    def __init__(self, index, opener=None, max_open=2):
        if max_open < 1:
            raise ValueError('Need at least one open shard')
        self.files = index['files']; self.ends = []
        total = 0
        for item in self.files:
            if item['rows'] < 0:
                raise ValueError('Negative row count')
            total += item['rows']; self.ends.append(total)
        if total != index['row_count']:
            raise ValueError('Raw indexed total mismatch')
        self.total, self.max_open = total, max_open
        self.opener = opener; self.cache = OrderedDict(); self.maximum_open = 0

    def __len__(self):
        return self.total

    def locate(self, row):
        if not isinstance(row, int) or not 0 <= row < self.total:
            raise IndexError(row)
        shard = bisect.bisect_right(self.ends, row)
        return shard, row - (self.ends[shard-1] if shard else 0)

    def open(self, shard):
        if shard not in self.cache:
            # Evict before opening to preserve the simultaneous-mapping bound.
            if len(self.cache) >= self.max_open:
                self.cache.popitem(last=False); gc.collect()
            if self.opener is None:
                import datasets
                pipeline.check_artifact(self.files[shard])
                value = datasets.Dataset.from_file(self.files[shard]['path'])
            else:
                value = self.opener(shard)
            if len(value) != self.files[shard]['rows']:
                raise ValueError('Canonical shard row count changed')
            self.cache[shard] = value
            self.maximum_open = max(self.maximum_open, len(self.cache))
        self.cache.move_to_end(shard)
        return self.cache[shard]

    def fetch(self, indices, columns=None):
        if len(indices) > 2048:
            raise ValueError('Raw selection exceeds bounded document batch')
        groups = OrderedDict(); output = [None] * len(indices)
        for slot, row in enumerate(indices):
            shard, local = self.locate(row)
            groups.setdefault(shard, []).append((slot, local))
        for shard, group in groups.items():
            pipeline.stop_if_requested()
            ds = self.open(shard)
            # Project before selecting; never concatenate corpus-wide Arrow tables.
            if columns is not None:
                ds = ds.select_columns(columns)
            selected = ds.select([local for _, local in group])
            for (slot, _), record in zip(group, selected):
                output[slot] = record
            del ds, selected
        if any(row is None for row in output):
            raise ValueError('Missing indexed document')
        return output

    def select(self, indices):
        return Selection(self, indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
            return self.select(indices)[:]
        if key < 0:
            key += len(self)
        return self.fetch([key])[0]

    def close(self):
        self.cache.clear(); gc.collect()
