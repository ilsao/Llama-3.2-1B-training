"""Bounded, one-producer/one-consumer per-rank benchmark spool (stdlib only)."""
import json
import os
import time
from pathlib import Path


def atomic_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as f:
        json.dump(obj, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def wait_for(path, deadline, abort_root):
    path, abort_root = Path(path), Path(abort_root)
    started = time.perf_counter()
    while not path.exists():
        if time.time() >= deadline:
            raise TimeoutError(str(path))
        if list(abort_root.glob('abort-*.json')):
            raise RuntimeError('Peer aborted; preserve unconsumed spool')
        time.sleep(.005)
    return time.perf_counter() - started


def wait_for_roles(root, roles, deadline):
    """All ranks wait for own rank0 publication as well as the external peer."""
    root = Path(root)
    for role in roles:
        wait_for(root/role/'ready.json', deadline, root)
    return {role: json.loads((root/role/'ready.json').read_text()) for role in roles}


class Spool:
    def __init__(self, root, rank, segment, payload_bytes, deadline, slots=2):
        self.root, self.rank, self.segment = Path(root), rank, segment
        self.directory = self.root / 'spool' / f'{segment}-rank{rank}'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.payload_bytes, self.deadline, self.slots = payload_bytes, deadline, slots
        if slots * payload_bytes * 8 > 1_000_000_000_000:
            raise ValueError('Spool capacity exceeds 1 TB')

    def path(self, key, extension):
        return self.directory / f'{key:06d}.{extension}'

    def publish(self, key, byteview, identity):
        if len(byteview) != self.payload_bytes:
            raise ValueError('Wrong payload length')
        backpressure = 0.
        if key >= self.slots:
            backpressure = wait_for(self.path(key-self.slots, 'ack'), self.deadline, self.root)
        target = self.path(key, 'bin')
        if target.exists() or self.path(key, 'ready').exists():
            raise ValueError('Never overwrite published data')
        started = time.perf_counter()
        with self.path(key, 'partial').open('xb', buffering=0) as f:
            remaining = byteview
            while remaining:
                written = f.write(remaining)
                if not written:
                    raise IOError('Short write')
                remaining = remaining[written:]
            os.fsync(f.fileno())
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.replace(self.path(key, 'partial'), target)
        atomic_json(self.path(key, 'ready'), dict(identity, payload_bytes=self.payload_bytes))
        return dict(write_seconds=time.perf_counter()-started, backpressure_seconds=backpressure)

    def read_into(self, key, byteview, expected):
        wait = wait_for(self.path(key, 'ready'), self.deadline, self.root)
        started = time.perf_counter()
        identity = json.loads(self.path(key, 'ready').read_text())
        if identity != dict(expected, payload_bytes=self.payload_bytes):
            raise ValueError('Teacher/student identity mismatch')
        if len(byteview) != self.payload_bytes or self.path(key, 'bin').stat().st_size != self.payload_bytes:
            raise ValueError('Truncated payload')
        with self.path(key, 'bin').open('rb', buffering=0) as f:
            remaining = byteview
            while remaining:
                n = f.readinto(remaining)
                if not n:
                    raise IOError('Short read')
                remaining = remaining[n:]
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return dict(ready_wait_seconds=wait, read_seconds=time.perf_counter()-started)

    def acknowledge(self, key):
        # Called only after this student's backward completed. Restrict deletion
        # to this run/segment/rank's consumed payload and READY metadata.
        self.path(key, 'bin').unlink()
        self.path(key, 'ready').unlink()
        atomic_json(self.path(key, 'ack'), dict(consumed=True, key=key))
