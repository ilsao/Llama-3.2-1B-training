#!/usr/bin/env python3
"""Retire redundant, manifest-owned corpus files only after complete verification."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as pipeline
from dataset import (DEFAULT_ROOT, PROJECT, atomic_json, check_artifact, load_prepared,
                     read_json, require_compute, sha256)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value) or value in {'.', '..'}:
        raise ValueError('Invalid preparation ID')
    return value


def contained_file(root, path):
    root, path = root.resolve(), Path(path).absolute()
    if not path.is_relative_to(root) or not path.resolve().is_relative_to(root):
        raise ValueError('Refuse file outside project storage: ' + str(path))
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Refuse symlink in deletion path: ' + str(current))
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError('Refuse non-file or multiply linked artifact: ' + str(path))
    return path


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str):
            yield Path(value['path']).resolve()
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def cleanup_plan(root, prep_id, retire_download_source=False):
    root = root.resolve()
    prep = root / 'preprocessed' / identifier(prep_id)
    if prep.is_symlink() or not prep.resolve().is_relative_to(root):
        raise ValueError('Preparation path escapes project storage')
    cfg = read_json(prep / 'configuration.json')
    verified = read_json(prep / 'verify/manifest.json')
    packed = read_json(prep / 'pack/manifest.json')
    split = read_json(prep / 'split/manifest.json')
    selected = read_json(prep / 'select/manifest.json')
    download = read_json(prep / 'download/manifest.json')
    manifests = [verified, packed, split, selected, download]
    if (any(m['fingerprint'] != cfg['fingerprint'] for m in manifests)
            or verified['parent_sha256'] != sha256(prep / 'pack/manifest.json')
            or verified['details']['pack_manifest_sha256'] != sha256(prep / 'pack/manifest.json')
            or verified['details'].get('evaluation_raw_documents') != cfg['config']['evaluation_documents']
            or split['details'].get('evaluation_raw_documents') != cfg['config']['evaluation_documents']
            or not verified['details'].get('evaluation_raw_files')):
        raise ValueError('Require complete, sealed, independent evaluation originals')
    for stage, expected in verified['details'].get('manifest_sha256', {}).items():
        if stage not in ('download', 'split', 'clean', 'score', 'select', 'pack'):
            raise ValueError('Unknown sealed manifest stage')
        if sha256(prep / stage / 'manifest.json') != expected:
            raise ValueError('Historical manifest changed after verification: ' + stage)
    eligible = {}
    def add(items, reason):
        for item in items:
            path = contained_file(root, item['path'])
            record = {**item, 'path': str(path), 'reason': reason}
            previous = eligible.get(path)
            if previous and (previous['sha256'], previous['size']) != (record['sha256'], record['size']):
                raise ValueError('Conflicting ownership metadata')
            eligible[path] = record
    add(split['details']['raw_cache_files'], 'native raw corpus cache; eval independently archived')
    add(selected['details']['train_files'], 'training document tokens duplicated by verified packed sequences')
    retiring = [prep]
    reuse = download['details'].get('reused_download')
    if reuse and retire_download_source:
        source = root / 'preprocessed' / identifier(reuse['preparation_id'])
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('Download source escapes project storage')
        if sha256(source / 'download/manifest.json') != reuse['manifest_sha256']:
            raise ValueError('Reused source download changed')
        if any((source / stage).exists() for stage in ('split', 'clean', 'score', 'select', 'pack', 'verify')):
            raise ValueError('Source preparation has downstream work; cannot retire it automatically')
        retiring.append(source)
    if not reuse or retire_download_source:
        add(download['artifacts'], 'downloaded corpus duplicated by sealed retained artifacts')
    protected = set()
    for manifest in (verified, packed):
        protected.update(references(manifest['artifacts']))
    protected.update(references(split['details']['evaluation_raw_files']))
    protected.update(references(selected['details']['eval_files']))
    tokenizer = Path(selected['details']['tokenizer_directory']).resolve()
    protected.update(Path(i['path']).resolve() for i in selected['artifacts']
                     if Path(i['path']).resolve().parent == tokenizer)
    # Other preparations, including interrupted stages, keep all their referenced files.
    # Already retired preparations protect only non-released assets.
    for other in (root / 'preprocessed').iterdir():
        if not other.is_dir() or other in retiring:
            continue
        if other.is_symlink():
            raise ValueError('Unknown symlink preparation; refuse cleanup')
        released = set()
        if (other / 'retention.json').exists():
            released = {Path(i['path']).resolve() for i in read_json(other / 'retention.json')['released']}
        for stage in ('download', 'split', 'clean', 'score', 'select', 'pack', 'verify'):
            for name in ('manifest.json', 'state.json'):
                path = other / stage / name
                if path.exists():
                    protected.update(set(references(read_json(path))) - released)
    candidates = [item for path, item in sorted(eligible.items()) if path.resolve() not in protected]
    return {'preparation_id': prep_id, 'fingerprint': cfg['fingerprint'],
            'verification_sha256': sha256(prep / 'verify/manifest.json'),
            'retiring_preparations': [p.name for p in retiring], 'candidates': candidates,
            'bytes': sum(i['size'] for i in candidates),
            'protected_candidate_count': len(eligible) - len(candidates)}


def assert_no_other_project_jobs():
    result = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', 'asciibase64'),
                             '-o', '%A|%Z'], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        job, cwd = line.split('|', 1)
        if job.strip() == os.environ.get('SLURM_JOB_ID'):
            continue
        path = Path(cwd.strip())
        if path.is_relative_to(PROJECT) or path.is_relative_to(DEFAULT_ROOT):
            raise ValueError('Project job still active/queued; defer cleanup: ' + job)


def apply_plan(root, plan):
    prep = root / 'preprocessed' / plan['preparation_id']
    # A receipt precedes deletion. Interrupted retirement is resumable and loaders
    # continue to require all retained immutable assets; original manifests stay intact.
    for item in plan['candidates']:
        pipeline.stop_if_requested()
        path = contained_file(root, item['path'])
        if path.exists():
            check_artifact(item, True)
    for name in plan['retiring_preparations']:
        destination = root / 'preprocessed' / name / 'retention.json'
        atomic_json(destination, {'state': 'retiring', 'released': plan['candidates'],
                                 'verification_sha256': plan['verification_sha256'],
                                 'retained_preparation': plan['preparation_id'], 'timestamp': time.time()})
    removed = 0
    for item in plan['candidates']:
        pipeline.stop_if_requested()
        path = contained_file(root, item['path'])
        if path.exists():
            path.unlink()
            removed += item['size']
    # Remove only snapshot links to blobs this operation has released.
    # Never follow or remove links outside this project's cache.
    released = {Path(i['path']) for i in plan['candidates']}
    hub = root / 'caches/hub/datasets--allenai--dolma3_mix-150B-1025'
    if hub.exists() and not hub.is_symlink():
        for link in (hub / 'snapshots').rglob('*'):
            if link.is_symlink() and link.resolve() in released and link.resolve().is_relative_to(root.resolve()):
                link.unlink()
    for name in plan['retiring_preparations']:
        destination = root / 'preprocessed' / name / 'retention.json'
        receipt = read_json(destination)
        atomic_json(destination, {**receipt, 'state': 'retired', 'completed_utc': time.time()})
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prep-id', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retire-download-source', action='store_true')
    args = parser.parse_args()
    require_compute()
    def stop_handler(signum, frame):
        pipeline.STOP = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    pipeline.stop_if_requested()
    root = DEFAULT_ROOT.resolve()
    if root != DEFAULT_ROOT:
        raise ValueError('Project storage must not be a symlink')
    assert_no_other_project_jobs()
    with contextlib.ExitStack() as stack:
        global_lock = stack.enter_context((root / '.retention.lock').open('a'))
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Lock every existing preparation while collecting references.
        for other in sorted((root / 'preprocessed').iterdir()):
            if other.is_dir():
                if other.is_symlink():
                    raise ValueError('Refuse symlink preparation')
                lock = stack.enter_context((other / '.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = cleanup_plan(root, args.prep_id, args.retire_download_source)
        # Retained eval originals, packed data and verification seal must be intact.
        load_prepared(root / 'preprocessed' / args.prep_id, 'train', verify_checksums=True)
        load_prepared(root / 'preprocessed' / args.prep_id, 'evaluation', verify_checksums=True)
        selected = read_json(root / 'preprocessed' / args.prep_id / 'select/manifest.json')
        tokenizer = Path(selected['details']['tokenizer_directory']).resolve()
        for item in selected['artifacts']:
            if Path(item['path']).resolve().parent == tokenizer or item in selected['details']['eval_files']:
                check_artifact(item, True)
        report = root / 'preprocessed' / args.prep_id / 'cleanup-plan.json'
        atomic_json(report, plan)
        if args.apply:
            removed = apply_plan(root, plan)
            load_prepared(root / 'preprocessed' / args.prep_id, 'train', verify_checksums=True)
            load_prepared(root / 'preprocessed' / args.prep_id, 'evaluation', verify_checksums=True)
            atomic_json(report.with_name('cleanup-result.json'), {**plan, 'removed_bytes_this_attempt': removed,
                                                                'offline_reload_passed': True})
        print(json.dumps({'candidates': len(plan['candidates']), 'bytes': plan['bytes'],
                          'protected_candidates': plan['protected_candidate_count'],
                          'applied': args.apply, 'report': str(report)}))

if __name__ == '__main__':
    try:
        main()
    except pipeline.StopRequested as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(75)
