#!/usr/bin/env python3
"""Slurm-only, read-only compressed shard integrity audit with a durable cursor."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset
from preparation_parallel import configure_processes, configure_threads, ordered_executor, worker_count
from training_state import identifier


def audit_one(task):
    entry, deadline = task
    result = {'path': entry['path'], 'repo_filename': entry['repo_filename'],
              'expected_sha256': entry['sha256'], 'expected_size': entry['size'],
              'decoded_bytes': 0, 'status': 'interrupted'}
    path = Path(entry['path'])
    try:
        result['actual_size'] = path.stat().st_size
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                if time.time() >= deadline:
                    return result
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        result['actual_sha256'] = digest.hexdigest()
        result['manifest_hash_matches'] = (result['actual_sha256'] == entry['sha256']
                                            and result['actual_size'] == entry['size'])
        import zstandard as zstd
        with path.open('rb') as stream:
            try:
                with zstd.ZstdDecompressor().stream_reader(stream, read_across_frames=True) as reader:
                    while True:
                        if time.time() >= deadline:
                            return result
                        result['compressed_position_before_last_read'] = stream.tell()
                        block = reader.read(8 * 1024 * 1024)
                        if not block:
                            break
                        result['decoded_bytes'] += len(block)
            except Exception:
                result['compressed_position_at_error'] = stream.tell() if not stream.closed else None
                raise
        result['status'] = 'ok' if result['manifest_hash_matches'] else 'hash_mismatch'
    except Exception as error:
        result.update(status='decode_or_io_error', error_type=type(error).__name__, error=str(error))
    return result


def validate_cursor(state, files):
    records = state['records']
    if state['next_file'] != len(records) or len(records) > len(files):
        raise ValueError('Audit cursor does not match committed ordered results')
    for record, source in zip(records, files):
        if (record['repo_filename'] != source['repo_filename'] or record['path'] != source['path']
                or record['expected_sha256'] != source['sha256']
                or record['expected_size'] != source['size']
                or record['status'] == 'interrupted'):
            raise ValueError('Audit committed result order/identity mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-prep', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    identifier(args.run_id)
    identifier(args.source_prep)
    dataset.require_compute()
    configure_threads()
    methods = configure_processes()
    root = dataset.setup_storage(args.root)
    source = root / 'preprocessed' / args.source_prep
    target = root / 'runs/cache-audit' / args.run_id
    target.mkdir(parents=True, exist_ok=True)
    with (source / '.lock').open('r') as source_lock, (target / '.lock').open('a') as run_lock:
        fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = source / 'download/manifest.json'
        manifest = dataset.read_json(manifest_path)
        cfg = dataset.read_json(source / 'configuration.json')
        original_identity = {k: v for k, v in cfg.items() if k not in {'fingerprint', 'code_commit', 'git_status'}}
        if (dataset.fingerprint(original_identity) != cfg['fingerprint']
                or manifest['fingerprint'] != cfg['fingerprint'] or manifest['stage'] != 'download'
                or manifest['parent_sha256'] is not None or (source / 'retention.json').exists()):
            raise ValueError('Require an authenticated, unretired completed download')
        files = manifest['details']['files']
        if ([item['repo_filename'] for item in files] != manifest['details']['canonical_files']
                or len(files) != len(manifest['artifacts'])
                or any(not Path(item['path']).resolve().is_relative_to(root) for item in files)):
            raise ValueError('Download membership/order/path mismatch')
        for item, recorded in zip(files, manifest['artifacts']):
            if any(item[k] != recorded[k] for k in ('path', 'size', 'sha256')):
                raise ValueError('Download artifact record mismatch')
        identity = {'source_preparation': args.source_prep,
                    'manifest_sha256': dataset.sha256(manifest_path),
                    'source_fingerprint': cfg['fingerprint'],
                    'code_sha256': dataset.sha256(Path(__file__)),
                    'zstandard_version': importlib.metadata.version('zstandard'),
                    'process_start_methods': methods}
        state_path = target / 'state.json'
        state = dataset.state_for(state_path, {'identity': identity, 'next_file': 0, 'records': [], 'jobs': []})
        if state['identity'] != identity:
            raise ValueError('Audit source/code/environment changed; use a new run ID')
        validate_cursor(state, files)
        job_id = os.environ['SLURM_JOB_ID']
        if job_id not in state['jobs']:
            state['jobs'].append(job_id)
        dataset.atomic_json(state_path, state)
        deadline = min(time.time() + 540, int(os.environ['PREP_STOP_EPOCH']))
        signal.signal(signal.SIGTERM, lambda *_: setattr(dataset, 'STOP', True))
        signal.signal(signal.SIGUSR1, lambda *_: setattr(dataset, 'STOP', True))
        tasks = ((entry, deadline) for entry in files[state['next_file']:])
        workers = worker_count({'preprocessing_workers': 7})
        print(json.dumps({'identity': identity, 'job_id': job_id, 'workers': workers,
                          'allocated_cpus': int(os.environ['SLURM_CPUS_PER_TASK']),
                          'next_file': state['next_file'], 'total_files': len(files)}), flush=True)
        try:
            with ordered_executor(workers, check=dataset.stop_if_requested) as ordered:
                for record in ordered(audit_one, tasks):
                    if record['status'] == 'interrupted':
                        break
                    state['records'].append(record)
                    state['next_file'] += 1
                    if record['status'] != 'ok' or state['next_file'] % 50 == 0:
                        dataset.atomic_json(state_path, state)
                        print(json.dumps({'checked': state['next_file'], 'total': len(files),
                                          'failure': record if record['status'] != 'ok' else None}), flush=True)
        except dataset.StopRequested:
            pass
        finally:
            dataset.atomic_json(state_path, state)
        failures = [record for record in state['records'] if record['status'] != 'ok']
        complete = state['next_file'] == len(files)
        result = {'identity': identity, 'job_id': job_id, 'jobs': state['jobs'],
                  'scan_complete': complete, 'checked_files': state['next_file'],
                  'total_files': len(files), 'failures': failures,
                  'passed': complete and not failures, 'production_split_validated': False}
        dataset.atomic_json(target / 'result.json', result)
        print(json.dumps(result), flush=True)
        return 0 if complete else 75


if __name__ == '__main__':
    sys.exit(main())
