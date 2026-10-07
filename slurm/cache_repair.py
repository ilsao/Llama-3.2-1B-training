#!/usr/bin/env python3
"""Validate one pinned replacement in an isolated cache; preserve historical bytes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset
from preparation_parallel import configure_processes, configure_threads
from training_state import identifier


def require_upstream_match(item, upstream):
    if (item['sha256'] != upstream['sha256'] or item['size'] != upstream['size']
            or not isinstance(upstream['sha256'], str) or len(upstream['sha256']) != 64):
        raise ValueError('Replacement differs from pinned upstream LFS hash/size')


def require_decoder_agreement(first, second):
    if (not first['passed'] or not second['passed'] or first['decoded_bytes'] <= 0
            or first['decoded_bytes'] != second['decoded_bytes']
            or first['decoded_sha256'] != second['decoded_sha256']):
        raise ValueError('Full decoded byte hashes differ or a decoder failed')


def decode(path, backend, deadline):
    digest, total = hashlib.sha256(), 0
    try:
        if backend == 'python-zstandard':
            import zstandard as zstd
            stream = zstd.ZstdDecompressor().stream_reader(path.open('rb'), read_across_frames=True)
        else:
            import pyarrow as pa
            pa.set_cpu_count(1)
            pa.set_io_thread_count(1)
            stream = pa.input_stream(str(path), compression='zstd')
        with stream:
            while True:
                if time.time() >= deadline:
                    raise TimeoutError('Decoder deadline reached')
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                total += len(block)
                digest.update(block)
        return {'backend': backend, 'passed': True, 'decoded_bytes': total,
                'decoded_sha256': digest.hexdigest()}
    except Exception as error:
        return {'backend': backend, 'passed': False, 'decoded_bytes': total,
                'error_type': type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-prep', required=True)
    parser.add_argument('--audit-run', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for value in (args.run_id, args.source_prep, args.audit_run):
        identifier(value)
    dataset.require_compute()
    configure_threads()
    methods = configure_processes()
    root = dataset.setup_storage(args.root)
    source = root / 'preprocessed' / args.source_prep
    target = root / 'runs/cache-repair' / args.run_id
    target.mkdir(parents=True, exist_ok=True)
    deadline = min(time.time() + 540, int(os.environ['PREP_STOP_EPOCH']))
    with (source / '.lock').open('r') as source_lock, (target / '.lock').open('a') as run_lock:
        fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit_path = root / 'runs/cache-audit' / args.audit_run / 'result.json'
        audit = dataset.read_json(audit_path)
        manifest_path = source / 'download/manifest.json'
        manifest = dataset.read_json(manifest_path)
        cfg = dataset.read_json(source / 'configuration.json')
        cfg_identity = {k: v for k, v in cfg.items() if k not in {'fingerprint', 'code_commit', 'git_status'}}
        if (dataset.fingerprint(cfg_identity) != cfg['fingerprint']
                or manifest['fingerprint'] != cfg['fingerprint'] or manifest['stage'] != 'download'
                or audit['identity']['source_preparation'] != args.source_prep
                or audit['identity']['source_fingerprint'] != cfg['fingerprint']
                or audit['identity']['manifest_sha256'] != dataset.sha256(manifest_path)
                or not audit['scan_complete'] or audit['checked_files'] != len(manifest['details']['files'])
                or len(audit['failures']) != 1 or (source / 'retention.json').exists()):
            raise ValueError('Require authenticated complete audit with exactly one failed shard')
        failure = audit['failures'][0]
        entries = [item for item in manifest['details']['files']
                   if item['repo_filename'] == failure['repo_filename']]
        if len(entries) != 1:
            raise ValueError('Failed file is not a unique canonical source member')
        original = entries[0]
        if (original['path'] != failure['path'] or original['sha256'] != failure['actual_sha256']
                or not Path(original['path']).resolve().is_relative_to(root)):
            raise ValueError('Original audited artifact changed')
        dataset.check_artifact(original, True)
        identity = {'source_preparation': args.source_prep, 'source_fingerprint': cfg['fingerprint'],
                    'source_manifest_sha256': dataset.sha256(manifest_path),
                    'audit_result_sha256': dataset.sha256(audit_path),
                    'code_sha256': dataset.sha256(Path(__file__)),
                    'dataset_id': cfg['config']['dataset_id'],
                    'dataset_revision': cfg['config']['dataset_revision'],
                    'process_start_methods': methods}
        state_path = target / 'state.json'
        state = dataset.state_for(state_path, {'identity': identity, 'jobs': []})
        if state['identity'] != identity:
            raise ValueError('Replacement source/code identity changed; use a new run ID')
        job = os.environ['SLURM_JOB_ID']
        if job not in state['jobs']:
            state['jobs'].append(job)
        state.update(phase='query_pinned_upstream', original=original)
        dataset.atomic_json(state_path, state)
        print(json.dumps({'job_id': job, 'repo_filename': original['repo_filename'],
                          'phase': state['phase']}), flush=True)
        try:
            from huggingface_hub import HfApi, hf_hub_download
            infos = HfApi().get_paths_info(identity['dataset_id'], [original['repo_filename']],
                                         repo_type='dataset', revision=identity['dataset_revision'])
            if len(infos) != 1 or infos[0].path != original['repo_filename'] or not infos[0].lfs:
                raise ValueError('Pinned source LFS metadata unavailable')
            upstream = {'sha256': infos[0].lfs.sha256, 'size': infos[0].size}
            state.update(upstream=upstream, phase='download_isolated_copy')
            dataset.atomic_json(state_path, state)
            print(json.dumps({'upstream': upstream, 'original_matches_upstream':
                              original['sha256'] == upstream['sha256'], 'phase': state['phase']}), flush=True)
            if time.time() >= deadline:
                raise TimeoutError('Replacement deadline before download')
            new_path = Path(hf_hub_download(identity['dataset_id'], original['repo_filename'],
                            repo_type='dataset', revision=identity['dataset_revision'],
                            cache_dir=str(target / 'hub'), force_download=True, etag_timeout=30)).resolve()
            if not new_path.is_relative_to(target.resolve()) or new_path == Path(original['path']).resolve():
                raise ValueError('Replacement must remain in the isolated run cache')
            replacement = {**dataset.artifact(new_path), 'repo_filename': original['repo_filename'],
                           'source': original['source']}
            require_upstream_match(replacement, upstream)
            state.update(replacement=replacement, phase='verify_decoders')
            dataset.atomic_json(state_path, state)
            old_arrow = decode(Path(original['path']), 'arrow-zstd', deadline)
            first = decode(new_path, 'python-zstandard', deadline)
            second = decode(new_path, 'arrow-zstd', deadline)
            require_decoder_agreement(first, second)
            documents = 0
            for line, document in dataset.raw_json_records(new_path):
                if documents % 256 == 0 and time.time() >= deadline:
                    raise TimeoutError('Replacement JSON deadline')
                dataset.lossless_record(line, document)
                documents += 1
            if documents == 0:
                raise ValueError('Replacement contains no documents')
            result = {'passed': True, 'approved_for_new_preparation_import': True,
                      'identity': identity, 'job_id': job, 'jobs': state['jobs'], 'upstream': upstream,
                      'original': original, 'replacement': replacement, 'document_count': documents,
                      'original_arrow_decode': old_arrow, 'replacement_decoders': [first, second],
                      'local_bytes_differ_from_pinned_upstream': original['sha256'] != upstream['sha256'],
                      'old_artifacts_modified': False, 'production_split_validated': False}
            state.update(phase='complete')
            dataset.atomic_json(state_path, state)
            dataset.atomic_json(target / 'result.json', result)
            print(json.dumps(result), flush=True)
            return 0
        except Exception as error:
            # HTTP exceptions may contain signed URLs. Persist type/phase only.
            result = {'passed': False, 'identity': identity, 'job_id': job,
                      'phase': state['phase'], 'error_type': type(error).__name__,
                      'old_artifacts_modified': False}
            dataset.atomic_json(target / 'result.json', result)
            print(json.dumps(result), flush=True)
            return 1


if __name__ == '__main__':
    sys.exit(main())
