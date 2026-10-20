"""Publish the completed BE60P20 export; large IO stays inside Slurm."""
import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import (DEFAULT_ROOT, PROJECT, TERMINAL, ExperimentLedger,
                            atomic_json, digest, file_digest, load_config, locked, read_json)
from slurm.submit_training import mark_uncertain, recover_submission

RUN = 'hf-be60p20-2b-20261010'
PARENT = 'be60p20-lrscaled-2b-20261010'
MODEL = 'asciibase64/Llama-3.2-1B-dolma3_mix'
DIRECTORY = DEFAULT_ROOT / 'runs/publications' / RUN
PLAN_DIRECTORY = PROJECT / 'experiments' / RUN


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def artifact_hashes(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha256.update(block)
            git_sha1.update(block)
    return dict(size=size, sha256=sha256.hexdigest(), git_sha1=git_sha1.hexdigest())


def complete_endpoint(summary, accounting, export):
    return (summary['global_step'] == summary['attempted_updates'] == summary['last_eval_step']
            == summary['export_global_step'] == 15259
            and summary['total_tokens_seen'] == summary['budget_tokens_charged']
            == summary['export_tokens_seen'] == 2000027648
            and summary['stop_reason'] == 'training_token_target'
            and summary['training_token_target_reached'] is True
            and not summary['initialization']['pretrained_weights_loaded']
            and accounting['all_jobs_reconciled'] and not accounting['budget_exceeded']
            and export['reload_verified'] is True and summary['export'] == export['path'])


def submit():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with locked(DIRECTORY / '.submission.lock', blocking=False):
        if (DIRECTORY / 'launch.json').exists():
            raise ValueError('Existing publication submission; use status/recovery, never retry blindly')
        parent = DEFAULT_ROOT / 'runs/experiments' / PARENT
        summary = read_json(parent / 'summary.json')
        accounting = read_json(parent / 'resource-accounting.json')
        export = read_json(parent / 'export.json')
        if not complete_endpoint(summary, accounting, export):
            raise ValueError('Parent endpoint or HF reload verification incomplete')
        plan = read_json(PLAN_DIRECTORY / 'plan.json')
        if plan['model_id'] != MODEL or plan['export'] != export['path']:
            raise ValueError('Publication target/export changed')
        if plan['status'] != 'approved_for_submission':
            raise ValueError('Require explicit approval of public target and full payload')
        cfg = load_config(PROJECT / plan['configuration'])
        if (cfg['category'], cfg['world_size'], cfg['max_h200_hours']) != ('validation', 1, .5):
            raise ValueError('Require a single half-hour/one-H200 publication allocation')
        sources = [PROJECT / 'slurm/hf_be60p20_upload.py', PROJECT / 'slurm/hf-be60p20-upload.sbatch',
                   PROJECT / 'slurm/common.sh', PROJECT / 'slurm/training-common.sh',
                   PROJECT / 'training_state.py', PROJECT / 'slurm/submit_training.py',
                   PROJECT / 'slurm/submission.py', PLAN_DIRECTORY / 'README.md',
                   PLAN_DIRECTORY / 'plan.json', PLAN_DIRECTORY / 'payload-manifest.json',
                   PROJECT / plan['configuration']]
        sources.append(PROJECT / 'experiments' / PARENT / 'plan.json')
        sources += [parent / name for name in ('summary.json', 'resource-accounting.json',
                    'export.json', 'initialization.json', 'data-provenance.json',
                    'optimizer-groups.json', 'execution-identity.json')]
        manifest = dict(model_id=MODEL, parent_run=PARENT, export=export['path'],
                        expected_parent_commit=plan['expected_parent_commit'],
                        sealed_files={str(path): file_digest(path) for path in sources},
                        export_file_stats={p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                           for p in Path(export['path']).iterdir() if p.is_file()},
                        config=cfg, user_authorization=plan['user_authorization'], requested_seconds=1800,
                        gpus=1, cpus=8, kind='validated export publication; no training')
        DEFAULT_ROOT.joinpath('logs').mkdir(parents=True, exist_ok=True)
        tag = 'hf-upload-' + uuid.uuid4().hex
        command = ['sbatch', '--parsable', '--comment=' + tag,
                   str(PROJECT / 'slurm/hf-be60p20-upload.sbatch'), RUN]
        ledger = ExperimentLedger(DEFAULT_ROOT)
        ledger.reserve(tag, RUN, cfg, 1800, 1, False)
        atomic_json(DIRECTORY / 'authorization.json', manifest)
        state = dict(status='submission_uncertain', reservation=tag,
                     submission_command=command, started_utc=now())
        atomic_json(DIRECTORY / 'launch.json', state)
        with ledger.transaction() as data:
            data['jobs'][tag].update(submission_tag=tag, submission_started_utc=state['started_utc'],
                submission_command=command, allocation_kind='validation_and_publication', parent_run=PARENT)
        environment = {**os.environ, 'TRAIN_SOURCE_ROOT': str(PROJECT), 'DATA_ROOT': str(DEFAULT_ROOT)}
        try:
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
        except OSError:
            ledger.cancel_reservation(tag)
            state['status'] = 'submission_not_executed'
            atomic_json(DIRECTORY / 'launch.json', state)
            raise
        state['submission_result'] = dict(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        atomic_json(DIRECTORY / 'launch.json', state)
        job = result.stdout.strip().split(';')[0]
        if result.returncode or not re.fullmatch(r'\d+', job):
            mark_uncertain(ledger, tag)
            raise ValueError('Uncertain submission; retain reservation and recover before retry')
        with ledger.transaction() as data:
            data['jobs'][tag]['returned_job_id'] = job
        ledger.bind(tag, job)
        state.update(status='submitted', job_id=job)
        atomic_json(DIRECTORY / 'launch.json', state)
        print('Submitted HF publication job', job, 'target=', MODEL)


def status():
    with locked(DIRECTORY / '.submission.lock', blocking=False):
        state = read_json(DIRECTORY / 'launch.json')
        if state['status'] == 'submission_not_executed':
            print(state['status'])
            return
        if not state.get('job_id'):
            state['job_id'] = recover_submission(ExperimentLedger(DEFAULT_ROOT), state['reservation'])
        if state['status'] != 'reconciled':
            result = subprocess.run(['sacct', '-X', '-j', state['job_id'], '--noheader',
                '--parsable2', '--format=JobIDRaw,State%30'], capture_output=True, text=True, check=True, timeout=20)
            rows = [row.split('|') for row in result.stdout.splitlines()]
            slurm = next((row[1].split()[0].rstrip('+') for row in rows if row[0] == state['job_id']), 'UNKNOWN')
            state['slurm_state'] = slurm
            if slurm in TERMINAL:
                state['accounting'] = ExperimentLedger(DEFAULT_ROOT).reconcile(state['job_id'])
                state['status'] = 'reconciled'
            atomic_json(DIRECTORY / 'launch.json', state)
        print('Job', state['job_id'], state.get('slurm_state'), state['status'])
        for name in ('progress.json', 'upload.json'):
            path = DIRECTORY / name
            if path.exists():
                value = read_json(path)
                print(name, {key: value[key] for key in ('stage', 'status', 'revision', 'commit_url') if key in value})


def progress(stage):
    atomic_json(DIRECTORY / 'progress.json', dict(stage=stage, recorded_utc=now()))
    print(stage, flush=True)


def worker():
    from dataset import require_compute
    require_compute()
    manifest = read_json(DIRECTORY / 'authorization.json')
    for name, expected in manifest['sealed_files'].items():
        if file_digest(Path(name)) != expected:
            raise ValueError('Pinned publication input changed: ' + name)
    allocation = ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'], RUN)
    if allocation['config_digest'] != digest(manifest['config']):
        raise ValueError('Publication resource reservation changed')
    if (DIRECTORY / 'upload.json').exists():
        raise ValueError('Existing upload receipt; verify/recover it rather than publish again')
    parent = DEFAULT_ROOT / 'runs/experiments' / PARENT
    summary = read_json(parent / 'summary.json')
    accounting = read_json(parent / 'resource-accounting.json')
    export_metadata = read_json(parent / 'export.json')
    if not complete_endpoint(summary, accounting, export_metadata):
        raise ValueError('Parent endpoint changed')
    export = Path(manifest['export'])
    for name, expected in manifest['export_file_stats'].items():
        path = export / name
        if dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) != expected:
            raise ValueError('Verified export file changed: ' + name)
    index = read_json(export / 'model.safetensors.index.json')['weight_map']
    if not index or any(not (export / filename).is_file() for filename in set(index.values())):
        raise ValueError('Incomplete weight shard index')
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download, hf_hub_url
    import requests
    config = AutoConfig.from_pretrained(export, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
    original = AutoTokenizer.from_pretrained(parent / 'assets', local_files_only=True)
    texts = ['This model was trained from random initialization.', '資料驗證與模型訓練。', 'def square(x):\n    return x * x']
    if config.model_type != 'llama' or config.vocab_size != 128256:
        raise ValueError('Unexpected exported architecture')
    if tokenizer(texts, add_special_tokens=False) != original(texts, add_special_tokens=False):
        raise ValueError('Export tokenizer changed')
    stage = DIRECTORY / 'metadata'
    stage.mkdir(exist_ok=True)
    (stage / 'README.md').write_bytes((PLAN_DIRECTORY / 'README.md').read_bytes())
    (stage / 'NOTICE').write_text('Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n')
    license_path = hf_hub_download('meta-llama/Llama-3.2-1B', 'LICENSE.txt', revision=summary['configuration']['model_revision'])
    (stage / 'LICENSE.txt').write_bytes(Path(license_path).read_bytes())
    atomic_json(stage / 'resolved-training-config.json', summary['configuration'])
    atomic_json(stage / 'training-metadata.json', summary)
    atomic_json(stage / 'training-details.json', dict(
        run_id=PARENT, global_step=15259, training_tokens=summary['total_tokens_seen'],
        training_h200_hours=accounting['h200_hours_measured'], training_allocations=accounting['jobs'],
        initialization=read_json(parent / 'initialization.json'), provenance=read_json(parent / 'data-provenance.json'),
        optimizer_groups=read_json(parent / 'optimizer-groups.json'), evaluation=summary['last_evaluation'],
        evaluation_scope='Fixed first 1000 reserved documents; full 50000 reserved/OJ PPL not measured',
        wandb_url=summary['wandb_url'], execution_identity=read_json(parent / 'execution-identity.json'),
        export_validation=export_metadata, lr_scaling=read_json(PROJECT / 'experiments' / PARENT / 'plan.json')['lr_scaling']))
    files = {p.name: p for p in export.iterdir() if p.is_file()}
    files.update({p.name: p for p in stage.iterdir() if p.is_file()})
    progress('hashing_verified_export')
    hashes = {name: artifact_hashes(path) for name, path in files.items()}
    approved = read_json(PLAN_DIRECTORY / 'payload-manifest.json')
    if approved['model_id'] != MODEL or set(approved['files']) != set(files):
        raise ValueError('Payload differs from approved public file list')
    for name, expected in approved['files'].items():
        if hashes[name]['size'] != expected['size']:
            raise ValueError('Approved payload size differs: ' + name)
        if 'sha256' in expected and hashes[name]['sha256'] != expected['sha256']:
            raise ValueError('Approved metadata checksum differs: ' + name)
    api = HfApi()
    info = api.model_info(MODEL)
    if info.sha != manifest['expected_parent_commit'] or info.private or info.gated:
        raise ValueError('Remote HEAD or public visibility changed; inspect before upload')
    progress('uploading')
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()]
    result = api.create_commit(MODEL, operations=operations,
        commit_message='Upload BE60P20 LR-scaled 2B-token final checkpoint at step 15259',
        repo_type='model', parent_commit=info.sha, num_threads=4)
    receipt = dict(model_id=MODEL, revision=result.oid, commit_url=result.commit_url,
        export=str(export), parent_run=PARENT, global_step=15259, training_tokens=2000027648,
        perplexity=summary['last_evaluation']['eval/perplexity'], upload_checksums=hashes,
        status='uploaded; public verification pending', slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(DIRECTORY / 'upload.json', receipt)
    progress('verifying_public_revision')
    public = HfApi(token=False)
    remote = public.get_paths_info(MODEL, list(files), revision=result.oid)
    if len(remote) != len(files):
        raise ValueError('Published file coverage incomplete')
    for item in remote:
        expected = hashes[item.path]
        if item.size != expected['size']:
            raise ValueError('Published file size differs: ' + item.path)
        if item.lfs:
            if item.lfs.sha256 != expected['sha256']:
                raise ValueError('Published LFS checksum differs: ' + item.path)
        elif item.blob_id != expected['git_sha1']:
            raise ValueError('Published Git blob differs: ' + item.path)
    remote_config = AutoConfig.from_pretrained(MODEL, revision=result.oid, token=False)
    remote_tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=result.oid, token=False)
    if (remote_config.to_dict() != config.to_dict()
            or remote_tokenizer(texts, add_special_tokens=False) != tokenizer(texts, add_special_tokens=False)):
        # Transformers tracks a different name/path after remote loading.
        local_cfg, public_cfg = config.to_dict(), remote_config.to_dict()
        for name in ('_name_or_path', '_commit_hash'):
            local_cfg.pop(name, None)
            public_cfg.pop(name, None)
        if local_cfg != public_cfg or remote_tokenizer(texts, add_special_tokens=False) != tokenizer(texts, add_special_tokens=False):
            raise ValueError('Public config/tokenizer differs')
    for filename in sorted(set(index.values())):
        response = requests.head(hf_hub_url(MODEL, filename, revision=result.oid), allow_redirects=True, timeout=60)
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('Public weight shard HEAD unavailable')
    public_info = public.model_info(MODEL, revision=result.oid)
    if public_info.private or public_info.gated or public_info.sha != result.oid:
        raise ValueError('Public revision visibility verification failed')
    receipt.update(status='complete', public_unauthenticated_access=True,
        remote_file_checksums_verified=True, remote_config_tokenizer_load_verified=True,
        public_weight_shard_access_verified=True, finished_utc=now())
    atomic_json(DIRECTORY / 'upload.json', receipt)
    progress('complete')
    print('Published', MODEL, 'revision=', result.oid, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status', 'worker'))
    args = parser.parse_args()
    {'submit': submit, 'status': status, 'worker': worker}[args.action]()
