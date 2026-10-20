"""One bounded publication allocation; preserve remote history and verify blobs."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'kd3b-hidden8-20261019/source'))
from training_state import DEFAULT_ROOT, ExperimentLedger, atomic_json, digest, file_digest, load_config, locked, read_json

RUN = 'hf-kd6b-r5-20261020'
PARENT = 'kd6b-r5-20261020-seed42'
MODEL = 'asciibase64/Llama-3.2-1B-dolma3_mix'
DEST = DEFAULT_ROOT/'runs/publications'/RUN
TRAIN = DEFAULT_ROOT/'runs/experiments'/PARENT


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def check():
    plan = read_json(HERE/'plan.json')
    assert plan['model_id'] == MODEL and plan['parent_run'] == PARENT
    for name, sha in plan['sealed_inputs'].items():
        assert file_digest(name) == sha, name
    complete = read_json(TRAIN/'complete.json')
    summary = read_json(TRAIN/'summary.json')
    export = read_json(TRAIN/'export.json')
    assert complete['tokens'] == summary['confirmed_tokens'] == 6000082944
    assert complete['updates'] == summary['global_step'] == 28619
    assert summary['complete'] and summary['stop_reason'] == 'target_reached'
    assert export['reload_verified'] and export['path'] == plan['export'] == complete['export']
    checkpoint = read_json(Path(complete['checkpoint'])/'complete.json')
    assert checkpoint['all_optimizer_states_reload_verified']
    for name, stat in plan['export_stats'].items():
        p = Path(plan['export'])/name
        assert {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} == stat, name
    return plan


def submit():
    plan = check()
    DEST.mkdir(parents=True, exist_ok=True)
    with locked(DEST/'.submission.lock', blocking=False):
        assert not (DEST/'launch.json').exists(), 'Existing submission; inspect instead of retry'
        cfg = load_config(HERE/'validation-config.json')
        ledger = ExperimentLedger(DEFAULT_ROOT)
        tag = RUN+'-'+uuid.uuid4().hex
        ledger.reserve(tag, RUN, cfg, 1800, 1, False)
        state = dict(reservation=tag, status='submission_uncertain', started_utc=now())
        atomic_json(DEST/'launch.json', state)
        with ledger.transaction() as data:
            data['jobs'][tag].update(allocation_kind='validation_and_publication', parent_run=PARENT,
                authorization='User explicitly requested uploading the completed R5 6B model to Hugging Face')
        command = ['sbatch', '--parsable', '--comment='+tag, str(REPO/'slurm/hf-kd6b-r5-upload.sbatch'), RUN]
        state['command'] = command
        atomic_json(DEST/'launch.json', state)
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        state['submission_result'] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        job = result.stdout.strip().split(';')[0]
        atomic_json(DEST/'launch.json', state)
        assert result.returncode == 0 and job.isdigit(), 'Uncertain submission; reservation retained'
        ledger.bind(tag, job)
        state.update(job_id=job, status='submitted')
        atomic_json(DEST/'launch.json', state)
        atomic_json(HERE/'launch.json', state)
        print(json.dumps(state))


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            sha.update(block)
            git.update(block)
    return dict(size=size, sha256=sha.hexdigest(), git_sha1=git.hexdigest())


def worker():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURMD_NODENAME')
    plan = check()
    allocation = ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'], RUN)
    assert allocation['config_digest'] == digest(load_config(HERE/'validation-config.json'))
    assert not (DEST/'upload-intent.json').exists(), 'Do not duplicate a publication intent'
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_url
    from transformers import AutoConfig, AutoTokenizer
    import requests
    export = Path(plan['export'])
    config = AutoConfig.from_pretrained(export, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
    assert config.model_type == 'llama' and config.vocab_size == 128256
    index = read_json(export/'model.safetensors.index.json')['weight_map']
    assert index and all((export/n).is_file() for n in set(index.values()))
    files = {p.name: p for p in export.iterdir() if p.is_file()}
    files.update({p.name: p for p in (HERE/'payload').iterdir() if p.is_file()})
    assert set(files) == set(plan['payload_files'])
    atomic_json(DEST/'progress.json', dict(stage='hashing_export', timestamp=now()))
    checksums = {n: hashes(p) for n, p in files.items()}
    checkpoint = read_json(Path(read_json(TRAIN/'complete.json')['checkpoint'])/'complete.json')
    saved = {Path(f['name']).name: f for f in checkpoint['files'] if f['name'].startswith('student/')}
    for name in set(plan['export_stats']) - {p.name for p in (HERE/'payload').iterdir()}:
        assert checksums[name]['sha256'] == saved[name]['sha256'], name
    atomic_json(DEST/'payload-checksums.json', checksums)
    api = HfApi()
    info = api.model_info(MODEL)
    assert info.sha == plan['expected_parent_commit'] and not info.private and not info.gated
    atomic_json(DEST/'upload-intent.json', dict(model_id=MODEL, parent_commit=info.sha,
        checksums=checksums, slurm_job_id=os.environ['SLURM_JOB_ID'], timestamp=now()))
    atomic_json(DEST/'progress.json', dict(stage='uploading', timestamp=now()))
    commit = api.create_commit(MODEL, repo_type='model', parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in files.items()],
        num_threads=4, commit_message='Publish R5 online KD 6B: 10k PPL 12.39892, step 28619')
    receipt = dict(model_id=MODEL, revision=commit.oid, commit_url=commit.commit_url,
        parent_revision=info.sha, parent_run=PARENT, training_tokens=6000082944, global_step=28619,
        combined_10k_perplexity=12.3989218778246, slurm_job_id=os.environ['SLURM_JOB_ID'],
        local_export=str(export), status='uploaded; verification pending')
    atomic_json(DEST/'upload.json', receipt)
    atomic_json(DEST/'progress.json', dict(stage='verifying_public_revision', timestamp=now()))
    public = HfApi(token=False)
    remote = public.get_paths_info(MODEL, list(files), revision=commit.oid)
    assert len(remote) == len(files)
    for item in remote:
        expected = checksums[item.path]
        assert item.size == expected['size'], item.path
        assert (item.lfs.sha256 == expected['sha256'] if item.lfs else item.blob_id == expected['git_sha1']), item.path
    rc = AutoConfig.from_pretrained(MODEL, revision=commit.oid, token=False)
    rt = AutoTokenizer.from_pretrained(MODEL, revision=commit.oid, token=False)
    lc, pc = config.to_dict(), rc.to_dict()
    for key in ['_name_or_path', '_commit_hash']:
        lc.pop(key, None); pc.pop(key, None)
    assert lc == pc
    texts = ['Training from random initialization.', '資料驗證與模型訓練。', 'def square(x):\n    return x * x']
    assert tokenizer(texts, add_special_tokens=False) == rt(texts, add_special_tokens=False)
    for name in set(index.values()):
        response = requests.head(hf_hub_url(MODEL, name, revision=commit.oid), allow_redirects=True, timeout=60)
        response.raise_for_status()
        assert response.status_code == 200
    final = public.model_info(MODEL)
    assert final.sha == commit.oid and not final.private and not final.gated
    receipt.update(status='complete', remote_file_checksums_verified=True,
        remote_config_tokenizer_load_verified=True, public_unauthenticated_access=True,
        public_weight_shard_access_verified=True, finished_utc=now())
    atomic_json(DEST/'upload.json', receipt)
    atomic_json(HERE/'upload.json', receipt)
    atomic_json(DEST/'progress.json', dict(stage='complete', timestamp=now()))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    {'check': check, 'submit': submit, 'worker': worker}[sys.argv[1]]()
