"""Small, dependency-free training contracts, durable budgets and Slurm ledger.

Safe to import on the login node. Tensor/model operations live in train.py.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

DEFAULT_ROOT = Path('/work/asciibase64/Llama-3.2-1B-training')
PROJECT = Path(__file__).resolve().parent
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value) or value in {'.', '..'}:
        raise ValueError('Use a simple run/preparation ID, at most 96 characters')
    return value


def storage_root(value):
    root = Path(value).resolve()
    if not root.is_relative_to(Path('/work/asciibase64')):
        raise ValueError('Large artifacts must be under /work/asciibase64/')
    return root


def load_config(path):
    cfg = read_json(path)
    required = set(read_json(PROJECT / 'configs/train-pilot.json'))
    if set(cfg) != required:
        raise ValueError(f'Configuration keys mismatch: missing={required - set(cfg)}, extra={set(cfg) - required}')
    if cfg['model_id'] != 'meta-llama/Llama-3.2-1B' or cfg['sequence_length'] != 8192:
        raise ValueError('Preserve Llama 3.2 1B and the 8192-token context')
    if not re.fullmatch(r'[0-9a-f]{40}', cfg['model_revision']):
        raise ValueError('Pin model config to a full commit revision')
    for key in ('world_size', 'microbatch_size', 'gradient_accumulation_steps', 'max_steps',
                'max_training_tokens', 'logging_steps', 'eval_steps', 'eval_documents', 'stop_margin_seconds'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['seed']) is not int or type(cfg['warmup_steps']) is not int:
        raise ValueError('Seed and warmup_steps must be integers')
    if cfg['world_size'] > 8 or cfg['max_training_tokens'] > 6_000_000_000:
        raise ValueError('Maximum 8 H200s and 6B processed input tokens per logical run')
    if not 0 < cfg['max_h200_hours'] <= 64 or not 0 < cfg['eval_documents'] <= 10_000:
        raise ValueError('Maximum 64 H200-hours and 10,000 held-out documents')
    if cfg['eval_steps'] > max(1, cfg['max_steps'] // 10):
        raise ValueError('Evaluate at least once per 10% of planned optimizer updates')
    if cfg['max_steps'] < 10 and cfg['eval_steps'] != 1:
        raise ValueError('Very short pilots must evaluate each update')
    if not 0 <= cfg['warmup_steps'] < cfg['max_steps'] or not 0 <= cfg['min_lr_ratio'] <= 1:
        raise ValueError('Invalid warmup or cosine floor')
    if len(cfg['betas']) != 2 or any(not 0 <= x < 1 for x in cfg['betas']):
        raise ValueError('Invalid AdamW betas')
    for key in ('learning_rate', 'epsilon', 'max_grad_norm'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not math.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0:
        raise ValueError('Invalid weight decay')
    if cfg['category'] not in {'pilot', 'scaling', 'full', 'validation'}:
        raise ValueError('Unknown experiment category')
    if cfg['wandb_project'] != 'lab5-training-llama' or cfg['wandb_mode'] not in {'online', 'offline'}:
        raise ValueError('Use lab5-training-llama; offline is allowed only for validation')
    if cfg['wandb_mode'] != 'online' and cfg['category'] != 'validation':
        raise ValueError('Training pilots require online W&B logging')
    return cfg


def lr_multiplier(update_index, max_steps, warmup_steps, min_ratio):
    """LR for zero-based *next* successful optimizer update; first update is nonzero."""
    if update_index < warmup_steps:
        return (update_index + 1) / warmup_steps
    progress = min(1.0, (update_index - warmup_steps) / max(1, max_steps - warmup_steps - 1))
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2


def window_indices(cursor, rows, world_size, rank, microbatch, accumulation):
    """Contiguous canonical packed order; no padding, repetition or reshuffling.

    Only complete distributed accumulation windows are used. An incomplete tail
    is deliberately left unused (reported in run metadata), never repeated.
    """
    width = world_size * microbatch
    end = cursor + width * accumulation
    if not 0 <= rank < world_size or cursor < 0 or min(width, accumulation) <= 0:
        raise ValueError('Invalid sampler geometry')
    if end > rows:
        return []
    return [list(range(start + rank * microbatch, start + (rank + 1) * microbatch))
            for start in range(cursor, end, width)]


class TokenJournal:
    """High-water accounting never rewinds with a model checkpoint.

    Reservations are fsynced before any forward pass. A crash leaves uncertain
    reservations charged conservatively; confirmed_tokens remains measured.
    """
    def __init__(self, path, limit):
        self.path, self.limit = Path(path), limit
        self.data = read_json(path) if self.path.exists() else {
            'limit': limit, 'confirmed_tokens': 0, 'charged_tokens': 0,
            'next_transaction': 0, 'pending': {}}
        if self.data['limit'] != limit:
            raise ValueError('Do not change a logical run token limit on resume')
        if (not 0 <= self.data['confirmed_tokens'] <= self.data['charged_tokens'] <= limit or
                self.data['charged_tokens'] - self.data['confirmed_tokens'] != sum(self.data['pending'].values())):
            raise ValueError('Corrupt token high-water accounting; refuse to reset or repair silently')
        if not self.path.exists():
            atomic_json(self.path, self.data)

    def reserve(self, tokens):
        if tokens <= 0 or self.data['charged_tokens'] + tokens > self.limit:
            raise ValueError('Insufficient remaining training token budget')
        key = str(self.data['next_transaction'])
        self.data['next_transaction'] += 1
        self.data['pending'][key] = tokens
        self.data['charged_tokens'] += tokens
        atomic_json(self.path, self.data)
        return key

    def confirm(self, key):
        tokens = self.data['pending'].pop(key)
        self.data['confirmed_tokens'] += tokens
        atomic_json(self.path, self.data)
        return tokens


@contextlib.contextmanager
def locked(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def slurm_record(job_id, root=None):
    if not re.fullmatch(r'\d+', str(job_id)):
        raise ValueError('Use a numeric, non-array job ID')
    raw = subprocess.check_output([
        'sacct', '-j', str(job_id), '-X', '--noheader', '--parsable2',
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%500,Start,End,NodeList%128'], text=True)
    rows = [row.split('|') for row in raw.splitlines() if row.split('|')[0] == str(job_id)]
    if len(rows) != 1:
        raise ValueError('Expected one Slurm allocation; retain reservation until accounting is available')
    job, state, elapsed, tres, start, end, nodes = rows[0][:7]
    resources = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
    count = int(resources.get('gres/gpu', 0))
    typed = {k[9:].lower(): int(v) for k, v in resources.items() if k.startswith('gres/gpu:')}
    if state.split()[0] not in TERMINAL:
        raise ValueError('Job has not terminated; its full reservation stays charged')
    evidence = 'typed AllocTRES'
    if count and not typed and root is not None:
        # nano4 records only generic GPU TRES. Hardware type must come from
        # startup evidence captured inside this exact allocation, never its partition name.
        directory = Path(root) / 'logs' / f'job-{job_id}'
        gpu_path = directory / 'gpus.txt'
        visible = gpu_path.read_text().splitlines() if gpu_path.exists() else []
        if (len(visible) == count and all('H200' in row and ', GPU-' in row for row in visible)
                and len(set(visible)) == count):
            typed = {'h200': count}
            evidence = {'source': 'startup nvidia-smi for allocated devices', 'sha256': file_digest(gpu_path)}
        else:
            # A startup failure may precede nvidia-smi. The recorded node must
            # match the accounting node and contain exclusively H200 GPUs.
            node_path = directory / 'node.txt'
            node = node_path.read_text() if node_path.exists() else ''
            node_name = re.search(r'\bNodeName=(\S+)', node)
            gres = re.search(r'\bGres=(\S+)', node)
            gpu_types = re.findall(r'gpu:([^:,()]+):(\d+)', gres[1]) if gres else []
            if (node_name and node_name[1] == nodes and gpu_types and
                    all(kind.lower() == 'h200' for kind, _ in gpu_types) and
                    sum(int(number) for _, number in gpu_types) >= count):
                typed = {'h200': count}
                evidence = {'source': 'startup scontrol node with exclusively H200 GPUs', 'sha256': file_digest(node_path)}
    if (count and typed != {'h200': count}) or (not count and typed):
        raise ValueError('Need typed H200 TRES or matching startup hardware evidence; reservation stays charged')
    return {'job_id': job, 'state': state, 'elapsed_seconds': int(elapsed),
            'gpu_count': count, 'h200_hours': count * int(elapsed) / 3600,
            'alloc_tres': tres, 'gpu_type_evidence': evidence, 'start': start, 'end': end, 'nodes': nodes}


class ExperimentLedger:
    def __init__(self, root):
        self.path = Path(root) / 'experiment-ledger.json'
        self.lock = Path(root) / '.experiment-ledger.lock'

    @contextlib.contextmanager
    def transaction(self):
        with locked(self.lock):
            data = read_json(self.path) if self.path.exists() else {'runs': {}, 'jobs': {}}
            yield data
            data['total_h200_hours_measured'] = sum(j.get('h200_hours', 0) for j in data['jobs'].values())
            data['total_h200_hours_reserved_or_measured'] = sum(
                j.get('h200_hours', j['reserved_h200_hours']) for j in data['jobs'].values())
            atomic_json(self.path, data)

    def reserve(self, reservation, run_id, config, seconds, gpus, resume):
        identifier(run_id)
        hours = gpus * seconds / 3600
        if gpus != config['world_size'] or seconds <= config['stop_margin_seconds']:
            raise ValueError('Allocation does not match configured world size or stopping margin')
        with self.transaction() as data:
            runs, jobs = data['runs'], data['jobs']
            if reservation in jobs:
                raise ValueError('Reservation ID already exists')
            if run_id in runs:
                if not resume or runs[run_id]['config_digest'] != digest(config):
                    raise ValueError('Existing run requires --resume and the identical configuration')
            else:
                if resume:
                    raise ValueError('Unknown run cannot resume')
                if config['category'] == 'full' and sum(r['category'] == 'full' for r in runs.values()) >= 3:
                    raise ValueError('Three committed full runs already exist in the shared ledger')
            previous = [j for j in jobs.values() if j['run_id'] == run_id]
            if any('h200_hours' not in j for j in previous):
                raise ValueError('Reconcile all earlier jobs for this run before resubmission')
            used = sum(j['h200_hours'] for j in previous)
            if used + hours > min(64, config['max_h200_hours']) + 1e-12:
                raise ValueError(f'Allocation exceeds remaining H200-hours ({config["max_h200_hours"] - used:.6f})')
            runs.setdefault(run_id, {'category': config['category'], 'config_digest': digest(config),
                                     'max_h200_hours': config['max_h200_hours'], 'created': time.time()})
            jobs[reservation] = {'run_id': run_id, 'reserved_h200_hours': hours, 'gpu_count': gpus,
                                 'requested_seconds': seconds, 'status': 'reserved'}

    def bind(self, reservation, job_id):
        if not re.fullmatch(r'\d+', job_id):
            raise ValueError('Expected a numeric sbatch job ID')
        with self.transaction() as data:
            if job_id in data['jobs']:
                raise ValueError('Job already registered')
            item = data['jobs'].pop(reservation)
            item.update(job_id=job_id, status='submitted')
            data['jobs'][job_id] = item

    def cancel_reservation(self, reservation):
        with self.transaction() as data:
            item = data['jobs'].pop(reservation)
            if not any(j['run_id'] == item['run_id'] for j in data['jobs'].values()):
                data['runs'].pop(item['run_id'])

    def reconcile(self, job_id):
        measured = slurm_record(job_id, self.path.parent)
        with self.transaction() as data:
            if job_id not in data['jobs']:
                raise ValueError('Unregistered training job; submit through submit_training.py')
            item = data['jobs'][job_id]
            never_allocated = measured['gpu_count'] == 0 and measured['elapsed_seconds'] == 0
            if measured['gpu_count'] != item['gpu_count'] and not never_allocated:
                raise ValueError('Slurm GPU count differs from the reserved allocation')
            item.update(measured, status='reconciled')
            run = data['runs'][item['run_id']]
            run['h200_hours_measured'] = sum(j.get('h200_hours', 0) for j in data['jobs'].values()
                                             if j['run_id'] == item['run_id'])
            run['budget_exceeded'] = run['h200_hours_measured'] > run['max_h200_hours']
            run_record = {'run_id': item['run_id'], **run,
                          'all_jobs_reconciled': all('h200_hours' in j for j in data['jobs'].values()
                                                     if j['run_id'] == item['run_id']),
                          'jobs': [j for j in data['jobs'].values() if j['run_id'] == item['run_id']]}
        run_path = self.path.parent / 'runs/experiments' / item['run_id']
        if run_path.exists():
            with locked(run_path / '.run.lock', blocking=False):
                atomic_json(run_path / 'resource-accounting.json', run_record)
                if (run_path / 'summary.json').exists():
                    summary = read_json(run_path / 'summary.json')
                    summary['h200_hours_final'] = (run_record['h200_hours_measured'] if run_record['all_jobs_reconciled']
                                                  else 'pending remaining allocation reconciliation')
                    summary['resource_accounting_path'] = str(run_path / 'resource-accounting.json')
                    atomic_json(run_path / 'summary.json', summary)
        return measured

    def allocation(self, job_id, run_id):
        with locked(self.lock):
            data = read_json(self.path)
            job = data['jobs'][str(job_id)]
            if job['run_id'] != run_id or 'h200_hours' in job:
                raise ValueError('Allocation does not belong to this active logical run')
            used = sum(j.get('h200_hours', j['reserved_h200_hours']) for key, j in data['jobs'].items()
                       if j['run_id'] == run_id and key != str(job_id))
            return {**job, 'previous_h200_hours': used,
                    'config_digest': data['runs'][run_id]['config_digest'],
                    'max_h200_hours': data['runs'][run_id]['max_h200_hours']}
