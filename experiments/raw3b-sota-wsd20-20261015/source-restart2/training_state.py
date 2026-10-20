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
BE60P20_2B_RUN_ID = 'be60p20-lrscaled-2b-20261010'
BE60P20_2B_CONFIG_DIGEST = 'c2ec561c2bae73988d292fa9fe1bc7093265a3c1a6f401930dceedea40da46bf'
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


def be60p20_2b_authorized(run_id, config, resume=False):
    """Explicitly approved fourth formal run, exact config and fresh weights only."""
    return (run_id == BE60P20_2B_RUN_ID and not resume
            and digest(config) == BE60P20_2B_CONFIG_DIGEST)


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
    optional = {'attention_backend', 'checkpoint_interval_seconds', 'final_evaluation_reserve_seconds', 'campaign_id', 'sweep_stage', 'scheduler_steps', 'optimizer', 'muon', 'sweep_id', 'sweep_purpose', 'parent_runs', 'lr_scheduler', 'wsd_decay_ratio', 'fixed_horizon', 'checkpoint_policy', 'export_on_completion', 'retain_all_checkpoints', 'sweep_calibration_digest', 'sweep_environment_digest', 'target_training_tokens', 'hold_schedule'}
    optional.update({'performance', 'phase_schedule', 'initialization', 'clipping', 'ema', 'lr_transfer'})
    from recipe_runtime import validate
    validate(cfg)
    if 'performance' in cfg:
        performance = cfg['performance']
        if (not isinstance(performance, dict) or set(performance) !=
                {'model_compile', 'compile_mode', 'muon_compile', 'async_transfer'}):
            raise ValueError('Require a complete performance configuration')
        if any(type(performance[key]) is not bool for key in
               ('model_compile', 'muon_compile', 'async_transfer')):
            raise ValueError('Performance switches must be boolean')
        if performance['compile_mode'] not in {
                'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'}:
            raise ValueError('Unsupported compile mode')
        if performance['muon_compile'] and cfg.get('optimizer') != 'muon':
            raise ValueError('Muon compilation requires native Muon')
    if not required <= set(cfg) or set(cfg) - required - optional:
        raise ValueError(f'Configuration keys mismatch: missing={required - set(cfg)}, extra={set(cfg) - required}')
    if cfg.get('lr_scheduler', 'cosine') not in {'linear', 'cosine', 'wsd', 'wsd_hold_zero', 'warmup_constant'}:
        raise ValueError('Unsupported LR scheduler')
    decay_ratio = cfg.get('wsd_decay_ratio', 0.2)
    if type(decay_ratio) not in (int, float) or not math.isfinite(decay_ratio) or not 0 < decay_ratio < 1:
        raise ValueError('WSD decay ratio must be finite and in (0, 1)')
    for key in ('fixed_horizon', 'export_on_completion', 'retain_all_checkpoints'):
        if key in cfg and type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if cfg.get('checkpoint_policy', 'best_and_interval') not in {'best_and_interval', 'final_only'}:
        raise ValueError('Unknown checkpoint policy')
    for key in ('sweep_calibration_digest', 'sweep_environment_digest'):
        if key in cfg and not re.fullmatch(r'[0-9a-f]{64}', cfg[key]):
            raise ValueError('Pin sweep calibration/environment to a SHA256 digest')
    optimizer = cfg.get('optimizer', 'adamw')
    if optimizer not in {'adamw', 'muon'}:
        raise ValueError('optimizer must be adamw or muon')
    if optimizer == 'muon':
        muon = cfg.get('muon', {})
        if set(muon) != {'learning_rate', 'momentum', 'nesterov', 'ns_steps', 'weight_decay', 'adjust_lr_fn', 'epsilon', 'ns_coefficients'}:
            raise ValueError('Muon requires an explicit complete configuration')
        for key in ('learning_rate', 'momentum', 'weight_decay', 'epsilon'):
            if type(muon[key]) not in (int, float) or not math.isfinite(muon[key]):
                raise ValueError(f'Invalid Muon {key}')
        if muon['learning_rate'] <= 0 or not 0 <= muon['momentum'] < 1 or muon['weight_decay'] < 0 or muon['epsilon'] <= 0:
            raise ValueError('Invalid Muon learning rate, momentum or decay')
        if type(muon['ns_steps']) is not int or not 1 <= muon['ns_steps'] <= 10:
            raise ValueError('Muon ns_steps must be an integer in [1, 10]')
        if type(muon['nesterov']) is not bool or muon['adjust_lr_fn'] not in {'original', 'match_rms_adamw'}:
            raise ValueError('Unsupported native Muon Nesterov/LR adjustment')
        if len(muon['ns_coefficients']) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in muon['ns_coefficients']):
            raise ValueError('Muon requires three finite Newton-Schulz coefficients')
    elif 'muon' in cfg:
        raise ValueError('Muon options are only valid with optimizer=muon')
    if cfg.get('attention_backend', 'sdpa') not in {'sdpa', 'flash_attention_3'}:
        raise ValueError('Unsupported attention backend')
    for key in ('checkpoint_interval_seconds', 'final_evaluation_reserve_seconds'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 0):
            raise ValueError('Invalid optional interval')
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
    target = cfg.get('target_training_tokens')
    if target is not None and (type(target) is not int or not 0 < target <= cfg['max_training_tokens']):
        raise ValueError('Training token target must be a positive integer within the hard token cap')
    if not 0 < cfg['max_h200_hours'] <= 64 or not 0 < cfg['eval_documents'] <= 10_000:
        raise ValueError('Maximum 64 H200-hours and 10,000 held-out documents')
    if cfg['eval_steps'] > max(1, cfg['max_steps'] // 10):
        raise ValueError('Evaluate at least once per 10% of planned optimizer updates')
    if cfg['max_steps'] < 10 and cfg['eval_steps'] != 1:
        raise ValueError('Very short pilots must evaluate each update')
    if 'scheduler_steps' in cfg and (type(cfg['scheduler_steps']) is not int or not cfg['warmup_steps'] < cfg['scheduler_steps'] <= cfg['max_steps']):
        raise ValueError('Scheduler steps must lie between warmup and the hard step ceiling')
    if not 0 <= cfg['warmup_steps'] < cfg['max_steps'] or not 0 <= cfg['min_lr_ratio'] <= 1:
        raise ValueError('Invalid warmup or cosine floor')
    if len(cfg['betas']) != 2 or any(not 0 <= x < 1 for x in cfg['betas']):
        raise ValueError('Invalid AdamW betas')
    for key in ('learning_rate', 'epsilon'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if type(cfg['max_grad_norm']) not in (int, float) or not math.isfinite(cfg['max_grad_norm']) or cfg['max_grad_norm'] < 0:
        raise ValueError('Clipping threshold must be finite; zero disables clipping')
    if 'phase_schedule' in cfg:
        from horizon_contracts import validate_phases
        validate_phases(cfg)
    if not math.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0:
        raise ValueError('Invalid weight decay')
    if cfg['category'] not in {'pilot', 'scaling', 'full', 'validation'}:
        raise ValueError('Unknown experiment category')
    if cfg['wandb_project'] != 'lab5-training-llama' or cfg['wandb_mode'] not in {'online', 'offline'}:
        raise ValueError('Use lab5-training-llama; offline is allowed only for validation')
    if cfg['wandb_mode'] != 'online' and cfg['category'] != 'validation':
        raise ValueError('Training pilots require online W&B logging')
    horizon = cfg.get('scheduler_steps', cfg['max_steps'])
    if cfg.get('fixed_horizon', False) and horizon != cfg['max_steps']:
        raise ValueError('Fixed-horizon schedules must finish on the final planned update')
    if cfg.get('lr_scheduler') == 'wsd':
        decay_steps = round(horizon * decay_ratio)
        if not 2 <= decay_steps <= horizon - cfg['warmup_steps']:
            raise ValueError('WSD needs a nonoverlapping stable/decay horizon with at least two decay updates')
    if cfg.get('lr_scheduler') == 'wsd_hold_zero':
        if not cfg.get('fixed_horizon', False) or cfg['min_lr_ratio'] != 0:
            raise ValueError('Hold/zero schedule requires a fixed horizon and terminal LR zero')
        settings = cfg.get('hold_schedule')
        hold_schedule_boundaries(horizon, cfg['warmup_steps'], settings)
        if not math.isclose(sum(settings[key] for key in
                                ('first_decay_ratio', 'hold_ratio', 'final_decay_ratio')),
                            decay_ratio, rel_tol=0, abs_tol=1e-12):
            raise ValueError('wsd_decay_ratio must equal the complete cooldown/hold/zero tail')
    elif 'hold_schedule' in cfg:
        raise ValueError('hold_schedule is only valid with wsd_hold_zero')
    return cfg


def training_target_reached(cfg, journal_data):
    """Only confirmed processed inputs can satisfy a requested training target."""
    target = cfg.get('target_training_tokens')
    return target is not None and journal_data['confirmed_tokens'] >= target


def schedule_multiplier(update_index, cfg):
    if 'phase_schedule' in cfg:
        from horizon_contracts import phase_multiplier
        return phase_multiplier(update_index, cfg)
    return lr_multiplier(update_index, cfg.get('scheduler_steps', cfg['max_steps']),
                         cfg['warmup_steps'], cfg['min_lr_ratio'],
                         cfg.get('lr_scheduler', 'cosine'), cfg.get('wsd_decay_ratio', 0.2),
                         cfg.get('hold_schedule'))


def hold_schedule_boundaries(max_steps, warmup_steps, settings):
    """Return zero-based starts for first decay, hold, and final zero decay."""
    keys = {'first_decay_ratio', 'hold_ratio', 'final_decay_ratio', 'plateau_lr_ratio'}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError('Hold/zero schedule requires explicit phase ratios and plateau LR')
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not 0 < value < 1 for value in settings.values()):
        raise ValueError('Hold/zero ratios must be finite and strictly between zero and one')
    ratios = [settings[key] for key in ('first_decay_ratio', 'hold_ratio', 'final_decay_ratio')]
    if sum(ratios) >= 1:
        raise ValueError('Hold/zero tail must leave room for warmup and stable training')
    first, hold, final = [round(max_steps * ratio) for ratio in ratios]
    start = max_steps - first - hold - final
    if first < 2 or hold < 1 or final < 2 or start < warmup_steps:
        raise ValueError('Hold/zero phases overlap warmup or contain too few updates')
    return start, start + first, start + first + hold, max_steps


def lr_multiplier(update_index, max_steps, warmup_steps, min_ratio, scheduler='cosine',
                  decay_ratio=0.2, hold_schedule=None):
    """LR for zero-based *next* successful optimizer update; first update is nonzero."""
    if update_index < warmup_steps:
        return (update_index + 1) / warmup_steps
    if scheduler == 'warmup_constant':
        return 1.0
    if scheduler == 'wsd_hold_zero':
        start, hold_start, final_start, end = hold_schedule_boundaries(
            max_steps, warmup_steps, hold_schedule)
        plateau = hold_schedule['plateau_lr_ratio']
        if update_index < start:
            return 1.0
        if update_index < hold_start:
            progress = (update_index - start) / (hold_start - start - 1)
            return 1.0 + (plateau - 1.0) * progress
        if update_index < final_start:
            return plateau
        progress = min(1.0, (update_index - final_start) / (end - final_start - 1))
        return plateau * (1.0 - progress)
    progress = min(1.0, (update_index - warmup_steps) / max(1, max_steps - warmup_steps - 1))
    if scheduler == 'linear':
        return min_ratio + (1 - min_ratio) * (1 - progress)
    if scheduler == 'wsd':
        decay_steps = round(max_steps * decay_ratio)
        decay_start = max_steps - decay_steps
        progress = min(1.0, max(0.0, (update_index - decay_start) / max(1, decay_steps - 1)))
        return min_ratio + (1 - min_ratio) * (1 - progress)
    if scheduler != 'cosine':
        raise ValueError('Unsupported LR scheduler')
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
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%500,Start,End,NodeList%128'], text=True, timeout=30)
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

    def reserve(self, reservation, run_id, config, seconds, gpus, resume, *, uninitialized_retry=False, fresh_retry=False):
        identifier(run_id)
        hours = gpus * seconds / 3600
        be60p20_authorization = be60p20_2b_authorized(run_id, config, resume)
        from pair3b_contracts import authorized as pair3b_authorized, RUNS as PAIR3B_RUNS
        pair3b = pair3b_authorized(run_id, config, resume)
        from sota3b_contracts import authorized as sota3b_authorized
        sota3b = sota3b_authorized(run_id, config, resume)
        from wsd20_contracts import authorized as wsd20_authorized
        wsd20 = wsd20_authorized(run_id, config, resume)
        filtered_authorization = (run_id == 'muon-wsd-filtered-2b-20261009' and (not resume or uninitialized_retry) and
            config.get('category') == 'full' and config.get('sweep_id') == run_id and
            config.get('target_training_tokens') == 2_000_000_000 and
            config.get('max_training_tokens') == 2_000_027_648 and
            config.get('max_h200_hours') == 64 and config.get('world_size') == 8)
        seconds_limit, hours_limit = (28800, 64) if filtered_authorization else (7200, 16)
        if pair3b or sota3b or wsd20: seconds_limit, hours_limit = 21600, 48
        if be60p20_authorization:
            seconds_limit, hours_limit = 14400, 32
        from horizon_contracts import campaign_allocation_limits
        limits = campaign_allocation_limits(run_id, config)
        recipe_final = run_id == 'init-recipe-20261014-final-2b' and not resume and config.get('sweep_id') == 'init-recipe-20261014' and config.get('category') == 'full' and config.get('max_h200_hours') == 32 and config.get('world_size') == 8 and config.get('target_training_tokens') in (2000027648,2000158720) and config.get('max_training_tokens') == config.get('target_training_tokens') and config.get('max_steps',0)*config.get('world_size',0)*config.get('microbatch_size',0)*config.get('gradient_accumulation_steps',0)*8192 == config.get('target_training_tokens')
        from followup_contracts import authorized as followup_authorized
        followup_final=followup_authorized(run_id,config,resume)
        if recipe_final or followup_final: seconds_limit, hours_limit = 14400,32
        if limits is not None:
            seconds_limit, hours_limit = limits
        if seconds > seconds_limit or hours > hours_limit or not 1 <= gpus <= 8:
            raise ValueError(f'Maximum {seconds_limit} seconds and {hours_limit} H200-hours per sbatch')
        if gpus != config['world_size'] or seconds <= config['stop_margin_seconds']:
            raise ValueError('Allocation does not match configured world size or stopping margin')
        with self.transaction() as data:
            runs, jobs = data['runs'], data['jobs']
            if fresh_retry:
                from wsd20_contracts import fresh_restart_authorized
                previous = [j for j in jobs.values() if j['run_id'] == run_id]
                if (resume or uninitialized_retry or seconds > 21540 or
                    not fresh_restart_authorized(run_id, config, previous, self.path.parent/'runs/experiments'/run_id)):
                    raise ValueError('Fresh restart is restricted to approved failed jobs553561/553577 and whole-minute21540seconds')

            if uninitialized_retry:
                previous = [j for j in jobs.values() if j['run_id'] == run_id]
                if (not filtered_authorization or not resume or not previous or
                    any(j.get('state') != 'FAILED' or 'h200_hours' not in j for j in previous) or
                    (self.path.parent / 'runs/experiments' / run_id).exists()):
                    raise ValueError('Retry requires reconciled failed allocations and no model/run initialization')
            if reservation in jobs:
                raise ValueError('Reservation ID already exists')
            if run_id in runs:
                if (not resume and not fresh_retry) or runs[run_id]['config_digest'] != digest(config):
                    raise ValueError('Existing run requires --resume and the identical configuration')
            else:
                if resume or fresh_retry:
                    raise ValueError('Unknown run cannot resume or restart')
                full_runs = sum(r['category'] == 'full' for r in runs.values())
                if (config['category'] == 'full' and full_runs >= 3 and not (pair3b and full_runs < 10) and not (sota3b and full_runs == 9) and not (wsd20 and full_runs == 10)
                        and not (be60p20_authorization and full_runs == 3) and limits is None and not (recipe_final and full_runs == 6) and not (followup_final and full_runs == 7)):
                    raise ValueError('Three committed full runs already exist in the shared ledger')
            previous = [j for j in jobs.values() if j['run_id'] == run_id]
            if any('h200_hours' not in j for j in previous):
                raise ValueError('Reconcile all earlier jobs for this run before resubmission')
            used = sum(j['h200_hours'] for j in previous) + runs.get(run_id, {}).get('inherited_h200_hours', 0)
            if used + hours > min(64, config['max_h200_hours']) + 1e-12:
                raise ValueError(f'Allocation exceeds remaining H200-hours ({config["max_h200_hours"] - used:.6f})')
            runs.setdefault(run_id, {'category': config['category'], 'config_digest': digest(config),
                                     'max_h200_hours': config['max_h200_hours'], 'created': time.time()})
            if pair3b: runs[run_id]['formal_run_number'] = PAIR3B_RUNS[run_id]
            if sota3b: runs[run_id]['formal_run_number'] = 10
            if wsd20: runs[run_id]['formal_run_number'] = 11
            jobs[reservation] = {'run_id': run_id, 'reserved_h200_hours': hours, 'gpu_count': gpus,
                                 'requested_seconds': seconds, 'status': 'reserved'}
            if limits is not None:
                runs[run_id]['formal_run_number'] = 5 if run_id.endswith('standard') else 6
                jobs[reservation]['authorization'] = 'Explicitly approved horizon-wsd-20261014 final run5/6; two sbatches per logical run, 32 H200-hours; preserves four historical formal runs'
            if followup_final:
                runs[run_id]['formal_run_number']=8
                jobs[reservation]['authorization']='User explicitly requested parallel with formal7: one matched fresh2B with residual-half/resplit0.1/EMA off,4h32H,shared256H budget'
            if recipe_final:
                runs[run_id]['formal_run_number'] = 7
                jobs[reservation]['authorization'] = 'User approved init-recipe-20261014 plan: one fresh final2B, 4h/32H, campaign256H; formal7 exception only'
            if be60p20_authorization:
                runs[run_id]['formal_run_number'] = 4
                jobs[reservation]['authorization'] = 'User explicitly approved on 2026-10-10: fourth formal BE60P20 2B run, 8 H200/8 CPUs, max 4h/32 H200-hours, fresh weights only'

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
            run['logical_h200_hours_measured'] = run['h200_hours_measured'] + run.get('inherited_h200_hours', 0)
            run['budget_exceeded'] = run['logical_h200_hours_measured'] > run['max_h200_hours']
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
            return {**job, 'previous_h200_hours': used + data['runs'][run_id].get('inherited_h200_hours', 0),
                    'config_digest': data['runs'][run_id]['config_digest'],
                    'max_h200_hours': data['runs'][run_id]['max_h200_hours']}
