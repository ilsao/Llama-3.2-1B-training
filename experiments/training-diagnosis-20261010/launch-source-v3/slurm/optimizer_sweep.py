#!/usr/bin/env python3
"""Stage-scoped, metadata-only sweep controller. Explicit activation; no GPU work here."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer_sweep import (SWEEP_ID, PREPARATION_ID, ENVIRONMENT, STAGES, candidates,
    freeze_steps, training_config, prune_initial, select_finalists, result_from_run)
from training_state import (DEFAULT_ROOT, PROJECT, ExperimentLedger, atomic_json, digest,
    file_digest, identifier, load_config, locked, read_json, storage_root)
from slurm.submit_training import recover_submission, mark_uncertain

SOURCE_FILES = ('train.py', 'training_state.py', 'optimizers.py', 'optimizer_sweep.py',
    'attention_backend.py', 'dataset.py', 'raw_baseline.py', 'preparation_parallel.py',
    'requirements-training.txt', 'slurm/optimizer_sweep.py', 'slurm/auto_optimizer_sweep.py', 'slurm/sweep.sbatch',
    'slurm/sweep-calibration.sbatch', 'slurm/training-common.sh', 'slurm/common.sh',
    'slurm/training_entry.py', 'slurm/submit_training.py', 'slurm/submission.py', 'tests/training_runtime.py', 'tests/fa3_runtime.py')


def snapshot():
    names = SOURCE_FILES + ('configs/optimizer-sweep.json', 'configs/train-sweep-adamw.json') + tuple(
        str(p.relative_to(PROJECT)) for p in sorted((PROJECT / 'configs/optimizer-sweep').glob('*.json')))
    return {name: file_digest(PROJECT / name) for name in names}


def base(root):
    return root / 'runs/optimizer-sweep' / SWEEP_ID


def stage_path(root, stage):
    return base(root) / stage


def calibration_plan(root):
    return [{'optimizer': optimizer, 'gpus': gpus,
        'run_id': f'sweep-calibration-{optimizer}-{gpus}-20261008',
        'command': [sys.executable, str(PROJECT / 'slurm/submit_training.py'), 'sweep-calibration',
            f'sweep-calibration-{optimizer}-{gpus}-20261008', '--prep-id', PREPARATION_ID,
            '--config', str(PROJECT / f'configs/optimizer-sweep/calibration-{optimizer}-{gpus}.json'),
            '--minutes', '10', '--root', str(root)]}
        for optimizer in ('adamw', 'muon') for gpus in (4, 8)]


def collect_calibration(root):
    """Validate full-size and tiny runtime evidence, then persist measured timing contracts."""
    samples, signatures, preparations, environments = [], set(), set(), set()
    for item in calibration_plan(root):
        run_id = item['run_id']; path = root / 'runs/experiments' / run_id
        summary = read_json(path / 'summary.json'); cfg = summary['configuration']
        if (cfg != load_config(PROJECT / f'configs/optimizer-sweep/calibration-{item["optimizer"]}-{item["gpus"]}.json') or
                summary['stop_reason'] != 'max_steps' or summary['global_step'] != 30 or
                summary['total_tokens_seen'] != 30 * 131072 or summary['budget_tokens_charged'] != 30 * 131072 or
                summary['initialization']['pretrained_weights_loaded'] or
                summary['initialization']['num_parameters'] != 1235814400):
            raise ValueError('Need complete random-init native full-size calibration at matched global batch')
        job_id = summary['slurm_job_id']
        measured = ExperimentLedger(root).reconcile(job_id)
        if measured['state'] != 'COMPLETED' or measured['gpu_count'] != item['gpus']:
            raise ValueError('Calibration allocation did not complete with verified H200 geometry')
        runtime = read_json(root / 'runs/runtime-validation' / run_id / 'result.json')
        if not runtime['passed'] or runtime['optimizer'] != item['optimizer'] or runtime['world_size'] != item['gpus'] or runtime.get('schedulers') != ['linear', 'cosine', 'wsd']:
            raise ValueError('Native optimizer/scheduler exact replay checks have not passed')
        kernel = read_json(root / 'logs' / f'job-{job_id}' / 'fa3-check.json')
        if not kernel['passed'] or not all(kernel['calls'][key] > 0 for key in ('dense', 'varlen')) or summary['attention_kernel_calls']['dense'] <= 0:
            raise ValueError('Need FA3 forward/backward, padding and 8192 kernel evidence')
        identity = read_json(path / 'execution-identity.json')
        for name, checksum in identity['source_hashes'].items():
            if name == 'fa3-build.json':
                current = file_digest(Path(ENVIRONMENT) / name)
            else:
                current = file_digest(PROJECT / name)
            if checksum != current:
                raise ValueError('Calibration source/build changed; do not reuse stale timing')
        if identity['environment'].get('torch') not in ('2.9.1', '2.9.1+cu126'):
            raise ValueError('Calibration must use native PyTorch 2.9.1')
        environments.add(digest(identity['environment']))
        signatures.add(summary['initialization']['sample_signature'])
        preparations.add(summary['preparation_fingerprint'])
        pause = read_json(path / 'calibration-pause.json')
        if pause['global_step'] != 1 or pause['cursor'] != 16 or pause['wandb_id'] != summary['wandb_id']:
            raise ValueError('Need the faithful full-size step-1 continuation receipt')
        with (path / 'timing-steps.jsonl').open() as stream:
            timings = [json.loads(line) for line in stream]
        steady = [r['update_seconds'] for r in timings if r['step'] > 5]
        if len(steady) != 25 or any(not math.isfinite(x) or x <= 0 for x in steady):
            raise ValueError('Need five excluded warmup updates and 25 finite measured steady updates')
        samples.append(dict(optimizer=item['optimizer'], gpus=item['gpus'], run_id=run_id,
            job_id=job_id, update_seconds_p95=sorted(steady)[-2],
            initialization_seconds=summary['timing_totals']['initialization_max_seconds'],
            evaluation_seconds=summary['evaluation_seconds'], checkpoint_seconds=summary['checkpoint_seconds'],
            h200_hours=measured['h200_hours']))
    if len(signatures) != 1 or len(preparations) != 1 or len(environments) != 1:
        raise ValueError('Calibration changed initialization/data/environment between optimizers or geometries')
    value = dict(schema_version=1, sweep_id=SWEEP_ID, passed=True, samples=samples,
        slowest_8gpu_update_seconds=max(r['update_seconds_p95'] for r in samples if r['gpus'] == 8),
        initialization_seconds=max(r['initialization_seconds'] for r in samples) + 15,
        evaluation_seconds=max(r['evaluation_seconds'] for r in samples),
        checkpoint_seconds=max(r['checkpoint_seconds'] for r in samples),
        source_hashes=snapshot(), environment_digest=environments.pop(),
        initialization_signature=signatures.pop(), preparation_fingerprint=preparations.pop(),
        preparation_manifest_sha256=file_digest(root / 'preprocessed' / PREPARATION_ID / 'raw/ready.json'),
        environment=ENVIRONMENT, measured_h200_hours=sum(r['h200_hours'] for r in samples),
        scaling={optimizer: next(r['update_seconds_p95'] for r in samples if r['optimizer'] == optimizer and r['gpus'] == 4) /
                 next(r['update_seconds_p95'] for r in samples if r['optimizer'] == optimizer and r['gpus'] == 8)
                 for optimizer in ('adamw', 'muon')},
        initialization_note='max single-launch initialization plus 15 seconds for common Slurm startup')
    target = base(root) / 'calibration.json'
    if target.exists() and read_json(target) != value:
        raise ValueError('Existing calibration is immutable; do not replace evidence for frozen stages')
    atomic_json(target, value)
    return value


def prepare(root, stage, calibration_path):
    directory = stage_path(root, stage)
    calibration = read_json(calibration_path)
    if (calibration.get('passed') is not True or calibration['sweep_id'] != SWEEP_ID or
            calibration['source_hashes'] != snapshot() or calibration['environment'] != ENVIRONMENT or
            calibration['preparation_manifest_sha256'] != file_digest(root / 'preprocessed' / PREPARATION_ID / 'raw/ready.json')):
        raise ValueError('Require successful current native calibration and unchanged canonical evaluation identity')
    selected = candidates(); parents = {}
    if stage != 'initial':
        previous = 'initial' if stage == 'short' else 'short'
        report = read_json(stage_path(root, previous) / 'report.json')
        if not report['all_terminal']:
            raise ValueError('Wait for the preceding stage to finish and reconcile')
        if stage == 'short':
            ids = report['promotion']['survivors']
        else:
            if not report['promotion']['needs_medium']:
                raise ValueError('Medium stage is unnecessary according to the frozen ranking criteria')
            ids = report['promotion']['finalists']
        parents = {r['id']: r['run_id'] for r in report['results']}
        selected = [c for c in selected if c['id'] in ids]
    if not selected:
        raise ValueError('No complete candidates eligible for promotion')
    progress = read_json(root / 'preprocessed' / PREPARATION_ID / 'raw/progress.json')
    rows = sum(item['rows'] for item in progress['train_files'])
    horizon = freeze_steps(stage, calibration, rows)
    manifest = dict(sweep_id=SWEEP_ID, stage=stage, preparation_id=PREPARATION_ID,
        environment=ENVIRONMENT, calibration_sha256=file_digest(calibration_path),
        source_hashes=calibration['source_hashes'], initialization_signature=calibration['initialization_signature'],
        preparation_fingerprint=calibration['preparation_fingerprint'],
        preparation_manifest_sha256=calibration['preparation_manifest_sha256'],
        horizon=horizon, allocation_seconds=STAGES[stage] * 60, gpus=8, cpus=8,
        candidates=[], reserved_stage_h200_hours=len(selected) * STAGES[stage] * 8 / 60)
    for candidate in selected:
        run_id = identifier(f'os-20261008-{stage}-{candidate["id"]}')
        cfg = training_config(candidate, stage, horizon['steps'], manifest['calibration_sha256'],
                              [parents[candidate['id']]] if candidate['id'] in parents else [])
        cfg.update(stop_margin_seconds=horizon['stop_margin_seconds'],
                   sweep_environment_digest=calibration['environment_digest'])
        target = directory / 'configs' / (candidate['id'] + '.json')
        if target.exists() and read_json(target) != cfg:
            raise ValueError('A frozen stage configuration already exists; refuse to change its horizon')
        atomic_json(target, cfg); load_config(target)
        manifest['candidates'].append({**candidate, 'run_id': run_id, 'config': str(target), 'config_digest': digest(cfg)})
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError('Stage manifest already frozen')
    atomic_json(manifest_path, manifest)
    state_path = directory / 'state.json'
    if not state_path.exists():
        atomic_json(state_path, dict(manifest_digest=digest(manifest), enabled=False, status='prepared', jobs={}, last_error=None))
    return manifest


def dispatch(root, manifest, state, candidate):
    """Persist controller intent AND shared reservation before sbatch. Never retry uncertainty."""
    ledger = ExperimentLedger(root)
    cfg = load_config(candidate['config'])
    if digest(cfg) != candidate['config_digest']:
        raise ValueError('Frozen training configuration changed')
    seconds = manifest['allocation_seconds']
    gpus = cfg['world_size']
    if seconds > 7200 or seconds * gpus / 3600 > 16:
        raise ValueError('Per-sbatch cap exceeded')
    tag = 'sweep-' + uuid.uuid4().hex
    entry = dict(status='dispatching', reservation=tag, run_id=candidate['run_id'])
    state['jobs'][candidate['id']] = entry
    state_path = stage_path(root, manifest['stage']) / 'state.json'
    atomic_json(state_path, state)
    ledger.reserve(tag, candidate['run_id'], cfg, seconds, gpus, False)
    root.joinpath('logs').mkdir(parents=True, exist_ok=True)
    command = ['sbatch', '--parsable', '--cpus-per-task=8', f'--gpus-per-node={gpus}',
        f'--time={seconds // 60}', f'--signal=B:USR1@{cfg["stop_margin_seconds"]}',
        '--exclude=25a-hgpn062', f'--job-name=os-{manifest["stage"]}', f'--comment={tag}',
        f'--output={root}/logs/%x-%j.out', f'--error={root}/logs/%x-%j.err',
        str(PROJECT / ('slurm/sweep-calibration.sbatch' if manifest['stage'] == 'calibration' else 'slurm/sweep.sbatch')),
        candidate['run_id'], *(['sweep-calibration'] if manifest['stage'] == 'calibration' else []),
        PREPARATION_ID, candidate['config']]
    with ledger.transaction() as data:
        data['jobs'][tag].update(status='submission_uncertain', submission_tag=tag,
            submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), submission_command=command,
            source_root=str(PROJECT), sweep_id=SWEEP_ID, sweep_stage=manifest['stage'])
    environment = {**os.environ, 'DATA_ROOT': str(root), 'TRAIN_SOURCE_ROOT': str(PROJECT), 'TRAIN_ENV_PREFIX': ENVIRONMENT}
    environment.pop('CAMPAIGN_END_EPOCH', None)
    if manifest['stage'] == 'calibration':
        environment['RUNTIME_OPTIMIZER'] = candidate['optimizer']
    try:
        response = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    except OSError:
        ledger.cancel_reservation(tag)
        entry.update(status='not_dispatched', error='sbatch could not execute')
        atomic_json(state_path, state); raise
    if response.returncode or not re.fullmatch(r'\d+(;[^\s]+)?', response.stdout.strip()):
        mark_uncertain(ledger, tag)
        raise RuntimeError('Sbatch outcome uncertain; retain reservation and recover its unique tag')
    job_id = response.stdout.strip().split(';')[0]
    with ledger.transaction() as data:
        data['jobs'][tag]['returned_job_id'] = job_id
    ledger.bind(tag, job_id)
    entry.update(job_id=job_id, status='submitted')
    atomic_json(state_path, state)


def recover_entry(root, entry):
    ledger = ExperimentLedger(root)
    data = read_json(ledger.path)
    tag = entry['reservation']
    bound = [j for j in data['jobs'].values() if j.get('submission_tag') == tag and j.get('job_id')]
    if len(bound) == 1:
        job_id = bound[0]['job_id']
    elif tag in data['jobs']:
        job_id = recover_submission(ledger, tag)
    else:
        raise RuntimeError('Controller intent has no matching reservation; audit before any retry')
    entry.update(job_id=job_id, status='submitted')


def report(root, manifest, state):
    all_terminal = len(state['jobs']) == len(manifest['candidates']) and all(
        j['status'] in ('terminal', 'not_dispatched') for j in state['jobs'].values())
    results = []
    for candidate in manifest['candidates']:
        entry = state['jobs'].get(candidate['id'], {})
        if entry.get('status') == 'terminal':
            cfg = load_config(candidate['config'])
            result = result_from_run(root, candidate, candidate['run_id'], cfg, entry['allocation'])
            if result['status'] == 'complete' and (
                    result['initialization_signature'] != manifest['initialization_signature'] or
                    result['preparation_fingerprint'] != manifest['preparation_fingerprint']):
                raise ValueError('Candidate initialization/data differs from native calibration')
            results.append(result)
        elif entry.get('status') == 'not_dispatched':
            results.append(dict(id=candidate['id'], recipe=candidate['recipe'], optimizer=candidate['optimizer'],
                                run_id=candidate['run_id'], status='not_dispatched'))
    if len({r['eval_tokens_100'] for r in results if r['status'] == 'complete'}) > 1:
        raise ValueError('Evaluation prediction-token counts differ between candidates')
    ledger_path = ExperimentLedger(root).path
    ledger_jobs = read_json(ledger_path)['jobs'].values() if ledger_path.exists() else []
    run_ids = {c['run_id'] for c in manifest['candidates']}
    allocations = [j for j in ledger_jobs if j['run_id'] in run_ids]
    measured_hours = sum(j.get('h200_hours', 0) for j in allocations)
    outstanding_hours = sum(j['reserved_h200_hours'] for j in allocations if 'h200_hours' not in j)
    all_terminal = all_terminal and outstanding_hours == 0
    promotion = (prune_initial(results) if manifest['stage'] == 'initial' else select_finalists(results)) if all_terminal else None
    value = dict(sweep_id=SWEEP_ID, stage=manifest['stage'], all_terminal=all_terminal,
        enabled=state['enabled'], results=results, promotion=promotion,
        ranking=[r['id'] for r in sorted((r for r in results if r['status'] == 'complete'),
                                       key=lambda r: (r['nll_100'], r['id']))],
        planned_training_tokens_per_candidate=manifest['horizon']['tokens'],
        measured_h200_hours=measured_hours, outstanding_reserved_h200_hours=outstanding_hours)
    atomic_json(stage_path(root, manifest['stage']) / 'report.json', value)
    return value


def tick(root, stage):
    directory = stage_path(root, stage)
    with locked(directory / 'controller.lock', blocking=False):
        manifest = read_json(directory / 'manifest.json'); state = read_json(directory / 'state.json')
        if digest(manifest) != state['manifest_digest']:
            raise ValueError('Frozen stage manifest changed')
        try:
            state['last_error'] = None
            for entry in state['jobs'].values():
                if entry['status'] == 'dispatching':
                    recover_entry(root, entry)
                    atomic_json(directory / 'state.json', state)
                if entry['status'] == 'submitted':
                    # Active or temporarily unavailable sacct remains fully reserved.
                    try:
                        measured = ExperimentLedger(root).reconcile(entry['job_id'])
                    except (ValueError, FileNotFoundError, subprocess.SubprocessError):
                        continue
                    entry.update(status='terminal', allocation=measured)
                    atomic_json(directory / 'state.json', state)
            value = report(root, manifest, state)
            if value['all_terminal']:
                state.update(enabled=False, status='complete')
                value['enabled'] = False
                atomic_json(directory / 'report.json', value)
            elif state['enabled']:
                if manifest['source_hashes'] != snapshot():
                    raise ValueError('Source changed; reconciliation continues but new submissions stop')
                if manifest['preparation_manifest_sha256'] != file_digest(root / 'preprocessed' / PREPARATION_ID / 'raw/ready.json'):
                    raise ValueError('Canonical evaluation identity changed')
                if (value['outstanding_reserved_h200_hours'] == 0 and
                        not any(j['status'] in ('submitted', 'dispatching') for j in state['jobs'].values())):
                    candidate = next(c for c in manifest['candidates'] if c['id'] not in state['jobs'])
                    dispatch(root, manifest, state, candidate)
                    state['status'] = 'running'
        except Exception as error:
            state['last_error'] = f'{type(error).__name__}: {error}'
            atomic_json(directory / 'state.json', state)
            raise
        atomic_json(directory / 'state.json', state)
        return state


def install_cron(root, stage):
    """Opt-in metadata controller only; preserve every other existing cron entry."""
    directory = stage_path(root, stage)
    if not (directory / 'manifest.json').exists():
        raise ValueError('Prepare and freeze the stage before installing its controller')
    marker = f'# llama-optimizer-sweep-{SWEEP_ID}-{stage}'
    existing = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if existing.returncode and not (existing.returncode == 1 and 'no crontab' in existing.stderr.lower()):
        raise RuntimeError('Could not read crontab; leave existing schedules untouched')
    command = shlex.join(['/usr/bin/python3', str(PROJECT / 'slurm/optimizer_sweep.py'), 'tick',
                          '--root', str(root), '--stage', stage])
    line = f'* * * * * {command} >> {shlex.quote(str(directory / "controller.log"))} 2>&1 {marker}'
    if any(marker in row for row in existing.stdout.splitlines()):
        raise ValueError('Controller cron already exists; do not duplicate it')
    subprocess.run(['crontab', '-'], input=existing.stdout.rstrip() + '\n' + line + '\n', text=True, check=True)
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'calibration-plan', 'collect-calibration', 'prepare', 'enable', 'disable', 'tick', 'status', 'install-cron'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--stage', choices=tuple(STAGES), default='initial')
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args(); root = storage_root(args.root)
    if args.action == 'plan':
        value = read_json(PROJECT / 'configs/optimizer-sweep.json')
    elif args.action == 'calibration-plan':
        value = calibration_plan(root)
        for item in value:
            print('TRAIN_ENV_PREFIX=' + shlex.quote(ENVIRONMENT) + ' ' + shlex.join(item['command']))
        return
    elif args.action == 'collect-calibration':
        value = collect_calibration(root)
    elif args.action == 'prepare':
        with locked(base(root) / 'prepare.lock'):
            value = prepare(root, args.stage, args.calibration or base(root) / 'calibration.json')
    elif args.action in ('enable', 'disable'):
        directory = stage_path(root, args.stage)
        with locked(directory / 'controller.lock'):
            manifest = read_json(directory / 'manifest.json'); value = read_json(directory / 'state.json')
            if digest(manifest) != value['manifest_digest']:
                raise ValueError('Stage manifest changed')
            if args.action == 'enable' and manifest['source_hashes'] != snapshot():
                raise ValueError('Frozen source changed')
            if args.action == 'enable' and value['status'] == 'complete':
                raise ValueError('Completed stage cannot be enabled again')
            value['enabled'] = args.action == 'enable'
            atomic_json(directory / 'state.json', value)
    elif args.action == 'tick':
        value = tick(root, args.stage)
    elif args.action == 'install-cron':
        value = install_cron(root, args.stage)
    else:
        value = read_json(stage_path(root, args.stage) / 'state.json')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
