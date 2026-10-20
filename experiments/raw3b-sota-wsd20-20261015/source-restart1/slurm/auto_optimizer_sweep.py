#!/usr/bin/env python3
"""Durable metadata-only owner of calibration and conditional optimizer sweep stages."""
import argparse
import datetime as dt
import json
import importlib.metadata
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import optimizer_sweep as stage
from optimizer_sweep import SWEEP_ID, PREPARATION_ID, ENVIRONMENT
from training_state import (DEFAULT_ROOT, PROJECT, ExperimentLedger, atomic_json, digest,
                            file_digest, load_config, locked, read_json, storage_root)


def path(root):
    return stage.base(root) / 'campaign.json'


def save(root, state):
    state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(path(root), state)


def install_cron(root):
    marker = '# llama-auto-optimizer-sweep-' + SWEEP_ID
    old = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if old.returncode and not (old.returncode == 1 and 'no crontab' in old.stderr.lower()):
        raise RuntimeError('Cannot safely read existing crontab')
    if any('# llama-optimizer-sweep-' + SWEEP_ID in row for row in old.stdout.splitlines()):
        raise ValueError('A stage controller already owns this sweep; remove that owner before launch')
    command = shlex.join(['/usr/bin/python3', str(PROJECT / 'slurm/auto_optimizer_sweep.py'),
                          'tick', '--root', str(root)])
    line = f'* * * * * {command} >> {shlex.quote(str(stage.base(root) / "controller.log"))} 2>&1 {marker}'
    matches = [row for row in old.stdout.splitlines() if marker in row]
    if matches:
        if matches != [line]:
            raise ValueError('Existing campaign cron differs; refuse duplicate ownership')
        return line
    subprocess.run(['crontab', '-'], input=old.stdout.rstrip() + '\n' + line + '\n',
                   text=True, check=True)
    return line


def environment_identity():
    """Read installed package metadata only; never import torch on the login node."""
    directories = list(Path(ENVIRONMENT).glob('lib/python*/site-packages'))
    versions = {d.metadata['Name']: d.version for d in
                importlib.metadata.distributions(path=[str(p) for p in directories])}
    if versions.get('torch') not in ('2.9.1', '2.9.1+cu126'):
        raise ValueError('Installed native PyTorch 2.9.1 metadata is missing')
    return dict(packages_digest=digest(versions),
                fa3_build_sha256=file_digest(Path(ENVIRONMENT) / 'fa3-build.json'))


def initialize(root):
    if path(root).exists():
        return read_json(path(root))
    if not (Path(ENVIRONMENT) / 'bin/python').exists():
        raise ValueError('Installed native Muon/FA3 environment is missing')
    ready = root / 'preprocessed' / PREPARATION_ID / 'raw/ready.json'
    progress = read_json(ready.parent / 'progress.json')
    if sum(f['rows'] for f in progress['train_files']) < 1600:
        raise ValueError('Need published canonical training buffer for at least 100 updates')
    manifest = dict(stage='calibration', sweep_id=SWEEP_ID, allocation_seconds=600,
                    gpus=8, cpus=8, candidates=[])
    for item in stage.calibration_plan(root):
        source = PROJECT / f'configs/optimizer-sweep/calibration-{item["optimizer"]}-{item["gpus"]}.json'
        cfg = load_config(source)
        target = stage.stage_path(root, 'calibration') / 'configs' / source.name
        if target.exists() and read_json(target) != cfg:
            raise ValueError('Calibration configuration was already frozen differently')
        atomic_json(target, cfg)
        manifest['candidates'].append(dict(id=f'{item["optimizer"]}-{item["gpus"]}',
            optimizer=item['optimizer'], run_id=item['run_id'], config=str(target), config_digest=digest(cfg)))
    directory = stage.stage_path(root, 'calibration')
    if (directory / 'state.json').exists():
        raise ValueError('Orphaned calibration state requires audit; refuse to reset submission intent')
    atomic_json(directory / 'manifest.json', manifest)
    atomic_json(directory / 'state.json', dict(manifest_digest=digest(manifest), enabled=False,
        status='prepared', jobs={}, last_error=None))
    state = dict(schema_version=1, sweep_id=SWEEP_ID, enabled=False, phase='calibration',
                 status='prepared', source_hashes=stage.snapshot(), environment_identity=environment_identity(),
                 preparation_manifest_sha256=file_digest(ready), last_error=None, stop_reason=None)
    save(root, state)
    return state


def reconcile_calibration(root):
    directory = stage.stage_path(root, 'calibration')
    manifest = read_json(directory / 'manifest.json')
    state = read_json(directory / 'state.json')
    if digest(manifest) != state['manifest_digest']:
        raise ValueError('Calibration manifest changed')
    # Persist every recovered binding and receipt before proceeding to another job.
    for entry in state['jobs'].values():
        if entry['status'] == 'dispatching':
            stage.recover_entry(root, entry)
            atomic_json(directory / 'state.json', state)
        if entry['status'] == 'submitted':
            try:
                measured = ExperimentLedger(root).reconcile(entry['job_id'])
            except (ValueError, FileNotFoundError, subprocess.SubprocessError):
                continue
            entry.update(status='terminal', allocation=measured)
            atomic_json(directory / 'state.json', state)
    return manifest, state


def check_calibration_receipt(root, candidate, entry):
    """Reject a completed allocation with incomplete/failed runtime evidence before next job."""
    run = root / 'runs/experiments' / candidate['run_id']
    summary = read_json(run / 'summary.json')
    cfg = load_config(candidate['config'])
    runtime = read_json(root / 'runs/runtime-validation' / candidate['run_id'] / 'result.json')
    kernel = read_json(root / 'logs' / ('job-' + entry['job_id']) / 'fa3-check.json')
    pause = read_json(run / 'calibration-pause.json')
    if (summary['configuration'] != cfg or summary['stop_reason'] != 'max_steps' or
            summary['global_step'] != 30 or summary['total_tokens_seen'] != 30 * 131072 or
            summary['budget_tokens_charged'] != summary['total_tokens_seen'] or
            summary['initialization']['pretrained_weights_loaded'] or
            summary['initialization']['num_parameters'] != 1235814400 or
            entry['allocation']['gpu_count'] != cfg['world_size'] or
            not runtime['passed'] or runtime['optimizer'] != candidate['optimizer'] or
            runtime['world_size'] != cfg['world_size'] or runtime.get('schedulers') != ['linear', 'cosine', 'wsd'] or
            not kernel['passed'] or not all(kernel['calls'][k] > 0 for k in ('dense', 'varlen')) or
            summary['attention_kernel_calls']['dense'] <= 0 or pause['global_step'] != 1 or
            pause['cursor'] != 16 or pause['wandb_id'] != summary['wandb_id']):
        raise ValueError('Incomplete random-init/native optimizer/scheduler/FA3/resume evidence')


def set_stage_enabled(root, name, enabled):
    directory = stage.stage_path(root, name)
    with locked(directory / 'controller.lock'):
        state = read_json(directory / 'state.json')
        state['enabled'] = enabled and state['status'] != 'complete'
        atomic_json(directory / 'state.json', state)


def finish(root, state, reason=None):
    reports = []
    for name in ('initial', 'short', 'medium'):
        report_path = stage.stage_path(root, name) / 'report.json'
        if report_path.exists():
            reports.append(read_json(report_path))
    latest = reports[-1] if reports else None
    ranked = sorted((r for r in (latest['results'] if latest else []) if r['status'] == 'complete'),
                    key=lambda r: (r['nll_100'], r['id']))
    best = ranked[0] if ranked else None
    if best:
        manifest = read_json(stage.stage_path(root, latest['stage']) / 'manifest.json')
        candidate = next(c for c in manifest['candidates'] if c['id'] == best['id'])
        atomic_json(stage.base(root) / 'best-config.json', read_json(candidate['config']))
    ledger_path = ExperimentLedger(root).path
    jobs = list(read_json(ledger_path)['jobs'].values()) if ledger_path.exists() else []
    jobs = [j for j in jobs if j.get('sweep_id') == SWEEP_ID]
    summary = dict(sweep_id=SWEEP_ID, status='complete' if best else 'failed',
        reason=reason, ranking_stage=latest['stage'] if latest else None, ranking=ranked,
        reports=reports, best=best, measured_h200_hours=sum(j.get('h200_hours', 0) for j in jobs),
        outstanding_h200_hours=sum(j['reserved_h200_hours'] for j in jobs if 'h200_hours' not in j),
        full_reserved_set_evaluation=False, automatic_final_training=False)
    atomic_json(stage.base(root) / 'summary.json', summary)
    state.update(enabled=False, status=summary['status'], phase='done', stop_reason=reason)


def tick_locked(root, state):
    try:
        changed = (state['source_hashes'] != stage.snapshot() or
            state['environment_identity'] != environment_identity() or
            state['preparation_manifest_sha256'] != file_digest(root / 'preprocessed' / PREPARATION_ID / 'raw/ready.json'))
    except (OSError, ValueError):
        changed = True
    if changed:
        state.update(enabled=False, status='blocked', stop_reason='Frozen source, environment or evaluation identity changed')
    phase = state['phase']
    if phase == 'done':
        return state
    if phase == 'calibration':
        manifest, calibration_state = reconcile_calibration(root)
        entries = calibration_state['jobs']
        failed = [j for j in entries.values() if j['status'] == 'not_dispatched' or
                  (j['status'] == 'terminal' and j['allocation']['state'] != 'COMPLETED')]
        if failed:
            state.update(enabled=False, status='failed', stop_reason='Calibration allocation failed; no automatic retry')
            return state
        for candidate in manifest['candidates']:
            entry = entries.get(candidate['id'], {})
            if entry.get('status') == 'terminal':
                try:
                    check_calibration_receipt(root, candidate, entry)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    state.update(enabled=False, status='failed', stop_reason=f'Calibration evidence: {error}')
                    return state
        active = any(j['status'] in ('dispatching', 'submitted') for j in entries.values())
        complete = len(entries) == len(manifest['candidates']) and not active
        if not state['enabled']:
            return state
        if not complete:
            if not active:
                candidate = next(c for c in manifest['candidates'] if c['id'] not in entries)
                stage.dispatch(root, manifest, calibration_state, candidate)
            state['status'] = 'running'
            return state
        try:
            stage.collect_calibration(root)
            stage.prepare(root, 'initial', stage.base(root) / 'calibration.json')
        except (ValueError, KeyError, FileNotFoundError) as error:
            state.update(enabled=False, status='failed', stop_reason=f'Calibration gate: {error}')
            return state
        state['phase'] = 'initial'
        save(root, state)  # Transition durable before activating or submitting the next stage.
        phase = 'initial'
    set_stage_enabled(root, phase, state['enabled'])
    stage.tick(root, phase)  # Disabled campaigns still reconcile all receipts.
    report = read_json(stage.stage_path(root, phase) / 'report.json')
    if not state['enabled'] or not report['all_terminal']:
        return state
    if phase == 'initial':
        next_phase = 'short' if report['promotion']['survivors'] else None
    elif phase == 'short':
        next_phase = 'medium' if report['promotion']['needs_medium'] else None
    else:
        next_phase = None
    if next_phase is None:
        finish(root, state, report.get('promotion', {}).get('reason'))
        return state
    try:
        stage.prepare(root, next_phase, stage.base(root) / 'calibration.json')
    except (ValueError, KeyError, FileNotFoundError) as error:
        state.update(enabled=False, status='failed', stop_reason=f'Promotion gate: {error}')
        return state
    state.update(phase=next_phase, status='running')
    save(root, state)
    set_stage_enabled(root, next_phase, True)
    stage.tick(root, next_phase)
    return state


def tick(root):
    with locked(stage.base(root) / 'campaign.lock', blocking=False):
        state = read_json(path(root))
        try:
            state['last_error'] = None
            tick_locked(root, state)
        except Exception as error:
            state['last_error'] = f'{type(error).__name__}: {error}'
            # Submission uncertainty is recoverable on later ticks, never a retry.
            save(root, state)
            raise
        save(root, state)
        return state


def launch(root):
    with locked(stage.base(root) / 'campaign.lock'):
        state = initialize(root)
        if state['source_hashes'] != stage.snapshot() or state['environment_identity'] != environment_identity():
            raise ValueError('Frozen campaign source/environment changed')
        if state['status'] in ('complete', 'failed', 'blocked'):
            raise ValueError('Campaign is terminal or blocked; refuse to reset run identities')
        state['cron'] = install_cron(root)
        state.update(enabled=True, status='running')
        save(root, state)
    return tick(root)


def disable(root):
    with locked(stage.base(root) / 'campaign.lock'):
        state = read_json(path(root))
        state.update(enabled=False, status='disabled')
        if state['phase'] in ('initial', 'short', 'medium'):
            set_stage_enabled(root, state['phase'], False)
        save(root, state)
    return tick(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'tick', 'status', 'disable'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = storage_root(args.root)
    try:
        value = (read_json(path(root)) if args.action == 'status' else
                 {'launch': launch, 'tick': tick, 'disable': disable}[args.action](root))
    except BlockingIOError:
        return  # Another owner tick holds the advisory lock.
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
