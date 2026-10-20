"""Bounded two-arm performance campaign; metadata-only on the login node."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import (PROJECT, DEFAULT_ROOT, TERMINAL, ExperimentLedger,
                            atomic_json, file_digest, load_config, locked, read_json)
from slurm.submit_training import mark_uncertain, recover_submission

CAMPAIGN = 'performance-ab-20261010'
DIRECTORY = PROJECT / 'experiments' / CAMPAIGN
CONFIGS = PROJECT / 'configs' / CAMPAIGN
BASELINE = PROJECT / 'experiments/be60p20-lrscaled-2b-20261010/launch-source'
RUNS = {stage: CAMPAIGN+'-'+stage.lower() for stage in ('runtime', 'A', 'B')}


def freeze():
    target = DIRECTORY / 'source-b'
    if target.exists():
        return
    target.mkdir(parents=True)
    for path in PROJECT.iterdir():
        if path.is_file() and (path.suffix == '.py' or path.name.startswith('requirements')):
            shutil.copy2(path, target / path.name)
    for name in ('configs', 'slurm', 'tests', 'third_party'):
        shutil.copytree(PROJECT/name, target/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    atomic_json(DIRECTORY/'source-b-seal.json', {
        str(p.relative_to(target)): file_digest(p) for p in target.rglob('*') if p.is_file()})


def check_sources():
    parent = read_json(PROJECT/'experiments/be60p20-lrscaled-2b-20261010/launch.json')
    for name, expected in parent['source_snapshot_sha256'].items():
        if file_digest(BASELINE/name) != expected:
            raise ValueError('Original source changed: '+name)
    for name, expected in read_json(DIRECTORY/'source-b-seal.json').items():
        if file_digest(DIRECTORY/'source-b'/name) != expected:
            raise ValueError('Optimized frozen source changed: '+name)


def runtime_launch_path():
    pointer = DIRECTORY/'active-runtime.json'
    return DIRECTORY/read_json(pointer)['launch_file'] if pointer.exists() else DIRECTORY/'runtime-launch.json'


def submit(stage, dependency=None, recovery=False):
    path = DIRECTORY / (stage+('-recovery-1' if recovery else '')+'-launch.json')
    if path.exists():
        raise ValueError('Existing submission; recover/reconcile, never resubmit blindly')
    check_sources()
    config = CONFIGS / (stage+'.json')
    cfg = load_config(config)
    seconds = 1800 if stage == 'runtime' else 2700
    assert cfg['category'] == 'validation' and cfg['world_size'] == 8
    assert cfg['max_h200_hours'] == (4 if stage == 'runtime' else 6)
    source = BASELINE if stage == 'A' else DIRECTORY/'source-b'
    tag = 'performance-' + uuid.uuid4().hex
    command = ['sbatch', '--parsable', '--comment='+tag, '--time='+str(seconds//60),
               '--job-name=perf-'+stage, str(DIRECTORY/'source-b/slurm/performance-ab.sbatch'),
               RUNS[stage], stage, str(config)]
    if dependency:
        command.insert(5, '--dependency=afterok:'+dependency)
    ledger = ExperimentLedger(DEFAULT_ROOT)
    DEFAULT_ROOT.joinpath('logs').mkdir(parents=True, exist_ok=True)
    run_id = RUNS[stage]+('-recovery-1' if recovery else '')
    command[-3] = run_id
    ledger.reserve(tag, run_id, cfg, seconds, 8, False)
    state = dict(stage=stage, run_id=run_id, reservation=tag, source_root=str(source),
                 configuration=str(config), configuration_sha256=file_digest(config),
                 status='submission_uncertain', submission_command=command,
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    launcher = (DIRECTORY/'source-b/slurm/performance_entry.py' if stage == 'runtime'
                else PROJECT/'slurm/performance_entry.py')
    state.update(measurement_launcher=str(launcher), measurement_launcher_sha256=file_digest(launcher))
    atomic_json(path, state)
    with ledger.transaction() as data:
        data['jobs'][tag].update(submission_tag=tag, submission_command=command,
            submission_started_utc=state['started_utc'], allocation_kind='performance_validation')
    environment = {**os.environ, 'DATA_ROOT': str(DEFAULT_ROOT), 'TRAIN_SOURCE_ROOT': str(source),
        'PERFORMANCE_LAUNCHER': str(launcher), 'PERFORMANCE_STAGE': stage,
        'PERFORMANCE_LAUNCHER_SHA256': file_digest(launcher)}
    try:
        receipt = subprocess.run(command, env=environment, capture_output=True, text=True)
    except OSError:
        ledger.cancel_reservation(tag)
        state['status'] = 'submission_not_executed'
        atomic_json(path, state)
        raise
    state['submission_result'] = dict(stdout=receipt.stdout, stderr=receipt.stderr, returncode=receipt.returncode)
    atomic_json(path, state)
    job = receipt.stdout.strip().split(';')[0]
    if receipt.returncode or not re.fullmatch(r'\d+', job):
        mark_uncertain(ledger, tag)
        raise ValueError('Uncertain submission; retain reservation, recover before proceeding')
    with ledger.transaction() as data:
        data['jobs'][tag]['returned_job_id'] = job
    ledger.bind(tag, job)
    state.update(status='submitted', job_id=job)
    atomic_json(path, state)
    print('Submitted', stage, job, 'source=', source)
    if recovery:
        atomic_json(DIRECTORY/'active-runtime.json', {'launch_file':path.name,
            'user_authorization':'現在應該好了，重新試試',
            'previous_submission':'runtime-launch.json',
            'historical_reservation_preserved':True})
    return job


def status():
    for path in sorted(DIRECTORY.glob('*-launch.json')):
        state = read_json(path)
        stage = state['stage']+(' recovery' if 'recovery' in path.name else '')
        if not state.get('job_id'):
            if state['status'] == 'submission_not_executed':
                print(stage, state['status']); continue
            try:
                state['job_id'] = recover_submission(ExperimentLedger(DEFAULT_ROOT), state['reservation'])
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                state['last_recovery_error'] = str(error)
                state['last_recovery_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
                atomic_json(path, state)
                print(stage, 'SUBMISSION_UNCERTAIN:', error)
                continue
        if state['status'] != 'reconciled':
            output = subprocess.run(['sacct','-X','-j',state['job_id'],'--noheader','--parsable2',
                '--format=JobIDRaw,State%30'],capture_output=True,text=True,check=True,timeout=20).stdout
            rows = [row.split('|') for row in output.splitlines()]
            state['slurm_state'] = next((r[1].split()[0].rstrip('+') for r in rows if r[0]==state['job_id']), 'UNKNOWN')
            if state['slurm_state'] in TERMINAL:
                state['accounting'] = ExperimentLedger(DEFAULT_ROOT).reconcile(state['job_id'])
                state['status'] = 'reconciled'
            atomic_json(path, state)
        print(stage, state['job_id'], state.get('slurm_state'), state['status'])


def benchmarks():
    status()
    runtime = read_json(runtime_launch_path())
    result = read_json(DEFAULT_ROOT/'runs/runtime-validation'/runtime['run_id']/'performance-result.json')
    if runtime.get('slurm_state') != 'COMPLETED' or not result['passed']:
        raise ValueError('Require completed/reconciled correctness allocation')
    cfg = read_json(CONFIGS/'B.json')
    cfg['performance']['compile_mode'] = result['compile_mode']
    atomic_json(CONFIGS/'B.json', cfg)
    atomic_json(DIRECTORY/'correctness-results.json', result)
    a = submit('A')
    submit('B', dependency=a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('runtime','retry-runtime','benchmarks','status'))
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with locked(DIRECTORY/'.controller.lock', blocking=False):
        if args.action == 'runtime':
            freeze(); submit('runtime')
        elif args.action == 'retry-runtime':
            # Explicit user-triggered retry. Never reached by a scheduler or
            # automatic controller; the old uncertain reservation is retained.
            original = read_json(DIRECTORY/'runtime-launch.json')
            if original.get('job_id') or original['status'] != 'submission_uncertain':
                raise ValueError('Recover/reconcile the original submission first')
            try:
                recovered = recover_submission(ExperimentLedger(DEFAULT_ROOT),original['reservation'])
            except ValueError as error:
                if 'found 0 matching jobs' not in str(error): raise
            else:
                raise ValueError('Original job recovered: '+recovered+'; do not duplicate')
            submit('runtime', recovery=True)
        elif args.action == 'benchmarks': benchmarks()
        else: status()
