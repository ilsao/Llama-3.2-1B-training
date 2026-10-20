#!/usr/bin/env python3
"""Two single-allocation validation roles, preserving all study costs."""
import argparse
import datetime
import re
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE.parent/'kd3b-online-pipeline-20261019/source'
sys.path.insert(0, str(SOURCE))
from training_state import DEFAULT_ROOT, ExperimentLedger, atomic_json, digest, file_digest, load_config, locked, read_json

BASE = 'kd3b-io8-validation-20261019'
SCRIPT = REPO/'slurm/kd3b-io8-repair.sbatch'
RECORD = HERE/'repair-submission.json'


def inventory():
    paths = [p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in (HERE/'source-v2').iterdir() if p.is_file()]
    paths += [HERE/'repair-plan.json', Path(__file__).resolve(), SCRIPT]
    return {str(p.relative_to(REPO)): file_digest(p) for p in sorted(paths)}


def check():
    assert inventory() == read_json(HERE/'repair-seal.json')['sha256'], 'Execution source changed'
    cfg = load_config(HERE/'source-v2/validation-config.json')
    assert cfg['world_size'] == 8 and cfg['category'] == 'validation'
    return cfg


def initialize():
    cfg = check(); ledger = ExperimentLedger(DEFAULT_ROOT)
    data = read_json(ledger.path)
    if RECORD.exists():
        raise ValueError('Repair already attempted; no further replacement')
    approval = read_json(HERE/'repair-approval.json')
    if approval.get('status') != 'approved' or not approval.get('user_approval_text') or approval.get('failed_jobs') != ['575177', '575178']:
        raise ValueError('Explicit one-time replacement approval required')
    original = read_json(HERE/'validation-submission.json')
    for role, item in original['roles'].items():
        prior = data['jobs'][item['job_id']]
        if prior.get('state') != 'FAILED' or 'h200_hours' not in prior:
            raise ValueError('Reconcile original failure before resource-only continuation')
        if data['runs'][item['run_id']]['config_digest'] != digest(cfg):
            raise ValueError('Validation configuration changed')
    prefixes = ('methods-parallel-20261019', 'kd-acceleration-', 'kd-pipeline-', 'kd3b-online-pipeline-', 'kd3b-io-validation-', 'kd3b-io8-validation-')
    jobs = {k: j for k, j in data['jobs'].items() if j['run_id'].startswith(prefixes)}
    used = sum(j.get('h200_hours', j['reserved_h200_hours']) for j in jobs.values())
    assert used+16*1080/3600 <= 128
    state = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_digest=digest(cfg), budget=dict(study_cap=128, prior_h200_hours=used,
        new_max_h200_hours=16*1080/3600, maximum_after_submission=used+16*1080/3600,
        preserved_jobs=sorted(jobs)), roles={})
    for role in ('producer', 'consumer'):
        state['roles'][role] = dict(run_id=BASE+'-'+role, reservation='kd3bio8repair-'+uuid.uuid4().hex, phase='intent')
    atomic_json(RECORD, state)
    for role, item in state['roles'].items():
        ledger.reserve(item['reservation'], item['run_id'], cfg, 1080, 8, True)
        item['phase'] = 'reserved'; atomic_json(RECORD, state)
    print('Reserved two 8-H200/8-CPU validation jobs, each <=18 minutes; same logical IDs, resource continuation only; no checkpoint load')


def submit(role):
    check(); state = read_json(RECORD); item = state['roles'][role]
    if item['phase'] != 'reserved':
        raise ValueError('Only an unsubmitted reservation can be submitted; no automatic retry')
    options = ['--exclude=25a-hgpn001']
    if role == 'consumer':
        producer = state['roles']['producer']
        assert producer['phase'] == 'submitted'
        result = subprocess.run(['squeue', '-h', '-j', producer['job_id'], '-o', '%T|%N'],
                                text=True, capture_output=True, check=True, timeout=15)
        parts = result.stdout.strip().split('|')
        if len(parts) != 2 or parts[0] != 'RUNNING' or not re.fullmatch(r'[a-zA-Z0-9-]+', parts[1]):
            raise ValueError('Wait until producer is allocated before choosing a different node')
        options = ['--exclude=25a-hgpn001,'+parts[1]]
        item['excluded_producer_node'] = parts[1]
    (DEFAULT_ROOT/'logs').mkdir(parents=True, exist_ok=True)
    item['phase'] = 'submitting'; atomic_json(RECORD, state)
    try:
        result = subprocess.run(['sbatch', '--parsable', '--comment='+item['reservation'],
            '--job-name=kd3b-io8-repair-'+role, *options, str(SCRIPT), role],
            capture_output=True, text=True, timeout=45)
        job = result.stdout.strip().split(';')[0]
        if result.returncode or not re.fullmatch(r'\d+', job):
            raise RuntimeError('Uncertain submission: '+result.stderr.strip())
        item.update(job_id=job, phase='accepted'); atomic_json(RECORD, state)
        ExperimentLedger(DEFAULT_ROOT).bind(item['reservation'], job)
        item['phase'] = 'submitted'; atomic_json(RECORD, state)
        print(role, job)
    except Exception as exc:
        item.update(error=repr(exc), recovery_required=True); atomic_json(RECORD, state)
        raise RuntimeError('Reservation retained; recover original tag, never resubmit') from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['seal', 'check', 'reserve', 'producer', 'consumer', 'reconcile'])
    args = parser.parse_args()
    if args.action == 'seal':
        if RECORD.exists():
            raise ValueError('Never reseal a submitted validation')
        atomic_json(HERE/'repair-seal.json', dict(sha256=inventory()))
    elif args.action == 'check':
        print('Validated', digest(check()))
    elif args.action == 'reserve':
        with locked(HERE/'.submit.lock', blocking=False): initialize()
    elif args.action in ('producer', 'consumer'):
        with locked(HERE/'.submit.lock', blocking=False): submit(args.action)
    else:
        state = read_json(RECORD)
        results = {role: ExperimentLedger(DEFAULT_ROOT).reconcile(item['job_id']) for role, item in state['roles'].items()}
        atomic_json(HERE/'repair-accounting.json', results)
        print(results)


if __name__ == '__main__':
    main()
