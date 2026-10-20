#!/usr/bin/env python3
"""Explicit one-allocation launch, integrity check and terminal reconciliation."""
import argparse
import datetime
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO/'experiments/kd3b-online-pipeline-20261019'
SOURCE = LIVE/'source'
RUN = 'kd3b-online-pipeline-20261019-seed42'
SCRIPT = REPO/'slurm/kd3b-online-pipeline.sbatch'
sys.path.insert(0, str(SOURCE))
from training_state import (DEFAULT_ROOT, ExperimentLedger, atomic_json,
                            digest, file_digest, load_config, locked, read_json)


def inventory():
    files = [p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [LIVE/'config.json', LIVE/'authorization.json', LIVE/'protocol.md',
              LIVE/'teacher-receipt.json', SCRIPT, Path(__file__).resolve()]
    return {str(p.relative_to(REPO)): file_digest(p) for p in sorted(files)}


def check():
    if inventory() != read_json(LIVE/'execution-seal.json')['sha256']:
        raise ValueError('Sealed execution source/configuration changed')
    cfg = load_config(LIVE/'config.json')
    auth = read_json(LIVE/'authorization.json')
    if auth['run_id'] != RUN or auth['config_digest'] != digest(cfg):
        raise ValueError('Authorization/config mismatch')
    if cfg['seed'] != 42 or cfg['world_size'] != 8 or cfg['max_h200_hours'] != 16:
        raise ValueError('Unexpected training geometry/budget')
    return cfg


def budget(data):
    prefixes = ('methods-parallel-20261019', 'kd-acceleration-', 'kd-pipeline-', 'kd3b-online-pipeline-')
    jobs = {k: j for k, j in data['jobs'].items() if j['run_id'].startswith(prefixes)}
    used = sum(j.get('h200_hours', j['reserved_h200_hours']) for j in jobs.values())
    return dict(study_limit_h200_hours=128, prior_charged_h200_hours=used,
                new_reserved_h200_hours=16, maximum_after_submission=used+16,
                preserved_job_ids=sorted(jobs))


def submit():
    cfg = check()
    ledger = ExperimentLedger(DEFAULT_ROOT)
    (DEFAULT_ROOT/'logs').mkdir(parents=True, exist_ok=True)
    record = LIVE/'submission.json'
    with locked(LIVE/'.submission.lock', blocking=False):
        existing = read_json(ledger.path)
        if record.exists() or RUN in existing['runs'] or (DEFAULT_ROOT/'runs/experiments'/RUN).exists():
            raise ValueError('Existing identity: inspect/recover this submission, never resubmit')
        costs = budget(existing)
        if costs['maximum_after_submission'] > 128:
            raise ValueError('Study budget exceeded')
        tag = 'kd3b-'+uuid.uuid4().hex
        state = dict(run_id=RUN, reservation=tag, submission_tag=tag,
            submission_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            config_digest=digest(cfg), seal_sha256=file_digest(LIVE/'execution-seal.json'),
            requested_seconds=7200, gpu_count=8, cpu_count=8,
            budget=costs, phase='intent', training_checkpoint_load=False)
        atomic_json(record, state)
        ledger.reserve(tag, RUN, cfg, 7200, 8, False)
        state['phase'] = 'reserved'
        atomic_json(record, state)
        command = ['sbatch', '--parsable', '--comment='+tag, str(SCRIPT), RUN]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            job = result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+', job):
                raise RuntimeError('Uncertain submission: '+result.stderr.strip())
            state.update(returned_job_id=job, phase='accepted')
            atomic_json(record, state)
            ledger.bind(tag, job)
            state['phase'] = 'submitted'
            atomic_json(record, state)
        except Exception as error:
            state.update(error=str(error), recovery_required=True)
            atomic_json(record, state)
            raise RuntimeError('Reservation retained; find/bind original tag without another sbatch') from error
        print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal', 'check', 'submit', 'reconcile'])
    args = parser.parse_args()
    if args.action == 'seal':
        atomic_json(LIVE/'execution-seal.json', dict(sha256=inventory()))
        print('Execution sealed; no allocation submitted')
    elif args.action == 'check':
        cfg = check()
        print(json.dumps(dict(run_id=RUN, config_digest=digest(cfg), checked=True)))
    elif args.action == 'submit':
        submit()
    else:
        job = read_json(LIVE/'submission.json')['returned_job_id']
        result = ExperimentLedger(DEFAULT_ROOT).reconcile(job)
        atomic_json(LIVE/'accounting.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
