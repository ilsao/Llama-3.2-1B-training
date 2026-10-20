#!/usr/bin/env python3
"""Submit one bounded baseline; metadata-only status also reconciles its costs."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import (DEFAULT_ROOT, PROJECT, ExperimentLedger, atomic_json,
                            file_digest, load_config, locked, read_json)
from slurm.submit_training import mark_uncertain, recover_submission
from slurm.auto_optimizer_sweep import environment_identity

RUN_ID = 'baseline-muon-wsd-raw-20261009'
PREP_ID = 'dolma-raw-baseline-20261008'
CONFIG = PROJECT / 'configs/baselines/muon-wsd-raw-20261009.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['submit', 'status'])
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    root = DEFAULT_ROOT
    ledger = ExperimentLedger(root)
    directory = root / 'runs/baselines' / RUN_ID
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / 'launch.json'
    with locked(directory / '.submission.lock', blocking=False):
        if args.action == 'status':
            state = read_json(record)
            if state.get('status') == 'reconciled':
                return
            if not state.get('job_id'):
                state['job_id'] = recover_submission(ledger, state['reservation'])
                atomic_json(record, state)
            result = subprocess.run(['sacct', '-X', '-j', state['job_id'], '--noheader',
                                     '--parsable2', '--format=JobIDRaw,State'],
                                    capture_output=True, text=True, check=True, timeout=30)
            terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}
            rows = [r.split('|') for r in result.stdout.splitlines()]
            status = next((r[1].split()[0].rstrip('+') for r in rows if r[0] == state['job_id']), '')
            if status in terminal:
                state['accounting'] = ledger.reconcile(state['job_id'])
                state['status'] = 'reconciled'
                evaluation = root / 'runs/experiments' / RUN_ID / 'full-evaluation.json'
                state['full_evaluation'] = read_json(evaluation) if evaluation.exists() else None
                state['full_ppl_below_20'] = bool(state['full_evaluation'] and
                    state['full_evaluation']['eval/perplexity'] < 20)
                atomic_json(record, state)
            print(json.dumps({'job_id': state['job_id'], 'slurm_status': status,
                              'record_status': state['status']}))
            return
        if record.exists() and not args.test_only:
            raise SystemExit(f'Baseline already reserved/submitted; use status to recover: {record}')
        cfg = load_config(CONFIG)
        assert cfg['world_size'] == 8 and cfg['max_h200_hours'] == 16
        assert cfg['max_training_tokens'] <= 6_000_000_000
        assert cfg['max_steps'] * 16 * 8192 <= cfg['max_training_tokens']
        ready = root / 'preprocessed' / PREP_ID / 'raw/ready.json'
        if not ready.is_file():
            raise SystemExit('Canonical reserved-set evaluation cache is not ready')
        progress = read_json(ready.parent / 'progress.json')
        if progress['training_tokens'] < cfg['max_steps'] * 16 * 8192:
            raise SystemExit('Existing raw buffer does not cover the fixed training horizon')
        environment = environment_identity()
        if environment['packages_digest'] != cfg['sweep_environment_digest']:
            raise SystemExit('Native Muon/FA3 environment differs from completed calibration')
        root.joinpath('logs').mkdir(parents=True, exist_ok=True)
        reservation = 'muon-baseline-' + uuid.uuid4().hex
        frozen = CONFIG if args.test_only else directory / 'config.json'
        command = ['sbatch', '--parsable', '--time=120', '--gpus-per-node=8',
                   '--cpus-per-task=8', '--exclude=25a-hgpn062']
        if args.test_only:
            command.append('--test-only')
        else:
            ledger.reserve(reservation, RUN_ID, cfg, 7200, 8, False)
            atomic_json(frozen, cfg)
            command.append('--comment=' + reservation)
        command += [str(PROJECT / 'slurm/muon-baseline.sbatch'), RUN_ID, PREP_ID, str(frozen)]
        state = dict(run_id=RUN_ID, preparation_id=PREP_ID, reservation=reservation,
                     status='submission_uncertain', source_root=str(PROJECT),
                     configuration=str(frozen), environment=environment,
                     ready_sha256=file_digest(ready), requested_seconds=7200, gpus=8,
                     source_commit=subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip(),
                     submission_command=command)
        if not args.test_only:
            state['submission_started_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(record, state)
            with ledger.transaction() as data:
                data['jobs'][reservation].update(submission_tag=reservation,
                    submission_started_utc=state['submission_started_utc'],
                    submission_command=command, source_root=str(PROJECT))
        try:
            result = subprocess.run(command, env={**os.environ, 'DATA_ROOT': str(root),
                                    'TRAIN_SOURCE_ROOT': str(PROJECT)}, capture_output=True, text=True)
        except OSError:
            if not args.test_only:
                ledger.cancel_reservation(reservation)
                state['status'] = 'submission_not_executed'
                atomic_json(record, state)
            raise
        if args.test_only:
            print((result.stdout + result.stderr).strip())
            raise SystemExit(result.returncode)
        job_id = result.stdout.strip().split(';')[0]
        if result.returncode or not re.fullmatch(r'\d+', job_id):
            mark_uncertain(ledger, reservation)
            raise SystemExit(f'Submission uncertain; reservation retained. Use status; do not resubmit. {record}')
        with ledger.transaction() as data:
            data['jobs'][reservation]['returned_job_id'] = job_id
        ledger.bind(reservation, job_id)
        state.update(job_id=job_id, status='submitted')
        atomic_json(record, state)
        # This cron only reconciles accounting/results; it never submits another job.
        old = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if old.returncode not in (0, 1) or (old.returncode == 1 and 'no crontab' not in old.stderr.lower()):
            raise SystemExit(f'Submitted {job_id}; cannot read crontab, use status for accounting')
        marker = '# llama-muon-baseline-accounting-' + RUN_ID
        if not any(marker in row for row in old.stdout.splitlines()):
            invocation = shlex.join(['/usr/bin/python3', str(Path(__file__).resolve()), 'status'])
            line = f'* * * * * {invocation} >> {shlex.quote(str(directory / "accounting.log"))} 2>&1 {marker}'
            subprocess.run(['crontab', '-'], input=old.stdout.rstrip() + '\n' + line + '\n', text=True, check=True)
        print(f'Submitted {job_id}: {RUN_ID}; 8 H200, 8 CPUs, 7200 seconds, 16 H200-hours maximum')


if __name__ == '__main__':
    main()
