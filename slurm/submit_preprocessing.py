#!/usr/bin/env python3
"""Durable preprocessing submission intents; login-node metadata only."""
import argparse
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import DEFAULT_ROOT, PROJECT, atomic_json, identifier, locked, read_json, storage_root
from slurm.submission import find_submission_job

STAGES = {'setup', 'download', 'split', 'clean', 'score', 'select', 'pack', 'verify', 'smoke', 'parallel-check'}


def receipt_path(root, tag):
    identifier(tag)
    return root / 'logs' / (tag + '.json')


def save_intent(state_path, root, state, item):
    # A returned job receipt precedes binding into central state. A crash between
    # writes leaves a recoverable intent, never permission to submit again.
    atomic_json(receipt_path(root, item['tag']), item)
    intents = state.setdefault('submission_intents', [])
    intents[:] = [old for old in intents if old.get('tag') != item['tag']]
    intents.append(dict(item))
    atomic_json(state_path, state)


def bind(state_path, root, state, item, job_id):
    if not re.fullmatch(r'\d+', job_id):
        raise ValueError('Expected a numeric job ID; retain intent')
    item.update(status='submitted', job_id=job_id, returned_job_id=job_id)
    jobs = state.setdefault('jobs', [])
    matching = [old for old in jobs if old.get('job_id') == job_id]
    if matching and any(old.get('tag') != item['tag'] for old in matching):
        raise ValueError('Job ID already belongs to another intent')
    if not matching:
        jobs.append({'job_id': job_id, 'tag': item['tag'], 'prep_id': item['prep_id'],
                     'stage': item['stage'], 'ledger': 'preprocessing', 'status': 'SUBMITTED',
                     'allocated_cpus': 8, 'allocated_gpu_count': 1,
                     'allocation_h200_hours_upper_bound': 24,
                     'accounting_status': 'reserved; reconcile after termination'})
    save_intent(state_path, root, state, item)
    return job_id


def submit(stage, prep_id, extra, root, state_path):
    if stage not in STAGES:
        raise ValueError('Invalid preprocessing stage')
    if stage != 'setup':
        identifier(prep_id)
    with locked(root / '.preprocessing-submission.lock', blocking=False):
        state = read_json(state_path)
        ledger_path = root / 'preprocessing-ledger.json'
        accounted = read_json(ledger_path)['jobs'] if ledger_path.exists() else {}
        # Includes legacy heartbeat intents; different stages must not overlap
        # within the same preparation, including uncertain or pending jobs.
        for old in state.get('submission_intents', []):
            if (old.get('prep_id') == prep_id and old.get('status') != 'not_dispatched'
                    and old.get('job_id') not in accounted):
                raise ValueError('Existing preprocessing intent must be recovered/reconciled before submission')
        tag = 'preprocessing-' + uuid.uuid4().hex
        script = 'setup.sbatch' if stage == 'setup' else 'score.sbatch' if stage == 'score' else 'prepare.sbatch'
        command = ['sbatch', '--parsable', '--cpus-per-task=8', '--gpus-per-node=1',
                   '--time=24:00:00', f'--comment={tag}',
                   f'--output={root}/logs/%x-%j.out', f'--error={root}/logs/%x-%j.err',
                   str(PROJECT / 'slurm' / script)]
        command += [] if stage == 'setup' else [prep_id] if stage == 'score' else [stage, prep_id, *extra]
        if stage in {'setup', 'score'} and extra:
            raise ValueError('setup/score do not accept dataset arguments')
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        item = {'tag': tag, 'submission_tag': tag, 'prep_id': prep_id, 'stage': stage,
                'ledger': 'preprocessing', 'status': 'submission_uncertain', 'timestamp': now,
                'submission_started_utc': now, 'submission_command': command,
                'reserved_h200_hours': 24, 'gpu_count': 1, 'allocated_cpus': 8}
        accounting = state.setdefault('accounting', {})
        key = 'shared_measured_plus_outstanding_reservations_upper_bound'
        accounting[key] = accounting.get(key, 0) + 24
        save_intent(state_path, root, state, item)  # before dispatch
        try:
            result = subprocess.run(command, env={**os.environ, 'DATA_ROOT': str(root)},
                                    capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'Sbatch reply timed out; retain intent {tag} and reservation. '
                               f'Recover with: bash slurm/submit.sh recover {tag}') from error
        except OSError:
            # The child did not execute. Leave explicit evidence, without a reservation.
            item['status'] = 'not_dispatched'
            accounting[key] -= 24
            save_intent(state_path, root, state, item)
            raise
        job_id = result.stdout.strip().split(';')[0]
        if result.returncode or not re.fullmatch(r'\d+', job_id):
            raise RuntimeError(f'Sbatch outcome uncertain; retain 24 H200-hour reservation. '
                               f'Do not resubmit; use: bash slurm/submit.sh recover {tag}')
        item['returned_job_id'] = job_id
        atomic_json(receipt_path(root, tag), item)
        return bind(state_path, root, state, item, job_id)


def recover(tag, root, state_path):
    with locked(root / '.preprocessing-submission.lock', blocking=False):
        state = read_json(state_path)
        item = read_json(receipt_path(root, tag))
        if item['tag'] != tag or item['status'] == 'not_dispatched':
            raise ValueError('Expected a dispatched submission intent')
        job_id = find_submission_job(item, root)
        return bind(state_path, root, state, item, job_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(STAGES | {'recover'}))
    parser.add_argument('prep_id', nargs='?')
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.stage != 'setup' and args.prep_id is None:
        parser.error('Supply a preparation ID or recovery tag')
    if args.stage == 'recover' and args.extra:
        parser.error('recover accepts only the submission tag')
    args.prep_id = args.prep_id or 'setup'
    root = storage_root(os.environ.get('DATA_ROOT', str(DEFAULT_ROOT)))
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    state_path = PROJECT / 'research-state.yaml'
    job_id = (recover(args.prep_id, root, state_path) if args.stage == 'recover'
              else submit(args.stage, args.prep_id, args.extra, root, state_path))
    print(f'Job {job_id}; stage={args.stage}; preparation={args.prep_id}; reconcile after termination')


if __name__ == '__main__':
    main()
