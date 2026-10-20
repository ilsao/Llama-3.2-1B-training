#!/usr/bin/env python3
"""Lightweight submission/reconciliation; never loads torch or corpus data."""
import argparse
import datetime as dt
import os
import re
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm.submission import find_submission_job
from training_state import (BE60P20_2B_RUN_ID, DEFAULT_ROOT, PROJECT, ExperimentLedger,
                            be60p20_2b_authorized, identifier,
                            load_config, storage_root)


def recover_submission(ledger, reservation):
    """One metadata snapshot; ambiguity never releases a charged reservation."""
    with ledger.transaction() as data:
        item = dict(data['jobs'][reservation])
    if item['status'] not in {'reserved', 'submission_uncertain'} or not item.get('submission_tag'):
        raise ValueError('Expected an unbound reservation with a durable submission tag')
    job_id = find_submission_job(item, ledger.path.parent)
    ledger.bind(reservation, job_id)
    return job_id


def mark_uncertain(ledger, reservation):
    with ledger.transaction() as data:
        data['jobs'][reservation]['status'] = 'submission_uncertain'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'runtime', 'validate', 'pilot', 'scaling', 'resume', 'accounting', 'recover', 'sweep-calibration'])
    parser.add_argument('id', help='Run ID, numeric accounting job ID, or recover reservation ID')
    parser.add_argument('--prep-id')
    parser.add_argument('--root', default=os.environ.get('DATA_ROOT', str(DEFAULT_ROOT)), type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dependency', help='Slurm afterok:JOB_ID dependency for validated prerequisites')
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw',
                        help='Optimizer for the built-in runtime correctness check')
    parser.add_argument('--gpus', type=int, choices=(4, 8), help='Runtime correctness-check GPU count; configs govern training')
    parser.add_argument('--minutes', type=int, help='Override allocation duration; must fit remaining budget')
    parser.add_argument('--exclude', help='Comma-separated verified Slurm node names to avoid')
    parser.add_argument('--test-only', action='store_true', help='Slurm validation only; no reservation or execution')
    args = parser.parse_args()
    if args.exclude and not re.fullmatch(r'[A-Za-z0-9_-]+(?:,[A-Za-z0-9_-]+)*', args.exclude):
        parser.error('Use comma-separated node names for --exclude')
    if args.dependency and not re.fullmatch(r'afterok:[0-9]+(?::[0-9]+)*', args.dependency):
        parser.error('Use afterok:JOB_ID[:JOB_ID]')
    root = storage_root(args.root)
    ledger = ExperimentLedger(root)
    if args.action == 'accounting':
        import json
        print(json.dumps(ledger.reconcile(args.id), indent=2))
        return
    if args.action == 'recover':
        print(f'Recovered job {recover_submission(ledger, args.id)}; reconcile after termination')
        return
    identifier(args.id)
    config_path = (args.config or PROJECT / 'configs' / (
        'train-smoke.json' if args.action == 'validate' else
        'train-scaling-4.json' if args.action == 'scaling' else 'train-pilot.json')).resolve()
    cfg = load_config(config_path)
    minutes = args.minutes if args.minutes is not None else {'setup': 60, 'runtime': 10, 'scaling': 10, 'sweep-calibration': 10}.get(args.action, 30)
    if args.id == BE60P20_2B_RUN_ID and args.action != 'pilot':
        parser.error('The authorized BE60P20 2B run requires a fresh pilot launcher; resume is unauthorized')
    be60p20_authorization = be60p20_2b_authorized(args.id, cfg)
    if args.id == BE60P20_2B_RUN_ID and not be60p20_authorization:
        parser.error('BE60P20 2B requires the exact approved configuration')
    if be60p20_authorization and args.prep_id != 'dolma-raw-baseline-20261008':
        parser.error('BE60P20 2B requires the parent raw preparation')
    if be60p20_authorization and minutes > 240:
        parser.error('BE60P20 2B allocation is limited to 240 minutes')
    if minutes > 120 and args.action != 'setup' and not be60p20_authorization:
        parser.error('Training/validation allocations are limited to 120 minutes per sbatch')
    if cfg.get('sweep_stage') in ('initial', 'short', 'medium') and not cfg.get('sweep_calibration_digest'):
        parser.error('Sweep templates are uncalibrated; freeze the stage through optimizer_sweep.py')
    if minutes <= 0:
        parser.error('minutes must be positive')
    if args.action in {'pilot', 'scaling', 'resume', 'validate', 'sweep-calibration'}:
        if not args.prep_id:
            parser.error('--prep-id is required after dataset verify completes')
        identifier(args.prep_id)
        if not ((root / 'preprocessed' / args.prep_id / 'verify/manifest.json').is_file() or
                (root / 'preprocessed' / args.prep_id / 'raw/ready.json').is_file()):
            parser.error('Preparation has no verification seal; complete dataset verify first')
    elif args.config:
        parser.error('setup/runtime use their built-in allocation contracts, without --config')
    if args.optimizer != 'adamw' and args.action != 'runtime':
        parser.error('--optimizer is for runtime checks; training uses the config')
    if args.gpus is not None and args.action != 'runtime':
        parser.error('--gpus is only for runtime; change the config for a new training run')
    if args.action in {'setup', 'runtime'}:
        cfg.update(world_size=1 if args.action == 'setup' else (args.gpus or 4), category='validation',
                   max_h200_hours=64, wandb_mode='offline')
        if args.action == 'runtime' and args.optimizer == 'muon':
            cfg.update(optimizer='muon', muon=dict(learning_rate=0.02, momentum=0.95,
                nesterov=True, ns_steps=5, weight_decay=0.1, adjust_lr_fn='original', epsilon=1e-7, ns_coefficients=[3.4445,-4.775,2.0315]))
    if args.action == 'validate' and cfg['category'] != 'validation':
        parser.error('validate requires a validation configuration')
    if args.action == 'scaling' and (cfg['category'] != 'scaling' or minutes > 10 or cfg['max_steps'] > 100):
        parser.error('Scaling requires category=scaling, at most 10 minutes and 100 updates')
    if args.action == 'sweep-calibration' and (cfg['category'] != 'validation' or minutes > 10):
        parser.error('Sweep calibration needs a validation config and at most 10 minutes')
    root.joinpath('logs').mkdir(parents=True, exist_ok=True)
    reservation = 'submission-' + uuid.uuid4().hex
    if not args.test_only:
        ledger.reserve(reservation, args.id, cfg, minutes * 60, cfg['world_size'], args.action == 'resume')
    if not args.test_only and args.action == 'setup':
        with ledger.transaction() as data:
            data['jobs'][reservation]['allocation_kind'] = 'dependency_installation'
            data['jobs'][reservation]['source_root'] = str(PROJECT)
    script = ('setup-training.sbatch' if args.action == 'setup' else
              'sweep-calibration.sbatch' if args.action == 'sweep-calibration' else 'train.sbatch')
    if be60p20_authorization:
        script = 'be60p20-2b.sbatch'
    command = ['sbatch', '--parsable', '--cpus-per-task=8', f'--time={minutes}', f'--gpus-per-node={cfg["world_size"]}',
               f'--job-name=llama-{args.action}', f'--signal=B:USR1@{cfg["stop_margin_seconds"]}', f'--output={root}/logs/%x-%j.out',
               f'--error={root}/logs/%x-%j.err']
    excluded = set(args.exclude.split(',')) if args.exclude else set()
    if args.action == 'sweep-calibration':
        excluded.add('25a-hgpn062')
    if excluded:
        command.append('--exclude=' + ','.join(sorted(excluded)))
    if not args.test_only:
        command.append(f'--comment={reservation}')
    if args.dependency:
        command.append(f'--dependency={args.dependency}')
    if args.test_only:
        command.append('--test-only')
    command.extend([str(PROJECT / 'slurm' / script), args.id])
    if be60p20_authorization:
        command.extend([args.prep_id, str(config_path)])
    elif args.action != 'setup':
        command.extend([args.action, args.prep_id or '-', str(config_path)])
    if not args.test_only:
        with ledger.transaction() as data:
            data['jobs'][reservation].update(
                submission_tag=reservation,
                submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                submission_command=command)
    environment = {**os.environ, 'DATA_ROOT': str(root), 'TRAIN_SOURCE_ROOT': str(PROJECT),
                   'RUNTIME_OPTIMIZER': cfg.get('optimizer', args.optimizer)}
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
    except OSError:
        if not args.test_only:
            ledger.cancel_reservation(reservation)
        raise
    if result.returncode:
        if not args.test_only:
            mark_uncertain(ledger, reservation)
            raise SystemExit(f'Sbatch outcome uncertain; reservation {reservation} remains charged. '
                             f'Do not resubmit. Recover with: python3 slurm/submit_training.py recover '
                             f'{reservation} --root {root}')
        raise SystemExit(result.stderr.strip() or result.stdout.strip())
    if args.test_only:
        print((result.stderr + result.stdout).strip())
        return
    job_id = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'\d+', job_id):
        mark_uncertain(ledger, reservation)
        raise SystemExit(f'Invalid sbatch response; reservation {reservation} remains charged. '
                         f'Recover with: python3 slurm/submit_training.py recover {reservation} --root {root}')
    with ledger.transaction() as data:
        data['jobs'][reservation]['returned_job_id'] = job_id
    ledger.bind(reservation, job_id)
    print(f'Submitted {job_id}; run={args.id}; reserved={cfg["world_size"] * minutes / 60:g} H200-hours')
    print(f'Logs: {root}/logs/llama-{args.action}-{job_id}.out and .err')
    print(f'After termination: python3 slurm/submit_training.py accounting {job_id} --root {root}')


if __name__ == '__main__':
    main()
