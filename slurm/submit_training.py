#!/usr/bin/env python3
"""Lightweight submission/reconciliation; never loads torch or corpus data."""
import argparse
import os
import re
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import (DEFAULT_ROOT, PROJECT, ExperimentLedger, identifier,
                            load_config, storage_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'runtime', 'validate', 'pilot', 'scaling', 'resume', 'accounting'])
    parser.add_argument('id', help='Unique run ID, or numeric job ID for accounting')
    parser.add_argument('--prep-id')
    parser.add_argument('--root', default=os.environ.get('DATA_ROOT', str(DEFAULT_ROOT)), type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dependency', help='Slurm afterok:JOB_ID dependency for validated prerequisites')
    parser.add_argument('--gpus', type=int, choices=(4, 8), help='Runtime correctness-check GPU count; configs govern training')
    parser.add_argument('--minutes', type=int, help='Override allocation duration; must fit remaining budget')
    parser.add_argument('--test-only', action='store_true', help='Slurm validation only; no reservation or execution')
    args = parser.parse_args()
    if args.dependency and not re.fullmatch(r'afterok:[0-9]+(?::[0-9]+)*', args.dependency):
        parser.error('Use afterok:JOB_ID[:JOB_ID]')
    root = storage_root(args.root)
    ledger = ExperimentLedger(root)
    if args.action == 'accounting':
        import json
        print(json.dumps(ledger.reconcile(args.id), indent=2))
        return
    identifier(args.id)
    config_path = (args.config or PROJECT / 'configs' / (
        'train-smoke.json' if args.action == 'validate' else
        'train-scaling-4.json' if args.action == 'scaling' else 'train-pilot.json')).resolve()
    cfg = load_config(config_path)
    minutes = args.minutes if args.minutes is not None else {'setup': 60, 'runtime': 10, 'scaling': 10}.get(args.action, 30)
    if minutes <= 0:
        parser.error('minutes must be positive')
    if args.action in {'pilot', 'scaling', 'resume', 'validate'}:
        if not args.prep_id:
            parser.error('--prep-id is required after dataset verify completes')
        identifier(args.prep_id)
        if not (root / 'preprocessed' / args.prep_id / 'verify/manifest.json').is_file():
            parser.error('Preparation has no verification seal; complete dataset verify first')
    elif args.config:
        parser.error('setup/runtime use their built-in allocation contracts, without --config')
    if args.gpus is not None and args.action != 'runtime':
        parser.error('--gpus is only for runtime; change the config for a new training run')
    if args.action in {'setup', 'runtime'}:
        cfg.update(world_size=1 if args.action == 'setup' else (args.gpus or 4), category='validation',
                   max_h200_hours=64, wandb_mode='offline')
    if args.action == 'validate' and cfg['category'] != 'validation':
        parser.error('validate requires a validation configuration')
    if args.action == 'scaling' and (cfg['category'] != 'scaling' or minutes > 10 or cfg['max_steps'] > 100):
        parser.error('Scaling requires category=scaling, at most 10 minutes and 100 updates')
    root.joinpath('logs').mkdir(parents=True, exist_ok=True)
    reservation = 'submission-' + uuid.uuid4().hex
    if not args.test_only:
        ledger.reserve(reservation, args.id, cfg, minutes * 60, cfg['world_size'], args.action == 'resume')
    script = 'setup-training.sbatch' if args.action == 'setup' else 'train.sbatch'
    command = ['sbatch', '--parsable', '--cpus-per-task=8', f'--time={minutes}', f'--gpus-per-node={cfg["world_size"]}',
               f'--job-name=llama-{args.action}', f'--output={root}/logs/%x-%j.out',
               f'--error={root}/logs/%x-%j.err']
    if args.dependency:
        command.append(f'--dependency={args.dependency}')
    if args.test_only:
        command.append('--test-only')
    command.extend([str(PROJECT / 'slurm' / script), args.id])
    if args.action != 'setup':
        command.extend([args.action, args.prep_id or '-', str(config_path)])
    environment = {**os.environ, 'DATA_ROOT': str(root)}
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
    except OSError:
        if not args.test_only:
            ledger.cancel_reservation(reservation)
        raise
    if result.returncode:
        if not args.test_only:
            ledger.cancel_reservation(reservation)
        raise SystemExit(result.stderr.strip() or result.stdout.strip())
    if args.test_only:
        print((result.stderr + result.stdout).strip())
        return
    job_id = result.stdout.strip().split(';')[0]
    ledger.bind(reservation, job_id)
    print(f'Submitted {job_id}; run={args.id}; reserved={cfg["world_size"] * minutes / 60:g} H200-hours')
    print(f'Logs: {root}/logs/llama-{args.action}-{job_id}.out and .err')
    print(f'After termination: python3 slurm/submit_training.py accounting {job_id} --root {root}')


if __name__ == '__main__':
    main()
