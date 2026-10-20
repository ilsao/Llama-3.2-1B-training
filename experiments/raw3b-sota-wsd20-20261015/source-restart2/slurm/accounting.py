#!/usr/bin/env python3
"""Lightweight login-node reconciliation after each allocation terminates."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('job_id')
parser.add_argument('--root', type=Path, default=Path('/work/asciibase64/Llama-3.2-1B-training'))
args = parser.parse_args()
root = args.root.resolve()
if not root.is_relative_to(Path('/work/asciibase64')) or not re.fullmatch(r'\d+', args.job_id):
    parser.error('Use the project /work storage and a numeric job ID')
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import slurm_record
measured = slurm_record(args.job_id, root)
job = measured['job_id']
record_dir = root / 'logs' / f'job-{job}'
record = {**measured, 'gpu_hours': measured['h200_hours'],
          'category': 'data_preprocessing'}
for name in ('stage', 'preparation-id'):
    path = record_dir / f'{name}.txt'
    record[name] = path.read_text().strip() if path.exists() else 'setup'
ledger = root / 'preprocessing-ledger.json'
with (root / '.accounting.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    data = json.loads(ledger.read_text()) if ledger.exists() else {'jobs': {}}
    data['jobs'][job] = record  # idempotent; do not count the same allocation twice
    data['total_gpu_hours'] = sum(item['gpu_hours'] for item in data['jobs'].values())
    data['total_h200_hours'] = sum(item['h200_hours'] for item in data['jobs'].values())
    temporary = ledger.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(ledger)
print(json.dumps(record, indent=2))
