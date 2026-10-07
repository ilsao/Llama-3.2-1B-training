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
raw = subprocess.check_output(['sacct', '-j', args.job_id, '-X', '--noheader', '--parsable2',
       '--format=JobIDRaw,State,ElapsedRaw,AllocTRES%500,Start,End,NodeList%128'], text=True)
records = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == args.job_id]
if len(records) != 1:
    raise SystemExit('Expected exactly one allocation accounting record')
job, state, elapsed, tres, start, end, nodes = records[0][:7]
if state.split()[0] not in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}:
    raise SystemExit('Allocation has not terminated; reconcile after it ends')
resources = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
gpus = int(resources.get('gres/gpu', '0'))
typed = {key[9:]: int(value) for key, value in resources.items() if key.startswith('gres/gpu:')}
if gpus and (not typed or sum(typed.values()) != gpus):
    raise SystemExit('GPU type/count unavailable in AllocTRES; obtain typed Slurm accounting before recording H200-hours')
record_dir = root / 'logs' / f'job-{job}'
record = {'job_id': job, 'state': state, 'elapsed_seconds': int(elapsed), 'alloc_tres': tres,
          'gpu_count': gpus, 'gpu_types': typed, 'gpu_hours': gpus * int(elapsed) / 3600,
          'h200_hours': sum(value for key, value in typed.items() if key.lower() == 'h200') * int(elapsed) / 3600,
          'start': start, 'end': end, 'nodes': nodes, 'category': 'data_preprocessing'}
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
