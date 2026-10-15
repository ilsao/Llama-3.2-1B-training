#!/usr/bin/env python3
"""Recover small accounting metadata from per-run and per-allocation originals.

Never submits jobs. Produces a reviewable candidate before an explicit --apply.
"""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import atomic_json, digest, file_digest, locked, read_json, load_config, TERMINAL


def merge_job(jobs, key, item):
    key = str(key)
    prior = jobs.setdefault(key, {})
    for field in ('run_id', 'config_digest', 'gpu_count', 'h200_hours', 'elapsed_seconds'):
        if field in prior and field in item and prior[field] != item[field]:
            raise ValueError(f'Conflicting {field} for allocation {key}')
    # Startup is older than a reconciled per-run snapshot.
    combined = {**item, **prior}
    jobs[key] = combined


def parse_accounting(text, root):
    records = {}
    for row in text.splitlines():
        if not row.strip():
            continue
        job, state, seconds, tres, start, end, nodes = row.split('|')[:7]
        state = state.split()[0].rstrip('+')
        if state not in TERMINAL or job in records:
            raise ValueError('Missing, duplicate or nonterminal accounting row: ' + job)
        resources = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
        count = int(resources.get('gres/gpu', 0))
        evidence = root / 'logs' / ('job-' + job) / 'gpus.txt'
        lines = evidence.read_text().splitlines() if evidence.exists() else []
        if count and not (len(lines) == count and len(set(lines)) == count and all('H200' in x and ', GPU-' in x for x in lines)):
            # Very early failures can precede nvidia-smi; allocation/node metadata
            # was captured before launching the worker.
            nodefile = evidence.parent / 'node.txt'
            node = nodefile.read_text() if nodefile.exists() else ''
            if not (f'NodeName={nodes}' in node and 'h200' in node.lower()):
                raise ValueError('H200 evidence unavailable for ' + job)
            proof = {'source': 'archived allocated node metadata', 'sha256': file_digest(nodefile)}
        else:
            proof = {'source': 'startup nvidia-smi for allocated devices', 'sha256': file_digest(evidence)} if count else 'zero GPU allocation'
        records[job] = dict(job_id=job, state=state, elapsed_seconds=int(seconds), alloc_tres=tres,
                            start=start, end=end, nodes=nodes, gpu_count=count,
                            h200_hours=count * int(seconds) / 3600, gpu_type_evidence=proof)
    return records


def build(root, project, accounting):
    jobs, runs, provenance = {}, {}, []
    for path in sorted((root / 'runs/experiments').glob('*/resource-accounting.json')):
        saved = read_json(path); run = saved['run_id']
        runs[run] = {k: v for k, v in saved.items() if k not in ('run_id', 'jobs', 'all_jobs_reconciled')}
        for item in saved['jobs']:
            key = item.get('job_id', item.get('submission_tag'))
            if not key:
                raise ValueError('Unidentified historical allocation: ' + str(path))
            merge_job(jobs, key, item)
        provenance.append({'path': str(path), 'sha256': file_digest(path)})
    startup = {}
    for path in sorted((root / 'logs').glob('job-*/training-reservation.json')):
        item = read_json(path); key = path.parent.name.removeprefix('job-')
        if item.get('job_id') != key:
            raise ValueError('Startup allocation identity differs: ' + str(path))
        startup[key] = item; merge_job(jobs, key, item)
        provenance.append({'path': str(path), 'sha256': file_digest(path)})
    # Read only small saved config JSONs. Never inspect checkpoint tensors.
    config_paths = set(project.glob('configs/**/*.json'))
    config_paths.update(project.glob('experiments/*/configs/*.json'))
    config_paths.update(project.glob('.worktrees/*/configs/*.json'))
    config_paths.update(project.glob('experiments/*/source/configs/*.json'))
    config_paths.update((root / 'runs/experiments').glob('*/resolved-config.json'))
    config_paths.update((root / 'runs/auto-baseline').glob('*/*.json'))
    for item in startup.values():
        config_paths.update(Path(x) for x in item.get('submission_command', []) if x.endswith('.json') and Path(x).is_file())
    configs = {}
    for path in sorted(config_paths):
        obj = read_json(path)
        if isinstance(obj, dict) and 'category' in obj:
            configs[digest(obj)] = (obj, path)
            try:
                resolved = load_config(path)
                configs[digest(resolved)] = (resolved, path)
                # submit_training.py setup/runtime contracts modify this base.
                if path.name == 'train-pilot.json':
                    for world in (1, 4, 8):
                        special = {**resolved, 'world_size': world, 'category': 'validation', 'max_h200_hours': 64, 'wandb_mode': 'offline'}
                        configs[digest(special)] = (special, path)
            except (ValueError, KeyError):
                pass
    for key, item in jobs.items():
        if key in accounting:
            measured = accounting[key]
            if measured['gpu_count'] != item['gpu_count']:
                raise ValueError('Measured/reserved GPU mismatch: ' + key)
            item.update(measured, status='reconciled')
        if 'h200_hours' not in item:
            raise ValueError('Accounting still missing for ' + key)
        run = item['run_id']
        if run not in runs:
            original = startup.get(key, item)
            cfg_hash = original.get('config_digest')
            if cfg_hash not in configs:
                raise ValueError('Original run configuration unavailable: ' + run + ' ' + str(cfg_hash))
            cfg, path = configs[cfg_hash]
            runs[run] = dict(category=cfg['category'], config_digest=cfg_hash,
                             max_h200_hours=original['max_h200_hours'], recovered_from_configuration=str(path))
    for name, run in runs.items():
        own = [j for j in jobs.values() if j['run_id'] == name]
        # Recover inherited branch costs from the first startup cursor. Later
        # previous_h200_hours contains earlier allocations of the SAME run.
        if 'inherited_h200_hours' not in run:
            first = min(own, key=lambda j: int(j.get('job_id', '999999999')))
            old = startup.get(first.get('job_id', ''), {})
            if old.get('previous_h200_hours', 0):
                run['inherited_h200_hours'] = old['previous_h200_hours']
        run['h200_hours_measured'] = sum(j['h200_hours'] for j in own)
        run['logical_h200_hours_measured'] = run['h200_hours_measured'] + run.get('inherited_h200_hours', 0)
        run['budget_exceeded'] = run['logical_h200_hours_measured'] > run['max_h200_hours']
    candidate = dict(jobs=jobs, runs=runs,
                     total_h200_hours_measured=sum(j['h200_hours'] for j in jobs.values()))
    candidate['total_h200_hours_reserved_or_measured'] = candidate['total_h200_hours_measured']
    return candidate, provenance


def apply(root, candidate, expected_current_hash, backup):
    path = root / 'experiment-ledger.json'
    with locked(root / '.experiment-ledger.lock'):
        if file_digest(path) != expected_current_hash:
            raise ValueError('Live ledger changed during review; rebuild recovery')
        current = read_json(path)
        if current.get('jobs') or current.get('runs'):
            raise ValueError('Recovery only replaces an independently verified empty ledger')
        atomic_json(backup, current)
        atomic_json(path, candidate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--project', type=Path, required=True)
    p.add_argument('--accounting', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--apply', action='store_true'); a = p.parse_args()
    ledger = a.root / 'experiment-ledger.json'; before = file_digest(ledger)
    candidate, sources = build(a.root, a.project, parse_accounting(a.accounting.read_text(), a.root))
    a.output.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output / 'candidate.json', candidate)
    audit = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), empty_ledger_sha256=before,
                 candidate_sha256=digest(candidate), jobs=len(candidate['jobs']), runs=len(candidate['runs']),
                 total_h200_hours_measured=candidate['total_h200_hours_measured'], sources=sources,
                 accounting_sha256=file_digest(a.accounting), cause='Unverified; no deletion cause inferred')
    if a.apply:
        apply(a.root, candidate, before, a.output / 'empty-ledger-original.json')
        audit['applied'] = True
    atomic_json(a.output / 'audit.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k != 'sources'}, indent=2))

if __name__ == '__main__': main()
