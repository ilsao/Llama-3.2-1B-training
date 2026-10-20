#!/usr/bin/env python3
"""Review prepared cooldown configs. No Slurm calls or shared-state writes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from training_state import hold_schedule_boundaries, load_config, schedule_multiplier


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('main', 'interaction', 'all'), default='main')
    args = parser.parse_args()
    plan = read(REPOSITORY / 'experiments/cooldown-grid-20261010/plan.json')
    root = Path(plan['storage_root'])
    for name, expected in plan['source_hashes'].items():
        if sha(REPOSITORY / name) != expected:
            raise ValueError('Pinned source changed: ' + name)
    if sha(plan['environment_build']) != plan['environment_build_sha256']:
        raise ValueError('FA3 build changed')
    reference = plan['reference']
    directory = Path(reference['directory'])
    if sha(directory / 'execution-identity.json') != reference['execution_identity_sha256']:
        raise ValueError('Reference execution identity changed')
    reference_config_path = REPOSITORY / reference['config']
    if sha(reference_config_path) != reference['config_sha256']:
        raise ValueError('Reference configuration changed')
    base = load_config(reference_config_path)
    summary = read(directory / 'summary.json')
    accounting = read(directory / 'resource-accounting.json')
    provenance = read(directory / 'data-provenance.json')
    initialization = read(directory / 'initialization.json')
    if (summary['global_step'] != 3051 or summary['attempted_updates'] != 3051
            or summary['last_eval_step'] != 3051
            or summary['total_tokens_seen'] != summary['budget_tokens_charged']
            or summary['total_tokens_seen'] != 399900672
            or not accounting['all_jobs_reconciled'] or accounting['budget_exceeded']
            or initialization['pretrained_weights_loaded']
            or initialization['sample_signature'] != reference['initialization_signature']
            or summary['environment_digest'] != reference['environment_digest']):
        raise ValueError('Reference endpoint incomplete or identity changed')
    for field, expected in (('fingerprint', reference['preparation_fingerprint']),
                            ('evaluation_indices_sha256', reference['evaluation_indices_sha256']),
                            ('evaluation_manifest_sha256', reference['evaluation_manifest_sha256'])):
        if provenance[field] != expected:
            raise ValueError('Reference provenance changed: ' + field)
    prep = root / 'preprocessed' / plan['preparation_id']
    if (sha(prep / 'raw/ready.json') != reference['evaluation_manifest_sha256']
            or sha(prep / 'configuration.json') != provenance['raw_identity_sha256']):
        raise ValueError('Preparation identity or evaluation manifest changed')
    progress = read(prep / 'raw/progress.json')
    if sum(item['rows'] for item in progress['train_files']) < 3051 * 16:
        raise ValueError('Canonical packed prefix does not cover the full token horizon')
    allowed = {'sweep_id', 'sweep_purpose', 'parent_runs', 'lr_scheduler',
               'min_lr_ratio', 'wsd_decay_ratio', 'hold_schedule'}
    invariant = {key: value for key, value in base.items() if key not in allowed}
    configs = {}
    for item in plan['candidates']:
        path = REPOSITORY / item['config']
        if sha(path) != item['config_sha256']:
            raise ValueError('Candidate config changed: ' + item['candidate'])
        cfg = load_config(path)
        if {key: value for key, value in cfg.items() if key not in allowed} != invariant:
            raise ValueError('Non-schedule recipe changed: ' + item['candidate'])
        if (cfg['max_steps'] != cfg['scheduler_steps'] or cfg['max_steps'] != 3051
                or cfg['max_training_tokens'] != cfg['target_training_tokens']
                or cfg['target_training_tokens'] != 399900672
                or not cfg['fixed_horizon'] or cfg['eval_steps'] > 305):
            raise ValueError('Matched horizon/evaluation settings changed')
        if cfg['lr_scheduler'] == 'wsd_hold_zero':
            phases = hold_schedule_boundaries(3051, 170, cfg['hold_schedule'])
            if list(phases) != item['phase_indices'] or phases[1:3] != (2441, 2746):
                raise ValueError('Hold/zero phase locations changed')
            if cfg['hold_schedule']['plateau_lr_ratio'] != item['plateau_ratio']:
                raise ValueError('Plateau level changed')
            if abs(schedule_multiplier(phases[1], cfg) - item['plateau_ratio']) > 1e-12:
                raise ValueError('LR is not at its plateau during hold')
            start = phases[0]
        else:
            raise ValueError('Only E/BE hold-zero candidates belong in this grid')
        if start != item['cooldown_start_index']:
            raise ValueError('Cooldown start changed')
        values = [schedule_multiplier(step, cfg) for step in range(3051)]
        if (any(not math.isfinite(value) for value in values)
                or any(a < b - 1e-12 for a, b in zip(values[170:], values[171:]))
                or abs(values[-1] - item['terminal_ratio']) > 1e-12):
            raise ValueError('Nonmonotonic cooldown or wrong terminal LR')
        configs[item['candidate']] = cfg
    ledger = read(root / 'experiment-ledger.json')
    prep_ledger = read(root / 'preprocessing-ledger.json')
    state = read(REPOSITORY / 'research-state.yaml')
    unresolved = [item for item in state.get('submission_intents', [])
                  if item.get('ledger') == 'preprocessing' and item.get('status') != 'not_dispatched'
                  and item.get('job_id') not in prep_ledger['jobs']]
    held = sum(item['reserved_h200_hours'] for item in unresolved)
    used = ledger['total_h200_hours_reserved_or_measured'] + prep_ledger['total_h200_hours'] + held
    outstanding = [key for key, item in ledger['jobs'].items() if 'h200_hours' not in item]
    selected = [item for item in plan['candidates'] if args.group == 'all' or item['group'] == args.group]
    selected.sort(key=lambda item: plan['candidate_priority'].index(item['candidate']))
    for item in selected:
        if item['run_id'] in ledger['runs'] or (root / 'runs/experiments' / item['run_id']).exists():
            raise ValueError('Run already exists; no implicit retry: ' + item['run_id'])
    print(f'Validated {len(configs)} configs; matched E40P10 reference; group={args.group}')
    print(f'Listed jobs={len(selected)}, allocation cap={sum(item["allocation_h200_hours"] for item in selected):g} H200-hours')
    print(f'Historical reserved/measured including unresolved preprocessing: {used:.6f}; held intents={held:g}; unresolved training allocations={len(outstanding)}')
    print('User override: remaining aggregate H200-hours do not prune this design; history and held intents are preserved.')
    for item in selected:
        command = ['python3', str(REPOSITORY / 'slurm/submit_training.py'), 'pilot', item['run_id'],
                   '--prep-id', plan['preparation_id'], '--config', str(REPOSITORY / item['config']),
                   '--minutes', '45', '--exclude', '25a-hgpn151']
        print(shlex.join(command))
    print('Commands are for review only. No Slurm call, submission, reservation, or shared-state write.')


if __name__ == '__main__':
    main()
