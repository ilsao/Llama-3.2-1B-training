#!/usr/bin/env python3
"""Read-only experiment review; prints commands and never dispatches them."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from training_state import load_config, schedule_multiplier


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--winner', choices=['B-earlier-decay', 'C-weaker-decay', 'A-combined', 'E-hold-zero'],
                        help='Review seed-43 commands after completed and reconciled seed-42 ranking')
    args = parser.parse_args()
    plan = read(REPOSITORY / 'experiments/training-diagnosis-20261010/plan.json')
    root = Path(plan['storage_root'])
    configs = {}
    for entry in plan['candidates']:
        path = REPOSITORY / entry['config']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['config_sha256']:
            raise ValueError(f'Configuration changed: {path}')
        cfg = load_config(path)
        configs[(entry['candidate'], entry['seed'])] = cfg
        if cfg['max_steps'] != plan['planned_updates'] or cfg['scheduler_steps'] != cfg['max_steps']:
            raise ValueError('Matched endpoint or scheduler horizon changed')
        if cfg['max_training_tokens'] != cfg['max_steps'] * plan['global_input_tokens_per_update']:
            raise ValueError('Token budget differs from the matched endpoint')
        if cfg['target_training_tokens'] != cfg['max_training_tokens'] or not cfg['fixed_horizon']:
            raise ValueError('Exact fixed token horizon required')
        if cfg['eval_steps'] > cfg['max_steps'] // 10:
            raise ValueError('Evaluation interval too long')
        if cfg['world_size'] != 8 or cfg['category'] != 'pilot' or cfg['max_h200_hours'] != 6:
            raise ValueError('Pilot allocation contract changed')
        if abs(schedule_multiplier(cfg['max_steps'] - 1, cfg) - cfg['min_lr_ratio']) > 1e-12:
            raise ValueError('Cooldown does not complete')
    for seed in (42, 43):
        control = configs[('A-control', seed)]
        lower = configs[('D-lower-update-lr', seed)]
        for step in range(control['max_steps']):
            a = control['muon']['learning_rate'] * control['muon']['weight_decay'] * schedule_multiplier(step, control)
            d = lower['muon']['learning_rate'] * lower['muon']['weight_decay'] * schedule_multiplier(step, lower)
            if abs(a - d) > 1e-15:
                raise ValueError('D must preserve the control decay coefficient at every step')
    for name, expected in plan['source_hashes'].items():
        if hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Pinned source changed: {name}; review the plan before execution')
    ledger = read(root / 'experiment-ledger.json')
    preprocessing = read(root / 'preprocessing-ledger.json')
    used = ledger['total_h200_hours_reserved_or_measured'] + preprocessing['total_h200_hours']
    ceiling = plan['budget_snapshot']['conservative_aggregate_ceiling_h200_hours']
    outstanding = [key for key, item in ledger['jobs'].items() if 'h200_hours' not in item]
    if outstanding:
        raise ValueError('Unreconciled training allocations: ' + ', '.join(outstanding))
    research = read(REPOSITORY / 'research-state.yaml')
    unresolved = [item.get('tag', '?') for item in research.get('submission_intents', [])
                  if item.get('ledger') == 'preprocessing' and item.get('status') != 'not_dispatched'
                  and item.get('job_id') not in preprocessing['jobs']]
    if unresolved:
        held = sum(item['reserved_h200_hours'] for item in research.get('submission_intents', [])
                   if item.get('tag') in unresolved)
        used += held
        print(f'Launch gate: retain {held:g} H200-hours for unresolved preprocessing intents; '
              'recover/reconcile before launch. This is a read-only design review.')
    stage = 'ranking'
    selected = [next(entry for entry in plan['candidates']
                         if entry['candidate'] == candidate and entry['stage'] == 'ranking')
                for candidate in plan['ranking_candidates']]
    if args.winner:
        ranking = {}
        for entry in selected:
            run = root / 'runs/experiments' / entry['run_id']
            summary = read(run / 'summary.json')
            account = read(run / 'resource-accounting.json')
            if (summary['global_step'] != plan['planned_updates']
                    or summary['attempted_updates'] != summary['global_step']
                    or summary['total_tokens_seen'] != plan['matched_training_input_tokens']
                    or summary['budget_tokens_charged'] != summary['total_tokens_seen']
                    or summary['last_eval_step'] != summary['global_step']
                    or not account.get('all_jobs_reconciled') or account.get('budget_exceeded')):
                raise ValueError('Ranking endpoint incomplete, skipped, or not reconciled')
            if summary['last_evaluation']['eval/prediction_tokens'] <= 0:
                raise ValueError('Missing endpoint prediction tokens')
            ranking[entry['candidate']] = summary['last_evaluation']['eval/nll'] / summary['last_evaluation']['eval/prediction_tokens']
        if any(not math.isfinite(value) for value in ranking.values()):
            raise ValueError('All ranking endpoint NLLs must be finite')
        control = 'A-control'
        priority = plan['promotion']['selection_order_for_near_ties']
        eligible = [candidate for candidate in priority
                    if ranking[control] - ranking[candidate]
                    >= plan['promotion']['minimum_seed42_control_improvement_nats']]
        if not eligible:
            raise ValueError('No modified recipe meets the seed-42 control improvement threshold')
        best = min(ranking[candidate] for candidate in eligible)
        chosen = next(candidate for candidate in eligible
                      if ranking[candidate] - best < plan['promotion']['near_tie_nats'])
        if args.winner != chosen:
            raise ValueError(f'Preregistered selection is {chosen}, not {args.winner}')
        selected = [next(entry for entry in plan['candidates']
                         if entry['seed'] == 43 and entry['candidate'] == candidate)
                    for candidate in (control, chosen)]
        print('Conditional confirmation: original control and ' + chosen)
        stage = 'confirmation'
    if stage == 'ranking':
        print('Seed-43 confirmation is deferred, not reserved: both jobs require '
              '12 additional H200-hours and a fresh budget check.')
    planned = sum(entry['allocation_h200_hours'] for entry in selected)
    reserve = (plan['budget_plan']['confirmation_h200_hours'] if stage == 'ranking' else 0) + plan['budget_plan']['diagnostic_design_only_h200_hours']
    if used + planned + reserve > ceiling + 1e-12:
        raise ValueError('Planned allocations exceed the conservative aggregate envelope')
    print(f'Review only: {stage}; {used:.6f} H200-hours already reserved/measured; '
          f'{planned:g} for listed jobs; {reserve:g} held for conditional work.')
    for entry in selected:
        if (root / 'runs/experiments' / entry['run_id']).exists() or entry['run_id'] in ledger['runs']:
            raise ValueError(f'Run ID already used: {entry["run_id"]}; no implicit retry or resume')
        command = ['python3', str(REPOSITORY / 'slurm/submit_training.py'), 'pilot', entry['run_id'],
                   '--prep-id', plan['preparation_id'], '--config', str(REPOSITORY / entry['config']),
                   '--minutes', str(entry['allocation_minutes'])]
        print(shlex.join(command))
    print('No submission, reservation, Slurm query, or shared-state write was performed.')


if __name__ == '__main__':
    main()
