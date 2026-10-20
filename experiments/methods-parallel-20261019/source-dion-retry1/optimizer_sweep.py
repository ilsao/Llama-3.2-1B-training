"""Dependency-free staged optimizer/scheduler experiment contracts.

No model imports, dataset scans or GPU operations: safe for login-node metadata work.
"""
import copy
import json
import math
from collections import defaultdict
from training_state import PROJECT, read_json

SWEEP_ID = 'optimizer-scheduler-raw-20261008'
PREPARATION_ID = 'dolma-raw-baseline-20261008'
ENVIRONMENT = '/work/asciibase64/Llama-3.2-1B-training/environments/training-muon-torch291-fa3-94e22c9'
STAGES = {'initial': 5, 'short': 15, 'medium': 20}
SCHEDULERS = ('linear', 'cosine', 'wsd')
TOKENS_PER_UPDATE = 16 * 8192


def candidates():
    result = []
    for optimizer, rates in [('adamw', [('3e-4', .0003), ('6e-4', .0006), ('1.2e-3', .0012)]),
                             ('muon', [('0.02', .02), ('0.04', .04), ('0.08', .08)])]:
        for label, lr in rates:
            recipe = f'{optimizer}-lr{label}'
            for scheduler in SCHEDULERS:
                result.append(dict(id=f'{recipe}-{scheduler}', recipe=recipe,
                    optimizer=optimizer, learning_rate=lr, scheduler=scheduler,
                    weight_decay=.002 / lr if optimizer == 'muon' else .1))
    return result


def training_config(candidate, stage, steps, calibration_digest=None, parent_runs=()):
    if stage not in STAGES or type(steps) is not int or steps < 100 or steps % 50:
        raise ValueError('Sweep stages need at least 100 updates and a multiple of 50')
    if steps * TOKENS_PER_UPDATE > 6_000_000_000:
        raise ValueError('Planned inputs exceed 6B tokens')
    cfg = copy.deepcopy(read_json(PROJECT / 'configs/train-sweep-adamw.json'))
    cfg.update(optimizer=candidate['optimizer'], learning_rate=candidate['learning_rate']
               if candidate['optimizer'] == 'adamw' else .0003,
        max_steps=steps, scheduler_steps=steps, warmup_steps=max(50, steps // 50),
        max_training_tokens=steps * TOKENS_PER_UPDATE, max_h200_hours=64,
        eval_steps=steps // 10, eval_documents=1000, stop_margin_seconds=60,
        category='pilot', lr_scheduler=candidate['scheduler'], wsd_decay_ratio=.2,
        fixed_horizon=True, checkpoint_policy='final_only', export_on_completion=False,
        retain_all_checkpoints=True, sweep_id=SWEEP_ID, sweep_stage=stage,
        sweep_purpose='optimizer x learning rate x scheduler at matched training tokens',
        parent_runs=list(parent_runs))
    cfg.pop('muon', None)
    if calibration_digest is not None:
        cfg['sweep_calibration_digest'] = calibration_digest
    if candidate['optimizer'] == 'muon':
        cfg['muon'] = dict(learning_rate=candidate['learning_rate'], momentum=.95,
            nesterov=True, ns_steps=5, weight_decay=candidate['weight_decay'],
            adjust_lr_fn='original', epsilon=1e-7, ns_coefficients=[3.4445, -4.775, 2.0315])
    return cfg


def calibration_config(optimizer, gpus):
    if optimizer not in ('adamw', 'muon') or gpus not in (4, 8):
        raise ValueError('Calibrate AdamW/Muon at matched batch on 4/8 GPUs')
    candidate = next(c for c in candidates() if c['optimizer'] == optimizer and c['scheduler'] == 'cosine')
    cfg = training_config(candidate, 'initial', 100)
    cfg.update(world_size=gpus, gradient_accumulation_steps=16 // gpus,
               max_steps=30, scheduler_steps=30, warmup_steps=1, eval_steps=3,
               max_training_tokens=6_000_000_000, category='validation', wandb_mode='offline',
               sweep_stage='validation', sweep_purpose='native runtime correctness and matched-batch scaling')
    return cfg


def freeze_steps(stage, calibration, available_rows):
    """Same common horizon for all candidates, measured including update overhead."""
    seconds = STAGES[stage] * 60
    steady = calibration['slowest_8gpu_update_seconds']
    margin = max(60, math.ceil(calibration['checkpoint_seconds'] + calibration['evaluation_seconds'] + 30))
    overhead = (calibration['initialization_seconds'] + 10 * calibration['evaluation_seconds'] +
                calibration['checkpoint_seconds'] + margin)
    if any(not math.isfinite(x) or x <= 0 for x in (steady, calibration['checkpoint_seconds'],
            calibration['evaluation_seconds'], calibration['initialization_seconds'])):
        raise ValueError('Calibration times must be finite and positive')
    capacity = math.floor(.8 * max(0, seconds - overhead) / steady)
    steps = min(capacity, available_rows // 16, 6_000_000_000 // TOKENS_PER_UPDATE) // 50 * 50
    if steps < 100:
        raise ValueError('Initialization/evaluation/save overhead or raw buffer cannot fit 100 updates')
    return dict(steps=steps, tokens=steps * TOKENS_PER_UPDATE, stop_margin_seconds=margin,
                overhead_reserve_seconds=overhead, slowest_update_seconds=steady)


def prune_initial(results):
    """Remove only the clearly worst scheduler within each optimizer/LR recipe.

    Called after all candidates terminate. Both points are matched training tokens;
    no WSD elimination occurs before its final decay update has been evaluated.
    Incomplete/unstable candidates never rank as successful or automatically retry.
    """
    groups = defaultdict(list)
    for result in results:
        if result['status'] == 'complete':
            groups[result['recipe']].append(result)
    survivors, eliminated = [], []
    for recipe, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (r['nll_100'], r['id']))
        if len(rows) == 3:
            best, worst = rows[0], rows[-1]
            best_90 = min(rows, key=lambda r: (r['nll_90'], r['id']))
            worst_90 = max(rows, key=lambda r: (r['nll_90'], r['id']))
            if (worst_90['id'] == worst['id'] and best_90['id'] == best['id'] and
                    worst['nll_90'] - best['nll_90'] >= .10 - 1e-12 and
                    worst['nll_100'] - best['nll_100'] >= .10 - 1e-12):
                eliminated.append(worst['id']); rows.pop()
        survivors.extend(r['id'] for r in rows)
    return dict(survivors=survivors, eliminated=eliminated,
                excluded=[r['id'] for r in results if r['status'] != 'complete'])


def select_finalists(results):
    rows = [r for r in results if r['status'] == 'complete']
    best = [min((r for r in rows if r['optimizer'] == optimizer),
                key=lambda r: (r['nll_100'], r['id']), default=None) for optimizer in ('adamw', 'muon')]
    if any(r is None for r in best):
        return dict(finalists=[r['id'] for r in best if r], needs_medium=False,
                    reason='Both optimizer families need complete evidence')
    reversal = (best[0]['nll_90'] - best[1]['nll_90']) * (best[0]['nll_100'] - best[1]['nll_100']) < 0
    close = abs(best[0]['nll_100'] - best[1]['nll_100']) < .05
    return dict(finalists=[r['id'] for r in best], needs_medium=reversal or close,
                reason='ranking reversal' if reversal else 'NLL gap below .05' if close else 'clear short-stage ranking')


def result_from_run(root, candidate, run_id, cfg, measured):
    run = root / 'runs/experiments' / run_id
    result = dict(id=candidate['id'], recipe=candidate['recipe'], optimizer=candidate['optimizer'],
                  run_id=run_id, status='incomplete', allocation=measured)
    if not (run / 'summary.json').exists():
        return result
    summary = read_json(run / 'summary.json')
    if summary['configuration'] != cfg:
        raise ValueError('Result configuration differs from frozen stage configuration')
    result.update(tokens=summary['total_tokens_seen'], charged_tokens=summary['budget_tokens_charged'],
        stop_reason=summary['stop_reason'], global_step=summary['global_step'],
        initialization_signature=summary['initialization']['sample_signature'],
        preparation_fingerprint=summary['preparation_fingerprint'], wandb_url=summary['wandb_url'])
    if summary['stop_reason'] == 'nonfinite_or_skipped_update':
        result['status'] = 'unstable'; return result
    if (measured['state'] != 'COMPLETED' or summary['stop_reason'] != 'max_steps' or
        summary['global_step'] != cfg['max_steps'] or summary['total_tokens_seen'] != cfg['max_steps'] * TOKENS_PER_UPDATE or
        summary['budget_tokens_charged'] != summary['total_tokens_seen']):
        return result
    points = {}
    with (run / 'metrics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if 'eval/nll' in row:
                tokens = row['eval/total_training_tokens_seen']
                if row['eval/documents'] != cfg['eval_documents'] or row['eval/prediction_tokens'] <= 0:
                    raise ValueError('Evaluation subset/count differs from the fixed comparison')
                points[tokens] = row
    for percentage in (90, 100):
        tokens = cfg['max_steps'] * percentage // 100 * TOKENS_PER_UPDATE
        if tokens not in points:
            return result
        row = points[tokens]
        nll = row['eval/nll'] / row['eval/prediction_tokens']
        if not math.isfinite(nll):
            result['status'] = 'unstable'; return result
        result[f'nll_{percentage}'] = nll
        result[f'eval_tokens_{percentage}'] = row['eval/prediction_tokens']
    hours = measured['h200_hours']
    accounting_path = run / 'resource-accounting.json'
    if accounting_path.exists():
        accounting = read_json(accounting_path)
        if not accounting['all_jobs_reconciled'] or accounting.get('budget_exceeded'):
            return result
        hours = accounting['h200_hours_measured']
        result['allocations'] = accounting['jobs']
    if not math.isfinite(hours) or hours <= 0:
        return result
    result.update(status='complete', h200_hours=hours,
                  allocation_tokens_per_h200_hour=result['tokens'] / hours)
    return result
