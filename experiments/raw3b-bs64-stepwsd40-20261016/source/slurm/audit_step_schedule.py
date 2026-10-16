"""Prepare and audit actual-step LR tables;never load tensors or models."""
import csv
import json
import math
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from training_state import DEFAULT_ROOT, atomic_json, load_config, schedule_multiplier
from batch3b_contracts import STUDY, PLANNED_UPDATES, STEP_SPLIT, token_prefix


def expected_rows(cfg):
    prefix = token_prefix()
    return [dict(global_step=i + 1, completed_updates=i, inputs_before=prefix[i],
                 inputs_after=prefix[i + 1], batch=(prefix[i + 1] - prefix[i]) // 8192,
                 muon_lr=cfg['muon']['learning_rate'] * schedule_multiplier(i, cfg),
                 adamw_lr=cfg['learning_rate'] * schedule_multiplier(i, cfg))
            for i in range(PLANNED_UPDATES)]


def audit_rows(rows, cfg):
    expected = expected_rows(cfg)
    seen = set()
    for row in rows:
        step = row['train/global_step']
        if not 1 <= step <= PLANNED_UPDATES:
            raise ValueError('Logged step outside actual horizon')
        e = expected[step - 1]
        checks = dict(
            tokens=row['train/total_tokens_seen'] == e['inputs_after'],
            batch=row['train/global_batch_sequences'] == e['batch'],
            beta2=row['train/adamw_beta2'] == .95,
            scheduler_index=row['train/scheduler_completed_updates'] == step - 1,
            muon_lr=math.isclose(row['train/lr_muon'],e['muon_lr'],rel_tol=1e-12,abs_tol=1e-15),
            adamw_lr=math.isclose(row['train/learning_rate'],e['adamw_lr'],rel_tol=1e-12,abs_tol=1e-15))
        if not all(checks.values()):
            raise ValueError(f'Actual-step schedule mismatch at update{step}: {checks}')
        seen.add(step)
    required = set(range(cfg['logging_steps'], PLANNED_UPDATES, cfg['logging_steps'])) | {PLANNED_UPDATES}
    if not required <= seen:
        raise ValueError('Missing actual LR logging steps')
    return dict(passed=True, logged_updates=len(rows), distinct_steps=len(seen),
                actual_updates=PLANNED_UPDATES, decay_clock='optimizer_updates',
                warmup_clock='optimizer_updates', phase_updates=[STEP_SPLIT,PLANNED_UPDATES-STEP_SPLIT],
                decay_start_completed_updates=[8014,13929],
                plateau_start_completed_updates=[11575,14184],
                final_ramp_start_completed_updates=[12465,14248])


def prepare(cfg):
    directory = DEFAULT_ROOT / 'runs' / STUDY
    directory.mkdir(parents=True,exist_ok=True)
    with (directory / 'expected-step-lr.csv').open('x',newline='') as stream:
        rows = expected_rows(cfg)
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def audit(cfg,job_id):
    path = DEFAULT_ROOT / 'logs' / f'raw3b-bs64-stepwsd40-{job_id}.out'
    rows=[]
    for line in path.open():
        if not line.startswith('{'):continue
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if 'train/lr_muon' in row:rows.append(row)
    result = audit_rows(rows,cfg)
    result.update(job_id=job_id,log_path=str(path))
    atomic_json(SOURCE.parent / 'schedule-audit.json',result)
    return result


if __name__ == '__main__':
    cfg=load_config(SOURCE.parent / 'config.json')
    if sys.argv[1]=='prepare':prepare(cfg)
    elif sys.argv[1]=='audit':audit(cfg,sys.argv[2])
    else:raise ValueError('Unknown audit action')
