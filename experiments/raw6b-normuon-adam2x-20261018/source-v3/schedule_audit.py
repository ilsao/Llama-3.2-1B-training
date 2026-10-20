"""Check observed LRs and actual token trajectory against the approved token clock."""
import json,math
from pathlib import Path
from ppl_contracts import batch_plan

def audit_log(path,cfg,require_complete=True):
    plan=batch_plan(cfg);records={}
    for line in Path(path).open():
        if not line.startswith('{'):continue
        try:row=json.loads(line)
        except ValueError:continue
        if 'train/lr_muon' not in row:continue
        step=row['train/global_step']
        if not 1<=step<=len(plan):raise ValueError('Unexpected update')
        expected=plan[step-1]
        if row['train/total_tokens_seen']!=expected['end_tokens']:raise ValueError('Actual token trajectory mismatch')
        for field,peak in [('train/lr_muon',cfg['muon']['learning_rate']),('train/learning_rate',cfg['learning_rate'])]:
            if not math.isclose(row[field],peak*expected['lr_multiplier'],rel_tol=1e-10,abs_tol=1e-14):raise ValueError('Observed LR differs from token-clock40% schedule')
        if step in records and records[step]!=row:raise ValueError('Conflicting logged update')
        records[step]=row
    expected_steps=set(range(cfg['logging_steps'],len(plan)+1,cfg['logging_steps']))|{len(plan)}
    if require_complete and set(records)!=expected_steps:raise ValueError('Missing LR evidence')
    if not records:raise ValueError('No LR evidence')
    return dict(passed=True,logged_updates=len(records),last_logged_step=max(records),last_logged_tokens=records[max(records)]['train/total_tokens_seen'],decay_clock='nonpadding_tokens',decay_ratio=.4,complete=require_complete)
