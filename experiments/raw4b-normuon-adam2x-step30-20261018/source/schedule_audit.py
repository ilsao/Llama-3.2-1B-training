"""Audit actual recorded step LRs and token cursor; metadata only."""
import json,math
from pathlib import Path
from ppl_contracts import token_prefix,multiplier_for_step

def audit_log(path,cfg,require_complete=True):
    prefix=token_prefix();records={}
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith('{'):continue
            try:row=json.loads(line)
            except ValueError:continue
            if 'train/lr_muon' not in row:continue
            step=row['train/global_step']
            if not 1<=step<=19080:raise ValueError('Unexpected logged update')
            if row['train/total_tokens_seen']!=prefix[step]:raise ValueError('Step/token cursor mismatch')
            multiplier=multiplier_for_step(step-1,cfg)
            for field,peak in [('train/lr_muon',cfg['muon']['learning_rate']),('train/learning_rate',cfg['learning_rate'])]:
                if not math.isclose(row[field],peak*multiplier,rel_tol=1e-10,abs_tol=1e-14):
                    raise ValueError(f'Actual LR disagrees with step30 schedule: step{step} {field}')
            if step in records and records[step]!=row:raise ValueError('Conflicting logged update')
            records[step]=row
    if require_complete and set(records)!=set(range(10,19081,10)):
        raise ValueError('Missing final recorded step LR evidence')
    if not records:raise ValueError('No LR records yet')
    return dict(passed=True,logged_updates=len(records),last_logged_step=max(records),
        last_logged_tokens=records[max(records)]['train/total_tokens_seen'],
        decay_clock='optimizer_updates',warmup_clock='nonpadding_tokens',
        decay_start_completed_updates=[12463,18697],phase_updates=[17804,1276],
        log_path=str(path),complete=require_complete)
