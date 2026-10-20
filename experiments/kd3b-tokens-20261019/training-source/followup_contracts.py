"""Pure contracts for the user-authorized residual-half/resplit matched2B run."""
import copy
import json
import math
from pathlib import Path

NAME='init-recipe-20261014'
RUN=NAME+'-followup-rh-resplit-2b'
PARENT=NAME+'-final-2b'
LIVE=Path(__file__).resolve().parent.parent


def build_configuration(parent):
    batch=parent['world_size']*parent['microbatch_size']*parent['gradient_accumulation_steps']*8192
    if (parent['world_size']!=8 or parent['sequence_length']!=8192 or parent['category']!='full' or
        parent.get('sweep_id')!=NAME or parent['target_training_tokens']!=math.ceil(2_000_000_000/batch)*batch or
        parent['max_training_tokens']!=parent['target_training_tokens'] or parent['max_steps']*batch!=parent['target_training_tokens'] or
        parent.get('initialization','gaussian')!='gaussian' or parent.get('ema',[]) or 'phase_schedule' in parent or
        parent['lr_scheduler']!='wsd_hold_zero' or parent['max_h200_hours']!=32):
        raise ValueError('Parent is not the expected fresh2B control recipe')
    cfg=copy.deepcopy(parent);end=cfg['max_steps'];split=round(end*12715/15259)
    cfg.update(initialization='residual_half',ema=[],parent_runs=[PARENT],sweep_stage='followup-final',
        sweep_purpose='Matched2B residual-half plus resplit0.1, EMA off; compare formal7')
    cfg['phase_schedule']=[dict(start=0,end=split,peak_ratio=1.,warmup_steps=cfg['warmup_steps'],hold_schedule=copy.deepcopy(cfg['hold_schedule'])),
        dict(start=split,end=end,peak_ratio=.1,warmup_steps=cfg['warmup_steps'],hold_schedule=copy.deepcopy(cfg['hold_schedule']))]
    return cfg


def authorized(run_id,cfg,resume):
    if run_id!=RUN or resume:return False
    manifest=LIVE/'resolved.json';auth=LIVE/'authorization.json'
    if not manifest.exists() or not auth.exists():return False
    from training_state import digest,file_digest
    m=json.loads(manifest.read_text());a=json.loads(auth.read_text())
    parent=json.loads(Path(m['parent_configuration']).read_text())
    return (a['run_id']==RUN and a['formal_run_number']==8 and a['max_allocation_seconds']==14400 and a['max_h200_hours']==32
        and a['fresh_random_weights'] and not a['checkpoint_resume'] and cfg==build_configuration(parent)
        and digest(cfg)==m['configuration_digest']
        and file_digest(m['parent_configuration'])==m['parent_configuration_sha256']
        and (m['parent_completed_full_evaluation'] or
             (a.get('parallel_training') is True and m.get('parallel_training') is True and m.get('parent_configuration_verified') is True)))
