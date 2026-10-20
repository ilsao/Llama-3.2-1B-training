"""Exact 4B token plan for the user-approved fresh formal16."""
from functools import lru_cache
STUDY = 'raw4b-normuon-adam2x-step30-20261018'
PREP = 'dolma-raw-baseline-20261008'
Q, END, SPLIT, SWITCH = 131072, 30518, 25431, 15259
TARGET = END * Q
MUON_LR, ADAM_LR = .004995613826805825, .00029973682960834946

def evaluation_units():
    return tuple(sorted({k * END // 10 for k in range(1, 10)} | {SWITCH, SPLIT, END}))

def boundaries():
    # Preserve all parent data-window events, including old 40% token knots.
    # These only split batches; they do not drive the new LR decay.
    from training_state import hold_schedule_boundaries
    tail = dict(first_decay_ratio=.4*2/3, hold_ratio=.4/6,
                final_decay_ratio=.4/6, plateau_lr_ratio=.2)
    result = {0, SWITCH, END}
    for start, end in ((0, SPLIT), (SPLIT, END)):
        result.update({start, start+170, end-1, end})
        result.update(start+x for x in hold_schedule_boundaries(end-start,170,tail))
    return tuple(sorted(result))

def events(cfg):
    return sorted(set(boundaries()) | set(evaluation_units()))

def window(cfg, tokens):
    if type(tokens) is not int or tokens % Q or not 0 <= tokens <= TARGET:
        raise ValueError('Incomplete BS16-equivalent token cursor')
    if tokens == TARGET: return 0, 0
    unit = tokens // Q
    units = min(1 if unit < SWITCH else 4, next(x for x in events(cfg) if x > unit)-unit)
    return 16*units, 2*units

def batch_plan(cfg):
    rows=[];tokens=0
    from training_state import schedule_multiplier
    while tokens < TARGET:
        width, accumulation=window(cfg,tokens)
        end=tokens+width*8192
        rows.append(dict(update=len(rows)+1,start_tokens=tokens,end_tokens=end,
                         width=width,accumulation=accumulation,
                         lr_multiplier=schedule_multiplier(len(rows),cfg)))
        tokens=end
    return rows

def count_updates(cfg,start_tokens=0):
    tokens=start_tokens;updates=0
    while tokens < TARGET:
        width,_=window(cfg,tokens);tokens+=width*8192;updates+=1
    return updates

def observer_knots(cfg):
    endpoints=[r['end_tokens'] for r in batch_plan(cfg)]
    return dict(ema_start=min(x for x in endpoints if x>=.95*TARGET),
                mean=[min(endpoints,key=lambda x:(abs(x-f*TARGET),x)) for f in (.98,.985,.99,.995,1.)])

@lru_cache(maxsize=1)
def token_prefix():
    tokens=[0]
    while tokens[-1] < TARGET:
        width,_=window(None,tokens[-1]);tokens.append(tokens[-1]+width*8192)
    return tuple(tokens)

def multiplier_for_step(completed_updates,cfg):
    """Next update LR: unchanged token warmup, last30% actual-update WSD tail."""
    from training_state import lr_multiplier
    if type(completed_updates) is not int or not 0 <= completed_updates <= 19080:
        raise ValueError('Invalid actual optimizer-update cursor')
    index=min(completed_updates,19079)
    tokens=token_prefix()[index]
    phase=cfg['phase_schedule'][0 if index<17804 else 1]
    token_start=0 if index<17804 else SPLIT*Q
    warmup_units=(tokens-token_start)//Q
    if warmup_units<phase['warmup_steps']:
        # This exactly preserves the parent warmup, including variable batches.
        return phase['peak_ratio']*(warmup_units+1)/phase['warmup_steps']
    return phase['peak_ratio']*lr_multiplier(index-phase['start'],phase['end']-phase['start'],0,0,
        'wsd_hold_zero',cfg['wsd_decay_ratio'],phase['hold_schedule'])

def beta2_for(cfg,width,tokens): return .95

def validate(cfg):
    from fourb_authorization import authorized
    if not authorized(STUDY,cfg):raise ValueError('Exact approved4B config digest changed')
    expected=dict(sweep_id=STUDY,category='full',world_size=8,microbatch_size=1,
                  gradient_accumulation_steps=2,sequence_length=8192,
                  target_training_tokens=TARGET,max_training_tokens=TARGET,
                  max_steps=19080,scheduler_steps=19080,max_h200_hours=56,
                  batch_switch_tokens=SWITCH*Q,schedule_clock='optimizer_updates',warmup_clock='nonpadding_tokens',
                  reference_batch_tokens=Q,adamw_beta2_policy='fixed',betas=[.9,.95],
                  learning_rate=ADAM_LR,eval_documents=1000,eval_steps=1900,
                  seed=42,initialization='residual_half',ema=[],clipping='global',
                  max_grad_norm=.5,stop_margin_seconds=300,
                  final_evaluation_reserve_seconds=900,checkpoint_policy='final_only',
                  fixed_horizon=True,export_on_completion=True,wandb_mode='online')
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Approved4B fixed recipe changed')
    if cfg['wsd_decay_ratio'] != .3 or cfg['hold_schedule'] != dict(first_decay_ratio=.2,hold_ratio=.05,final_decay_ratio=.05,plateau_lr_ratio=.2):
        raise ValueError('Require last30% step tail with unchanged 4:1:1 shape')
    if token_prefix()[17804] != SPLIT*Q or token_prefix()[-1] != TARGET:
        raise ValueError('Phase/data cursor changed')
    p=cfg['ppl']
    if set(p)!={'stage','arm','mask','matrix'}:raise ValueError('Incomplete method identity')
    if p != dict(stage=3,arm='nor',mask='intradoc',matrix='nor'):
        raise ValueError('Only the approved intradoc NorMuon NS5 method is allowed')
    if cfg['muon']['learning_rate']!=MUON_LR or cfg['eval_steps']>count_updates(cfg)//10:
        raise ValueError('Incorrect LR or eval frequency')
    if cfg.get('matrix_optimizer') or cfg.get('fork_snapshot_tokens'):
        raise ValueError('Fresh initialization, isolated adapters only')

def check_fork(*_): raise ValueError('No training checkpoint loading in this study')
