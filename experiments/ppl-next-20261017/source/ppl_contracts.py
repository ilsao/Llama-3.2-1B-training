"""Dependency-free exact 1B token plan for the two-stage mask/optimizer study."""
STUDY = 'ppl-next-20261017'
PREP = 'dolma-raw-baseline-20261008'
Q, END, SPLIT, SWITCH = 131072, 7630, 6358, 3815
TARGET = END * Q
MUON_LR, ADAM_LR = .008614891542046706, .00025844674626140116

def evaluation_units():
    return tuple(sorted({k * END // 10 for k in range(1, 10)} | {SWITCH, SPLIT, END}))

def boundaries():
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
                         lr_multiplier=schedule_multiplier(tokens//Q,cfg)))
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

def beta2_for(cfg,width,tokens): return .95

def validate(cfg):
    import json
    from pathlib import Path
    approved=json.loads((Path(__file__).parent/'approved-configs.json').read_text())
    key=f"{cfg['ppl']['arm']}-seed{cfg['seed']}"
    if key not in approved or cfg!=approved[key]:raise ValueError('Configuration is not a locked factorial arm')
    if count_updates(cfg)!=4775:raise ValueError('Wrong 1B endpoint')

def check_fork(*_):raise ValueError('No training checkpoint load')
