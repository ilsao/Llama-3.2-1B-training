"""Pure exact approved configuration, token windows and scoped budget."""
from bootstrap import HERE
from training_state import read_json,digest,hold_schedule_boundaries,schedule_multiplier
RUN='kd6b-r5-20261020-seed42'
CONFIG_DIGEST='3080c174a87a531814a8e26b5caf122b66806046c51c57cf9b58329278ecc24c'
Q=131072
TARGET=6000082944
SECONDS=32400
STUDY_CAP=248.1786111111111
PREFIXES=('methods-parallel-20261019','kd-acceleration-','kd-pipeline-',
          'kd3b-online-pipeline-','kd3b-io-validation-','kd3b-io8-validation-',
          'kd3b-hidden8-','kd6b-lossless-','kd6b-lossless2-','kd6b-r5-')


def config():
    cfg=read_json(HERE/'config.json')
    if digest(cfg)!=CONFIG_DIGEST:raise ValueError('Approved configuration changed')
    auth=read_json(HERE/'authorization.json')
    if not auth['approved'] or auth['configuration_digest']!=CONFIG_DIGEST or auth['run_id']!=RUN:
        raise ValueError('Missing exact authorization')
    return cfg


def eval_units(cfg):
    end=cfg['scheduler_steps']
    return sorted({k*end//10 for k in range(1,10)} | {
        cfg['batch_switch_tokens']//Q,cfg['phase_schedule'][0]['end'],end})


def event_units(cfg):
    result={0,cfg['batch_switch_tokens']//Q,cfg['scheduler_steps'],*eval_units(cfg)}
    for p in cfg['phase_schedule']:
        a,b=p['start'],p['end']
        result.update((a,a+p['warmup_steps'],b-1,b))
        result.update(a+x for x in hold_schedule_boundaries(b-a,p['warmup_steps'],p['hold_schedule']))
    return sorted(result)


def plan(cfg):
    events=event_units(cfg);rows=[];tokens=0
    while tokens<TARGET:
        u=tokens//Q
        units=min(1 if tokens<cfg['batch_switch_tokens'] else 4,next(x for x in events if x>u)-u)
        width=16*units
        rows.append(dict(update=len(rows)+1,start_tokens=tokens,end_tokens=tokens+width*8192,
                         width=width,accumulation=units,lr_multiplier=schedule_multiplier(u,cfg)))
        tokens+=width*8192
    if len(rows)!=28619 or tokens!=TARGET:raise ValueError('6B window plan changed')
    return rows


def study_used(data):
    return sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values()
               if j['run_id'].startswith(PREFIXES))

