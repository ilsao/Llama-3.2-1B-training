"""Exact formal20 scope; all schedule/probe/checkpoint events use input tokens."""
from bootstrap import HERE
from training_state import read_json,digest
from critical_batch import Q
CFG=read_json(HERE/'config.json')
RUN=CFG['run_id'];TARGET=CFG['max_training_tokens'];CONFIG_DIGEST=digest(CFG)
SECONDS=32400

def config():
    auth=read_json(HERE/'authorization.json')
    assert auth['approved'] and auth['configuration_digest']==CONFIG_DIGEST
    assert TARGET==6000082944 and TARGET%Q==0
    assert CFG['world_size']==8 and CFG['microbatch_size']==2
    assert CFG['cbs']['mode']=='adaptive' and CFG['cbs']['fraction']==1.0
    assert CFG['batch_switch_tokens'] is None and CFG['training_checkpoint_load'] is False
    assert CFG['token_phases'][0]['decay_start_tokens']==2399928320
    return CFG

def eval_units(cfg):
    end=TARGET//Q
    minimum_updates=(end+cfg['cbs']['max_units']-1)//cfg['cbs']['max_units']
    gap=max(1,min(570,minimum_updates//10))
    return sorted({k*end//10 for k in range(1,11)}|set(range(gap,end+1,gap))|{p[k]//Q for p in cfg['token_phases'] for k in ('decay_start_tokens','plateau_start_tokens','final_decay_start_tokens','end_tokens')})

def event_units(cfg):
    end=TARGET//Q
    result={0,end,*eval_units(cfg)}
    result.update(t//Q for t in cfg['branch_checkpoint_tokens'] if t<=TARGET)
    result.update(range(cfg['cbs']['probe_interval_tokens']//Q,end+1,cfg['cbs']['probe_interval_tokens']//Q))
    for p in cfg['token_phases']:
        for k in ('start_tokens','warmup_end_tokens','decay_start_tokens','plateau_start_tokens','final_decay_start_tokens','end_tokens'):
            assert p[k]%Q==0
            result.add(p[k]//Q)
        result.add(p['end_tokens']//Q-1)
    return sorted(result)
