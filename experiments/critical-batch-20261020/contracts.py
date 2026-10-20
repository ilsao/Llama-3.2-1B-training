"""Independent, token-matched experiments. No inherited formal19 exception."""
import os
from pathlib import Path
from bootstrap import HERE
from training_state import read_json, digest, hold_schedule_boundaries
from critical_batch import Q

CONFIG_PATH = Path(os.environ['CBS_CONFIG_PATH']).resolve()
CFG = read_json(CONFIG_PATH)
RUN = CFG['run_id']
TARGET = CFG['max_training_tokens']
CONFIG_DIGEST = digest(CFG)


def config():
    if TARGET % Q or CFG['world_size'] != 8 or CFG['microbatch_size'] != 2:
        raise ValueError('Require complete8rank microbatch2 token windows')
    if TARGET > 6000000000:
        raise ValueError('Prior rounded6B exception does not authorize this experiment')
    if CFG['checkpoint_policy'] != 'token_events_and_final':
        raise ValueError('Require useful branch checkpoints')
    return CFG


def eval_units(cfg):
    end = TARGET//Q
    # Minimum possible number of updates uses the largest permitted batch.
    # A gap of this many token quanta can never span more updates than quanta.
    # Hence this remains <=10% of planned updates under ANY adaptive trajectory.
    minimum_updates = (end+cfg['cbs']['max_units']-1)//cfg['cbs']['max_units']
    gap = max(1, minimum_updates//10)
    return sorted({k*end//10 for k in range(1, 11)} | set(range(gap,end+1,gap)))


def event_units(cfg):
    end = TARGET//Q
    result = {0, end, *eval_units(cfg)}
    for token in cfg['branch_checkpoint_tokens']:
        if token <= TARGET:
            result.add(token//Q)
    result.update(range(cfg['cbs']['probe_interval_tokens']//Q, end+1,
                        cfg['cbs']['probe_interval_tokens']//Q))
    for phase in cfg['phase_schedule']:
        a, b = phase['start'], phase['end']
        result.update((a, a+phase['warmup_steps'], b-1, b))
        result.update(a+x for x in hold_schedule_boundaries(b-a, phase['warmup_steps'], phase['hold_schedule']))
    result.add(cfg['batch_switch_tokens']//Q)
    return sorted(x for x in result if x <= end)
