"""Slurm-only explicit LR fork overlay; sealed numerical training code unchanged."""
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'slurm'))
from resplit_lr_runtime import fork_checkpoint
sys.path.insert(0,str(REPO/'experiments/horizon-wsd-20261014/source-v2'))

def main():
    from training_state import file_digest,read_json,atomic_json
    plan=read_json(REPO/'experiments/horizon-wsd-20261014/resplit-lr-plan.json')
    for name,expected in plan['overlay_hashes'].items():
        if file_digest(REPO/name)!=expected:raise ValueError('LR fork overlay changed: '+name)
    import train
    train.require_compute()
    train.fork_checkpoint=fork_checkpoint
    try:train.main()
    except SystemExit as error:code=error.code or 0
    else:code=0
    atomic_json(Path(os.environ['JOB_RECORD'])/f'worker-rank-{os.environ.get("RANK","0")}-exit.json',
                {'exit_code':code,'checkpointed_stop':code==75})
    return 0 if code==75 else code

if __name__=='__main__':sys.exit(main())
