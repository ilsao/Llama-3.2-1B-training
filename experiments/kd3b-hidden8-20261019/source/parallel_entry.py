"""Keep incomplete/failed exit codes and bound Python shutdown after local logs."""
import os,sys,traceback
from pathlib import Path

def main():
    from train import require_compute
    require_compute()
    import train
    from parallel_runtime import build_optimizers, Scheduler
    train.build_optimizers=build_optimizers
    train.TokenScheduler=Scheduler
    train.configure_update=lambda optimizers,cfg,width,tokens: .95
    return train.main()

if __name__=='__main__':
    code=0
    try:
        value=main()
        if isinstance(value,int):code=value
    except SystemExit as exc:code=exc.code if isinstance(exc.code,int) else 1
    except BaseException as exc:
        import time
        from training_state import DEFAULT_ROOT,atomic_json
        from hidden_contracts import RUN
        root=DEFAULT_ROOT/'runs/experiments'/RUN/'transport'
        root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/f'abort-consumer-{os.environ.get("RANK","unknown")}.json',dict(error=repr(exc),timestamp=time.time()))
        traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
