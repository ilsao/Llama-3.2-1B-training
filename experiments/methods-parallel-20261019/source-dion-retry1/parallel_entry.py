"""Keep incomplete/failed exit codes and bound Python shutdown after local logs."""
import os,sys,traceback
from pathlib import Path

def main():
    from train import require_compute
    require_compute()
    from parallel_runtime import install
    install()
    import train
    return train.main()

if __name__=='__main__':
    code=0
    try:
        value=main()
        if isinstance(value,int):code=value
    except SystemExit as exc:code=exc.code if isinstance(exc.code,int) else 1
    except BaseException:
        traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
