"""Keep a checkpointed stop (75) from becoming torchrun's generic failure (1)."""
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from train import main as train_main
    from training_state import atomic_json
    try:
        train_main()
    except SystemExit as exit_status:
        code=exit_status.code or 0
        if not isinstance(code,int): raise
    else:
        code=0
    record=Path(os.environ['JOB_RECORD'])/f'worker-rank-{os.environ.get("RANK","0")}-exit.json'
    atomic_json(record,{'exit_code':code,'checkpointed_stop':code==75})
    # All ranks finished their collectives and saved a faithful checkpoint before 75.
    return 0 if code==75 else code


if __name__=='__main__':
    sys.exit(main())
