import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from recipe_tail_runtime import fork_checkpoint
sys.path.insert(0,str(Path(os.environ['RECIPE_REPO'])/'experiments/horizon-wsd-20261014/source-v2'))
def main():
 import train
 from training_state import atomic_json
 train.fork_checkpoint=fork_checkpoint
 try:train.main()
 except SystemExit as error:code=error.code or 0
 else:code=0
 atomic_json(Path(os.environ['JOB_RECORD'])/f'worker-rank-{os.environ.get("RANK","0")}-exit.json',{'exit_code':code,'checkpointed_stop':code==75})
 return 0 if code==75 else code
if __name__=='__main__':sys.exit(main())
