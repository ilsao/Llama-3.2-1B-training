"""One explicitly approved replacement of uninitialized failed job566121."""
from pathlib import Path
from training_state import read_json
from fourb_authorization import authorized,RUN_ID,CONFIG_DIGEST

def authorized_retry(run_id,cfg,seconds,gpus,resume,uninitialized_retry,jobs,root):
    approval=read_json(Path(__file__).resolve().parent.parent/'recovery-authorization.json')
    previous=[v for v in jobs.values() if v['run_id']==run_id]
    return bool(approval.get('approved') is True and approval['failed_job']=='566121'
        and approval['configuration_digest']==CONFIG_DIGEST and approval['replacement_seconds']==35940
        and authorized(run_id,cfg,resume) and not uninitialized_retry
        and seconds==35940 and gpus==8 and len(previous)==1
        and previous[0].get('job_id')=='566121' and previous[0].get('state')=='FAILED'
        and previous[0].get('elapsed_seconds')==5
        and previous[0].get('h200_hours')==8*5/3600
        and not (Path(root)/'runs/experiments'/RUN_ID).exists())
