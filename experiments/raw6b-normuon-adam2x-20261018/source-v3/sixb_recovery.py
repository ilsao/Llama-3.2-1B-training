"""Exactly one approved metadata-only replacement after566121 and567063."""
from pathlib import Path
from training_state import read_json,file_digest
from fourb_authorization import authorized,RUN_ID,CONFIG_DIGEST

def approval():
    return read_json(Path(__file__).resolve().parent.parent/'recovery2-authorization.json')

def metadata_only_directory(directory,cfg):
    directory=Path(directory);a=approval()
    if not directory.is_dir():return False
    allowed=set(a['permitted_metadata_hashes'])|{'.run.lock','resource-accounting.json'}
    if not set(p.name for p in directory.iterdir())<=allowed:return False
    if any(not (directory/name).is_file() or file_digest(directory/name)!=sha for name,sha in a['permitted_metadata_hashes'].items()):return False
    return read_json(directory/'resolved-config.json')==cfg

def authorized_retry(run_id,cfg,seconds,gpus,resume,uninitialized_retry,jobs,root):
    a=approval();previous={v.get('job_id'):v for v in jobs.values() if v['run_id']==run_id}
    if not (a.get('approved') is True and a['failed_jobs']==['566121','567063']
        and a['configuration_digest']==CONFIG_DIGEST and a['replacement_seconds']==35880
        and authorized(run_id,cfg,resume) and not uninitialized_retry
        and seconds==35880 and gpus==8 and set(previous)=={'566121','567063'}):return False
    for job,elapsed in [('566121',5),('567063',97)]:
        j=previous[job]
        if j.get('state')!='FAILED' or j.get('elapsed_seconds')!=elapsed or j.get('h200_hours')!=8*elapsed/3600:return False
    return metadata_only_directory(Path(root)/'runs/experiments'/RUN_ID,cfg)
