"""Exact user-approved fresh formal11, single6h/48H allocation."""
RUN_ID='raw3b-sota-wsd20-20261015'
CONFIG_DIGEST='efd9b2a00aa2d225d9bc6b84e433384093288ed8ed266417591b46f8f432b5a7'

def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST


def fresh_restart_authorized(run_id,cfg,previous,run_path):
    """Only the approved zero-token import failure; no later attempt qualifies."""
    from pathlib import Path
    from training_state import read_json
    if not authorized(run_id,cfg) or len(previous)!=2:return False
    old={item.get('job_id'):item for item in previous}
    if set(old)!={'553561','553577'}:return False
    for job,seconds in [('553561',20),('553577',0)]:
        item=old[job]
        if (item.get('state')!='FAILED' or item.get('elapsed_seconds')!=seconds or
            item.get('gpu_count')!=8 or item.get('h200_hours')!=8*seconds/3600):return False
    path=Path(run_path)
    if path.exists():
        if any(p.name not in {'.run.lock','resolved-config.json','resource-accounting.json'} or not p.is_file() for p in path.iterdir()):return False
        config_path=path/'resolved-config.json'
        if config_path.exists() and read_json(config_path)!=cfg:return False
    return True
