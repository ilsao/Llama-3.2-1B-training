#!/usr/bin/env python3
"""Single authorized teacher/student submission; no automatic retries or resumes."""
import argparse
import datetime
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
LIVE=REPO/'experiments/kd3b-hidden8-20261019'
SOURCE=LIVE/'source'
SCRIPT=REPO/'slurm/kd3b-hidden8.sbatch'
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,atomic_json,digest,file_digest,load_config,locked,read_json
from hidden_contracts import RUN
PREFIXES=('methods-parallel-20261019','kd-acceleration-','kd-pipeline-',
          'kd3b-online-pipeline-','kd3b-io-validation-','kd3b-io8-validation-', 'kd3b-hidden8-')


def inventory():
    files=[p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [LIVE/name for name in ('config.json','authorization.json','protocol.md','schedule-plan.json')]
    files += [SCRIPT,Path(__file__).resolve()]
    return {str(p.relative_to(REPO)):file_digest(p) for p in sorted(files)}


def check():
    if inventory()!=read_json(LIVE/'execution-seal.json')['sha256']:raise ValueError('Execution seal changed')
    cfg=load_config(LIVE/'config.json');auth=read_json(LIVE/'authorization.json')
    if (auth['status']!='approved' or auth['user_reply']!='好，開始訓練' or auth['run_id']!=RUN or
        auth['config_digest']!=digest(cfg) or cfg['max_h200_hours']!=96 or auth['formal_run_number']!=18 or
        auth['seconds_per_role']!=21600 or auth['max_h200_hours_combined']!=96):
        raise ValueError('Authorization/configuration mismatch')
    return cfg,auth


def reserve_pair(ledger,cfg,auth,tags):
    # A single locked transaction reserves both roles against one formal identity.
    # The ordinary global ledger submission gates remain unchanged.
    if (set(tags)!={'producer','consumer'} or len(set(tags.values()))!=2 or
        auth['run_id']!=RUN or auth['config_digest']!=digest(cfg) or
        cfg['category']!='full' or cfg['world_size']!=8 or cfg['max_h200_hours']!=96):
        raise ValueError('Require exact approved pair/configuration')
    with ledger.transaction() as data:
        if RUN in data['runs'] or any(t in data['jobs'] for t in tags.values()):
            raise ValueError('Existing identity; no retry or second allocation')
        full={k:v for k,v in data['runs'].items() if v['category']=='full'}
        if len(full)!=17:raise ValueError('Expected all17 prior formal identities')
        historical={k:j for k,j in data['jobs'].items() if j['run_id'].startswith(PREFIXES)}
        used=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in historical.values())
        if used+96>auth['study_cap_h200_hours']+1e-9:raise ValueError('Combined methods-study cap exceeded')
        data['runs'][RUN]=dict(category='full',config_digest=digest(cfg),max_h200_hours=96,
            created=time.time(),formal_run_number=18,roles=['producer','consumer'],
            authorization=str(LIVE/'authorization.json'),study='methods-parallel-20261019')
        for role,tag in tags.items():
            if role not in ('producer','consumer'):raise ValueError('Unknown role')
            data['jobs'][tag]=dict(run_id=RUN,role=role,reserved_h200_hours=48,gpu_count=8,cpu_count=8,
                requested_seconds=21600,status='reserved',authorization=str(LIVE/'authorization.json'))
        return dict(prior_h200_hours=used,maximum_after_submission=used+96,
            study_cap_h200_hours=auth['study_cap_h200_hours'],preserved_job_ids=sorted(historical),
            preserved_formal_run_ids=sorted(full),new_reserved_h200_hours=96)


def submit():
    cfg,auth=check();ledger=ExperimentLedger(DEFAULT_ROOT)
    with locked(LIVE/'.submission.lock',blocking=False):
        record=LIVE/'submission.json'
        if record.exists() or (DEFAULT_ROOT/'runs/experiments'/RUN).exists():
            raise ValueError('Existing launch/artifacts; inspect instead of submitting again')
        (DEFAULT_ROOT/'logs').mkdir(parents=True,exist_ok=True)
        before=read_json(ledger.path)
        atomic_json(LIVE/'ledger-before.json',before)
        tags={r:'kd3bh8-'+r+'-'+uuid.uuid4().hex for r in ('producer','consumer')}
        state=dict(run_id=RUN,config_digest=digest(cfg),execution_seal_sha256=file_digest(LIVE/'execution-seal.json'),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='intent',roles={},
            checkpoint_load=False,automatic_retry=False)
        atomic_json(record,state)
        state['budget']=reserve_pair(ledger,cfg,auth,tags);state['phase']='reserved'
        atomic_json(record,state)
        try:
            for role,tag in tags.items():
                state['roles'][role]=dict(reservation=tag,phase='submitting_held')
                atomic_json(record,state)
                command=['sbatch','--parsable','--hold','--job-name=kd3bh8-'+role,'--comment='+tag,str(SCRIPT),RUN,role]
                result=subprocess.run(command,text=True,capture_output=True,timeout=45)
                job=result.stdout.strip().split(';')[0]
                if result.returncode or not re.fullmatch(r'\d+',job):raise RuntimeError('Uncertain submission: '+result.stderr)
                state['roles'][role].update(job_id=job,phase='accepted_held');atomic_json(record,state)
                ledger.bind(tag,job)
                state['roles'][role]['phase']='bound_held';atomic_json(record,state)
            state['phase']='releasing';atomic_json(record,state)
            subprocess.run(['scontrol','release',','.join(x['job_id'] for x in state['roles'].values())],check=True,timeout=30)
            state['phase']='submitted';atomic_json(record,state)
        except BaseException as error:
            state.update(error=repr(error),recovery_required=True);atomic_json(record,state)
            raise RuntimeError('Retain reservations; recover recorded tags/jobs without another sbatch') from error
        print(json.dumps(state,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','check','submit','reconcile','status'])
    args=parser.parse_args()
    if args.action=='seal':
        if (LIVE/'submission.json').exists():raise ValueError('Cannot reseal submitted code')
        atomic_json(LIVE/'execution-seal.json',dict(sha256=inventory()))
        check();print('Sealed; no allocation submitted')
    elif args.action=='check':
        cfg,auth=check();print(json.dumps(dict(checked=True,run_id=RUN,config_digest=digest(cfg))))
    elif args.action=='submit':submit()
    else:
        state=read_json(LIVE/'submission.json');jobs=[x['job_id'] for x in state['roles'].values()]
        if args.action=='status':
            subprocess.run(['sacct','-X','-j',','.join(jobs),'--format=JobIDRaw,JobName,State,Elapsed,Timelimit,NodeList'],check=True)
        else:
            from training_state import slurm_record
            for job in jobs:slurm_record(job,DEFAULT_ROOT)
            result={job:ExperimentLedger(DEFAULT_ROOT).reconcile(job) for job in jobs}
            atomic_json(LIVE/'accounting.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
