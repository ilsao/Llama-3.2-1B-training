#!/usr/bin/env python3
"""Recover only the two cancelled, never-allocated held submissions; manual invocation."""
import json,subprocess,sys,uuid,datetime
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'slurm'))
from kd3b_hidden8 import LIVE,SCRIPT,RUN,check,PREFIXES
from training_state import DEFAULT_ROOT,ExperimentLedger,read_json,atomic_json,locked,slurm_record,file_digest,digest

def main():
    cfg,auth=check();ledger=ExperimentLedger(DEFAULT_ROOT)
    with locked(LIVE/'.submission.lock',blocking=False):
        record=LIVE/'submission.json';state=read_json(record)
        old={'producer':'575359','consumer':'575360'}
        if state.get('never_allocated_recovery') or {r:x['job_id'] for r,x in state['roles'].items()}!=old:
            raise ValueError('This recovery is restricted to the original held pair')
        if (DEFAULT_ROOT/'runs/experiments'/RUN).exists():raise ValueError('Run artifacts exist')
        proofs={r:slurm_record(job,DEFAULT_ROOT) for r,job in old.items()}
        if any(p['state'].split()[0]!='CANCELLED' or p['elapsed_seconds']!=0 or p['gpu_count']!=0 for p in proofs.values()):
            raise ValueError('Both jobs must be confirmed cancelled with zero allocation before recovery')
        for job in old.values():ledger.reconcile(job)
        tags={r:'kd3bh8-direct-'+r+'-'+uuid.uuid4().hex for r in old}
        with ledger.transaction() as data:
            if data['runs'][RUN]['config_digest']!=digest(cfg):raise ValueError('Run changed')
            prior={k:j for k,j in data['jobs'].items() if j['run_id']==RUN}
            if set(prior)!=set(old.values()) or any(j.get('h200_hours')!=0 for j in prior.values()):
                raise ValueError('Unknown or resource-consuming prior job; no allocation retry')
            cost=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j['run_id'].startswith(PREFIXES))
            if cost+96>auth['study_cap_h200_hours']+1e-9:raise ValueError('Study budget exceeded')
            for role,tag in tags.items():
                data['jobs'][tag]=dict(run_id=RUN,role=role,reserved_h200_hours=48,gpu_count=8,cpu_count=8,
                    requested_seconds=21600,status='reserved',authorization=str(LIVE/'authorization.json'),
                    never_allocated_held_predecessor=old[role])
        state['never_allocated_recovery']=dict(reason='Slurm refused release of held jobs; both original IDs cancelled before any allocation',
            evidence=proofs,original_roles=state['roles'],script_sha256=file_digest(__file__),
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        state['roles']={r:dict(reservation=t,phase='reserved_direct') for r,t in tags.items()}
        state['phase']='direct_submission';atomic_json(record,state)
        try:
            for role,tag in tags.items():
                result=subprocess.run(['sbatch','--parsable','--job-name=kd3bh8-'+role,'--comment='+tag,str(SCRIPT),RUN,role],
                    text=True,capture_output=True,timeout=45)
                job=result.stdout.strip().split(';')[0]
                if result.returncode or not job.isdigit():raise RuntimeError('Uncertain submission: '+result.stderr)
                state['roles'][role].update(job_id=job,phase='accepted');atomic_json(record,state)
                ledger.bind(tag,job)
                state['roles'][role]['phase']='submitted';atomic_json(record,state)
            state['phase']='submitted';state['recovery_required']=False;atomic_json(record,state)
        except BaseException as error:
            state['recovery_error']=repr(error);state['recovery_required']=True;atomic_json(record,state)
            raise
        print(json.dumps(state,indent=2))

if __name__=='__main__':main()
