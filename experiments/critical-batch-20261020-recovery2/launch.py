"""Prepared startup replacement; explicit replacement approval required."""
import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path
from bootstrap import HERE,REPO,SOURCE
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,digest,file_digest,locked,read_json
ORIGINAL=HERE.parent/'critical-batch-20261020'
FAILED={'critical-batch-20261020-c0-shadow':{'587009','587032'},'critical-batch-20261020-c1-cbs-half':{'587010','587033'}}
SECONDS=7080


def inventory():
    previous=read_json(ORIGINAL/'source-seal.json')['sha256']
    paths=[REPO/n for n in previous if not n.startswith('experiments/critical-batch-20261020/')
           and n!='slurm/critical-batch-20261020.sbatch']
    paths+=list(HERE.glob('*.py'))+list((HERE/'configs').glob('*.json'))
    paths+=[REPO/'slurm/critical-batch-20261020-recovery2.sbatch']
    return {str(p.relative_to(REPO)):file_digest(p) for p in sorted(set(paths))}


def verify_source_seal():
    assert inventory()==read_json(HERE/'source-seal.json')['sha256']
    assert file_digest(ORIGINAL/'source-seal.json')==read_json(HERE/'plan.json')['original_source_seal_sha256']


def prepare():
    assert not (HERE/'submissions.json').exists()
    atomic_json(HERE/'source-seal.json',dict(sha256=inventory()))


def submit(path):
    verify_source_seal();auth=read_json(HERE/'authorization.json')
    assert auth['approved'],'Replacement allocations require explicit approval; none submitted'
    assert auth['source_seal_sha256']==file_digest(HERE/'source-seal.json')
    cfg=read_json(path);run=cfg['run_id']
    assert run in FAILED and path.parent==(HERE/'configs').resolve()
    assert digest(cfg)==read_json(ORIGINAL/'authorization.json')['configs'][run]==auth['configs'][run]
    output=DEFAULT_ROOT/'runs/experiments'/(run+'-startup-replacement2')
    with locked(HERE/'.submission.lock',blocking=False):
        state=read_json(HERE/'submissions.json') if (HERE/'submissions.json').exists() else {}
        assert run not in state and not output.exists(),'Never retry this replacement automatically'
        ledger=ExperimentLedger(DEFAULT_ROOT)
        with ledger.transaction() as data:
            previous=[j for j in data['jobs'].values() if j['run_id']==run]
            assert {j['job_id'] for j in previous}==FAILED[run] and all(j['state']=='FAILED' for j in previous)
            hours=8*SECONDS/3600
            assert sum(j['h200_hours'] for j in previous)+hours<=data['runs'][run]['max_h200_hours']
            shared=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values())
            assert shared+hours<=read_json(ORIGINAL/'authorization.json')['shared_ledger_cap_h200_hours']
            campaign=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values()
                         if j['run_id'].startswith('critical-batch-20261020'))
            assert campaign+hours<=read_json(ORIGINAL/'authorization.json')['campaign_cap_h200_hours']
            prior=DEFAULT_ROOT/'runs/experiments'/run
            for role in ('student','teacher'):
                journal=read_json(prior/f'{role}-input-journal.json')
                assert journal['confirmed_tokens']==0 and journal['charged_tokens']==131072
            assert auth['logical_input_cap_per_role']==cfg['max_training_tokens']+131072
            tag=run+'-startup-replacement2-'+uuid.uuid4().hex
            data['jobs'][tag]=dict(run_id=run,reserved_h200_hours=hours,gpu_count=8,cpu_count=8,
                requested_seconds=SECONDS,status='reserved',authorization_file=str(HERE/'authorization.json'),
                no_automatic_retry=True,prior_charged_inputs_per_role=131072,attempt_output=str(output))
            data['runs'][run].update(logical_input_cap_per_role=auth['logical_input_cap_per_role'],
                                    prior_charged_inputs_per_role=131072)
        item=dict(status='submission_uncertain',reservation=tag,config=str(path),output=str(output))
        state[run]=item;atomic_json(HERE/'submissions.json',state)
        cmd=['sbatch','--parsable','--comment='+tag,str(REPO/'slurm/critical-batch-20261020-recovery2.sbatch'),run]
        env={**os.environ,'CBS_CONFIG_PATH':str(path),'TRAIN_SOURCE_ROOT':str(SOURCE)}
        env.pop('CBS_VALIDATION_ONLY',None)
        result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=45)
        item['result']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        atomic_json(HERE/'submissions.json',state)
        job=result.stdout.strip().split(';')[0]
        assert result.returncode==0 and job.isdigit(),'Submission uncertain: reservation retained, do not retry'
        ledger.bind(tag,job);item.update(status='submitted',job_id=job)
        atomic_json(HERE/'submissions.json',state);print(json.dumps(item))


def status():
    path=HERE/'submissions.json'
    if not path.exists():print('No replacement allocations submitted');return
    with locked(HERE/'.submission.lock',blocking=False):
        state=read_json(path)
        for run,item in state.items():
            if not item.get('job_id'):
                print(run,'Submission uncertain; retain reservation and inspect Slurm comment');continue
            result=subprocess.run(['sacct','-X','-j',item['job_id'],'-n','-P','-o','JobIDRaw,State'],
                                  capture_output=True,text=True,check=True,timeout=30)
            states=[r.split('|')[1].split()[0].rstrip('+') for r in result.stdout.splitlines()
                    if r.split('|')[0]==item['job_id']]
            item['slurm_state']=states[0] if states else 'UNKNOWN'
            if item['slurm_state'] in TERMINAL:
                ledger=ExperimentLedger(DEFAULT_ROOT)
                item.update(accounting=ledger.reconcile(item['job_id']),status='reconciled')
                summary=Path(item['output'])/'summary.json'
                if summary.exists():
                    data=read_json(ledger.path);s=read_json(summary)
                    s['h200_hours_final']=data['runs'][run]['logical_h200_hours_measured']
                    atomic_json(summary,s)
        atomic_json(path,state);print(json.dumps(state,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','submit','status']);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':verify_source_seal();print('Replacement source seal verified')
    elif a.action=='status':status()
    else:submit(a.config.resolve())
