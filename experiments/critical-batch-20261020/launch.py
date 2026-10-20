"""Prepare-only by default. New scoped approval is required to submit pilots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from bootstrap import HERE, REPO, SOURCE
from training_state import DEFAULT_ROOT, ExperimentLedger, TERMINAL, atomic_json, digest, file_digest, locked, read_json

CAMPAIGN = 'critical-batch-20261020'


def inventory():
    # Borrow exactly the preserved R5 dependency versions; never modify/reseal R5.
    prior = read_json(HERE.parent/'kd6b-r5-20261020/execution-seal.json')['sha256']
    files = [REPO/n for n in prior if n.startswith(str(SOURCE.relative_to(REPO))+'/')]
    files += [HERE.parent/'kd6b-r5-20261020/checkpointing.py',
              HERE.parent/'kd6b-lossless-20261020/fused_kd.py',
              HERE.parent/'kd6b-lossless2-20261020/accelerations.py']
    files += list(HERE.glob('*.py')) + list((HERE/'configs').glob('*.json'))
    files += [REPO/'slurm/critical-batch-20261020.sbatch']
    return {str(p.relative_to(REPO)):file_digest(p) for p in sorted(set(files))}


def verify_source_seal():
    assert inventory() == read_json(HERE/'source-seal.json')['sha256'], 'Source/config changed; prepare again before approval'


def prepare():
    assert not (HERE/'submissions.json').exists(), 'Never reseal after submission'
    atomic_json(HERE/'source-seal.json', dict(sha256=inventory()))
    print('Prepared only; no jobs submitted')


def submit(path):
    verify_source_seal()
    cfg=read_json(path)
    assert path.parent == (HERE/'configs').resolve()
    assert cfg['category'] in ('pilot','validation') and cfg['allocation_seconds'] <= 7200
    auth=read_json(HERE/'authorization.json')
    assert auth['approved'], 'Experiment design/implementation is authorized; new compute has not been approved'
    assert auth['source_seal_sha256'] == file_digest(HERE/'source-seal.json')
    assert auth['configs'].get(cfg['run_id']) == digest(cfg)
    if cfg['category']=='pilot':
        gate=read_json(DEFAULT_ROOT/'runs/experiments'/(CAMPAIGN+'-gpu-gate')/'gpu-gate.json')
        assert gate['passed'] and gate['world_size']==8
        assert gate['source_seal_sha256']==file_digest(HERE/'source-seal.json')
    seconds=cfg['allocation_seconds'];hours=8*seconds/3600
    assert hours == cfg['max_h200_hours']
    statefile=HERE/'submissions.json'
    with locked(HERE/'.submission.lock',blocking=False):
        state=read_json(statefile) if statefile.exists() else {}
        assert cfg['run_id'] not in state, 'No automatic retry/requeue/resume'
        ledger=ExperimentLedger(DEFAULT_ROOT)
        # Check both ceilings and reserve atomically against all lab submitters.
        with ledger.transaction() as data:
            used=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values())
            assert used+hours <= auth['shared_ledger_cap_h200_hours'], 'Shared approved cap exceeded'
            campaign=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values()
                         if j['run_id'].startswith(CAMPAIGN))
            assert campaign+hours <= auth['campaign_cap_h200_hours']
            assert cfg['run_id'] not in data['runs']
            assert not (DEFAULT_ROOT/'runs/experiments'/cfg['run_id']).exists()
            assert seconds>cfg['stop_margin_seconds']+cfg['final_evaluation_reserve_seconds']
            assert cfg['max_training_tokens'] <= 6000000000 and cfg['world_size']==8
            tag=cfg['run_id']+'-'+uuid.uuid4().hex
            data['runs'][cfg['run_id']]=dict(category=cfg['category'],config_digest=digest(cfg),
                max_h200_hours=hours,created=time.time(),study=CAMPAIGN)
            data['jobs'][tag]=dict(run_id=cfg['run_id'],reserved_h200_hours=hours,gpu_count=8,
                cpu_count=8,requested_seconds=seconds,status='reserved',
                authorization_file=str(HERE/'authorization.json'),no_automatic_retry=True)
        intent=dict(status='submission_uncertain',reservation=tag,config=str(path),config_digest=digest(cfg))
        state[cfg['run_id']]=intent;atomic_json(statefile,state)
        (DEFAULT_ROOT/'logs').mkdir(parents=True,exist_ok=True)
        cmd=['sbatch','--parsable','--comment='+tag,'--time='+str(seconds//60),
             str(REPO/'slurm/critical-batch-20261020.sbatch'),cfg['run_id']]
        env={**os.environ,'CBS_CONFIG_PATH':str(path),'TRAIN_SOURCE_ROOT':str(SOURCE)}
        if cfg.get('validation_only'):env['CBS_VALIDATION_ONLY']='1'
        else:env.pop('CBS_VALIDATION_ONLY',None)
        result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=45)
        intent['result']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        atomic_json(statefile,state)
        job=result.stdout.strip().split(';')[0]
        assert result.returncode==0 and job.isdigit(), 'Submission uncertain; reservation retained'
        ledger.bind(tag,job);intent.update(job_id=job,status='submitted');atomic_json(statefile,state)
        print(json.dumps(intent))


def status():
    path=HERE/'submissions.json'
    if not path.exists():
        print('No jobs submitted');return
    with locked(HERE/'.submission.lock',blocking=False):
        state=read_json(path)
        for run,item in state.items():
            if not item.get('job_id'):
                print(run, 'Uncertain submission: retain reservation; inspect Slurm comment before any action')
                continue
            result=subprocess.run(['sacct','-X','-j',item['job_id'],'-n','-P','-o','JobIDRaw,State'],
                                  capture_output=True,text=True,check=True,timeout=30)
            states=[r.split('|')[1].split()[0].rstrip('+') for r in result.stdout.splitlines()
                    if r.split('|')[0]==item['job_id']]
            item['slurm_state']=states[0] if states else 'UNKNOWN'
            if item['slurm_state'] in TERMINAL:
                item['accounting']=ExperimentLedger(DEFAULT_ROOT).reconcile(item['job_id'])
                item['status']='reconciled'
        atomic_json(path,state);print(json.dumps(state,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','submit','status'])
    p.add_argument('--config',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':verify_source_seal();print('Source seal verified')
    elif a.action=='status':status()
    else:submit(a.config.resolve())
