#!/usr/bin/env python3
"""Exact formal19 authorization; single allocation, no automatic retry."""
import argparse
import json
import re
import subprocess
import time
import uuid
from bootstrap import HERE,REPO,SOURCE,FUSED,ACCEL
from contracts import RUN,CONFIG_DIGEST,SECONDS,STUDY_CAP,config,study_used,plan
from training_state import DEFAULT_ROOT,ExperimentLedger,atomic_json,read_json,file_digest,digest,locked
SCRIPT=REPO/'slurm/kd6b-r5-20261020.sbatch'
RECORD=HERE/'submission.json'


def inventory():
    previous=read_json(ACCEL/'execution-seal.json')['sha256']
    paths=[REPO/p for p in previous if p.startswith(str(SOURCE.relative_to(REPO))+'/')]
    paths += [FUSED/'fused_kd.py',ACCEL/'accelerations.py',SCRIPT]
    paths += [HERE/n for n in ('bootstrap.py','contracts.py','tail_data.py','checkpointing.py','runner.py',
                               'launch.py','test_contracts.py','config.json','authorization.json','protocol.md')]
    paths += [HERE.parent/'raw6b-normuon-adam2x-20261018'/n for n in ('raw-metadata-seal.json','raw-prefix.json')]
    return {str(p.relative_to(REPO)):file_digest(p) for p in sorted(set(paths))}


def check():
    cfg=config()
    assert inventory()==read_json(HERE/'execution-seal.json')['sha256'],'Sealed source changed'
    assert len(plan(cfg))==28619
    return cfg


def reserve_entries(data,cfg,reservation):
    if digest(cfg)!=CONFIG_DIGEST:raise ValueError('Only the approved configuration may reserve resources')
    if RUN in data['runs'] or reservation in data['jobs']:raise ValueError('Existing run/reservation; no retry')
    if sum(r['category']=='full' for r in data['runs'].values())!=18:raise ValueError('Expected preserved18 prior formal runs')
    if study_used(data)+72>STUDY_CAP+1e-9:raise ValueError('Scoped study budget exceeded')
    previous=dict(run_count=len(data['runs']),job_count=len(data['jobs']),
                  study_cost=study_used(data),formal_count=18,
                  historical_identity_digest=digest(sorted(data['runs'])))
    data['runs'][RUN]=dict(category='full',formal_run_number=19,config_digest=CONFIG_DIGEST,
        max_h200_hours=72,created=time.time(),study='methods-parallel-20261019',
        authorization='Explicit user 好，開始訓練 after reviewed R5 6B/9h72H/formal19/study248.178611H proposal')
    data['jobs'][reservation]=dict(run_id=RUN,reserved_h200_hours=72,gpu_count=8,cpu_count=8,
        requested_seconds=SECONDS,status='reserved',source_root=str(SOURCE),
        authorization_file=str(HERE/'authorization.json'),no_automatic_retry=True)
    return previous


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['seal','check','submit','status','reconcile'])
    action=parser.parse_args().action
    if action=='seal':
        assert not RECORD.exists(),'Never reseal after submission'
        config();atomic_json(HERE/'execution-seal.json',dict(sha256=inventory(),configuration_digest=CONFIG_DIGEST));return
    cfg=check()
    if action=='check':print('Exact authorization, source seal and28619-update plan verified');return
    if action in ('status','reconcile'):
        state=read_json(RECORD)
        if action=='reconcile':
            actual=ExperimentLedger(DEFAULT_ROOT).reconcile(state['job_id'])
            atomic_json(HERE/'accounting.json',actual);print(json.dumps(actual));return
        print(json.dumps(state,indent=2))
        if state.get('job_id'):
            p=subprocess.run(['sacct','-j',state['job_id'],'-X','--noheader','--parsable2',
                  '--format=JobIDRaw,State,ElapsedRaw,Start,End,NodeList'],capture_output=True,text=True,timeout=30,check=True)
            print(p.stdout.strip())
        return
    with locked(HERE/'.submit.lock',blocking=False):
        if RECORD.exists():raise ValueError('Submission already attempted; inspect identity, never retry automatically')
        assert not (DEFAULT_ROOT/'runs/experiments'/RUN).exists(),'Existing run artifacts'
        reservation='kd6br5-'+uuid.uuid4().hex
        state=dict(run_id=RUN,reservation=reservation,configuration_digest=CONFIG_DIGEST,
                   phase='intent',timestamp=time.time(),requested_seconds=SECONDS,h200_hours_maximum=72)
        atomic_json(RECORD,state)
        ledger=ExperimentLedger(DEFAULT_ROOT)
        with ledger.transaction() as data:prior=reserve_entries(data,cfg,reservation)
        state.update(phase='reserved',prior=prior);atomic_json(RECORD,state)
        (DEFAULT_ROOT/'logs').mkdir(parents=True,exist_ok=True)
        command=['sbatch','--parsable','--comment='+reservation,str(SCRIPT)]
        state.update(phase='submitting',command=command);atomic_json(RECORD,state)
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=45)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                raise RuntimeError('Submission uncertain; reservation retained: '+result.stderr)
            state.update(job_id=job,phase='accepted');atomic_json(RECORD,state)
            ledger.bind(reservation,job)
            state['phase']='submitted';atomic_json(RECORD,state)
            atomic_json(HERE/'ledger-after-submission.json',read_json(ledger.path))
            print(json.dumps(state,indent=2))
        except Exception as exc:
            state.update(error=repr(exc),recovery_required=True);atomic_json(RECORD,state);raise


if __name__=='__main__':main()

