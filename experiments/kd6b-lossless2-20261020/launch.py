#!/usr/bin/env python3
"""One validation allocation. Reserve/bind/account; never retry or start a full run."""
import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=HERE.parent/'kd3b-hidden8-20261019/source'
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,atomic_json,file_digest,locked,read_json
RUN='kd6b-lossless2-validation-20261020'
SCRIPT=REPO/'slurm/kd6b-lossless2-validation.sbatch'
RECORD=HERE/'submission.json'
CONFIG=dict(category='validation',world_size=8,stop_margin_seconds=120,
            max_h200_hours=4,max_training_tokens=180_000_000,
            seed=42,sweep_id='kd6b-lossless2-20261020',
            objective='mathematically equivalent full-vocabulary KD performance validation',
            no_checkpoint_load=True,no_model_export=True,no_automatic_retry=True)


def inventory():
    paths=[p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [HERE/n for n in ('accelerations.py','benchmark.py','launch.py','protocol.md')]
    paths += [SCRIPT,HERE.parent/'kd3b-hidden8-20261019/config.json', HERE.parent/'kd6b-lossless-20261020/fused_kd.py']
    return {str(p.relative_to(REPO)):file_digest(p) for p in sorted(paths)}


def check():
    assert inventory()==read_json(HERE/'execution-seal.json')['sha256'],'Changed sealed source'
    return CONFIG


def budget():
    data=read_json(ExperimentLedger(DEFAULT_ROOT).path)
    prefixes=('methods-parallel-20261019','kd-acceleration-','kd-pipeline-',
              'kd3b-online-pipeline-','kd3b-io-validation-','kd3b-io8-validation-',
              'kd3b-hidden8-','kd6b-lossless-','kd6b-lossless2-')
    jobs={k:j for k,j in data['jobs'].items() if j['run_id'].startswith(prefixes)}
    used=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in jobs.values())
    return dict(prior_measured_or_reserved_h200_hours=used,new_max_h200_hours=4,
                maximum_after_submission=used+4,study_cap=191.419,
                preserved_jobs=sorted(jobs),formal_count=sum(r['category']=='full' for r in data['runs'].values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','check','plan','submit','reconcile'])
    action=p.parse_args().action
    if action=='seal':
        assert not RECORD.exists(),'Never reseal a submitted validation'
        atomic_json(HERE/'execution-seal.json',dict(sha256=inventory(),config=CONFIG));return
    check()
    if action=='check':print('Source/config seal verified');return
    if action=='plan':print(json.dumps(budget(),indent=2));return
    if action=='reconcile':
        state=read_json(RECORD)
        result=ExperimentLedger(DEFAULT_ROOT).reconcile(state['job_id'])
        atomic_json(HERE/'accounting.json',result);print(json.dumps(result));return
    with locked(HERE/'.submit.lock',blocking=False):
        assert not RECORD.exists(),'Existing identity; inspect, never automatically retry'
        data=read_json(ExperimentLedger(DEFAULT_ROOT).path)
        assert RUN not in data['runs'],'Existing validation identity'
        prior=read_json(HERE.parent/'kd3b-hidden8-20261019/status.json')
        assert prior['terminal'],'Finish/reconcile current training before a new allocation'
        b=budget();assert b['maximum_after_submission']<=b['study_cap']
        token='kd6blossless2-'+uuid.uuid4().hex
        state=dict(run_id=RUN,reservation=token,config=CONFIG,budget=b,phase='intent',timestamp=time.time())
        atomic_json(RECORD,state)
        ExperimentLedger(DEFAULT_ROOT).reserve(token,RUN,CONFIG,1800,8,False)
        state['phase']='reserved';atomic_json(RECORD,state)
        (DEFAULT_ROOT/'logs').mkdir(parents=True,exist_ok=True)
        state['phase']='submitting';atomic_json(RECORD,state)
        try:
            result=subprocess.run(['sbatch','--parsable','--comment='+token,str(SCRIPT)],
                                  text=True,capture_output=True,timeout=45)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                raise RuntimeError('Uncertain submission: '+result.stderr)
            state.update(job_id=job,phase='accepted');atomic_json(RECORD,state)
            ExperimentLedger(DEFAULT_ROOT).bind(token,job)
            state['phase']='submitted';atomic_json(RECORD,state)
            print(json.dumps(state))
        except Exception as e:
            state.update(error=repr(e),recovery_required=True);atomic_json(RECORD,state)
            raise


if __name__=='__main__':main()
