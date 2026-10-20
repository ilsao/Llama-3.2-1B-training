"""One explicit continuation; no automatic resubmission or future resume permission."""
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

REPOSITORY=Path(__file__).resolve().parents[1]
SOURCE=REPOSITORY/'.worktrees/muon-sweep'
ROOT=Path('/work/asciibase64/Llama-3.2-1B-training')
RUN='baseline-muon-wsd-raw-20261009'
ID='best-raw-one-hour-20261009'
PRODUCER='raw-buffer-once-20261009'
DIRECTORY=ROOT/'runs/experiments'/RUN/'continuations'/ID


def extended_config(original):
    config=copy.deepcopy(original)
    config.update(max_steps=16000,max_h200_hours=23,eval_steps=400,
                  fixed_horizon=False,final_evaluation_reserve_seconds=0)
    # The original WSD horizon stays 8500: continuing uses its terminal LR floor.
    return config


def shared_upper():
    train=json.loads((ROOT/'experiment-ledger.json').read_text())
    prep=json.loads((ROOT/'preprocessing-ledger.json').read_text())
    state=json.loads((REPOSITORY/'research-state.yaml').read_text())
    outstanding={i['tag']:i['reserved_h200_hours'] for i in state.get('submission_intents',[])
        if i.get('ledger')=='preprocessing' and i.get('status')!='not_dispatched' and i.get('job_id') not in prep['jobs']}
    return train['total_h200_hours_reserved_or_measured']+prep['total_h200_hours']+sum(outstanding.values())


def source_check(authorization):
    from training_state import file_digest
    for path,expected in authorization['sealed_files'].items():
        if file_digest(Path(path))!=expected:raise ValueError('Pinned continuation source/metadata changed: '+path)


def install_status_schedule():
    old=subprocess.run(['crontab','-l'],capture_output=True,text=True,check=False)
    if old.returncode not in (0,1):raise RuntimeError('Cannot read accounting schedule')
    marker='# codex-best-raw-one-hour-accounting'
    lines=[x for x in old.stdout.splitlines() if marker not in x]
    command=shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'status'])
    lines.append(f'* * * * * {command} >> {shlex.quote(str(DIRECTORY/"accounting.log"))} 2>&1 {marker}')
    subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)


def submit(test_only=False):
    from training_state import ExperimentLedger,atomic_json,read_json,locked,digest,file_digest,load_config
    ledger=ExperimentLedger(ROOT);run=ROOT/'runs/experiments'/RUN
    authorization_path=DIRECTORY/'authorization.json';launch=DIRECTORY/'launch.json'
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with locked(DIRECTORY/'.submission.lock',blocking=False):
        if launch.exists():raise ValueError('One-time submission already exists; use status, never resubmit')
        original=read_json(run/'resolved-config.json');cfg=extended_config(original)
        best=read_json(run/'best.json');checkpoint=run/'checkpoints'/best['checkpoint']
        saved=read_json(checkpoint/'training-state.json');complete=read_json(checkpoint/'complete.json')
        journal=read_json(run/'token-journal.json');execution=read_json(run/'execution-identity.json')
        if digest(saved)!=complete['state_digest'] or saved['global_step']!=8500:raise ValueError('Best checkpoint changed')
        if saved['config_digest']!=digest(original) or saved['code_digest']!=digest(execution['source_hashes']):raise ValueError('Parent execution identity mismatch')
        if journal['pending'] or journal['charged_tokens']!=saved['charged_tokens_at_checkpoint'] or journal['confirmed_tokens']!=saved['confirmed_tokens_at_checkpoint']:
            raise ValueError('Best checkpoint differs from token high-water mark; no budget rollback')
        if best['confirmed_tokens']!=journal['confirmed_tokens']:raise ValueError('Best token count differs')
        for item in complete['files']:
            p=checkpoint/item['name']
            if not p.is_file() or p.stat().st_size!=item['size']:raise ValueError('Incomplete best checkpoint')
        environment=ROOT/'environments/training-muon-torch291-fa3-94e22c9'
        sealed={str(SOURCE/name):sha for name,sha in execution['source_hashes'].items() if name!='fa3-build.json'}
        sealed[str(environment/'fa3-build.json')]=execution['source_hashes']['fa3-build.json']
        for name in ['slurm/continue_best_once.py','slurm/raw_continue_once.py','slurm/continue-best-once.sbatch','slurm/raw-continue-once.sbatch',
                     'filtering_parallel.py','filtering_batch_reader.py','filtering_reader.py','filtering_resources.py','packing_parallel.py','dataset.py','preparation_parallel.py']:
            sealed[str(REPOSITORY/name)]=file_digest(REPOSITORY/name)
        for name in ['best.json','token-journal.json','initialization.json','raw-prefix-receipt.json']:
            sealed[str(run/name)]=file_digest(run/name)
        sealed[str(checkpoint/'training-state.json')]=file_digest(checkpoint/'training-state.json')
        sealed[str(checkpoint/'complete.json')]=file_digest(checkpoint/'complete.json')
        raw=ROOT/'preprocessed/dolma-raw-baseline-20261008'
        raw_seals={str(p):file_digest(p) for p in [raw/'configuration.json',raw/'raw/ready.json',ROOT/'preprocessed/dolma-dclm10-005/split/manifest.json']}
        progress=read_json(raw/'raw/progress.json')
        authorization=dict(id=ID,root=str(ROOT),run_id=RUN,source_root=str(SOURCE),preparation_id=raw.name,
            checkpoint=str(checkpoint),checkpoint_state_digest=complete['state_digest'],original_configuration=original,
            configuration=cfg,original_execution_identity=execution,sealed_files=sealed,raw_metadata_hashes=raw_seals,
            raw_prefix=progress['train_files'],raw_target_tokens=3_000_000_000,training_phase_seconds=3600,
            training_allocation_seconds=4200,producer_allocation_seconds=5400,
            authorization='User explicitly authorized best checkpoint + unfiltered data for one hour only; future training starts randomly',
            scope='Continuation of the same logical run, same optimizer/scheduler/RNG/cursor/W&B and cumulative budgets; no reset',
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        source_check(authorization)
        if shared_upper()+8*4200/3600+5400/3600>192:raise ValueError('Shared budget insufficient')
        with ledger.transaction() as data:
            parent=data['runs'][RUN];jobs=[j for j in data['jobs'].values() if j['run_id']==RUN]
            if any('h200_hours' not in j for j in jobs):raise ValueError('Parent allocations must be reconciled')
            used=sum(j['h200_hours'] for j in jobs)
            if parent['config_digest']!=digest(original) or used+8*4200/3600>cfg['max_h200_hours']:raise ValueError('Logical run budget/config changed')
        atomic_json(DIRECTORY/'config.json',cfg);load_config(DIRECTORY/'config.json')
        commands={
            'producer':['sbatch','--parsable','--comment='+ID+'-producer',str(REPOSITORY/'slurm/raw-continue-once.sbatch'),str(authorization_path)],
            'training':['sbatch','--parsable','--comment='+ID+'-training',str(REPOSITORY/'slurm/continue-best-once.sbatch'),RUN,str(authorization_path)]}
        environment_vars={**os.environ,'DATA_ROOT':str(ROOT),'TRAIN_SOURCE_ROOT':str(SOURCE),'TRAIN_ENV_PREFIX':str(environment)}
        if test_only:
            for command in commands.values():
                result=subprocess.run(command[:1]+['--test-only']+command[1:],env=environment_vars,capture_output=True,text=True,check=True)
                print(result.stdout+result.stderr)
            return
        atomic_json(authorization_path,authorization)
        with ledger.transaction() as data:
            parent=data['runs'][RUN]
            parent.setdefault('configuration_extensions',[]).append(dict(authorization=str(authorization_path),old_config_digest=parent['config_digest'],new_config_digest=digest(cfg),previous_h200_hours=used,one_time_only=True))
            parent.update(config_digest=digest(cfg),max_h200_hours=cfg['max_h200_hours'])
        producer_cfg={**cfg,'category':'validation','world_size':1,'max_h200_hours':1.5,'wandb_mode':'offline'}
        reservations={'producer':ID+'-producer','training':ID+'-training'}
        ledger.reserve(reservations['producer'],PRODUCER,producer_cfg,5400,1,False)
        ledger.reserve(reservations['training'],RUN,cfg,4200,8,True)
        state=dict(authorization=str(authorization_path),jobs={},reservations=reservations,status='submission_uncertain',submission_owner=str(Path(__file__).resolve()))
        atomic_json(launch,state);install_status_schedule()
        for kind,command in commands.items():
            tag=reservations[kind]
            with ledger.transaction() as data:
                data['jobs'][tag].update(submission_tag=tag,submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),submission_command=command,
                    allocation_kind='preprocessing' if kind=='producer' else 'training',source_root=str(SOURCE),continuation_authorization=str(authorization_path))
            result=subprocess.run(command,env=environment_vars,capture_output=True,text=True)
            state.setdefault('submission_results',{})[kind]=dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
            atomic_json(launch,state)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):raise ValueError('Submission uncertain; preserve reservations, use status')
            ledger.bind(tag,job);state['jobs'][kind]=job;atomic_json(launch,state)
        state['status']='submitted';atomic_json(launch,state);print(json.dumps(state))


def status():
    from training_state import ExperimentLedger,read_json,atomic_json,locked,TERMINAL
    path=DIRECTORY/'launch.json'
    if not path.exists():return
    with locked(DIRECTORY/'.submission.lock',blocking=False):
        s=read_json(path);ledger=ExperimentLedger(ROOT)
        for kind,job in s['jobs'].items():
            if kind in s.get('accounting',{}):continue
            r=subprocess.run(['sacct','-X','-j',job,'--noheader','--parsable2','--format=State%30'],capture_output=True,text=True,check=True,timeout=20)
            value=r.stdout.strip().split('|')[0].split()[0] if r.stdout.strip() else 'UNKNOWN'
            s.setdefault('slurm_states',{})[kind]=value
            if value in TERMINAL:s.setdefault('accounting',{})[kind]=ledger.reconcile(job)
        if len(s.get('accounting',{}))==2:
            s['status']='reconciled';old=subprocess.run(['crontab','-l'],capture_output=True,text=True,check=True)
            lines=[x for x in old.stdout.splitlines() if '# codex-best-raw-one-hour-accounting' not in x]
            subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)
        atomic_json(path,s);print(json.dumps({k:s.get(k) for k in ('jobs','status','slurm_states')}))


def worker(receipt):
    from training_state import read_json,atomic_json,locked,digest
    import train
    import wandb
    a=read_json(receipt);source_check(a)
    run=ROOT/'runs/experiments'/RUN;rank=int(os.environ.get('RANK','0'));job=os.environ['SLURM_JOB_ID']
    if rank==0:
        with locked(run/'.run.lock',blocking=False):
            if (DIRECTORY/'consumed.json').exists():raise ValueError('One-time checkpoint permission already consumed')
            if read_json(run/'resolved-config.json')!=a['original_configuration'] or read_json(run/'execution-identity.json')!=a['original_execution_identity']:
                raise ValueError('Original logical run changed before continuation')
            originals={name:read_json(run/name) for name in ['resolved-config.json','execution-identity.json','best.json','latest.json','summary.json','resource-accounting.json','full-evaluation.json']}
            atomic_json(DIRECTORY/'original-run-metadata.json',originals)
            atomic_json(run/'resolved-config.json',a['configuration'])
            atomic_json(run/'execution-identity.json',{**a['original_execution_identity'],'config_digest':digest(a['configuration'])})
            atomic_json(DIRECTORY/'consumed.json',dict(job_id=job,checkpoint=a['checkpoint'],started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    else:
        deadline=time.monotonic()+120
        while not (DIRECTORY/'consumed.json').exists():
            if time.monotonic()>deadline:raise ValueError('Main rank did not seal continuation')
            time.sleep(.1)
    if read_json(DIRECTORY/'consumed.json')['job_id']!=job:raise ValueError('Authorization belongs to a different job')
    timer=None;started=None
    def best_state(_):
        checkpoint=Path(a['checkpoint']);saved=read_json(checkpoint/'training-state.json');complete=read_json(checkpoint/'complete.json')
        if digest(saved)!=a['checkpoint_state_digest'] or complete['state_digest']!=digest(saved):raise ValueError('Original best checkpoint changed')
        for item in complete['files']:
            p=checkpoint/item['name']
            if not p.is_file() or p.stat().st_size!=item['size']:raise ValueError('Incomplete best checkpoint')
        saved.update(config_digest=digest(a['configuration']),planned_steps=a['configuration']['max_steps'],
            continuation_id=ID,continuation_authorization=str(receipt),continuation_original_config_digest=digest(a['original_configuration']))
        return str(checkpoint),saved
    train.checkpoint_state=best_state
    original_restore=train.restore_rng
    def restore_and_start(checkpoint,accelerator):
        nonlocal timer,started
        original_restore(checkpoint,accelerator);started=time.monotonic()
        timer=threading.Timer(a['training_phase_seconds'],train.request_stop);timer.daemon=True;timer.start()
        if rank==0:
            atomic_json(DIRECTORY/'training-phase.json',dict(job_id=job,started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stop_epoch=time.time()+a['training_phase_seconds'],seconds=a['training_phase_seconds']))
    train.restore_rng=restore_and_start
    original_init=wandb.init
    wandb.init=lambda *args,**kwargs:original_init(*args,**{**kwargs,'allow_val_change':True})
    args=SimpleNamespace(root=ROOT,run_id=RUN,prep_id=a['preparation_id'],resume=True,stop_after_step=None,validate_export=False)
    try:code=train.run(args,a['configuration'])
    finally:
        if timer is not None:timer.cancel()
    atomic_json(Path(os.environ['JOB_RECORD'])/f'worker-rank-{rank}-exit.json',dict(exit_code=code,checkpointed_stop=code==75))
    if rank==0:atomic_json(DIRECTORY/'result.json',dict(exit_code=code,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),phase_and_finalization_seconds=time.monotonic()-started if started else None,summary=read_json(run/'summary.json')))
    return 0 if code==75 else code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('submit','status','worker'));parser.add_argument('receipt',nargs='?',type=Path);parser.add_argument('--test-only',action='store_true');args=parser.parse_args()
    sys.path.insert(0,str(SOURCE))
    if args.action=='submit':submit(args.test_only)
    elif args.action=='status':status()
    else:sys.exit(worker(args.receipt))
