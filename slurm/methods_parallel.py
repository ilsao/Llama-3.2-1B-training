#!/usr/bin/env python3
"""Prepare/review or explicitly submit independent jobs; no automatic campaign loop."""
import argparse,datetime,hashlib,json,os,re,shlex,subprocess,sys,uuid
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
LIVE=REPO/'experiments/methods-parallel-20261019'
SOURCE=LIVE/'source'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(LIVE))
sys.path.insert(0,str(SOURCE/'slurm'))
from training_state import DEFAULT_ROOT,ExperimentLedger,atomic_json,digest,file_digest,load_config,locked,read_json,slurm_record,TERMINAL
SCRIPT=REPO/'slurm/methods-parallel.sbatch'
RECORDS=DEFAULT_ROOT/'runs/parallel-submissions'/LIVE.name

def inventory():
    files=[*SOURCE.rglob('*.py'),*SOURCE.rglob('*.json'),*SOURCE.rglob('*.sh'),
           *SOURCE.rglob('*.txt'),*SOURCE.rglob('*.toml'),*SOURCE.rglob('LICENSE'),
           *LIVE.glob('configs/*.json'),LIVE/'matrix.json',LIVE/'vendor-manifest.json',
           LIVE/'repair_unallocated_submission.py',LIVE/'build_matrix.py',LIVE/'control_reference.py',LIVE/'reference-control.json',SCRIPT,Path(__file__).resolve()]
    return {str(p.relative_to(REPO)):file_digest(p) for p in sorted(set(files))}

def verify():
    seal=read_json(LIVE/'execution-seal.json')
    current=inventory()
    if current!=seal['sha256']:raise ValueError('Execution files changed; review and reseal before allocating')
    from control_reference import validate
    validate()
    matrix=read_json(LIVE/'matrix.json')
    for row in matrix['jobs']:load_config(LIVE/row['config'])
    return matrix

def select(matrix,arms,seeds):
    known={r['arm'] for r in matrix['jobs']}
    chosen=known if arms=='all' else set(arms.split(','))
    seedset={int(x) for x in seeds.split(',')}
    if not chosen or not chosen<=known or not seedset or seedset!={42}:raise ValueError('Unknown arm or seed')
    return [r for r in matrix['jobs'] if r['arm'] in chosen and r['seed'] in seedset]

def command(row,tag):
    return ['sbatch','--parsable','--comment='+tag,'--job-name=mp-'+row['key'],
            '--export=ALL,TRAIN_SOURCE_ROOT='+str(SOURCE),str(SCRIPT),row['run_id'],row['key']]

def plan(rows):
    return dict(submitted=False,independent_jobs=len(rows),max_new_h200_hours=16*len(rows),
        maximum_simultaneous_gpus=8*len(rows),maximum_simultaneous_cpus=8*len(rows),
        concurrency_throttle=False,dependencies=[],
        jobs=[dict(key=r['key'],run_id=r['run_id'],config_digest=digest(load_config(LIVE/r['config'])),
                   example_sbatch=shlex.join(command(r,'<unique-ledger-reservation>'))) for r in rows])

def submit(rows,budget):
    needed=16*len(rows)
    if budget is None or budget!=needed:raise ValueError(f'Explicit --budget-h200-hours {needed} must match this submission exactly')
    # Log dirs must exist BEFORE sbatch opens output paths. No cleanup or model I/O.
    (DEFAULT_ROOT/'logs').mkdir(parents=True,exist_ok=True)
    RECORDS.mkdir(parents=True,exist_ok=True)
    ledger=ExperimentLedger(DEFAULT_ROOT)
    with locked(RECORDS/'.submission.lock',blocking=False):
        existing=read_json(ledger.path)
        for row in rows:
            if (RECORDS/(row['key']+'.json')).exists() or row['run_id'] in existing['runs'] or (DEFAULT_ROOT/'runs/experiments'/row['run_id']).exists():
                raise ValueError('Existing identity: '+row['key']+'; recover/reconcile, never retry')
        for row in rows:
            cfg=load_config(LIVE/row['config'])
            reservation='mp-'+uuid.uuid4().hex
            state=dict(key=row['key'],run_id=row['run_id'],reservation=reservation,
                 submission_tag=reservation,submission_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 config_digest=digest(cfg),seal_sha256=file_digest(LIVE/'execution-seal.json'),
                 requested_seconds=7200,gpu_count=8,cpu_count=8,batch_budget_h200_hours=budget,
                 phase='intent')
            record=RECORDS/(row['key']+'.json')
            # Persist intent before side effects; a crash cannot silently resubmit.
            atomic_json(record,state)
            ledger.reserve(reservation,row['run_id'],cfg,7200,8,False)
            state['phase']='reserved';atomic_json(record,state)
            try:
                result=subprocess.run(command(row,reservation),capture_output=True,text=True,timeout=45)
                value=result.stdout.strip().split(';')[0]
                if result.returncode or not re.fullmatch(r'\d+',value):
                    raise RuntimeError('sbatch result uncertain: '+result.stderr.strip())
                state.update(returned_job_id=value,phase='accepted');atomic_json(record,state)
                ledger.bind(reservation,value)
                # Worker waits for this binding before any model/data work.
                state['phase']='submitted';atomic_json(record,state)
                print(json.dumps(state),flush=True)
            except (OSError,subprocess.TimeoutExpired,RuntimeError,ValueError) as error:
                state.update(error=str(error),recovery_required=True)
                atomic_json(record,state)
                raise RuntimeError('Reservation retained. Stop new submissions; use recover --key '+row['key']) from error

def recover(key):
    from submission import find_submission_job
    with locked(RECORDS/'.submission.lock',blocking=False):
        path=RECORDS/(key+'.json');state=read_json(path)
        job=find_submission_job(state,DEFAULT_ROOT)
        ledger=ExperimentLedger(DEFAULT_ROOT);data=read_json(ledger.path)
        if state['reservation'] in data['jobs']:ledger.bind(state['reservation'],job)
        elif job not in data['jobs'] or data['jobs'][job]['run_id']!=state['run_id']:
            raise ValueError('Cannot safely bind recovered job')
        # Explicit recovery only identifies/binds; never releases or resubmits.
        state.update(returned_job_id=job,phase='recovered_check_scheduler',recovery_required=False)
        atomic_json(path,state)
        print(json.dumps(dict(state=state,next_action='Check scontrol show job; release this same job only if held. No new sbatch.')))

def reconcile(rows):
    ledger=ExperimentLedger(DEFAULT_ROOT)
    for row in rows:
        path=RECORDS/(row['key']+'.json')
        if not path.exists():print(json.dumps(dict(key=row['key'],state='not_submitted')));continue
        state=read_json(path);job=state.get('returned_job_id')
        if not job:print(json.dumps(dict(key=row['key'],state='recovery_required')));continue
        measured=slurm_record(job,DEFAULT_ROOT)
        if measured['state'] in TERMINAL:
            ledger.reconcile(job);state['accounting']=measured;atomic_json(path,state)
        print(json.dumps(dict(key=row['key'],accounting=measured)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','seal','check','submit','recover','reconcile'])
    parser.add_argument('--arms',default='all');parser.add_argument('--seeds',default='42')
    parser.add_argument('--budget-h200-hours',type=int);parser.add_argument('--key')
    args=parser.parse_args()
    if args.action=='seal':
        atomic_json(LIVE/'execution-seal.json',dict(sha256=inventory(),meaning='Integrity seal, not compute authorization'))
        print('Sealed execution files; no reservation or submission');return
    matrix=verify()
    if args.key:
        matches=[r for r in matrix['jobs'] if r['key']==args.key]
        if len(matches)!=1:raise ValueError('Unknown key')
        rows=matches
    else:rows=select(matrix,args.arms,args.seeds)
    if args.action in ('plan','check'):print(json.dumps(plan(rows),indent=2));return
    if args.action=='submit':submit(rows,args.budget_h200_hours)
    elif args.action=='recover':
        if not args.key:parser.error('recover requires --key')
        recover(args.key)
    else:reconcile(rows)
if __name__=='__main__':main()
