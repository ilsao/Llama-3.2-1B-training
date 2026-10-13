#!/usr/bin/env python3
"""Bounded campaign orchestration on login; tensors/data run only in sbatch."""
import argparse
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = 'horizon-wsd-20261014'
DIRECTORY = REPO / 'experiments' / CAMPAIGN
SOURCE = DIRECTORY / 'source-v2'
CONFIGS = REPO / 'configs' / CAMPAIGN
sys.path.insert(0, str(SOURCE))
from training_state import (DEFAULT_ROOT, ExperimentLedger, TERMINAL, atomic_json,
                            digest, file_digest, load_config, locked, read_json)
from horizon_contracts import (BATCH_TOKENS, SWEEP_STEPS, FINAL_STEPS, PREFIX_STEP,
                               FINAL_SPLIT, scaled_lr, tail_settings)

PLAN = DIRECTORY / 'plan.json'
STATE = DIRECTORY / 'controller-state.json'
DISABLED = DIRECTORY / 'disabled'
MARKER = '# horizon-wsd-20261014-controller'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def plan():
    return read_json(PLAN)


def state():
    return read_json(STATE) if STATE.exists() else dict(campaign=CAMPAIGN, stage='validation', jobs={}, selections={}, created_utc=stamp())


def check():
    p = plan()
    for name, expected in read_json(DIRECTORY/'recovery-plan.json')['source_hashes'].items():
        if file_digest(REPO/name) != expected:
            raise ValueError('Recovery source changed: '+name)
    for name, expected in p['source_hashes'].items():
        if file_digest(REPO/name) != expected:
            raise ValueError('Sealed source changed: ' + name)
    for name, expected in p['metadata_hashes'].items():
        if file_digest(name) != expected:
            raise ValueError('Pinned preparation/environment/reference changed: ' + name)
    progress = read_json(DEFAULT_ROOT/'preprocessed'/p['preparation_id']/'raw/progress.json')
    if sum(i['rows'] for i in progress['train_files']) < FINAL_STEPS * 16:
        raise ValueError('Raw packed prefix insufficient for both final arms')
    for name, expected in p['initial_configuration_hashes'].items():
        if file_digest(REPO/name) != expected:
            raise ValueError('Initial configuration changed')
        load_config(REPO/name)
    if p['sweep_tokens'] != SWEEP_STEPS * BATCH_TOKENS or p['final_tokens'] != FINAL_STEPS * BATCH_TOKENS:
        raise ValueError('Token horizons changed')
    print('Seals, metadata, environment, canonical prefix, and 0.7B/2B contracts passed.')


def config_for(name, lr_ratio=1, clipping=1, tail=.6, final=False, resplit=False):
    cfg = copy.deepcopy(plan()['base_configuration'])
    steps = FINAL_STEPS if final else SWEEP_STEPS
    cfg.update(max_steps=steps, scheduler_steps=steps, max_training_tokens=steps*BATCH_TOKENS,
        target_training_tokens=steps*BATCH_TOKENS, category='full' if final else 'pilot',
        max_h200_hours=32 if final else 16, eval_steps=max(1, steps//10),
        export_on_completion=final, retain_all_checkpoints=True,
        final_evaluation_reserve_seconds=600 if final else 120, stop_margin_seconds=180,
        sweep_id=CAMPAIGN, sweep_purpose='Token-scaled LR, clipping, shared-prefix decay, then two final arms',
        sweep_stage='final' if final else 'short', max_grad_norm=clipping,
        parent_runs=[CAMPAIGN+'-stable-prefix'] if name.startswith('decay') else ['cg-20261010-be60p20-s42'], hold_schedule=tail_settings(tail), wsd_decay_ratio=tail)
    horizon = (FINAL_SPLIT if resplit else steps) * BATCH_TOKENS
    cfg['learning_rate'] = scaled_lr(.0003,399900672,horizon,lr_ratio)
    cfg['muon']['learning_rate'] = scaled_lr(.02,399900672,horizon,lr_ratio)
    if resplit:
        cfg['phase_schedule'] = [
            dict(start=0,end=FINAL_SPLIT,peak_ratio=1.,warmup_steps=170,hold_schedule=tail_settings(tail)),
            dict(start=FINAL_SPLIT,end=FINAL_STEPS,
                 peak_ratio=((FINAL_STEPS-FINAL_SPLIT)/FINAL_SPLIT)**(-.32),
                 warmup_steps=170,hold_schedule=tail_settings(tail))]
    path=CONFIGS/(name+'.json')
    if path.exists() and read_json(path)!=cfg:
        raise ValueError('Resolved candidate configuration changed')
    atomic_json(path,cfg)
    return path,load_config(path)


def submit(s, key, run_name, path, cfg, minutes, mode='fresh', pause=0, parent='-', full_eval=False, test_only=False):
    run_id=CAMPAIGN+'-'+run_name
    if key in s['jobs']:
        return
    check()
    if not test_only and not (DIRECTORY/'protocol-commit.json').exists():
        raise ValueError('Commit protocol before any compute')
    if mode=='fresh' and (DEFAULT_ROOT/'runs/experiments'/run_id).exists():
        raise ValueError('Run already initialized; no automatic retry')
    tag='horizon-'+uuid.uuid4().hex
    command=['sbatch','--parsable',f'--time={minutes}', '--comment='+tag,
             '--job-name=hw-'+key, str(REPO/'slurm/horizon-wsd-v2.sbatch'),run_id,str(path),mode,str(pause),parent,'1' if full_eval else '0']
    env={**os.environ,'DATA_ROOT':str(DEFAULT_ROOT),'TRAIN_SOURCE_ROOT':str(SOURCE),
         'HORIZON_CONTROLLER':str(Path(__file__).resolve())}
    if test_only:
        result=subprocess.run(command[:1]+['--test-only']+command[1:],env=env,capture_output=True,text=True,timeout=30,check=True)
        print(result.stdout+result.stderr)
        return
    ledger=ExperimentLedger(DEFAULT_ROOT)
    DEFAULT_ROOT.joinpath('logs').mkdir(parents=True,exist_ok=True)
    ledger.reserve(tag,run_id,cfg,minutes*60,8,mode=='resume')
    if parent!='-':
        with ledger.transaction() as data:
            original=data['runs'][parent]
            if original.get('budget_exceeded') or any('h200_hours' not in j for j in data['jobs'].values() if j['run_id']==parent):
                raise ValueError('Parent must be reconciled')
            inherited=original['h200_hours_measured']
            if inherited+minutes*8/60>cfg['max_h200_hours']:
                raise ValueError('Logical branch resource cap exceeded')
            data['runs'][run_id].update(inherited_h200_hours=inherited,parent_run=parent,
                shared_prefix_cost='Parent physically charged once; included in each branch logical cap')
    record=dict(run_id=run_id,reservation=tag,status='submission_uncertain',mode=mode,pause=pause,parent=parent,
                full_eval=full_eval,configuration=str(path),configuration_sha256=file_digest(path),
                minutes=minutes,command=command,started_utc=stamp())
    s['jobs'][key]=record
    atomic_json(STATE,s)
    with ledger.transaction() as data:
        data['jobs'][tag].update(submission_tag=tag,submission_command=command,submission_started_utc=record['started_utc'],
            allocation_kind='horizon_wsd_campaign',source_root=str(SOURCE),
            authorization='User requested autoresearch with independent sbatches, checkpoint branches, and two final two-sbatch experiments on 2026-10-14')
    try:
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    except OSError:
        ledger.cancel_reservation(tag)
        record['status']='submission_not_executed'
        atomic_json(STATE,s)
        raise
    except subprocess.TimeoutExpired:
        raise ValueError('Uncertain submission; preserve reservation and recover, never retry')
    record['receipt']=dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
    atomic_json(STATE,s)
    job=result.stdout.strip().split(';')[0]
    if result.returncode or not re.fullmatch(r'\d+',job):
        raise ValueError('Uncertain sbatch outcome; reservation retained')
    with ledger.transaction() as data:
        data['jobs'][tag]['returned_job_id']=job
    ledger.bind(tag,job)
    record.update(job_id=job,status='submitted')
    atomic_json(STATE,s)
    print('Submitted',key,job,run_id)


def reconcile(s):
    sys.path.insert(0,str(REPO))
    from slurm.submission import find_submission_job
    ledger=ExperimentLedger(DEFAULT_ROOT)
    for record in s['jobs'].values():
        if record['status'] in ('reconciled','submission_not_executed'):
            continue
        if not record.get('job_id'):
            with ledger.transaction() as data:
                reservation=data['jobs'][record['reservation']]
            job=find_submission_job(reservation,DEFAULT_ROOT)
            ledger.bind(record['reservation'],job)
            record.update(job_id=job,status='submitted')
            atomic_json(STATE,s)
        result=subprocess.run(['sacct','-X','-j',record['job_id'],'--noheader','--parsable2',
            '--format=JobIDRaw,State%30'],capture_output=True,text=True,check=True,timeout=30)
        rows=[r.split('|') for r in result.stdout.splitlines()]
        record['slurm_state']=next((r[1].split()[0].rstrip('+') for r in rows if r[0]==record['job_id']),'UNKNOWN')
        if record['slurm_state'] in TERMINAL:
            record['accounting']=ledger.reconcile(record['job_id'])
            record['status']='reconciled'
        atomic_json(STATE,s)


def verify_job(run_id, pause=0):
    path=DEFAULT_ROOT/'runs/experiments'/run_id
    summary=read_json(path/'summary.json')
    cfg=summary['configuration']
    endpoint=pause or cfg['max_steps']
    if (summary['global_step']!=endpoint or summary['attempted_updates']!=endpoint or
            summary['last_eval_step']!=endpoint or summary['total_tokens_seen']!=endpoint*BATCH_TOKENS or
            summary['budget_tokens_charged']!=summary['total_tokens_seen'] or
            summary['cursor']!=endpoint*16 or summary['initialization']['pretrained_weights_loaded'] or
            summary['stop_reason']!=('validation_pause' if pause else 'training_token_target')):
        raise ValueError('Incomplete/mismatched endpoint; stop promotion, no retry')
    ev=summary['last_evaluation']
    if (ev['eval/prediction_tokens']<=0 or not math.isfinite(ev['eval/perplexity']) or
            ev['eval/documents']!=1000):
        raise ValueError('Invalid endpoint evaluation')
    return summary


def completed(s, keys):
    if not all(k in s['jobs'] and s['jobs'][k]['status']=='reconciled' for k in keys):
        return False
    for key in keys:
        r=s['jobs'][key]
        if r.get('slurm_state')!='COMPLETED':
            raise ValueError('Stage failed: '+key+'; diagnose without automatic retry')
        if file_digest(r['configuration'])!=r['configuration_sha256']:
            raise ValueError('Configuration changed')
        verify_job(r['run_id'],r['pause'])
    return True


def choose(s, keys, field, values):
    observations=[]
    provenance=None
    for key,value in zip(keys,values):
        r=s['jobs'][key]
        summary=verify_job(r['run_id'])
        identity=summary['data_provenance']
        if provenance is None:provenance=identity
        if provenance!=identity:raise ValueError('Endpoint evaluation identities differ')
        ev=summary['last_evaluation']
        observations.append(dict(candidate=key,value=value,nll=ev['eval/nll']/ev['eval/prediction_tokens'],
            perplexity=ev['eval/perplexity'],training_tokens=summary['total_tokens_seen'],
            physical_h200_hours=r['accounting']['h200_hours'],wandb_url=summary['wandb_url']))
    # Strict endpoint argmin; deterministic listed order for exact numeric ties.
    best=min(observations,key=lambda x:x['nll'])
    s['selections'][field]=best['value']
    s.setdefault('trajectory',[]).extend(observations)
    atomic_json(DIRECTORY/'results'/('ranking-'+field+'.json'),dict(best=best,observations=observations))
    report(s)


def report(s):
    rows=s.get('trajectory',[])
    lines=['# Horizon-scaled WSD campaign','',f'Stage: {s["stage"]}; selections: {s["selections"]}','',
           '| Candidate | PPL | Inputs | Physical H200-hours |','|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["candidate"]} | {row["perplexity"]:.6f} | {row["training_tokens"]} | {row["physical_h200_hours"]:.6f} |')
    lines+=['','Matched endpoint held-out NLL determines winners. Shared prefix is physically charged once; branches retain inherited tokens and compute.','No significance claim from a single seed. Final 50,000-document evaluation is separate from tuning.']
    (REPO/'to_human'/('horizon-wsd-20261014.md')).write_text('\n'.join(lines)+'\n')


def run_once():
    with locked(DIRECTORY/'.controller.lock',blocking=False):
        s=state()
        reconcile(s)
        if DISABLED.exists():
            report(s);return
        check()
        if s['stage']=='validation':
            cfg=copy.deepcopy(plan()['base_configuration'])
            cfg.update(category='validation',max_steps=20,scheduler_steps=20,warmup_steps=1,
                eval_steps=1,max_training_tokens=20*BATCH_TOKENS,target_training_tokens=20*BATCH_TOKENS,
                max_grad_norm=0,max_h200_hours=1.5,wandb_mode='offline',
                export_on_completion=False,retain_all_checkpoints=True,
                hold_schedule=tail_settings(.6),wsd_decay_ratio=.6,
                stop_margin_seconds=180,final_evaluation_reserve_seconds=30,sweep_id=CAMPAIGN,sweep_stage='runtime')
            path=CONFIGS/'validation.json';atomic_json(path,cfg)
            submit(s,'validation','validation-recovery1',path,load_config(path),10)
            if completed(s,['validation']):s['stage']='lr'
        if s['stage']=='lr':
            keys=['lr05','lr10','lr20'];ratios=[.5,1.,2.]
            for key,ratio in zip(keys,ratios):
                path,cfg=config_for(key,lr_ratio=ratio)
                submit(s,key,key,path,cfg,79 if key=='lr05' else 90)
            if completed(s,keys):
                choose(s,keys,'lr_ratio',ratios);s['stage']='clipping'
        if s['stage']=='clipping':
            keys=['clip05','clip10','clip20','clipoff'];values=[.5,1.,2.,0]
            for key,value in zip(keys,values):
                path,cfg=config_for(key,lr_ratio=s['selections']['lr_ratio'],clipping=value)
                submit(s,key,key,path,cfg,79 if key=='lr05' else 90)
            if completed(s,keys):
                choose(s,keys,'clipping',values);s['stage']='prefix'
        if s['stage']=='prefix':
            path,cfg=config_for('stable-prefix',lr_ratio=s['selections']['lr_ratio'],clipping=s['selections']['clipping'])
            submit(s,'prefix','stable-prefix',path,cfg,45,pause=PREFIX_STEP)
            if completed(s,['prefix']):s['stage']='decay'
        if s['stage']=='decay':
            keys=['decay60','decay50','decay40','decay20','decay10'];values=[.6,.5,.4,.2,.1]
            for key,value in zip(keys,values):
                path,cfg=config_for(key,lr_ratio=s['selections']['lr_ratio'],clipping=s['selections']['clipping'],tail=value)
                submit(s,key,key,path,load_config(path),75,parent=CAMPAIGN+'-stable-prefix')
            if completed(s,keys):
                choose(s,keys,'decay_tail',values);s['stage']='final'
        if s['stage']=='final':
            for arm in ('standard','resplit'):
                name='final-'+arm
                path,cfg=config_for(name,lr_ratio=s['selections']['lr_ratio'],clipping=s['selections']['clipping'],
                    tail=s['selections']['decay_tail'],final=True,resplit=arm=='resplit')
                submit(s,name+'-a',name,path,cfg,180,pause=FINAL_SPLIT)
                if completed(s,[name+'-a']):
                    submit(s,name+'-b',name,path,cfg,60,mode='resume',full_eval=True)
            keys=['final-standard-b','final-resplit-b']
            if completed(s,keys):
                final_results=[]
                for key in keys:
                    r=s['jobs'][key];full=read_json(DEFAULT_ROOT/'runs/experiments'/r['run_id']/'full-evaluation.json')
                    if full['training_global_step']!=FINAL_STEPS or full['eval/documents']!=50000:
                        raise ValueError('Final full-set evaluation incomplete')
                    final_results.append(dict(arm=key,perplexity=full['eval/perplexity'],nll=full['eval/nll']/full['eval/prediction_tokens'],wandb_url=verify_job(r['run_id'])['wandb_url']))
                s['final_full_results']=final_results
                s['selections']['final_arm']=min(final_results,key=lambda r:r['nll'])['arm']
                atomic_json(DIRECTORY/'results/final-full.json',final_results)
                s['stage']='complete'
                DISABLED.touch()
        s['updated_utc']=stamp();atomic_json(STATE,s);report(s)
        print(json.dumps(dict(stage=s['stage'],selections=s['selections'],jobs={k:r.get('slurm_state',r['status']) for k,r in s['jobs'].items()})))


def install():
    import shlex
    old=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if old.returncode not in (0,1):raise ValueError('Cannot inspect crontab')
    lines=[line for line in old.stdout.splitlines() if not line.rstrip().endswith(MARKER)]
    log=DEFAULT_ROOT/'runs'/CAMPAIGN/'controller.log';log.parent.mkdir(parents=True,exist_ok=True)
    lines.append('*/20 * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    desired='\n'.join(lines)+'\n'
    subprocess.run(['crontab','-'],input=desired,text=True,check=True)
    actual=subprocess.run(['crontab','-l'],capture_output=True,text=True,check=True).stdout
    if actual!=desired:raise ValueError('Schedule verification failed')
    DISABLED.unlink(missing_ok=True)
    print('Installed bounded controller, every 20 minutes; preserved other schedules.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','run-once','status','install','disable','verify-job','test-only'])
    parser.add_argument('run_id',nargs='?');parser.add_argument('--pause',type=int,default=0)
    args=parser.parse_args()
    if args.action in ('install', 'run-once') and not (DIRECTORY/'recovery-authorization.json').exists():
        raise ValueError('One additional recovery allocation requires explicit approval recorded in recovery-authorization.json')
    if args.action=='check':check()
    elif args.action=='run-once':run_once()
    elif args.action=='verify-job':verify_job(args.run_id,args.pause)
    elif args.action=='status':
        with locked(DIRECTORY/'.controller.lock',blocking=False):
            s=state();reconcile(s);atomic_json(STATE,s);report(s);print(json.dumps(s,indent=2))
    elif args.action=='install':check();install()
    elif args.action=='disable':DISABLED.touch();print('New submissions disabled; status still reconciles terminal jobs.')
    else:
        s=state();path,cfg=config_for('lr10');submit(s,'lr10','lr10',path,cfg,90,test_only=True)


if __name__=='__main__':main()
