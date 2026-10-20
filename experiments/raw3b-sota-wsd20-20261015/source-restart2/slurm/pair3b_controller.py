#!/usr/bin/env python3
"""Sole owner of the explicitly requested fresh raw/filtered3B comparison."""
import datetime as dt
import getpass
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import uuid
SOURCE=Path(__file__).resolve().parents[1];LIVE=SOURCE.parent
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,digest,file_digest,locked,load_config,read_json,slurm_record
from slurm.submission import find_submission_job
from pair3b_contracts import authorized,RUNS
STUDY='raw-filtered-3b-20261015';STATE=LIVE/'controller-state.json';MARKER='# '+STUDY+'-controller'


def owner():
    p=read_json(LIVE/'submission-owner.json')
    return p['hostname']==socket.gethostname().split('.')[0] and p['script']==str(Path(__file__).resolve()) and p['sha256']==file_digest(__file__)


def check():
    plan=read_json(LIVE/'plan.json')
    for f,h in read_json(LIVE/'source-seal.json')['files'].items():
        if file_digest(SOURCE/f)!=h:raise ValueError('Sealed source changed: '+f)
    for f,h in plan['immutable_inputs'].items():
        if file_digest(f)!=h:raise ValueError('Immutable input changed: '+f)
    for arm in (['raw','filtered'] if plan.get('filtered_enabled',False) else ['raw']):
        if not authorized(STUDY+'-'+arm,load_config(LIVE/'configs'/f'{arm}.json')):raise ValueError('Exact3B formal authorization changed')
    ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    if len(ledger['jobs'])<199 or len(ledger['runs'])<181 or sum(j.get('h200_hours',0) for j in ledger['jobs'].values())<663.4386111111111-1e-8:
        raise ValueError('Historical accounting floor violated')
    full={k for k,v in ledger['runs'].items() if v['category']=='full'}
    if full-set(RUNS)!=set(plan['previous_formal_runs']):raise ValueError('Historical formal run identities changed')


def training_check(run_id,path):
    cfg=load_config(path);d=read_json(DEFAULT_ROOT/'runs/experiments'/run_id/'summary.json')
    if (not authorized(run_id,cfg) or d['configuration']!=cfg or d['global_step']!=22889 or d['attempted_updates']!=22889 or
        d['total_tokens_seen']!=3000107008 or d['budget_tokens_charged']!=3000107008 or d['stop_reason']!='training_token_target' or
        d['checkpoint_loaded_for_training'] or d['initialization']['pretrained_weights_loaded'] or d['export_global_step']!=22889 or not math.isfinite(d['last_evaluation']['eval/perplexity'])):
        raise ValueError('Incomplete3B training; no automatic retry/resume')
    return d


def receipt(run_id,path):
    check();d=training_check(run_id,path);full=read_json(DEFAULT_ROOT/'runs/experiments'/run_id/'full-evaluation.json')
    control=read_json(read_json(LIVE/'plan.json')['control_summary'])
    for key in ['dataset_revision','evaluation_indices_sha256','pack_settings']:
        if d['data_provenance'][key]!=control['data_provenance'][key]:raise ValueError('Matched evaluation provenance changed')
    if full['eval/documents']!=50000 or full['eval/prediction_tokens']!=76930208 or full['global_step']!=22889 or not math.isfinite(full['eval/perplexity']):raise ValueError('Incomplete full reserved-set evaluation')
    job=os.environ['SLURM_JOB_ID']
    for rank in range(8):
        if read_json(DEFAULT_ROOT/'logs'/('job-'+job)/f'worker-rank-{rank}-exit.json')['exit_code']!=0:raise ValueError('Training rank failure')
    atomic_json(LIVE/'receipts'/f'{job}.json',dict(job_id=job,run_id=run_id,tokens=d['total_tokens_seen'],subset_perplexity=d['last_evaluation']['eval/perplexity'],full_perplexity=full['eval/perplexity'],wandb_url=d['wandb_url'],checkpoint=d['checkpoint'],export=d['export']))


def reconcile(s):
    ledger=ExperimentLedger(DEFAULT_ROOT)
    for key,j in s['jobs'].items():
        if j['status']=='reconciled':continue
        if not j.get('job_id'):
            job=find_submission_job(j,DEFAULT_ROOT)
            if j['kind']=='training':
                d=read_json(ledger.path)
                if j['tag'] in d['jobs']:ledger.bind(j['tag'],job)
            j.update(job_id=job,status='submitted')
        result=subprocess.run(['sacct','-X','-j',j['job_id'],'-n','-P','-o','JobIDRaw,State'],capture_output=True,text=True,timeout=30,check=True)
        states=[x.split('|')[1].split()[0].rstrip('+') for x in result.stdout.splitlines() if x.split('|')[0]==j['job_id']]
        j['slurm_state']=states[0] if states else 'UNKNOWN'
        if j['slurm_state'] not in TERMINAL:continue
        if j['kind']=='training':j['accounting']=ledger.reconcile(j['job_id'])
        else:
            measured=slurm_record(j['job_id'],DEFAULT_ROOT)
            with locked(DEFAULT_ROOT/'.accounting.lock'):
                path=DEFAULT_ROOT/'preprocessing-ledger.json';data=read_json(path)
                data['jobs'][j['job_id']]=dict(**measured,gpu_hours=measured['h200_hours'],category='data_preprocessing',stage=key,preparation_id='dolma-dclm10-005-rawmix3b-20261015')
                data['total_h200_hours']=sum(x['h200_hours'] for x in data['jobs'].values());data['total_gpu_hours']=data['total_h200_hours'];atomic_json(path,data)
            j['accounting']=measured
        j['status']='reconciled'
        p=LIVE/'receipts'/f"{j['job_id']}.json"
        if p.exists():j['endpoint']=read_json(p)
    atomic_json(STATE,s)


def ready(s,key):
    j=s['jobs'].get(key,{})
    if j.get('status')!='reconciled':return False
    if j.get('slurm_state')!='COMPLETED':raise ValueError('Failed '+key+'; no automatic retry/resume')
    return True


def submit(s,key,kind,minutes,arm=None):
    if key in ('mixprep','filtered') and not read_json(LIVE/'plan.json').get('filtered_enabled',False):return
    if key in s['jobs'] or (LIVE/'disabled').exists():return
    queue=subprocess.run(['squeue','-u',getpass.getuser(),'-p','8gpus','-h','-o','%i'],capture_output=True,text=True,timeout=30,check=True)
    if len(queue.stdout.splitlines())>=20:return
    tag=STUDY+'-'+uuid.uuid4().hex;run=STUDY+'-'+arm if arm else None
    j=dict(kind=kind,tag=tag,submission_tag=tag,submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='submission_uncertain',minutes=minutes,gpu_count=8 if arm else 1)
    if arm:
        path=LIVE/'configs'/f'{arm}.json';cfg=load_config(path)
        if (DEFAULT_ROOT/'runs/experiments'/run).exists():raise ValueError('No overwrite/retry of initialized run')
        ledger=ExperimentLedger(DEFAULT_ROOT);ledger.reserve(tag,run,cfg,minutes*60,8,False)
        with ledger.transaction() as data:data['jobs'][tag].update(submission_tag=tag,submission_started_utc=j['submission_started_utc'],allocation_kind=STUDY,source_root=str(SOURCE),authorization='User explicitly requested matched fresh3B raw and old filtered restored-token-mixture formal9/10;6h48H each')
        prep='dolma-raw-baseline-20261008' if arm=='raw' else 'dolma-dclm10-005-rawmix3b-20261015'
        args=[str(SOURCE/'slurm/pair3b-train.sbatch'),run,str(path),prep];j.update(run_id=run,configuration=str(path))
    else:args=[str(SOURCE/'slurm/pair3b-data.sbatch'),key,read_json(LIVE/'plan.json').get('raw_extension_authorization',str(LIVE/'raw-extension.json'))]
    command=['sbatch','--parsable','--time='+str(minutes),'--job-name=p3-'+key,'--comment='+tag,'--exclude='+','.join(read_json(LIVE/'plan.json')['excluded_nodes']),*args]
    j['submission_command']=command;s['jobs'][key]=j;atomic_json(STATE,s)
    if arm:
        with ledger.transaction() as data:data['jobs'][tag]['submission_command']=command
    env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),'PAIR3B_CONTROLLER':str(Path(__file__).resolve())}
    try:result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired) as e:j['submission_error']=str(e);atomic_json(STATE,s);return
    j['submission_receipt']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr);job=result.stdout.strip().split(';')[0]
    if result.returncode==0 and job.isdigit():
        j['returned_job_id']=job
        if arm:
            with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=job
            ledger.bind(tag,job)
        j.update(job_id=job,status='submitted')
    atomic_json(STATE,s)


def report(s):
    out=dict(state=s)
    out['actual_or_reserved_h200_hours']=sum(j.get('accounting',{}).get('h200_hours',j['minutes']*j['gpu_count']/60) for j in s['jobs'].values())
    atomic_json(LIVE/'results.json',out)
    lines=['# Raw3B training; filtered arm cancelled','',f"Stage: {s['stage']}",'']+[f"- {k}: {j.get('job_id','uncertain')}, {j.get('slurm_state',j['status'])}" for k,j in s['jobs'].items()]
    (LIVE/'report.md').write_text('\n'.join(lines)+'\n')


def run_once():
    if not owner():return
    try:
        with locked(LIVE/'.controller.lock',blocking=False):
            s=read_json(STATE) if STATE.exists() else dict(stage='prepare',jobs={})
            try:
                check();reconcile(s)
                if s.get('blocked') or (LIVE/'disabled').exists():report(s);return
                submit(s,'rawprep','preprocessing',240)
                if ready(s,'rawprep'):
                    progress=read_json(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008/raw/progress.json')
                    if sum(f['rows'] for f in progress['train_files'])<3000107008//8192:raise ValueError('Raw published prefix shorter than3B')
                    submit(s,'raw','training',360,'raw')
                    if read_json(LIVE/'plan.json').get('filtered_enabled',False):
                        submit(s,'mixprep','preprocessing',960);s['stage']='train_and_reweight'
                    else:s['stage']='raw_only'
                if read_json(LIVE/'plan.json').get('filtered_enabled',False) and ready(s,'mixprep'):
                    validation=read_json(DEFAULT_ROOT/'preprocessed/dolma-dclm10-005-rawmix3b-20261015/verify/report.json')
                    if not validation['passed'] or validation['maximum_absolute_share_error']>.0001:raise ValueError('Token-mixture verification failed')
                    submit(s,'filtered','training',360,'filtered');s['stage']='training'
                if not read_json(LIVE/'plan.json').get('filtered_enabled',False) and ready(s,'raw'):
                    if 'endpoint' not in s['jobs']['raw']:raise ValueError('Missing raw3B/full-eval receipt')
                    s['stage']='complete_raw_only'
                if read_json(LIVE/'plan.json').get('filtered_enabled',False) and ready(s,'raw') and ready(s,'filtered'):
                    if any('endpoint' not in s['jobs'][k] for k in ['raw','filtered']):raise ValueError('Missing3B/full-eval receipt')
                    s['stage']='complete'
                atomic_json(STATE,s);report(s)
            except subprocess.TimeoutExpired as e:
                command=e.cmd[0] if isinstance(e.cmd,(tuple,list)) else ''
                if command in ('sacct','squeue'):s['scheduler_deferred']=str(e)
                else:s.update(blocked=True,error=str(e))
                atomic_json(STATE,s);report(s)
            except Exception as e:s.update(blocked=True,error=str(e));atomic_json(STATE,s);report(s);raise
    except BlockingIOError:return


def install():
    check()
    if not owner():raise ValueError('Wrong submission owner')
    old=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if old.returncode not in (0,1):raise ValueError('Cannot read cron')
    rows=[r for r in old.stdout.splitlines() if not r.rstrip().endswith(MARKER)]
    log=DEFAULT_ROOT/'runs'/STUDY/'controller.log';log.parent.mkdir(parents=True,exist_ok=True)
    rows.append('* * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    subprocess.run(['crontab','-'],input='\n'.join(rows)+'\n',text=True,check=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='check':check()
    elif action=='install':install()
    elif action=='run-once':run_once()
    elif action=='training-check':training_check(sys.argv[2],sys.argv[3])
    elif action=='receipt':receipt(sys.argv[2],sys.argv[3])
    else:raise ValueError('Unknown action')
