#!/usr/bin/env python3
"""Single matched2B submission; configuration and ledger checks are metadata-only."""
import copy
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
SOURCE=Path(__file__).resolve().parents[1];LIVE=SOURCE.parent;REPO=SOURCE.parents[2]
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,digest,file_digest,locked,read_json,load_config
from followup_contracts import NAME,RUN,PARENT,build_configuration
CONTROL=REPO/'experiments'/NAME
STATE=LIVE/'controller-state.json';MARKER='# '+RUN+'-controller'


def owner():
    a=read_json(LIVE/'submission-owner.json')
    return a['script']==str(Path(__file__).resolve()) and a['sha256']==file_digest(__file__) and a['hostname']==socket.gethostname().split('.')[0]

def check():
    a=read_json(LIVE/'authorization.json');seal=read_json(LIVE/'source-seal.json')
    for f,h in seal['files'].items():
        if file_digest(SOURCE/f)!=h:raise ValueError('Sealed followup source changed: '+f)
    for f,h in a['parent_numerical_payload'].items():
        if file_digest(SOURCE/f)!=h or file_digest(CONTROL/'source'/f)!=h:raise ValueError('Numerical payload changed: '+f)
    if a['run_id']!=RUN or a['budget_h200_hours']!=256 or a['max_h200_hours']!=32:raise ValueError('Authorization changed')

def parent_completed(s):
    parent=read_json(CONTROL/'controller-state.json');r=parent['jobs'].get('final-2b')
    if not r or r.get('status')!='reconciled':s['stage']='waiting_parent_result';return None
    if r.get('slurm_state')!='COMPLETED' or 'endpoint' not in r:s['stage']='parent_failed';return None
    e=r['endpoint'];run=DEFAULT_ROOT/'runs/experiments'/PARENT
    cfg=read_json(run/'resolved-config.json');summary=read_json(run/'summary.json');full=read_json(run/'full-evaluation.json')
    accounting=read_json(run/'resource-accounting.json')
    if (r['run_id']!=PARENT or file_digest(run/'resolved-config.json')!=r['config_sha256'] or summary['configuration']!=cfg or accounting['config_digest']!=digest(cfg) or summary['checkpoint_loaded_for_training'] or summary['initialization']['pretrained_weights_loaded'] or
        e['tokens']!=cfg['target_training_tokens'] or summary['stop_reason']!='training_token_target' or
        full['eval/documents']!=50000 or full['global_step']!=cfg['max_steps'] or full['eval/prediction_tokens']!=76930208 or
        not math.isfinite(full['eval/perplexity']) or not accounting['all_jobs_reconciled'] or accounting['budget_exceeded']):
        raise ValueError('Parent complete2B/full-evaluation/accounting contract failed')
    return cfg,full,r

def parent_ready(s):
    auth=read_json(LIVE/'authorization.json')
    if not auth.get('parallel_training'):
        return parent_completed(s)
    parent=read_json(CONTROL/'controller-state.json');r=parent['jobs'].get('final-2b')
    if not r or r.get('status') not in ('submitted','reconciled') or not r.get('job_id'):
        s['stage']='waiting_parent_configuration';return None
    if r.get('slurm_state') not in ('PENDING','RUNNING','COMPLETING','COMPLETED'):
        s['stage']='parent_failed';return None
    run=DEFAULT_ROOT/'runs/experiments'/PARENT;path=run/'resolved-config.json'
    if not path.exists():s['stage']='waiting_parent_configuration';return None
    cfg=read_json(path)
    if (r['run_id']!=PARENT or file_digest(path)!=r['config_sha256'] or
        file_digest(r['configuration'])!=r['config_sha256'] or read_json(r['configuration'])!=cfg):
        raise ValueError('Parent immutable configuration contract failed')
    build_configuration(cfg)
    with ExperimentLedger(DEFAULT_ROOT).transaction() as data:
        job=data['jobs'].get(r['job_id']);registered=data['runs'].get(PARENT)
        if (not job or job['run_id']!=PARENT or not registered or registered['config_digest']!=digest(cfg) or
            registered.get('formal_run_number')!=7):
            raise ValueError('Parent configuration is not the registered formal7')
    return cfg,None,r


def cost():
    with ExperimentLedger(DEFAULT_ROOT).transaction() as data:
        return sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j['run_id'].startswith(NAME+'-'))

def receipt(config):
    check();cfg=load_config(config);run=DEFAULT_ROOT/'runs/experiments'/RUN
    summary=read_json(run/'summary.json');full=read_json(run/'full-evaluation.json');job=os.environ['SLURM_JOB_ID']
    if (summary['global_step']!=cfg['max_steps'] or summary['attempted_updates']!=cfg['max_steps'] or
        summary['total_tokens_seen']!=cfg['target_training_tokens'] or summary['budget_tokens_charged']!=summary['total_tokens_seen'] or
        summary['last_eval_step']!=cfg['max_steps'] or summary['stop_reason']!='training_token_target' or
        summary['checkpoint_loaded_for_training'] or summary['initialization']['pretrained_weights_loaded'] or
        summary['initialization']['recipe']['variant']!='residual_half' or full['eval/documents']!=50000 or
        full['eval/prediction_tokens']!=76930208 or full['global_step']!=cfg['max_steps'] or not math.isfinite(full['eval/perplexity'])):raise ValueError('Incomplete/incorrect followup endpoint')
    original=read_json(DEFAULT_ROOT/'runs/experiments'/PARENT/'data-provenance.json')
    if any(summary['data_provenance'][k]!=original[k] for k in ('dataset_revision','evaluation_indices_sha256','pack_settings')):raise ValueError('Followup evaluation identity differs from parent')
    exits=[read_json(DEFAULT_ROOT/'logs'/('job-'+job)/f'worker-rank-{i}-exit.json')['exit_code'] for i in range(8)]
    if any(exits):raise ValueError('Followup workers did not complete')
    atomic_json(LIVE/'endpoint.json',{'run_id':RUN,'job_id':job,'configuration_digest':digest(cfg),'tokens':summary['total_tokens_seen'],
        'checkpoint':summary['checkpoint'],'wandb_url':summary['wandb_url'],'subset':summary['last_evaluation'],'full':full,
        'evaluation_identity':summary['data_provenance'],'checkpoint_complete_sha256':file_digest(Path(summary['checkpoint'])/'complete.json')})

def report(s):
    out={'state':s,'campaign_actual_or_held_h200_hours':cost(),'campaign_limit':256,'followup_limit':32}
    if (LIVE/'endpoint.json').exists():
        e=read_json(LIVE/'endpoint.json');completed=parent_completed({})
        if completed is None:
            out['comparison_status']='waiting_parent_full_result'
        else:
            parent=completed[1]
            a=e['full']['eval/nll']/e['full']['eval/prediction_tokens'];b=parent['eval/nll']/parent['eval/prediction_tokens']
            out['comparison']={'baseline_full_perplexity':parent['eval/perplexity'],'followup_full_perplexity':e['full']['eval/perplexity'],'nll_delta':a-b,'improved':a<b,'followup_wandb':e['wandb_url'],'followup_checkpoint':e['checkpoint']}
    atomic_json(LIVE/'results.json',out)
    lines=['# Residual-half＋resplit 2B 對照','',f"狀態：{s['stage']}；這個 run 最多32 H200-hours，包含在原256H總預算。",'', 'EMA 關閉，從隨機權重開始。沿用當輪最終 LR、Muon、seed、data 和 batch。']
    if 'comparison' in out:
        d=out['comparison'];lines += ['',f"完整50k PPL：原配方 {d['baseline_full_perplexity']:.6f}；residual-half＋resplit {d['followup_full_perplexity']:.6f}。",'',f"[W&B]({d['followup_wandb']})"]
    (LIVE/'report.md').write_text('\n'.join(lines)+'\n')

def reconcile(s):
    if not s.get('reservation') or s.get('status')=='reconciled':return
    ledger=ExperimentLedger(DEFAULT_ROOT)
    if not s.get('job_id'):
        from slurm.submission import find_submission_job
        with ledger.transaction() as d:
            found=[x for k,x in d['jobs'].items() if k==s['reservation'] or x.get('submission_tag')==s['reservation']]
            if len(found)!=1:raise ValueError('Ambiguous followup reservation')
            reserved=copy.deepcopy(found[0])
        job=find_submission_job(reserved,DEFAULT_ROOT)
        if not reserved.get('job_id'):ledger.bind(s['reservation'],job)
        s['job_id']=job;s['status']='submitted'
    r=subprocess.run(['sacct','-X','-j',s['job_id'],'-n','-P','-o','JobIDRaw,State'],capture_output=True,text=True,timeout=30,check=True)
    states=[x.split('|')[1].split()[0].rstrip('+') for x in r.stdout.splitlines() if x.split('|')[0]==s['job_id']]
    s['slurm_state']=states[0] if states else 'UNKNOWN'
    if s['slurm_state'] in TERMINAL:
        s['accounting']=ledger.reconcile(s['job_id']);s['status']='reconciled'
        s['stage']='complete' if s['slurm_state']=='COMPLETED' and (LIVE/'endpoint.json').exists() else 'failed'


def run_once():
    if not owner():return
    with locked(LIVE/'.controller.lock',blocking=False):
        check();s=read_json(STATE) if STATE.exists() else {'stage':'waiting_parent_result'}
        reconcile(s)
        if s.get('reservation') or (LIVE/'disabled').exists():atomic_json(STATE,s);report(s);return
        parent=parent_ready(s)
        if parent is None:atomic_json(STATE,s);report(s);return
        parentcfg,full,record=parent;cfg=build_configuration(parentcfg)
        if cost()+32>256:raise ValueError('Insufficient original campaign budget for one32H followup')
        queues=subprocess.run(['squeue','-u',getpass.getuser(),'-p','8gpus','-h','-o','%i'],capture_output=True,text=True,timeout=30,check=True)
        limit=subprocess.run(['sacctmgr','-n','-P','show','qos','p_8gpus','format=Name,MaxSubmitPU'],capture_output=True,text=True,timeout=30,check=True).stdout.strip().split('|')
        if len(queues.stdout.splitlines())>=int(limit[1]):s['stage']='waiting_qos_slot';atomic_json(STATE,s);report(s);return
        if (DEFAULT_ROOT/'runs/experiments'/RUN).exists():raise ValueError('Followup already initialized; no retry')
        path=LIVE/'resolved-config.json';atomic_json(path,cfg);load_config(path)
        atomic_json(LIVE/'resolved.json',{'configuration_digest':digest(cfg),'parent_configuration':str(DEFAULT_ROOT/'runs/experiments'/PARENT/'resolved-config.json'),
            'parent_configuration_sha256':file_digest(DEFAULT_ROOT/'runs/experiments'/PARENT/'resolved-config.json'),'parent_configuration_verified':True,
            'parent_completed_full_evaluation':full is not None,'parallel_training':read_json(LIVE/'authorization.json').get('parallel_training',False),
            'parent_full_perplexity':None if full is None else full['eval/perplexity'],'parent_job':record['job_id']})
        ledger=ExperimentLedger(DEFAULT_ROOT);tag='followup-'+uuid.uuid4().hex
        ledger.reserve(tag,RUN,cfg,14400,8,False)
        auth=read_json(LIVE/'authorization.json')
        command=['sbatch','--parsable','--time=240','--job-name=rh-resplit-2b','--comment='+tag,'--exclude='+','.join(auth['excluded_nodes']),str(SOURCE/'slurm/followup.sbatch'),RUN,str(path)]
        s.update(stage='submitted',status='submission_uncertain',reservation=tag,command=command,run_id=RUN)
        atomic_json(STATE,s)
        with ledger.transaction() as d:d['jobs'][tag].update(allocation_kind=NAME,submission_tag=tag,submission_command=command,submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_root=str(SOURCE),controller_script=str(Path(__file__).resolve()))
        env={**os.environ,'DATA_ROOT':str(DEFAULT_ROOT),'TRAIN_SOURCE_ROOT':str(SOURCE),'FOLLOWUP_CONTROLLER':str(Path(__file__).resolve())}
        try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
        except (OSError,subprocess.TimeoutExpired) as e:s['submission_error']=str(e);atomic_json(STATE,s);return
        s['submission_receipt']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        job=r.stdout.strip().split(';')[0]
        if r.returncode==0 and job.isdigit():
            with ledger.transaction() as d:d['jobs'][tag]['returned_job_id']=job
            ledger.bind(tag,job);s.update(status='submitted',job_id=job)
        elif r.returncode==1 and not r.stdout.strip() and 'QOSMaxSubmitJobPerUserLimit' in r.stderr and 'Batch job submission failed' in r.stderr:
            ledger.cancel_reservation(tag);s.setdefault('submission_rejections',[]).append(s['submission_receipt']);s.pop('reservation');s['stage']='waiting_qos_slot'
        atomic_json(STATE,s);report(s)


def install():
    check()
    if not owner():raise ValueError('Wrong followup owner')
    old=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if old.returncode not in (0,1):raise ValueError('Cannot read crontab')
    rows=[x for x in old.stdout.splitlines() if not x.rstrip().endswith(MARKER)]
    log=DEFAULT_ROOT/'runs'/RUN/'controller.log';log.parent.mkdir(parents=True,exist_ok=True)
    rows.append('* * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    wanted='\n'.join(rows)+'\n';subprocess.run(['crontab','-'],input=wanted,text=True,check=True)
    if subprocess.check_output(['crontab','-l'],text=True)!=wanted:raise ValueError('Followup cron did not persist')

if __name__=='__main__':
    a=sys.argv[1]
    if a=='check':check()
    elif a=='run-once':run_once()
    elif a=='install':install()
    elif a=='receipt':receipt(sys.argv[2])
    elif a=='status':print(json.dumps(read_json(STATE),indent=2))
    else:raise ValueError('Unknown action')
