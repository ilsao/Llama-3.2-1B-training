#!/usr/bin/env python3
"""Bounded WSD bracketing extension; user removed the aggregate256H time budget."""
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
import slurm.lr_transfer_controller as base
import lr_transfer as transfer
from training_state import DEFAULT_ROOT,ExperimentLedger,atomic_json,read_json,digest,file_digest,locked,load_config
STUDY='lr-transfer-wsd-bracket-20261015';PREFIX='init-recipe-20261014-transfer-wsdbracket-'
STATE=LIVE/'controller-state.json';MARKER='# '+STUDY+'-controller';STAGE_LIMIT=24.
base.LIVE=LIVE;base.STATE=STATE;base.PREFIX=PREFIX


def owner():
    a=read_json(LIVE/'submission-owner.json')
    return a['script']==str(Path(__file__).resolve()) and a['sha256']==file_digest(__file__) and a['hostname']==socket.gethostname().split('.')[0]


def check():
    p=read_json(LIVE/'plan.json')
    if p['study']!=STUDY or p['aggregate_time_budget_enforced'] or p['stage_h200_hours_limit']!=24:raise ValueError('Extension authorization changed')
    for f,h in read_json(LIVE/'source-seal.json')['files'].items():
        if file_digest(SOURCE/f)!=h:raise ValueError('Source changed: '+f)
    for f,h in p['immutable_inputs'].items():
        if file_digest(f)!=h:raise ValueError('Input changed: '+f)

base.check=check


def make_configuration(kind,muon):
    cfg=transfer.configuration(read_json(SOURCE/'configs/lr-transfer-base.json'),kind,muon/transfer.ANCHOR_MUON)
    cfg.update(sweep_id=STUDY,sweep_purpose='PhaseA: find actual400M WSD optimum above failed EMA prediction; fixed AdamW/Muon ratio')
    return cfg


def submit(s,key,kind,muon):
    if key in s['jobs'] or (LIVE/'disabled').exists():return
    minutes={'screen':15,'decay':45}[kind];hours=minutes*8/60
    _,used=base.charges()
    if used+hours>STAGE_LIMIT+1e-9:raise ValueError('PhaseA24H allocation limit exhausted')
    queue=subprocess.run(['squeue','-u',getpass.getuser(),'-p','8gpus','-h','-o','%i'],capture_output=True,text=True,timeout=30,check=True)
    limit=subprocess.run(['sacctmgr','-n','-P','show','qos','p_8gpus','format=Name,MaxSubmitPU'],capture_output=True,text=True,timeout=30,check=True).stdout.strip().split('|')
    if len(queue.stdout.splitlines())>=int(limit[1]):return
    run=PREFIX+key
    if (DEFAULT_ROOT/'runs/experiments'/run).exists():raise ValueError('No automatic retry/resume for an initialized run')
    cfg=make_configuration(kind,muon);path=LIVE/'configs'/f'{key}.json';atomic_json(path,cfg);load_config(path)
    tag=STUDY+'-'+uuid.uuid4().hex;ledger=ExperimentLedger(DEFAULT_ROOT);ledger.reserve(tag,run,cfg,minutes*60,8,False)
    command=['sbatch','--parsable','--time='+str(minutes),'--job-name=wb-'+key,'--comment='+tag,'--exclude='+','.join(read_json(LIVE/'plan.json')['excluded_nodes']),str(SOURCE/'slurm/lr_transfer.sbatch'),run,str(path)]
    s['jobs'][key]={'run_id':run,'kind':kind,'multiplier':muon/transfer.ANCHOR_MUON,'config':str(path),'status':'submission_uncertain','reservation':tag,'command':command}
    atomic_json(STATE,s)
    with ledger.transaction() as data:data['jobs'][tag].update(allocation_kind=transfer.POOL,transfer_study=STUDY,submission_tag=tag,submission_command=command,submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_root=str(SOURCE),controller_script=str(Path(__file__).resolve()),authorization='User: disregard aggregate time budget and continue designing/performing LR transfer experiments; PhaseA bounded24H, no extra formal run')
    env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),'TRANSFER_CONTROLLER':str(Path(__file__).resolve())}
    try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired) as e:s['jobs'][key]['submission_error']=str(e);atomic_json(STATE,s);return
    receipt={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};s['jobs'][key]['submission_receipt']=receipt;job=r.stdout.strip().split(';')[0]
    if r.returncode==0 and job.isdigit():
        with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=job
        ledger.bind(tag,job);s['jobs'][key].update(job_id=job,status='submitted')
    elif r.returncode==1 and not r.stdout.strip() and 'QOSMaxSubmitJobPerUserLimit' in r.stderr and 'Batch job submission failed' in r.stderr:
        ledger.cancel_reservation(tag);s.setdefault('qos_rejections',[]).append(receipt);s['jobs'].pop(key)
    atomic_json(STATE,s)


def bracket_result(s):
    p=read_json(LIVE/'plan.json');observations=list(p['previous_wsd_points'])
    for i in range(3):
        r=s['jobs']['decay'+str(i)];e=r['endpoint'];ev=e['evaluation']
        if e['tokens']!=400031744 or ev['eval/prediction_tokens']!=1479302 or e['evaluation_identity']!=p['evaluation_identity']:
            raise ValueError('Unmatched400M WSD observation')
        observations.append({'muon_lr':r['multiplier']*transfer.ANCHOR_MUON,'adamw_lr':r['multiplier']*transfer.ANCHOR_MUON*.03,
            'mean_nll':ev['eval/nll']/ev['eval/prediction_tokens'],'perplexity':ev['eval/perplexity'],'job_id':r['job_id'],'wandb_url':e['wandb_url']})
    observations.sort(key=lambda x:x['muon_lr']);index=min(range(len(observations)),key=lambda i:observations[i]['mean_nll'])
    result={'observations':observations,'best_measured':observations[index],'bracketed':0<index<len(observations)-1,'target_tokens':400031744,'lr_ratio':.03}
    if result['bracketed']:
        neighbors=observations[index-1:index+2]
        try:result['local_quadratic']=transfer.parabola([(v['muon_lr'],v['mean_nll']) for v in neighbors],min_points=3)
        except ValueError as e:result.update(bracketed=False,fit_error=str(e))
    atomic_json(LIVE/'bracket-result.json',result)
    return result


def report(s):
    pool,study=base.charges();out={'state':s,'historical_pool_h200_hours':pool,'aggregate_time_budget_enforced':False,'stage_actual_or_held_h200_hours':study,'stage_limit':24}
    if (LIVE/'bracket-result.json').exists():out['bracket']=read_json(LIVE/'bracket-result.json')
    atomic_json(LIVE/'results.json',out)
    lines=['# WSD LR谷底校準','',f"階段：{s['stage']}；本階段已用/保留{study:.3f}/24H。舊總時間預算依使用者指示解除，歷史累計{pool:.3f}H仍保留。",'']
    lines += [f"- {k}: {r.get('job_id','uncertain')}，{r.get('slurm_state',r['status'])}" for k,r in s['jobs'].items()]
    if 'bracket' in out:
        b=out['bracket'];lines += ['',f"谷底已包住：{b['bracketed']}；400M最佳實測PPL={b['best_measured']['perplexity']:.6f}。"]
    (LIVE/'report.md').write_text('\n'.join(lines)+'\n')


def run_once():
    if not owner():return
    with locked(LIVE/'.controller.lock',blocking=False):
        check();s=read_json(STATE) if STATE.exists() else {'stage':'screen','jobs':{}}
        try:
            base.reconcile(s)
            if s.get('blocked') or (LIVE/'disabled').exists():report(s);return
            grid=read_json(LIVE/'plan.json')['new_muon_lrs']
            if s['stage']=='screen':
                for i,muon in enumerate(grid):submit(s,'screen'+str(i),'screen',muon)
                if not base.completed(s,['screen'+str(i) for i in range(3)]):report(s);return
                s['stage']='decay'
            if s['stage']=='decay':
                for i,muon in enumerate(grid):submit(s,'decay'+str(i),'decay',muon)
                if not base.completed(s,['decay'+str(i) for i in range(3)]):report(s);return
                result=bracket_result(s);s['stage']='complete' if result['bracketed'] else 'needs_wider_bracket'
            atomic_json(STATE,s);report(s)
        except subprocess.TimeoutExpired as error:
            command=error.cmd[0] if isinstance(error.cmd,(list,tuple)) else ''
            if command in ('squeue','sacct','sacctmgr'):
                s['scheduler_deferred']=str(error);atomic_json(STATE,s);report(s);return
            s.update(blocked=True,error=str(error));atomic_json(STATE,s);report(s);raise
        except Exception as error:s.update(blocked=True,error=str(error));atomic_json(STATE,s);report(s);raise


def install():
    check()
    if not owner():raise ValueError('Wrong owner')
    old=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if old.returncode not in (0,1):raise ValueError('Cannot read cron')
    rows=[r for r in old.stdout.splitlines() if not r.rstrip().endswith(MARKER)]
    log=DEFAULT_ROOT/'runs'/STUDY/'controller.log';log.parent.mkdir(parents=True,exist_ok=True)
    rows.append('* * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    wanted='\n'.join(rows)+'\n';subprocess.run(['crontab','-'],input=wanted,text=True,check=True)
    if subprocess.check_output(['crontab','-l'],text=True)!=wanted:raise ValueError('Cron did not persist')

if __name__=='__main__':
    action=sys.argv[1]
    if action=='check':check()
    elif action=='install':install()
    elif action=='run-once':run_once()
    elif action=='receipt':base.receipt(sys.argv[2],sys.argv[3])
    elif action=='status':print(json.dumps(read_json(STATE),indent=2))
    else:raise ValueError('Unknown action')
