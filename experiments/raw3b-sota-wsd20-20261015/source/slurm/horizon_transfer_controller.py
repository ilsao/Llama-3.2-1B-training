#!/usr/bin/env python3
"""Bounded fresh WSD calibration at 0.2/0.6B; metadata-only owner."""
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
STUDY='lr-transfer-horizons-20261015';PREFIX='init-recipe-20261014-transfer-horizons-'
STATE=LIVE/'controller-state.json';MARKER='# '+STUDY+'-controller';STAGE_LIMIT=62.
base.LIVE=LIVE;base.STATE=STATE;base.PREFIX=PREFIX


def owner():
    a=read_json(LIVE/'submission-owner.json')
    return a['script']==str(Path(__file__).resolve()) and a['sha256']==file_digest(__file__) and a['hostname']==socket.gethostname().split('.')[0]


def check():
    p=read_json(LIVE/'plan.json')
    if p['study']!=STUDY or p['aggregate_time_budget_enforced'] or p['stage_h200_hours_limit']!=STAGE_LIMIT:
        raise ValueError('Horizon calibration authorization changed')
    for f,h in read_json(LIVE/'source-seal.json')['files'].items():
        if file_digest(SOURCE/f)!=h:raise ValueError('Source changed: '+f)
    for f,h in p['immutable_inputs'].items():
        if file_digest(f)!=h:raise ValueError('Input changed: '+f)
    # An empty or truncated ledger must NEVER appear to be a fresh free budget.
    ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    if (len(ledger['jobs'])<188 or len(ledger['runs'])<170 or
        sum(j.get('h200_hours',0) for j in ledger['jobs'].values())+1e-9<614.2808333333338 or
        sum(r['category']=='full' for r in ledger['runs'].values())!=8):
        raise ValueError('Historical ledger integrity floor violated; stop new submissions')

base.check=check


def make_configuration(case):
    kind=case['kind'];cfg=transfer.configuration(read_json(SOURCE/'configs/lr-transfer-base.json'),kind,case['muon_lr']/transfer.ANCHOR_MUON)
    tokens=transfer.align_tokens(case['tokens']);steps=tokens//transfer.BATCH
    cfg.update(sweep_id=STUDY,sweep_purpose='True WSD horizon calibration, fixed LR ratio; no EMA proxy optimum assumption',
        target_training_tokens=tokens,max_training_tokens=tokens,max_steps=steps,scheduler_steps=steps,
        max_h200_hours=case['minutes']*8/60,eval_steps=max(1,steps//10),logging_steps=10)
    return cfg


def submit(s,case):
    key=case['key']
    if key in s['jobs'] or (LIVE/'disabled').exists():return
    prerequisite=case.get('requires')
    if prerequisite:
        job=s['jobs'].get(prerequisite,{})
        if job.get('status')!='reconciled':return
        if job.get('slurm_state')!='COMPLETED' or 'endpoint' not in job:
            raise ValueError('Required stability screen failed; no retry: '+prerequisite)
    minutes=case['minutes'];hours=minutes*8/60
    _,used=base.charges()
    if used+hours>STAGE_LIMIT+1e-9:raise ValueError('Bounded62H calibration allocation limit exhausted')
    queue=subprocess.run(['squeue','-u',getpass.getuser(),'-p','8gpus','-h','-o','%i'],capture_output=True,text=True,timeout=30,check=True)
    limit=subprocess.run(['sacctmgr','-n','-P','show','qos','p_8gpus','format=Name,MaxSubmitPU'],capture_output=True,text=True,timeout=30,check=True).stdout.strip().split('|')
    if len(queue.stdout.splitlines())>=int(limit[1]):return
    run=PREFIX+key
    if (DEFAULT_ROOT/'runs/experiments'/run).exists():raise ValueError('No automatic retry/resume for an initialized run')
    cfg=make_configuration(case);path=LIVE/'configs'/f'{key}.json';atomic_json(path,cfg);load_config(path)
    tag=STUDY+'-'+uuid.uuid4().hex;ledger=ExperimentLedger(DEFAULT_ROOT);ledger.reserve(tag,run,cfg,minutes*60,8,False)
    command=['sbatch','--parsable','--time='+str(minutes),'--job-name=ht-'+key,'--comment='+tag,'--exclude='+','.join(read_json(LIVE/'plan.json')['excluded_nodes']),str(SOURCE/'slurm/lr_transfer.sbatch'),run,str(path)]
    s['jobs'][key]={'run_id':run,'kind':case['kind'],'multiplier':case['muon_lr']/transfer.ANCHOR_MUON,'config':str(path),'status':'submission_uncertain','reservation':tag,'command':command}
    atomic_json(STATE,s)
    with ledger.transaction() as data:
        data['jobs'][tag].update(allocation_kind=transfer.POOL,transfer_study=STUDY,submission_tag=tag,submission_command=command,
            submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_root=str(SOURCE),controller_script=str(Path(__file__).resolve()),
            authorization='User: continue; 2026-10-15 approved bounded fresh LR-transfer research beyond aggregate time gate; no new formal run')
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


def linear_law(points):
    if len(points)<2:raise ValueError('Need two distinct horizons')
    xs=[math.log(t/1e9) for t,lr in points];ys=[math.log(lr) for t,lr in points]
    mx=sum(xs)/len(xs);my=sum(ys)/len(ys);ss=sum((x-mx)**2 for x in xs)
    if ss<=0:raise ValueError('Horizon variation absent')
    beta=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/ss;intercept=my-beta*mx
    return dict(beta=beta,muon_lr_at_1b=math.exp(intercept),log_muon_lr_at_1b=intercept,
        r_squared=transfer.r_squared(ys,[intercept+beta*x for x in xs]),points=points)


def analyze(s):
    plan=read_json(LIVE/'plan.json');groups={400031744:plan['previous_wsd_points']};fits={};errors={}
    for case in plan['cases']:
        if case['kind']!='decay':continue
        job=s['jobs'][case['key']];e=job['endpoint'];ev=e['evaluation'];tokens=transfer.align_tokens(case['tokens'])
        if e['tokens']!=tokens or ev['eval/prediction_tokens']!=1479302 or e['evaluation_identity']!=plan['evaluation_identity']:
            raise ValueError('Unmatched horizon evaluation/data identity')
        groups.setdefault(tokens,[]).append(dict(muon_lr=case['muon_lr'],mean_nll=ev['eval/nll']/ev['eval/prediction_tokens'],perplexity=ev['eval/perplexity'],job_id=job['job_id'],wandb_url=e['wandb_url']))
    for tokens,rows in groups.items():
        rows.sort(key=lambda x:x['muon_lr']);index=min(range(len(rows)),key=lambda i:rows[i]['mean_nll'])
        if index in (0,len(rows)-1):errors[str(tokens)]='Measured optimum on boundary';continue
        try:
            fits[str(tokens)]=dict(best_measured=rows[index],local_fit=transfer.parabola([(v['muon_lr'],v['mean_nll']) for v in rows[index-1:index+2]],min_points=3))
        except ValueError as e:errors[str(tokens)]=str(e)
    result=dict(observations=groups,horizon_fits=fits,errors=errors,actual_wsd_calibration_only=True,validated=False)
    if not errors:
        points=[(int(t),v['local_fit']['muon_lr']) for t,v in sorted(fits.items(),key=lambda x:int(x[0]))]
        law=linear_law(points);loo=[]
        for i,(tokens,lr) in enumerate(points):
            fit=linear_law(points[:i]+points[i+1:]);pred=transfer.predict(fit,tokens)
            loo.append(dict(tokens=tokens,predicted_muon_lr=pred,local_fitted_muon_lr=lr,relative_lr_error=abs(pred/lr-1)))
        result.update(direct_wsd_law=law,leave_one_horizon_out=loo,
            predictions=[dict(tokens=t,muon_lr=transfer.predict(law,t),adamw_lr=.03*transfer.predict(law,t),status='unvalidated_extrapolation') for t in [800000000,1000000000,2000000000,3000000000,4000000000,5000000000,6000000000]],
            warning='Three calibration horizons and one seed; local three-point R2 is not independent evidence;0.8/1B not tested; EMA transfer not validated')
    atomic_json(LIVE/'calibration-result.json',result)
    return result


def report(s):
    pool,study=base.charges();out=dict(state=s,historical_pool_h200_hours=pool,aggregate_time_budget_enforced=False,stage_actual_or_held_h200_hours=study,stage_limit=STAGE_LIMIT)
    if (LIVE/'calibration-result.json').exists():out['calibration']=read_json(LIVE/'calibration-result.json')
    atomic_json(LIVE/'results.json',out)
    lines=['# True WSD horizon calibration','',f"Stage: {s['stage']}; measured/reserved {study:.3f}/{STAGE_LIMIT} H200-hours.",'']
    lines += [f"- {k}: {r.get('job_id','uncertain')}, {r.get('slurm_state',r['status'])}" for k,r in s['jobs'].items()]
    (LIVE/'report.md').write_text('\n'.join(lines)+'\n')


def run_once():
    if not owner():return
    try:
        with locked(LIVE/'.controller.lock',blocking=False):
            check();s=read_json(STATE) if STATE.exists() else {'stage':'calibration','jobs':{}}
            try:
                base.reconcile(s)
                if s.get('blocked') or (LIVE/'disabled').exists():report(s);return
                cases=read_json(LIVE/'plan.json')['cases']
                for case in cases:submit(s,case)
                if base.completed(s,[c['key'] for c in cases]):
                    result=analyze(s);s['stage']='needs_wider_bracket' if result['errors'] else 'calibration_complete'
                atomic_json(STATE,s);report(s)
            except subprocess.TimeoutExpired as error:
                command=error.cmd[0] if isinstance(error.cmd,(list,tuple)) else ''
                if command in ('squeue','sacct','sacctmgr'):
                    s['scheduler_deferred']=str(error);atomic_json(STATE,s);report(s);return
                s.update(blocked=True,error=str(error));atomic_json(STATE,s);report(s);raise
            except Exception as error:s.update(blocked=True,error=str(error));atomic_json(STATE,s);report(s);raise
    except BlockingIOError:return


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
