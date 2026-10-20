#!/usr/bin/env python3
"""Single owner: reserve, submit, reconcile and fit within the existing256H pool."""
import copy
import datetime as dt
import getpass
import json
import math
import statistics
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import uuid
SOURCE=Path(__file__).resolve().parents[1];LIVE=SOURCE.parent
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,digest,file_digest,locked,read_json,load_config
from lr_transfer import STUDY,POOL,PREFIX,MULTIPLIERS,MINUTES,MAX_STUDY_HOURS,ANCHOR_MUON,configuration,analyze,parabola,predict
STATE=LIVE/'controller-state.json';MARKER='# '+STUDY+'-controller'


def owner():
    a=read_json(LIVE/'submission-owner.json')
    return a['script']==str(Path(__file__).resolve()) and a['sha256']==file_digest(__file__) and a['hostname']==socket.gethostname().split('.')[0]


def check():
    plan=read_json(LIVE/'plan.json');seal=read_json(LIVE/'source-seal.json')
    if plan['pool_cap_h200_hours']!=256 or plan['study_max_h200_hours']!=MAX_STUDY_HOURS or plan['scheduler_target']!='single_wsd':raise ValueError('Study authorization changed')
    for f,h in seal['files'].items():
        if file_digest(SOURCE/f)!=h:raise ValueError('Sealed source changed: '+f)
    for f,h in plan['immutable_inputs'].items():
        if file_digest(f)!=h:raise ValueError('Immutable input changed: '+f)


def charges():
    with ExperimentLedger(DEFAULT_ROOT).transaction() as data:
        pool=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j['run_id'].startswith(POOL+'-'))
        study=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j['run_id'].startswith(PREFIX))
    return pool,study


def receipt(path,run_id):
    check();cfg=load_config(path);run=DEFAULT_ROOT/'runs/experiments'/run_id;s=read_json(run/'summary.json');job=os.environ['SLURM_JOB_ID']
    if (s['configuration']!=cfg or s['stop_reason']!='training_token_target' or s['global_step']!=cfg['max_steps'] or s['attempted_updates']!=cfg['max_steps'] or
        s['total_tokens_seen']!=cfg['target_training_tokens'] or s['budget_tokens_charged']!=s['total_tokens_seen'] or s['checkpoint_loaded_for_training'] or
        s['initialization']['pretrained_weights_loaded'] or s['initialization']['recipe']['variant']!='residual_half' or s['last_eval_step']!=cfg['max_steps'] or
        s['last_evaluation']['eval/documents']!=cfg['eval_documents'] or s['last_evaluation']['eval/prediction_tokens']<=0 or
        (cfg['lr_transfer']['kind']!='validation' and s['last_evaluation']['eval/prediction_tokens']!=1479302) or not math.isfinite(s['last_evaluation']['eval/perplexity'])):
        raise ValueError('Incomplete/nonfinite/mismatched LR transfer endpoint')
    exits=[read_json(DEFAULT_ROOT/'logs'/('job-'+job)/f'worker-rank-{i}-exit.json')['exit_code'] for i in range(8)]
    if any(exits):raise ValueError('Nonzero worker exit')
    original=read_json(Path(read_json(LIVE/'plan.json')['control_summary']))
    for k in ('dataset_revision','pack_settings'):
        if s['data_provenance'][k]!=original['data_provenance'][k]:raise ValueError('Evaluation/data identity differs from fixed control')
    expected_indices=original['data_provenance']['evaluation_indices'][:cfg['eval_documents']]
    if s['data_provenance']['evaluation_indices']!=expected_indices or s['data_provenance']['evaluation_indices_sha256']!=digest(expected_indices):raise ValueError('Evaluation indices are not the fixed control prefix')
    if cfg['ema'] and ('0.6' not in s['ema_evaluations'] or not math.isfinite(s['ema_evaluations']['0.6']['eval/perplexity'])):raise ValueError('Missing proxy EMA evaluation')
    if cfg['lr_transfer']['kind']=='validation':
        rows=[json.loads(row) for row in (run/'timing-steps.jsonl').read_text().splitlines()]
        median=statistics.median(r['update_seconds'] for r in rows if r['step']>=8)
        if median>1.:raise ValueError('Correctness passed but full-model update too slow for94min1B: '+str(median))
    atomic_json(LIVE/'receipts'/f'{job}.json',{'job_id':job,'run_id':run_id,'config_digest':digest(cfg),'tokens':s['total_tokens_seen'],
        'global_step':s['global_step'],'evaluation':s['last_evaluation'],'ema_evaluations':s['ema_evaluations'],'initialization_signature':s['initialization']['sample_signature'],
        'evaluation_identity':digest({k:s['data_provenance'][k] for k in ('dataset_revision','evaluation_indices_sha256','pack_settings')}),
        'checkpoint':s['checkpoint'],'wandb_url':s['wandb_url']})


def reconcile(state):
    ledger=ExperimentLedger(DEFAULT_ROOT)
    for key,r in state['jobs'].items():
        if r['status']=='reconciled':continue
        if not r.get('job_id'):
            from slurm.submission import find_submission_job
            with ledger.transaction() as d:
                found=[x for k,x in d['jobs'].items() if k==r['reservation'] or x.get('submission_tag')==r['reservation']]
                if len(found)!=1:raise ValueError('Ambiguous transfer reservation')
                reserved=copy.deepcopy(found[0])
            job=find_submission_job(reserved,DEFAULT_ROOT)
            if not reserved.get('job_id'):ledger.bind(r['reservation'],job)
            r.update(job_id=job,status='submitted')
        result=subprocess.run(['sacct','-X','-j',r['job_id'],'-n','-P','-o','JobIDRaw,State'],capture_output=True,text=True,timeout=30,check=True)
        states=[x.split('|')[1].split()[0].rstrip('+') for x in result.stdout.splitlines() if x.split('|')[0]==r['job_id']]
        r['slurm_state']=states[0] if states else 'UNKNOWN'
        if r['slurm_state'] in TERMINAL:
            r['accounting']=ledger.reconcile(r['job_id']);r['status']='reconciled'
            p=LIVE/'receipts'/f"{r['job_id']}.json"
            if r['slurm_state']=='COMPLETED' and p.exists():
                e=read_json(p)
                if e['run_id']!=r['run_id'] or e['config_digest']!=digest(load_config(r['config'])):raise ValueError('Receipt binding changed')
                r['endpoint']=e
    atomic_json(STATE,state)


def submit(state,key,kind,multiplier):
    if key in state['jobs']:return
    if (LIVE/'disabled').exists():return
    cfg=configuration(read_json(LIVE/'base-config.json'),kind,multiplier);minutes=MINUTES[kind];hours=minutes*8/60
    pool,study=charges()
    if pool+hours>256+1e-9 or study+hours>MAX_STUDY_HOURS+1e-9:raise ValueError('Original shared256H/study allocation budget exhausted')
    queue=subprocess.run(['squeue','-u',getpass.getuser(),'-p','8gpus','-h','-o','%i'],capture_output=True,text=True,timeout=30,check=True)
    limit=subprocess.run(['sacctmgr','-n','-P','show','qos','p_8gpus','format=Name,MaxSubmitPU'],capture_output=True,text=True,timeout=30,check=True).stdout.strip().split('|')
    if len(queue.stdout.splitlines())>=int(limit[1]):state['waiting_qos']=True;return
    run_id=PREFIX+key
    if (DEFAULT_ROOT/'runs/experiments'/run_id).exists():raise ValueError('Run already exists; no automatic retry/resume')
    path=LIVE/'configs'/f'{key}.json';atomic_json(path,cfg);load_config(path)
    tag=STUDY+'-'+uuid.uuid4().hex;ledger=ExperimentLedger(DEFAULT_ROOT);ledger.reserve(tag,run_id,cfg,minutes*60,8,False)
    exclusions=read_json(LIVE/'plan.json')['excluded_nodes']
    command=['sbatch','--parsable','--time='+str(minutes),'--job-name=lt-'+key,'--comment='+tag,'--exclude='+','.join(exclusions),str(SOURCE/'slurm/lr_transfer.sbatch'),run_id,str(path)]
    state['jobs'][key]={'run_id':run_id,'kind':kind,'multiplier':multiplier,'config':str(path),'status':'submission_uncertain','reservation':tag,'command':command}
    atomic_json(STATE,state)
    with ledger.transaction() as d:d['jobs'][tag].update(allocation_kind=POOL,transfer_study=STUDY,submission_tag=tag,submission_command=command,submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_root=str(SOURCE),controller_script=str(Path(__file__).resolve()))
    env={**os.environ,'DATA_ROOT':str(DEFAULT_ROOT),'TRAIN_SOURCE_ROOT':str(SOURCE),'TRANSFER_CONTROLLER':str(Path(__file__).resolve())}
    try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired) as error:
        state['jobs'][key]['submission_error']=str(error);atomic_json(STATE,state);return
    receipt={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};state['jobs'][key]['submission_receipt']=receipt
    job=r.stdout.strip().split(';')[0]
    if r.returncode==0 and job.isdigit():
        with ledger.transaction() as d:d['jobs'][tag]['returned_job_id']=job
        ledger.bind(tag,job);state['jobs'][key].update(job_id=job,status='submitted')
    elif r.returncode==1 and not r.stdout.strip() and 'QOSMaxSubmitJobPerUserLimit' in r.stderr and 'Batch job submission failed' in r.stderr:
        ledger.cancel_reservation(tag);state.setdefault('qos_rejections',[]).append(receipt);state['jobs'].pop(key)
    atomic_json(STATE,state)


def completed(state,keys):
    if not all(k in state['jobs'] and state['jobs'][k]['status']=='reconciled' for k in keys):return False
    failed=[k for k in keys if 'endpoint' not in state['jobs'][k]]
    if failed:raise ValueError('Incomplete/failed stage; no automatic retry: '+','.join(failed))
    signatures={state['jobs'][k]['endpoint']['initialization_signature'] for k in keys}
    identities={state['jobs'][k]['endpoint']['evaluation_identity'] for k in keys}
    if len(signatures)!=1 or len(identities)!=1:raise ValueError('Matched cohort random initialization/evaluation identity differs')
    return True


def fit_proxy(state):
    cohorts={}
    for index,multiplier in enumerate(MULTIPLIERS):
        r=state['jobs']['proxy'+str(index)];e=r['endpoint'];rows={}
        with (DEFAULT_ROOT/'runs/experiments'/r['run_id']/'metrics.jsonl').open() as f:
            for line in f:
                d=json.loads(line)
                if 'eval/ema_0.6_nll' not in d:continue
                if (d['eval/total_training_tokens_seen']!=d['train/global_step']*131072 or d['eval/ema_0.6_prediction_tokens']!=1479302 or not math.isfinite(d['eval/ema_0.6_nll'])):raise ValueError('Invalid matched EMA measurement')
                rows[d['train/global_step']]={'tokens':d['eval/total_training_tokens_seen'],'nll':d['eval/ema_0.6_nll']/d['eval/ema_0.6_prediction_tokens'],'evaluation_identity':e['evaluation_identity']}
        cohorts[multiplier]=rows
    result=analyze(cohorts);atomic_json(LIVE/'proxy-fit.json',result)
    return result


def publish(result,wsd=None):
    output=copy.deepcopy(result)
    if wsd is not None:output['single_wsd_check']=wsd
    output['validated']=result['proxy_validation_passed'] and wsd is not None and wsd['passed']
    for row in output['targets']:row['status']='validated_proxy_based_estimate' if output['validated'] else 'unvalidated_estimate'
    atomic_json(LIVE/'predictions.json',output)
    rows=['# 1–6B token LR transfer','',f"狀態：{'通過proxy與短程WSD檢查' if output['validated'] else '尚未通過所有驗證'}。固定模型與AdamW/Muon比例，不是二維全域最佳值。",'',
        '| Token horizon | Muon peak LR | AdamW peak LR | 狀態 |','|---|---:|---:|---|']
    rows += [f"| {r['tokens']/1e9:g}B | {r['muon_peak_lr']:.9g} | {r['adamw_peak_lr']:.9g} | {r['status']} |" for r in output['targets']]
    rows += ['',f"log LR = {result['final_fit']['beta']:.8g} log(tokens/1B) + {result['final_fit']['log_muon_lr_at_1b']:.8g}",'',
        '1B附近為proxy實測範圍；2–6B為外推。WSD檢查只在400M，不宣稱已逐一實測1–6B。單seed；EMA只用於proxy量測。',
        'fit-window range為擬合視窗敏感度，不是統計95%信賴區間。']
    (LIVE/'lr-table.md').write_text('\n'.join(rows)+'\n')


def report(state):
    pool,study=charges();out={'state':state,'shared_pool_actual_or_held':pool,'shared_pool_cap':256,'study_actual_or_held':study,'study_cap':MAX_STUDY_HOURS}
    atomic_json(LIVE/'results.json',out)
    lines=['# LR horizon transfer','',f"階段：{state['stage']}；study {study:.3f}/{MAX_STUDY_HOURS:.3f} H200-hours；共享pool {pool:.3f}/256。",'',
        '固定1.236B dense model、residual-half、raw、8192 context、batch16、seed42。只測固定AdamW/Muon比例的共同LR倍率。',
        '先15分鐘GPU正確性驗證（16篇），再4個15分鐘穩定性screen、4個94分鐘1B stable proxy、3個45分鐘400M完整單段WSD。舊驗證失敗成本保留。',
        'EMA beta0.6每32updates更新一次，只供proxy量測；單段WSD validation不使用EMA。','']
    for k,r in state['jobs'].items():
        lines.append(f"- {k}: job {r.get('job_id','uncertain')}，{r.get('slurm_state',r['status'])}"+(f"；[W&B]({r['endpoint']['wandb_url']})" if 'endpoint' in r else ''))
    if (LIVE/'lr-table.md').exists():lines+=['',(LIVE/'lr-table.md').read_text()]
    (LIVE/'report.md').write_text('\n'.join(lines)+'\n')


def run_once():
    if not owner():return
    with locked(LIVE/'.controller.lock',blocking=False):
        check();state=read_json(STATE) if STATE.exists() else {'stage':'validation','jobs':{}}
        try:
            reconcile(state)
            if state.get('blocked') or (LIVE/'disabled').exists():report(state);return
            if state['stage']=='validation':
                submit(state,'validation-r1','validation',1.)
                if not completed(state,['validation-r1']):report(state);return
                state['stage']='screen'
            if state['stage']=='screen':
                keys=['screen'+str(i) for i in range(4)]
                for key,m in zip(keys,MULTIPLIERS):submit(state,key,'screen',m)
                if not completed(state,keys):report(state);return
                state['stage']='proxy'
            if state['stage']=='proxy':
                keys=['proxy'+str(i) for i in range(4)]
                for key,m in zip(keys,MULTIPLIERS):submit(state,key,'proxy',m)
                if not completed(state,keys):report(state);return
                result=fit_proxy(state);publish(result)
                if not result['proxy_validation_passed']:raise ValueError('Held-out token LR extrapolation failed prespecified checks')
                multiplier=predict(result['final_fit'],configuration(read_json(LIVE/'base-config.json'),'decay')['target_training_tokens'])/ANCHOR_MUON
                state['decay_multipliers']=[multiplier/math.sqrt(2),multiplier,multiplier*math.sqrt(2)];state['stage']='decay'
                atomic_json(STATE,state)
            if state['stage']=='decay':
                keys=['decay'+str(i) for i in range(3)]
                for key,m in zip(keys,state['decay_multipliers']):submit(state,key,'decay',m)
                if not completed(state,keys):report(state);return
                points=[]
                for key in keys:
                    r=state['jobs'][key];ev=r['endpoint']['evaluation'];points.append((r['multiplier']*ANCHOR_MUON,ev['eval/nll']/ev['eval/prediction_tokens']))
                result=read_json(LIVE/'proxy-fit.json');predicted=state['decay_multipliers'][1]*ANCHOR_MUON
                try:
                    measured=parabola(points,min_points=3);relative=abs(measured['muon_lr']/predicted-1)
                    wsd={'passed':relative<=.25,'predicted_muon_lr':predicted,'measured_quadratic_optimum':measured,'relative_error':relative,'tokens':configuration(read_json(LIVE/'base-config.json'),'decay')['target_training_tokens']}
                except ValueError as e:wsd={'passed':False,'reason':str(e),'points':points}
                publish(result,wsd);state['stage']='complete' if wsd['passed'] else 'wsd_check_failed';state['blocked']=not wsd['passed']
            atomic_json(STATE,state);report(state)
        except Exception as error:
            state.update(blocked=True,error=str(error));atomic_json(STATE,state);report(state);raise


def install():
    check()
    if not owner():raise ValueError('Wrong submission owner')
    old=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if old.returncode not in (0,1):raise ValueError('Cannot read cron')
    rows=[x for x in old.stdout.splitlines() if not x.rstrip().endswith(MARKER)]
    log=DEFAULT_ROOT/'runs'/STUDY/'controller.log';log.parent.mkdir(parents=True,exist_ok=True)
    rows.append('* * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    wanted='\n'.join(rows)+'\n';subprocess.run(['crontab','-'],input=wanted,text=True,check=True)
    if subprocess.check_output(['crontab','-l'],text=True)!=wanted:raise ValueError('Cron changed unexpectedly')

if __name__=='__main__':
    action=sys.argv[1]
    if action=='check':check()
    elif action=='install':install()
    elif action=='run-once':run_once()
    elif action=='receipt':receipt(sys.argv[2],sys.argv[3])
    elif action=='status':print(json.dumps(read_json(STATE),indent=2))
    else:raise ValueError('Unknown action')
