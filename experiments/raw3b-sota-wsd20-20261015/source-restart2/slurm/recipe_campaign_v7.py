#!/usr/bin/env python3
"""Single-owner, budgeted parallel sweep controller; no tensor imports on login."""
import argparse
import copy
import csv
import datetime as dt
import json
import getpass
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import uuid

SOURCE=Path(__file__).resolve().parents[1]
LIVE=SOURCE.parent
REPO=SOURCE.parents[2]
NAME='init-recipe-20261014'
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,digest,file_digest,load_config,locked,read_json
from recipe_runtime import INITIALIZERS
STATE=LIVE/'controller-state.json'
PLAN=LIVE/'plan.json'
OWNER=LIVE/'submission-owner.json'
MARKER='# '+NAME+'-controller'
PARENT='horizon-wsd-20261014-final-resplit'
SHORT_FAMILIES={
 'clip':[('clip-late',{'clipping':'late_one'}),('clip-groups',{'clipping':'per_optimizer'})],
 'wd':[('wd003',{'muon.weight_decay':.03}),('wd005',{'muon.weight_decay':.05})],
 'batch':[('batch8',{'gradient_accumulation_steps':1}),('batch32',{'gradient_accumulation_steps':4})],
 'data':[('filtered',{'preparation_id':'dolma-dclm10-005'})]}
EARLY={
 'init':[(i,{'initialization':i}) for i in INITIALIZERS],
 'adam':[('adam05',{'learning_rate':8.961576616821191e-5*.5}),('adam20',{'learning_rate':8.961576616821191e-5*2})],
 'muon':[('muon07',{'muon.learning_rate':.005974384411214128*.7}),('muon13',{'muon.learning_rate':.005974384411214128*1.3})]}


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()
def plan():return read_json(PLAN)
def validation_key():
    key=plan().get('validation_key','validation')
    if key not in ('validation','validation-r1'):raise ValueError('Unauthorized validation replacement')
    return key
def state():return read_json(STATE) if STATE.exists() else {'campaign':NAME,'stage':'validation','jobs':{},'aliases':{},'selections':{},'created_utc':stamp()}
def owner():
    p=read_json(OWNER)
    return p['script']==str(Path(__file__).resolve()) and p['sha256']==file_digest(__file__) and p['hostname']==socket.gethostname().split('.')[0]
def check():
    p=plan()
    roots=[SOURCE]
    active=os.environ.get('TRAIN_SOURCE_ROOT')
    if active and Path(active)!=SOURCE:roots.append(Path(active))
    for root in roots:
        for f,h in p['source_hashes'].items():
            if file_digest(root/f)!=h:raise ValueError('Frozen source changed: '+str(root/f))
    for f,h in p['metadata_hashes'].items():
        if file_digest(f)!=h:raise ValueError('Pinned metadata changed: '+f)
    if p['budget_h200_hours']!=256 or p['final_reserve_h200_hours']!=32:raise ValueError('Budget changed')

def patch_cfg(cfg,changes):
    for k,v in changes.items():
        if k=='preparation_id':continue
        if '.' in k:
            a,b=k.split('.',1);cfg[a][b]=copy.deepcopy(v)
        else:cfg[k]=copy.deepcopy(v)
    return cfg

def configuration(changes,tokens,stage='short',seed=42):
    cfg=patch_cfg(copy.deepcopy(plan()['base_configuration']),changes)
    cal=DEFAULT_ROOT/'runs/experiments'/(NAME+'-'+validation_key())/'calibration.json'
    if stage!='validation' and cal.exists():cfg['sweep_calibration_digest']=file_digest(cal)
    width=cfg['world_size']*cfg['microbatch_size']*cfg['gradient_accumulation_steps']
    batch=width*8192
    steps=tokens//batch
    if tokens%batch:raise ValueError('Endpoint is not a complete batch')
    cfg.update(sweep_id=NAME,sweep_purpose='Independent initialization/recipe sweeps, combined verification, one final2B',
        sweep_stage=stage,parent_runs=[],category='full' if stage=='final' else ('validation' if stage=='validation' else 'pilot'),
        seed=seed,max_steps=steps,scheduler_steps=steps,max_training_tokens=tokens,target_training_tokens=tokens,
        max_h200_hours=32 if stage=='final' else 16,warmup_steps=22282240//batch,
        eval_steps=max(1,steps//10),export_on_completion=stage=='final',retain_all_checkpoints=True,
        final_evaluation_reserve_seconds=600 if stage in ('short','final') else 120,
        stop_margin_seconds=180,initialization=cfg.get('initialization','gaussian'),clipping=cfg.get('clipping','global'),ema=cfg.get('ema',[]))
    if 'phase_schedule' in cfg:
        cut=round(steps*12715/15259)
        scale=cfg['phase_schedule'][-1]['peak_ratio']
        cfg['phase_schedule']=[dict(start=0,end=cut,peak_ratio=1.,warmup_steps=cfg['warmup_steps'],hold_schedule=cfg['hold_schedule']),
            dict(start=cut,end=steps,peak_ratio=scale,warmup_steps=min(cfg['warmup_steps'],max(1,int((steps-cut)*.1))),hold_schedule=cfg['hold_schedule'])]
    return cfg

def charges():
    with ExperimentLedger(DEFAULT_ROOT).transaction() as data:
        return sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j.get('allocation_kind')==NAME or j['run_id'].startswith(NAME+'-'))

def submission_slot():
    limits=subprocess.run(['sacctmgr','-n','-P','show','qos','p_8gpus','format=Name,MaxSubmitPU'],capture_output=True,text=True,timeout=30,check=True)
    fields=limits.stdout.strip().split('|')
    if len(fields)<2 or fields[0]!='p_8gpus' or not fields[1].isdigit():raise ValueError('Cannot verify partition QoS submission limit')
    queued=subprocess.run(['squeue','-u',getpass.getuser(),'-p','8gpus','-h','-o','%i'],capture_output=True,text=True,timeout=30,check=True)
    return len([x for x in queued.stdout.splitlines() if x.strip()])<int(fields[1])

def known_qos_rejection(result):
    return result.returncode==1 and not result.stdout.strip() and 'sbatch: error: QOSMaxSubmitJobPerUserLimit' in result.stderr and 'Batch job submission failed' in result.stderr

def submit(s,key,changes,tokens,minutes,stage='short',seed=42,tail=False):
    if key in s['jobs'] or key in s['aliases']:return
    check()
    if not owner():raise ValueError('Only the registered login host may submit')
    if not submission_slot():return  # Keep this candidate in the controller backlog.
    run_id=NAME+'-'+plan().get('run_name_overrides',{}).get(key,key)
    prep=changes.get('preparation_id','dolma-raw-baseline-20261008')
    if tail:
        cfg=read_json(REPO/'configs/horizon-wsd-20261014/tail-lr030.json')
        cfg['sweep_id']=NAME;cfg['phase_schedule'][1]['peak_ratio']=changes['scale']
        cfg['sweep_purpose']='Three authorized lower-LR forks from frozen phase1 checkpoint'
        cfg['parent_runs']=[PARENT]
    else:cfg=configuration(changes,tokens,stage,seed)
    path=LIVE/'configs'/(key+'.json')
    if path.exists() and read_json(path)!=cfg:raise ValueError('Candidate config changed')
    atomic_json(path,cfg)
    if not tail:load_config(path)
    if (DEFAULT_ROOT/'runs/experiments'/run_id).exists():raise ValueError('Existing run; no retry')
    cost=minutes*8/60
    if charges()+cost+(0 if stage=='final' else 32)>256+1e-9:raise ValueError('Campaign budget/final reserve exhausted')
    ledger=ExperimentLedger(DEFAULT_ROOT)
    tag=NAME+'-'+uuid.uuid4().hex
    ledger.reserve(tag,run_id,cfg,minutes*60,8,False)
    if tail:
        with ledger.transaction() as data:
            parent=data['runs'][PARENT]
            if parent.get('budget_exceeded') or any('h200_hours' not in j for j in data['jobs'].values() if j['run_id']==PARENT):raise ValueError('Unreconciled parent')
            inherited=parent['h200_hours_measured']
            tree=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j['run_id']==PARENT or j['run_id'].startswith('horizon-wsd-20261014-tail-lr') or data['runs'].get(j['run_id'],{}).get('parent_run')==PARENT)
            if inherited+cost>32 or tree+cost>64:raise ValueError('Resplit lineage cap exceeded')
            data['runs'][run_id].update(inherited_h200_hours=inherited,parent_run=PARENT)
            forks=sum(v.get('parent_run')==PARENT for v in data['runs'].values())
            parent_tokens=read_json(DEFAULT_ROOT/'runs/experiments'/PARENT/'token-journal.json')['charged_tokens']
            if parent_tokens+forks*333447168>6_000_000_000:raise ValueError('Resplit tree processed-input cap exceeded')
    branch='integration'
    if key in INITIALIZERS or key.removeprefix('promote-') in INITIALIZERS:branch='initialization'
    elif key.startswith(('adam','muon','promote-adam','promote-muon','wd','clip')):branch='optimizer'
    elif key=='filtered':branch='data'
    elif key.startswith(('batch','tail-')) or key=='short-baseline':branch='tail'
    job_source=SOURCE if branch=='integration' else LIVE/'worktrees'/branch
    failures=s.get('failed_allocations',{}).get(key,[])
    inherited_failed=sum(x['accounting']['h200_hours'] for x in failures)
    if inherited_failed:
        with ledger.transaction() as data:
            current=data['runs'][run_id].get('inherited_h200_hours',0)
            if current+inherited_failed+cost>cfg['max_h200_hours']:raise ValueError('Hardware-repair logical cap exceeded')
            data['runs'][run_id]['inherited_h200_hours']=current+inherited_failed
            data['runs'][run_id]['failed_allocations_retained']=[x['job_id'] for x in failures]
    command=['sbatch','--parsable',f'--time={minutes}','--comment='+tag,'--job-name=ir-'+key,
             '--exclude='+','.join(plan().get('excluded_nodes',['25a-hgpn001','25a-hgpn027','25a-hgpn062','25a-hgpn151'])),
             str(SOURCE/'slurm/recipe.sbatch'),run_id,str(path),prep,'tail' if tail else ('validation' if stage=='validation' else 'train'),'1' if tail or stage=='final' else '0']
    record={'run_id':run_id,'reservation':tag,'status':'submission_uncertain','configuration':str(path),'config_sha256':file_digest(path),
            'changes':changes,'stage':stage,'seed':seed,'tail':tail,'minutes':minutes,'branch':'campaign/'+NAME+'/'+branch,'source_root':str(job_source),'command':command,'started_utc':stamp()}
    s['jobs'][key]=record;atomic_json(STATE,s)
    with ledger.transaction() as data:
        data['jobs'][tag].update(submission_tag=tag,submission_command=command,submission_started_utc=stamp(),allocation_kind=NAME,source_root=str(job_source),
            authorization='User: Implement the approved256H parallel recipe campaign and one fresh final2B',controller_script=str(Path(__file__).resolve()),controller_host=read_json(OWNER)['hostname'])
    env={**os.environ,'DATA_ROOT':str(DEFAULT_ROOT),'TRAIN_SOURCE_ROOT':str(job_source),'RECIPE_CONTROLLER':str(Path(__file__).resolve()),'RECIPE_LIVE':str(LIVE),'RECIPE_REPO':str(REPO)}
    try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired) as e:
        record['submission_error']=str(e);atomic_json(STATE,s);return
    record['submission_receipt']={'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode}
    if r.returncode==0 and r.stdout.strip().split(';')[0].isdigit():
        job=r.stdout.strip().split(';')[0]
        with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=job
        ledger.bind(tag,job);record.update(job_id=job,status='submitted')
    elif known_qos_rejection(r):
        # Slurm explicitly refused before creating a job; no model/run exists.
        ledger.cancel_reservation(tag)
        record['status']='submission_rejected_before_job_creation'
        s.setdefault('submission_rejections',[]).append(record)
        del s['jobs'][key]
    # Other nonzero results can be ambiguous. Retain reservation, never retry.
    atomic_json(STATE,s)

def reconcile(s):
    from slurm.submission import find_submission_job
    ledger=ExperimentLedger(DEFAULT_ROOT)
    for key,r in s['jobs'].items():
        if r['status']=='reconciled':continue
        if 'job_id' not in r:
            with ledger.transaction() as data:
                matches=[v for k,v in data['jobs'].items() if k==r['reservation'] or v.get('submission_tag')==r['reservation']]
                if len(matches)!=1:raise ValueError('Ambiguous reservation binding')
                reservation=copy.deepcopy(matches[0])
            job=find_submission_job(reservation,DEFAULT_ROOT)
            if not job:continue
            if not reservation.get('job_id'):ledger.bind(r['reservation'],job)
            r.update(job_id=job,status='submitted')
        out=subprocess.run(['sacct','-X','-j',r['job_id'],'--noheader','--parsable2','--format=JobIDRaw,State'],capture_output=True,text=True,timeout=30,check=True)
        statuses=[line.split('|')[1].split()[0].rstrip('+') for line in out.stdout.splitlines() if line.split('|')[0]==r['job_id']]
        r['slurm_state']=statuses[0] if statuses else 'UNKNOWN'
        if r['slurm_state'] in TERMINAL:
            r['accounting']=ledger.reconcile(r['job_id']);r['status']='reconciled'
            try:
                receipt=read_json(LIVE/'receipts'/(r['job_id']+'.json'))
                if receipt['run_id']!=r['run_id'] or receipt['config_sha256']!=r['config_sha256']:raise ValueError('Endpoint receipt identity mismatch')
                r['endpoint']=receipt
            except Exception as e:r['verification_error']=str(e)

def record(s,key):return s['jobs'][s['aliases'].get(key,key)]
def done(s,keys):return all((k in s['jobs'] or k in s['aliases']) and record(s,k)['status']=='reconciled' for k in keys)
def valid(s,k):
    r=record(s,k)
    return r.get('slurm_state')=='COMPLETED' and 'endpoint' in r and 'verification_error' not in r

def score(s,k,variant='raw'):
    e=record(s,k)['endpoint']
    result=e['evaluation'] if variant=='raw' else e['ema_evaluations'][variant]
    return result['eval/nll']/result['eval/prediction_tokens']
def best(s,keys):
    eligible=[k for k in keys if valid(s,k)]
    if not eligible:raise ValueError('No complete, stable candidates: '+str(keys))
    return min(eligible,key=lambda k:score(s,k))

def receipt(run_id,job_id):
    check();r=DEFAULT_ROOT/'runs/experiments'/run_id
    cfg_path=Path(sys.argv[3])
    if run_id in (NAME+'-validation',NAME+'-validation-r1'):
        data=read_json(r/'calibration.json')
        if not data['passed']:raise ValueError('Calibration failed')
        output={'run_id':run_id,'job_id':job_id,'config_sha256':file_digest(cfg_path),'calibration':data}
    else:
        s=read_json(r/'summary.json');cfg=read_json(cfg_path)
        if (s['global_step']!=cfg['max_steps'] or s['attempted_updates']!=cfg['max_steps'] or s['last_eval_step']!=cfg['max_steps'] or
            s['total_tokens_seen']!=cfg['max_training_tokens'] or s['budget_tokens_charged']!=s['total_tokens_seen'] or
            s['cursor']!=cfg['max_steps']*cfg['world_size']*cfg['microbatch_size']*cfg['gradient_accumulation_steps'] or s['stop_reason']!='training_token_target'):
            raise ValueError('Incomplete endpoint: no promotion')
        if s['initialization']['pretrained_weights_loaded'] or s['last_evaluation']['eval/documents']!=1000 or not math.isfinite(s['last_evaluation']['eval/perplexity']):raise ValueError('Invalid evaluation/initialization')
        if bool(s['checkpoint_loaded_for_training']) != ('tail-lr' in run_id):raise ValueError('Unexpected checkpoint load')
        complete=Path(s['checkpoint'])/'complete.json'
        if not complete.exists():raise ValueError('Incomplete checkpoint')
        workers=[read_json(DEFAULT_ROOT/'logs'/('job-'+job_id)/f'worker-rank-{rank}-exit.json')['exit_code'] for rank in range(8)]
        if any(workers):raise ValueError('Worker checkpointed stop or failure')
        if cfg['export_on_completion']:
            full=read_json(r/'full-evaluation.json')
            if full['eval/documents']!=50000 or full['global_step']!=cfg['max_steps']:raise ValueError('Missing full reserved evaluation')
        output={'run_id':run_id,'job_id':job_id,'config_sha256':file_digest(cfg_path),'step':s['global_step'],'cursor':s['cursor'],'tokens':s['total_tokens_seen'],
            'checkpoint':s['checkpoint'],'checkpoint_complete_sha256':file_digest(complete),'evaluation':s['last_evaluation'],'ema_evaluations':s.get('ema_evaluations',{}),
            'evaluation_identity':{**{k:s['data_provenance'][k] for k in ('dataset_revision','evaluation_indices_sha256')},'evaluation_loss':s['data_provenance']['pack_settings']['evaluation_loss'],'pad_token_id':s['data_provenance']['pack_settings']['pad_token_id']},
            'wandb_url':s['wandb_url'],'full_evaluation':read_json(r/'full-evaluation.json') if (r/'full-evaluation.json').exists() else None,'worker_exit_codes':workers}
    path=LIVE/'receipts'/(job_id+'.json')
    if path.exists() and read_json(path)!=output:raise ValueError('Immutable receipt changed')
    atomic_json(path,output)

def promote(s,h):
    for family,items in EARLY.items():
        keys=[k for k,_ in items]
        if family!='init':keys=['gaussian']+keys
        if not done(s,keys):continue
        selection_key='early-'+family
        if selection_key not in s['selections']:
            eligible=[k for k in keys if valid(s,k)]
            s['selections'][selection_key]=sorted(eligible,key=lambda k:score(s,k))[:2]
        for k in s['selections'][selection_key]:
            pk='promote-'+k
            if k=='gaussian':s['aliases'][pk]='short-baseline'
            else:submit(s,pk,record(s,k)['changes'],h,60)

def ranked_short(s):
    result={'init':['short-baseline'],'adam':['short-baseline'],'muon':['short-baseline']}
    for f in result:
        result[f]+=['promote-'+k for k in s['selections'].get('early-'+f,[]) if k!='gaussian']
    for f,items in SHORT_FAMILIES.items():result[f]=['short-baseline']+[k for k,_ in items]
    return result

def validate_comparability(s,keys):
    identities=[record(s,k)['endpoint']['evaluation_identity'] for k in keys if valid(s,k)]
    if any(i!=identities[0] for i in identities[1:]):raise ValueError('Evaluation identities differ')
    counts=[record(s,k)['endpoint']['evaluation']['eval/prediction_tokens'] for k in keys if valid(s,k)]
    if len(set(counts))>1:raise ValueError('Evaluation masks/token counts differ')

def report(s):
    rows=[]
    for k,r in s['jobs'].items():
        e=r.get('endpoint',{})
        if 'evaluation' not in e:continue
        for variant,ev in [('raw',e['evaluation'])]+list(e.get('ema_evaluations',{}).items()):
            nll=ev['eval/nll']/ev['eval/prediction_tokens']
            rows.append({'candidate':k,'variant':variant,'stage':r['stage'],'seed':r['seed'],'tokens':e['tokens'],'nll':nll,'perplexity':math.exp(nll),
                'h200_hours':r.get('accounting',{}).get('h200_hours'),'status':r.get('slurm_state'),'wandb':e['wandb_url'],'checkpoint':e['checkpoint'],'config_sha256':r['config_sha256'],
                'full_perplexity':(e.get('full_evaluation') or {}).get('all_variants',{}).get(variant,{}).get('eval/perplexity'),
                'configuration':r['configuration'],'recipe':json.dumps(r['changes'],sort_keys=True)})
    atomic_json(LIVE/'results/ranking.json',{'rows':rows,'selections':s['selections'],'cost':charges(),'budget':256,'final_reserve':32 if s['stage']!='complete' else 0,'source_commit':plan()['source_commit'],'failed_allocations':s.get('failed_allocations',{})})
    path=LIVE/'results/ranking.tsv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    lines=['# 初始化與配方 sweep','',f"狀態：{s['stage']}；已使用或保留 {charges():.3f} / 256 H200-hours。",'',f"Source commit：{plan()['source_commit']}",'','| 候選 | variant | tokens | subset PPL | full50k PPL | H200-hours | W&B |','|---|---|---:|---:|---:|---:|---|']
    lines += [f"| {r['candidate']} | {r['variant']} | {r['tokens']} | {r['perplexity']:.5f} | {r['full_perplexity'] if r['full_perplexity'] is not None else '—'} | {r['h200_hours']} | [run]({r['wandb']}) |" for r in rows]
    lines += ['','## 分組與組合選擇','',json.dumps(s['selections'],ensure_ascii=False,indent=2)]
    (LIVE/'results/report.md').write_text('\n'.join(lines)+'\n')

def run_once():
    if not owner():print('Different owner host or sealed script; status only.');return
    with locked(LIVE/'.controller.lock',blocking=False):
        check();s=state();reconcile(s)
        atomic_json(STATE,s)
        if (LIVE/'disabled').exists():report(s);return
        vk=validation_key()
        if vk not in s['jobs']:submit(s,vk,{},131072*1000,20,'validation')
        if not done(s,[vk]):report(s);return
        if not valid(s,vk):s['stage']='validation_failed';atomic_json(STATE,s);report(s);return
        cal=record(s,vk)['endpoint']['calibration'];h=cal['short_tokens'];s['short_tokens']=h
        s['stage']='parallel_sweeps'
        for items in EARLY.values():
            for k,c in items:submit(s,k,c,cal['initial_tokens'],20,'initial')
        submit(s,'short-baseline',{'ema':[.99,.999,.9995]},h,60)
        for items in SHORT_FAMILIES.values():
            for k,c in items:submit(s,k,c,h,60)
        for scale in (.05,.1,.2):submit(s,'tail-lr'+str(int(scale*100)).zfill(3),{'scale':scale},2000027648,55,tail=True)
        promote(s,h)
        groups=ranked_short(s)
        early_done=all('early-'+f in s['selections'] for f in EARLY)
        keys=list(dict.fromkeys(k for ks in groups.values() for k in ks))
        tails=['tail-lr005','tail-lr010','tail-lr020']
        if not early_done or not done(s,keys+tails):atomic_json(STATE,s);report(s);return
        if not valid(s,'short-baseline'):s['stage']='baseline_failed';atomic_json(STATE,s);report(s);return
        validate_comparability(s,keys)
        if 'combined_changes' not in s:
            changes={}
            for f,ks in groups.items():
                winner=best(s,ks);s['selections']['short-'+f]=winner
                if f=='batch' and record(s,winner)['changes'].get('gradient_accumulation_steps',2)==1 and not cal['batch8_final_feasible']:
                    winner=best(s,[k for k in ks if record(s,k)['changes'].get('gradient_accumulation_steps',2)!=1])
                    s['selections']['batch_final_feasible']=winner
                if winner!='short-baseline':changes.update(record(s,winner)['changes'])
            ema=record(s,'short-baseline')['endpoint']['ema_evaluations']
            beta=min(ema,key=lambda b:score(s,'short-baseline',b)) if ema else None
            s['selections']['ema']=beta if beta and score(s,'short-baseline',beta)<score(s,'short-baseline') else 'raw'
            if s['selections']['ema']!='raw':changes['ema']=[float(beta)]
            full=[(k,record(s,k)['endpoint']['full_evaluation']['eval/perplexity']) for k in tails if valid(s,k)]
            full+=[('historical-lr030',17.699298979273298)]
            tw=min(full,key=lambda x:x[1]);s['selections']['tail']=dict(candidate=tw[0],full_perplexity=tw[1],historical_standard=17.2127001491)
            if tw[1]<17.2127001491:
                scale=record(s,tw[0])['changes']['scale'];changes['phase_schedule']=[{},dict(peak_ratio=scale)]
            s['combined_changes']=changes
        s['stage']='combination'
        submit(s,'combo-baseline',{},h,60)
        submit(s,'combo-merged',s['combined_changes'],h,60)
        if not done(s,['combo-baseline','combo-merged']):atomic_json(STATE,s);report(s);return
        if not valid(s,'combo-baseline'):
            s['stage']='combination_control_failed';atomic_json(STATE,s);report(s);return
        eligible=[k for k in keys+['combo-baseline','combo-merged'] if valid(s,k)]
        if not valid(s,'combo-merged') or score(s,'combo-merged')>=score(s,'combo-baseline'):
            a=copy.deepcopy(s['combined_changes']);a['initialization']='gaussian'
            b={k:v for k,v in s['combined_changes'].items() if k not in ('learning_rate','muon.learning_rate','muon.weight_decay','clipping')}
            submit(s,'combo-baseline-init',a,h,60);submit(s,'combo-baseline-optimizer',b,h,60)
            if not done(s,['combo-baseline-init','combo-baseline-optimizer']):atomic_json(STATE,s);report(s);return
            eligible += [k for k in ['combo-baseline-init','combo-baseline-optimizer'] if valid(s,k)]
        validate_comparability(s,eligible)
        eligible=[k for k in eligible if record(s,k)['changes'].get('gradient_accumulation_steps',2)!=1 or cal['batch8_final_feasible']]
        if 'candidate' not in s['selections']:
            s['selections']['candidate']=min(eligible,key=lambda k:min([score(s,k)]+[score(s,k,b) for b in record(s,k)['endpoint']['ema_evaluations']]))
            k=s['selections']['candidate'];s['candidate_changes']=copy.deepcopy(record(s,k)['changes'])
            ema=record(s,k)['endpoint']['ema_evaluations']
            beta=min(ema,key=lambda b:score(s,k,b)) if ema else None
            s['candidate_variant']=beta if beta and score(s,k,beta)<score(s,k) else 'raw'
            s['candidate_changes']['ema']=[float(beta)] if s['candidate_variant']!='raw' else []
        s['stage']='paired_seed'
        submit(s,'seed43-baseline',{},h,60,seed=43);submit(s,'seed43-candidate',s['candidate_changes'],h,60,seed=43)
        if not done(s,['seed43-baseline','seed43-candidate']):atomic_json(STATE,s);report(s);return
        win=s['selections']['candidate'];variant=s['candidate_variant']
        delta=score(s,win,variant)-score(s,'combo-baseline')
        if valid(s,'seed43-candidate') and valid(s,'seed43-baseline'):
            delta += score(s,'seed43-candidate',variant)-score(s,'seed43-baseline')
        else:delta=float('inf')
        final=s['candidate_changes'] if delta<0 else {}
        if final.get('gradient_accumulation_steps',2)==1 and not cal['batch8_final_feasible']:final={};s['selections']['batch8_final_fallback']='baseline: four-hour feasibility'
        s['selections']['final_changes']=final;s['stage']='final'
        batch=8*final.get('gradient_accumulation_steps',2)*8192
        tokens=math.ceil(2_000_000_000/batch)*batch
        submit(s,'final-2b',final,tokens,240,'final')
        if done(s,['final-2b']):s['stage']='complete' if valid(s,'final-2b') else 'final_failed'
        s['updated_utc']=stamp();atomic_json(STATE,s);report(s)

def install():
    check()
    if not owner():raise ValueError('Wrong owner')
    r=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if r.returncode not in (0,1):raise ValueError('Cannot read crontab')
    lines=[x for x in r.stdout.splitlines() if not x.rstrip().endswith(MARKER)]
    log=DEFAULT_ROOT/'runs'/NAME/'controller.log';log.parent.mkdir(parents=True,exist_ok=True)
    lines.append('* * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    wanted='\n'.join(lines)+'\n';subprocess.run(['crontab','-'],input=wanted,text=True,check=True)
    if subprocess.check_output(['crontab','-l'],text=True)!=wanted:raise ValueError('Cron verification failed')

if __name__=='__main__':
    action=sys.argv[1]
    if action=='receipt':receipt(sys.argv[2],os.environ['SLURM_JOB_ID'])
    elif action=='check':check()
    elif action=='install':install()
    elif action=='run-once':run_once()
    elif action=='status':print(json.dumps(state(),indent=2))
    elif action=='disable':(LIVE/'disabled').touch()
    else:raise ValueError('Unknown action')
