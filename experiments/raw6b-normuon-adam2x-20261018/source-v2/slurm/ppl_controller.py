#!/usr/bin/env python3
"""Sole submission owner: two parallel mask runs, then two winner-mask optimizers."""
import datetime as dt,json,math,os,socket,subprocess,sys,uuid
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]
LIVE=SOURCE.parent
REPO=LIVE.parents[1]
sys.path.insert(0,str(SOURCE))
from training_state import (DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,
                            read_json,file_digest,digest,locked,load_config)
from ppl_contracts import STUDY,PREP,TARGET,Q,count_updates
from slurm.submission import find_submission_job
STATE=LIVE/'controller-state-r2.json'
SECONDS=7200

def check():
    seal=read_json(LIVE/'execution-seal-r2.json')
    for path,sha in seal['files'].items():
        if file_digest(path)!=sha:raise ValueError('Sealed source/config changed: '+path)
    baseline=read_json(LIVE/'ledger-before.json');ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    for name,sha in baseline['run_digests'].items():
        if ledger['runs'].get(name,{}).get('config_digest')!=sha:raise ValueError('Historical run identity lost: '+name)
    for job,cost in baseline['measured_costs'].items():
        if ledger['jobs'].get(job,{}).get('h200_hours')!=cost:raise ValueError('Historical allocation cost changed: '+job)
    prefix=read_json(LIVE/'raw-prefix.json')
    progress=read_json(DEFAULT_ROOT/'preprocessed'/PREP/'raw/progress.json')
    if progress['train_files'][:len(prefix['files'])]!=prefix['files']:raise ValueError('Canonical raw prefix changed')
    if sum(x['rows'] for x in prefix['files'])<TARGET//8192:raise ValueError('Insufficient published training data')

def config_path(stage,arm,mask=None):
    return LIVE/'configs'/(f's{stage}-{arm}'+(f'-{mask}' if stage==2 else '')+'.json')

def run_id(stage,arm):return f'{STUDY}-s{stage}-{arm}'

def preflight_check(run,path):
    cfg=load_config(path);gate=read_json(Path(os.environ['JOB_RECORD'])/'ppl-gate.json')
    if not gate['passed'] or gate['configuration']!=cfg or gate['seconds']>600:raise ValueError('Preflight failed')
    if run!=run_id(cfg['ppl']['stage'],cfg['ppl']['arm']):raise ValueError('Run/config identity mismatch')

def endpoint(run,path):
    cfg=load_config(path);directory=DEFAULT_ROOT/'runs/experiments'/run
    s=read_json(directory/'summary.json');ev=s['last_evaluation'];steps=count_updates(cfg)
    if (s['configuration']!=cfg or s['stop_reason']!='training_token_target' or
        s['total_tokens_seen']!=TARGET or s['budget_tokens_charged']!=TARGET or
        s['global_step']!=steps or s['attempted_updates']!=steps or s['cursor']!=TARGET//8192 or
        s['checkpoint_loaded_for_training'] or s['initialization']['pretrained_weights_loaded'] or
        s['export_global_step']!=steps or s['export_tokens_seen']!=TARGET or
        ev['eval/documents']!=1000 or ev['eval/prediction_tokens']!=1479302 or
        not math.isfinite(ev['eval/perplexity']) or s['mean_snapshot_count']!=5 or
        set(s['ema_evaluations'])!={'token_ema','finite_window_mean'} or
        set(s['ema_exports'])!={'token_ema','finite_window_mean'}):
        raise ValueError('Incomplete or mismatched 1B endpoint: '+run)
    if not read_json(directory/'export.json')['reload_verified']:raise ValueError('Export not verified')
    if cfg['ppl']['mask']=='intradoc' and s['attention_kernel_calls']['varlen']==0:raise ValueError('Intradoc kernel never used')
    return s

def receipt(run,path):
    check();preflight_check(run,path);s=endpoint(run,path)
    for rank in range(8):
        if read_json(Path(os.environ['JOB_RECORD'])/f'worker-rank-{rank}-exit.json')['exit_code']!=0:
            raise ValueError('Worker failed')
    atomic_json(LIVE/'receipts'/f'{run}.json',dict(run_id=run,job_id=os.environ['SLURM_JOB_ID'],
        config_path=str(path),config_digest=digest(s['configuration']),
        tokens=TARGET,raw_perplexity=s['last_evaluation']['eval/perplexity'],
        mean_nll=s['last_evaluation']['eval/nll']/s['last_evaluation']['eval/prediction_tokens'],
        evaluation=s['last_evaluation'],observer_evaluations=s['ema_evaluations'],
        initialization=s['initialization']['sample_signature'],provenance=s['data_provenance'],
        source_digest=s['code_digest'],environment_digest=s['environment_digest'],
        export=s['export'],wandb_url=s['wandb_url']))

def submit(state,stage,arm,mask=None):
    run=run_id(stage,arm)
    if run in state['jobs']:return
    ledger=ExperimentLedger(DEFAULT_ROOT)
    prior=[(k,v) for k,v in read_json(ledger.path)['jobs'].items() if v['run_id']==run]
    old_ids={'dense':{'559836','559859'},'intradoc':{'559837','559860'}}
    repair=(stage==1 and {k for k,v in prior}==old_ids[arm]
            and all(v.get('state')=='FAILED' for k,v in prior))
    if prior and not repair:
        if len(prior)!=1:raise ValueError('No retry/resume permitted')
        key,item=prior[0]
        state['jobs'][run]={**item,'reservation':key,'config_path':str(config_path(stage,arm,mask))}
        atomic_json(STATE,state);return
    path=config_path(stage,arm,mask);cfg=load_config(path)
    tag=run+'-'+uuid.uuid4().hex
    seconds=7080 if stage==1 else SECONDS
    ledger.reserve(tag,run,cfg,seconds,8,False,fresh_retry=repair)
    excludes=read_json(LIVE/'execution-plan.json')['excluded_nodes']
    command=['sbatch','--parsable','--time='+str(seconds//60),'--job-name='+f'ppl-s{stage}-{arm}-r2','--comment='+tag,
             '--exclude='+','.join(excludes),str(REPO/'slurm/ppl-next.sbatch'),run,str(path)]
    item=dict(status='submission_uncertain',reservation=tag,submission_tag=tag,
              submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
              config_path=str(path),submission_command=command)
    state['jobs'][run]=item;atomic_json(STATE,state)
    with ledger.transaction() as data:
        data['jobs'][tag].update(item,study=STUDY,source_root=str(SOURCE),
            ppl_metadata_only_repair=repair,
            authorization='User-requested two-stage campaign; repair missing vendored import before model initialization, preserving logical run/config/all failed costs;118min stage1 or120min stage2,total<=16H/run; no checkpoint load; no generic retries')
    env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),'PPL_CONTROLLER':str(Path(__file__).resolve())}
    try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired) as e:
        item['submission_error']=str(e);atomic_json(STATE,state);return
    item['submission_receipt']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    job=r.stdout.strip().split(';')[0]
    if r.returncode==0 and job.isdigit():
        with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=job
        item['returned_job_id']=job;ledger.bind(tag,job);item.update(job_id=job,status='submitted')
    atomic_json(STATE,state)

def reconcile(state):
    ledger=ExperimentLedger(DEFAULT_ROOT)
    for run,item in state['jobs'].items():
        if item.get('status') in ('complete','failed'):continue
        if not item.get('job_id'):
            job=find_submission_job(item,DEFAULT_ROOT)
            current=read_json(ledger.path)['jobs']
            if job not in current:ledger.bind(item['reservation'],job)
            item['job_id']=job;item['status']='submitted';atomic_json(STATE,state)
        job=item['job_id']
        query=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State'],capture_output=True,text=True,timeout=30,check=True)
        rows=[x.split('|') for x in query.stdout.splitlines() if x.startswith(job+'|')]
        if not rows:continue
        status=rows[0][1].split()[0].rstrip('+');item['slurm_state']=status
        if status not in TERMINAL:continue
        measured=ledger.reconcile(job);item['accounting']=measured
        item['status']='failed'
        if status=='COMPLETED' and (LIVE/'receipts'/f'{run}.json').exists():
            endpoint(run,item['config_path']);item['status']='complete'
        atomic_json(STATE,state)

def selection(state):
    runs=[run_id(1,x) for x in ('dense','intradoc')]
    if not all(state['jobs'].get(r,{}).get('status')=='complete' for r in runs):return None
    records=[read_json(LIVE/'receipts'/f'{r}.json') for r in runs]
    for field in ('initialization','provenance','source_digest','environment_digest'):
        if records[0][field]!=records[1][field]:raise ValueError('Unmatched stage1 '+field)
    # Strict lower endpoint NLL/PPL wins; exact ties deterministically keep dense.
    index=0 if records[0]['mean_nll']<=records[1]['mean_nll'] else 1
    decision=dict(mask=('dense','intradoc')[index],control_run=runs[index],
                  criterion='lower raw final mean NLL on fixed 1000 docs at exact 1B inputs; exact tie dense',
                  mean_nll={x['run_id']:x['mean_nll'] for x in records},
                  ppl={x['run_id']:x['raw_perplexity'] for x in records},
                  practically_small_gap=abs(records[0]['mean_nll']-records[1]['mean_nll'])<.005,
                  checkpoint_training_load=False)
    path=LIVE/'stage1-selection.json'
    if path.exists() and read_json(path)!=decision:raise ValueError('Locked stage1 selection changed')
    atomic_json(path,decision);return decision

def tick():
    with locked(LIVE/'.controller.lock',blocking=False):
        state=read_json(STATE) if STATE.exists() else dict(jobs={},status='ready')
        reconcile(state)
        if (LIVE/'disabled').exists():
            state['status']='disabled';atomic_json(STATE,state);return state
        check()
        if not read_json(LIVE/'authorization.json')['approved']:raise ValueError('No campaign authorization')
        if any(x.get('status')=='failed' for x in state['jobs'].values()):
            state['status']='stopped_on_failure';atomic_json(STATE,state);return state
        for arm in ('dense','intradoc'):submit(state,1,arm)
        chosen=selection(state)
        if chosen:
            for arm in ('pe','nor'):submit(state,2,arm,chosen['mask'])
        complete=len(state['jobs'])==4 and all(x.get('status')=='complete' for x in state['jobs'].values())
        state['status']='complete' if complete else ('stage2' if chosen else 'stage1')
        if complete:
            baseline=read_json(LIVE/'receipts'/f"{chosen['control_run']}.json")
            results={}
            all_jobs=read_json(DEFAULT_ROOT/'experiment-ledger.json')['jobs'].values()
            for run,item in state['jobs'].items():
                rec=read_json(LIVE/'receipts'/f'{run}.json')
                for field in ('initialization','provenance','source_digest','environment_digest'):
                    if rec[field]!=baseline[field]:raise ValueError('Unmatched comparison '+field)
                results[run]={**rec,'h200_hours':sum(j['h200_hours'] for j in all_jobs if j['run_id']==run),
                              'delta_nll_vs_selected_ns':rec['mean_nll']-baseline['mean_nll']}
            atomic_json(LIVE/'results.json',dict(selected_mask=chosen['mask'],runs=results,
                        actual_h200_hours=sum(x['h200_hours'] for x in results.values()),
                        single_seed=True,adaptive_mask_selection=True))
        atomic_json(STATE,state);return state

def main():
    command=sys.argv[1]
    if command=='check':check()
    elif command=='receipt':receipt(sys.argv[2],sys.argv[3])
    elif command=='preflight-check':preflight_check(sys.argv[2],sys.argv[3])
    elif command=='run-once':
        state=tick();print(json.dumps(dict(status=state['status'],jobs={k:{f:v.get(f) for f in ('job_id','status','slurm_state')} for k,v in state['jobs'].items()})))
    elif command=='status':print(json.dumps(read_json(STATE),indent=2))
    else:raise ValueError('Unknown command')

if __name__=='__main__':main()
