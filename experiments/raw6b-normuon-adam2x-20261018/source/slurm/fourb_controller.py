#!/usr/bin/env python3
"""Single user-approved formal17 allocation; no retry, resume, or promotion."""
import datetime as dt,json,math,os,socket,subprocess,sys,uuid
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]
LIVE=SOURCE.parent
REPO=LIVE.parents[1]
sys.path.insert(0,str(SOURCE))
from training_state import (DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,
                            read_json,file_digest,digest,locked,load_config)
from ppl_contracts import STUDY,PREP,TARGET,Q,count_updates
from fourb_authorization import authorized
from slurm.submission import find_submission_job
STATE=LIVE/'controller-state.json'
CONFIG=LIVE/'config.json'
SECONDS=36000

def check():
    seal=read_json(LIVE/'source-seal.json')
    for path,sha in seal['files'].items():
        if file_digest(path)!=sha:raise ValueError('Sealed source/config changed: '+path)
    cfg=load_config(CONFIG)
    if not authorized(STUDY,cfg):raise ValueError('Exact6B authorization mismatch')
    baseline=read_json(LIVE/'ledger-before.json');ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    for name,sha in baseline['run_digests'].items():
        if ledger['runs'].get(name,{}).get('config_digest')!=sha:raise ValueError('Historical run identity lost: '+name)
    for job,cost in baseline['measured_costs'].items():
        if ledger['jobs'].get(job,{}).get('h200_hours')!=cost:raise ValueError('Historical cost changed: '+job)
    formal={name for name,v in ledger['runs'].items() if v['category']=='full'}
    if formal-{STUDY}!=set(baseline['formal_runs']):raise ValueError('Unexpected formal-run identities')
    prefix=read_json(LIVE/'raw-prefix.json')
    progress=read_json(DEFAULT_ROOT/'preprocessed'/PREP/'raw/progress.json')
    if progress['train_files'][:len(prefix['files'])]!=prefix['files']:raise ValueError('Canonical raw prefix changed')
    if sum(x['rows'] for x in prefix['files'])!=732421:raise ValueError('Unexpected base prefix; tail extension required')
    if count_updates(cfg)!=28619:raise ValueError('Unexpected6B batch plan')

def preflight_check():
    cfg=load_config(CONFIG);gate=read_json(Path(os.environ['JOB_RECORD'])/'ppl-gate.json')
    if not gate['passed'] or gate['configuration']!=cfg or gate['seconds']>600:raise ValueError('Preflight failed')

def training_check():
    cfg=load_config(CONFIG);directory=DEFAULT_ROOT/'runs/experiments'/STUDY
    s=read_json(directory/'summary.json');ev=s['last_evaluation'];steps=count_updates(cfg)
    if (s['configuration']!=cfg or s['stop_reason']!='training_token_target' or
        s['total_tokens_seen']!=TARGET or s['budget_tokens_charged']!=TARGET or
        s['global_step']!=steps or s['attempted_updates']!=steps or s['cursor']!=TARGET//8192 or
        s['checkpoint_loaded_for_training'] or s['initialization']['pretrained_weights_loaded'] or
        s['export_global_step']!=steps or s['export_tokens_seen']!=TARGET or
        ev['eval/documents']!=1000 or ev['eval/prediction_tokens']!=1479302 or
        not math.isfinite(ev['eval/perplexity']) or s['ema_evaluations'] or s['ema_exports'] or
        s['attention_kernel_calls']['varlen']==0):
        raise ValueError('Incomplete fresh6B endpoint; no automatic retry/resume')
    if not read_json(directory/'export.json')['reload_verified']:raise ValueError('HF reload not verified')
    parent=DEFAULT_ROOT/'runs/experiments/ppl-next-20261016-s2-nor'
    if s['initialization']['sample_signature']!=read_json(parent/'initialization.json')['sample_signature']:
        raise ValueError('Random initialization differs from approved seed42 recipe')
    if s['data_provenance']!=read_json(parent/'data-provenance.json'):
        raise ValueError('Data/evaluation provenance differs from1B winner')
    groups=read_json(directory/'optimizer-groups.json');old=read_json(parent/'optimizer-groups.json')
    for key in ('parameter_names','parameter_counts','tied_aliases','matrix_method','normuon_beta2','attention_mask'):
        if groups[key]!=old[key]:raise ValueError('Unexpected optimizer routing/method change: '+key)
    from raw6b_tail import check as tail_check
    tail_check()
    from schedule_audit import audit_log
    atomic_json(LIVE/'schedule-audit.json',audit_log(DEFAULT_ROOT/'logs'/('raw6b-normuon-adam2x-'+os.environ['SLURM_JOB_ID']+'.out'),cfg))
    return s

def receipt():
    check();preflight_check();s=training_check()
    directory=DEFAULT_ROOT/'runs/experiments'/STUDY
    extra=read_json(directory/'confirmation-evaluation.json')
    if (extra['eval/documents']!=9000 or extra['eval/prediction_tokens']!=13666812 or
        extra['evaluation_slots']!=[1000,10000] or extra['global_step']!=28619 or
        extra['training_global_step']!=28619 or extra['export']!=s['export'] or extra['variant']!='raw' or
        not math.isfinite(extra['eval/perplexity']) or
        not math.isclose(math.exp(extra['eval/nll']/extra['eval/prediction_tokens']),extra['eval/perplexity'],rel_tol=1e-12)):
        raise ValueError('9000-document endpoint confirmation incomplete')
    for rank in range(8):
        if read_json(Path(os.environ['JOB_RECORD'])/f'worker-rank-{rank}-exit.json')['exit_code']!=0:
            raise ValueError('Training worker failed')
    baseline=read_json(DEFAULT_ROOT/'runs/experiments/raw4b-normuon-adam2x-20261017/summary.json')
    previous=baseline['last_evaluation']['eval/perplexity']
    atomic_json(LIVE/'receipt.json',dict(run_id=STUDY,job_id=os.environ['SLURM_JOB_ID'],
        tokens=TARGET,updates=28619,subset_perplexity=s['last_evaluation']['eval/perplexity'],
        confirmation_perplexity=extra['eval/perplexity'],export=s['export'],wandb_url=s['wandb_url'],
        previous_4b_subset_perplexity=previous,
        delta_subset_ppl_vs_previous_4b=s['last_evaluation']['eval/perplexity']-previous,
        comparison_note='Different token budgets (6B versus4B), same seed42 and fixed1k subset;6B lookup LRs with AdamW x2; eval cadence2800 versus1900',
        confirmation_evaluation=extra,schedule_audit=read_json(LIVE/'schedule-audit.json'),training_tail=read_json(directory/'training-tail-provenance.json')))

def submit(state):
    if 'job' in state:return
    if not read_json(LIVE/'authorization.json')['approved']:raise ValueError('Missing user approval')
    ledger=ExperimentLedger(DEFAULT_ROOT)
    prior=[(k,v) for k,v in read_json(ledger.path)['jobs'].items() if v['run_id']==STUDY]
    if prior:
        if len(prior)!=1:raise ValueError('Only one allocation permitted; no retries')
        key,item=prior[0]
        state['job']={**item,'reservation':key}
        atomic_json(STATE,state);return
    cfg=load_config(CONFIG);tag=STUDY+'-'+uuid.uuid4().hex
    ledger.reserve(tag,STUDY,cfg,SECONDS,8,False)
    plan=read_json(LIVE/'plan.json')
    command=['sbatch','--parsable','--comment='+tag,'--exclude='+','.join(plan['excluded_nodes']),
             str(REPO/'slurm/raw6b-normuon-adam2x.sbatch'),STUDY,str(CONFIG)]
    item=dict(status='submission_uncertain',reservation=tag,submission_tag=tag,
              submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
              submission_command=command)
    state['job']=item;atomic_json(STATE,state)
    with ledger.transaction() as data:
        data['jobs'][tag].update(item,study=STUDY,source_root=str(SOURCE),
            authorization='User explicitly approved reviewed6B configuration:可以，開始訓練; upward6B endpoint and10h/80H exception; formal17 exception only;8H200/8CPU,one10h/80H allocation;fresh weights,no retry/resume/cleanup/HF upload')
    env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),'FOURB_CONTROLLER':str(Path(__file__).resolve())}
    try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
    except (OSError,subprocess.TimeoutExpired) as e:
        item['submission_error']=str(e);atomic_json(STATE,state);return
    item['submission_receipt']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    job=r.stdout.strip().split(';')[0]
    if r.returncode==0 and job.isdigit():
        with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=job
        item['returned_job_id']=job;ledger.bind(tag,job);item.update(job_id=job,status='submitted')
    atomic_json(STATE,state)

def reconcile(state):
    item=state.get('job')
    if not item or item.get('status') in ('complete','failed'):return
    ledger=ExperimentLedger(DEFAULT_ROOT)
    if not item.get('job_id'):
        job=find_submission_job(item,DEFAULT_ROOT)
        if job not in read_json(ledger.path)['jobs']:ledger.bind(item['reservation'],job)
        item['job_id']=job;item['status']='submitted';atomic_json(STATE,state)
    job=item['job_id']
    q=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State'],capture_output=True,text=True,timeout=180,check=True)
    rows=[x.split('|') for x in q.stdout.splitlines() if x.startswith(job+'|')]
    if not rows:return
    status=rows[0][1].split()[0].rstrip('+');item['slurm_state']=status
    if status in TERMINAL:
        item['accounting']=ledger.reconcile(job)
        item['status']='complete' if status=='COMPLETED' and (LIVE/'receipt.json').exists() else 'failed'
        if item['status']=='complete':
            measured=item['accounting']
            atomic_json(LIVE/'results.json',{**read_json(LIVE/'receipt.json'),
                'h200_hours':measured['h200_hours'],'allocation':measured})
    atomic_json(STATE,state)

def tick():
    with locked(LIVE/'.controller.lock',blocking=False):
        state=read_json(STATE) if STATE.exists() else dict(status='ready')
        reconcile(state)
        if (LIVE/'disabled').exists():state['status']='disabled'
        elif state.get('job',{}).get('status') in ('complete','failed'):
            state['status']=state['job']['status']
        else:
            check();submit(state);state['status']='submitted'
        atomic_json(STATE,state)
        snapshot=read_json(DEFAULT_ROOT/'experiment-ledger.json')
        if not snapshot.get('jobs') or not snapshot.get('runs'):raise ValueError('Empty ledger; preserve snapshot')
        atomic_json(LIVE/'ledger-latest.json',snapshot)
        return state

def main():
    cmd=sys.argv[1]
    if cmd=='check':check()
    elif cmd=='preflight-check':preflight_check()
    elif cmd=='training-check':training_check()
    elif cmd=='receipt':receipt()
    elif cmd=='run-once':
        state=tick();print(json.dumps(dict(status=state['status'],job={k:state.get('job',{}).get(k) for k in ('job_id','status','slurm_state')})))
    elif cmd=='status':print(json.dumps(read_json(STATE),indent=2))
    else:raise ValueError('Unknown command')

if __name__=='__main__':main()
