#!/usr/bin/env python3
"""Bounded submission owner for the approved CWD x auxiliary-LR factorial."""
import datetime as dt,json,math,os,subprocess,sys,uuid
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]
LIVE=SOURCE.parent
REPO=LIVE.parents[1]
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,read_json,file_digest,digest,locked,load_config
from ppl_contracts import STUDY,PREP,TARGET,count_updates
from slurm.submission import find_submission_job
STATE=LIVE/'controller-state.json'

def check():
    for path,sha in read_json(LIVE/'execution-seal.json')['files'].items():
        if file_digest(path)!=sha:raise ValueError('Sealed file changed: '+path)
    old=read_json(LIVE/'ledger-before.json');now=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    for key,val in old['run_digests'].items():
        if now['runs'].get(key,{}).get('config_digest')!=val:raise ValueError('Historical identity lost: '+key)
    for key,val in old['measured_costs'].items():
        if now['jobs'].get(key,{}).get('h200_hours')!=val:raise ValueError('Historical cost changed: '+key)
    prefix=read_json(LIVE/'raw-prefix.json')
    progress=read_json(DEFAULT_ROOT/'preprocessed'/PREP/'raw/progress.json')
    if progress['train_files'][:len(prefix['files'])]!=prefix['files']:raise ValueError('Raw prefix changed')
    if sum(x['rows'] for x in prefix['files'])<TARGET//8192:raise ValueError('Insufficient data')
    if not read_json(LIVE/'authorization.json')['approved']:raise ValueError('Authorization missing')

def endpoint(key):
    cfg=load_config(LIVE/'configs'/f'{key}.json')
    run=DEFAULT_ROOT/'runs/experiments'/f'{STUDY}-{key}'
    s=read_json(run/'summary.json');ev=s['last_evaluation']
    if (s['configuration']!=cfg or s['total_tokens_seen']!=TARGET or s['budget_tokens_charged']!=TARGET
        or s['global_step']!=4775 or s['attempted_updates']!=4775 or s['cursor']!=TARGET//8192
        or s['stop_reason']!='training_token_target' or s['checkpoint_loaded_for_training']
        or s['initialization']['pretrained_weights_loaded'] or s['export_global_step']!=4775
        or s['export_tokens_seen']!=TARGET or ev['eval/documents']!=1000
        or ev['eval/prediction_tokens']!=1479302 or not math.isfinite(ev['eval/perplexity'])
        or not read_json(run/'export.json')['reload_verified'] or s['attention_kernel_calls']['varlen']==0):
        raise ValueError('Incomplete/mismatched endpoint '+key)
    return s

def receipt(key):
    check();s=endpoint(key)
    for rank in range(8):
        if read_json(Path(os.environ['JOB_RECORD'])/f'worker-rank-{rank}-exit.json')['exit_code']!=0:raise ValueError('Worker failed')
    atomic_json(LIVE/'receipts'/f'{key}.json',dict(run_id=f'{STUDY}-{key}',job_id=os.environ['SLURM_JOB_ID'],
        mean_nll=s['last_evaluation']['eval/nll']/1479302,perplexity=s['last_evaluation']['eval/perplexity'],
        initialization=s['initialization']['sample_signature'],provenance=s['data_provenance'],
        source_digest=s['code_digest'],environment_digest=s['environment_digest'],evaluation=s['last_evaluation'],
        export=s['export'],wandb_url=s['wandb_url'],config_digest=digest(s['configuration'])))

def submit(state,key,kind='train'):
    if key in state['jobs']:return
    run=f'{STUDY}-{key}';ledger=ExperimentLedger(DEFAULT_ROOT)
    if any(j['run_id']==run for j in read_json(ledger.path)['jobs'].values()):raise ValueError('Existing allocation without controller state; manual recovery required')
    path=LIVE/'configs'/f'{key}.json' if kind=='train' else LIVE/'validation-config.json'
    cfg=load_config(path);seconds=7200 if kind=='train' else 900
    used=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in read_json(ledger.path)['jobs'].values() if j['run_id'].startswith(STUDY+'-'))
    seed43='seed43' in key
    cap=100 if seed43 else 68
    if used+8*seconds/3600>cap+1e-9:raise ValueError('Campaign cap exceeded')
    tag=run+'-'+uuid.uuid4().hex
    # Persist intent before reservation; ambiguous outcomes retain the reservation.
    item=dict(kind=kind,status='submission_uncertain',reservation=tag,submission_tag=tag,
              submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),config_path=str(path))
    state['jobs'][key]=item;atomic_json(STATE,state)
    ledger.reserve(tag,run,cfg,seconds,8,False)
    script=REPO/'slurm'/('ppl-factorial-train.sbatch' if kind=='train' else 'ppl-factorial-eval.sbatch')
    command=['sbatch','--parsable','--time='+str(seconds//60),'--job-name=pf17-'+key,'--comment='+tag,
             '--exclude='+','.join(read_json(LIVE/'execution-plan.json')['excluded_nodes']),str(script),run,key]
    item['submission_command']=command;atomic_json(STATE,state)
    with ledger.transaction() as data:data['jobs'][tag].update(item,study=STUDY,authorization='Approved fixed 2x2 <=68H; conditional paired seed43 +32H; fresh only, no retries')
    env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),'FACTORIAL_CONTROLLER':str(Path(__file__).resolve())}
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
    for key,item in state['jobs'].items():
        if item['status'] in ('complete','failed'):continue
        if not item.get('job_id'):
            job=find_submission_job(item,DEFAULT_ROOT)
            if job not in read_json(ledger.path)['jobs']:ledger.bind(item['reservation'],job)
            item.update(job_id=job,status='submitted');atomic_json(STATE,state)
        job=item['job_id']
        q=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State'],capture_output=True,text=True,check=True,timeout=30)
        rows=[x.split('|') for x in q.stdout.splitlines() if x.startswith(job+'|')]
        if not rows:continue
        status=rows[0][1].split()[0].rstrip('+');item['slurm_state']=status
        if status not in TERMINAL:continue
        item['accounting']=ledger.reconcile(job);item['status']='failed'
        if status=='COMPLETED':
            if item['kind']=='train':
                endpoint(key);read_json(LIVE/'receipts'/f'{key}.json')
            elif key=='gate':
                gate=read_json(DEFAULT_ROOT/'logs'/f'job-{job}'/'ppl-gate.json')
                if not gate['passed'] or gate['seconds']>780:raise ValueError('Gate failed')
            else:
                if read_json(LIVE/'confirmation.json')['status']!='complete':raise ValueError('Confirmation incomplete')
            item['status']='complete'
        atomic_json(STATE,state)

def selection(state):
    records={k:read_json(LIVE/'receipts'/f'{k}-seed42.json') for k in 'ABCD'}
    for r in records.values():
        for f in ('initialization','provenance','source_digest','environment_digest'):
            if r[f]!=records['A'][f]:raise ValueError('Unmatched factorial '+f)
    losses={k:r['mean_nll'] for k,r in records.items()}
    winner=min(losses,key=lambda k:(losses[k],k));delta=losses[winner]-losses['A']
    costs={k:state['jobs'][k+'-seed42']['accounting']['h200_hours'] for k in 'ABCD'}
    result=dict(winner=winner,delta_nll=delta,qualified=winner!='A' and delta<=-.005 and costs[winner]<=1.05*costs['A'],
                cost_ratio=costs[winner]/costs['A'],records=records,
                interaction_nll=losses['D']-losses['B']-losses['C']+losses['A'],single_seed=True)
    path=LIVE/'selection.json'
    if path.exists() and read_json(path)!=result:raise ValueError('Locked selection changed')
    atomic_json(path,result);return result

def tick():
    with locked(LIVE/'.controller.lock',blocking=False):
        state=read_json(STATE) if STATE.exists() else dict(jobs={},status='ready')
        reconcile(state)
        if (LIVE/'disabled').exists():state['status']='disabled'
        elif any(j['status']=='failed' for j in state['jobs'].values()):state['status']='stopped_on_failure'
        else:
            check()
            done=lambda key:state['jobs'].get(key,{}).get('status')=='complete'
            submit(state,'gate','gate');state['status']='gate'
            if done('gate'):
                for arm in 'AB':submit(state,arm+'-seed42')
                state['status']='wave1'
            if all(done(a+'-seed42') for a in 'AB'):
                for arm in 'CD':submit(state,arm+'-seed42')
                state['status']='wave2'
            if all(done(a+'-seed42') for a in 'ABCD'):
                result=selection(state);state['status']='complete_no_promotion'
                if result['qualified']:
                    submit(state,'confirm','confirm');state['status']='confirmation'
                    if done('confirm'):
                        conf=read_json(LIVE/'confirmation.json')
                        state['status']='complete_confirmation_negative'
                        if conf['delta_nll']<0 and conf['paired_document_ci95'][1]<0:
                            for arm in ('A',result['winner']):submit(state,arm+'-seed43')
                            state['status']='seed43'
                            if all(done(a+'-seed43') for a in ('A',result['winner'])):
                                a=read_json(LIVE/'receipts'/'A-seed43.json');b=read_json(LIVE/'receipts'/f"{result['winner']}-seed43.json")
                                for f in ('initialization','provenance','source_digest','environment_digest'):
                                    if a[f]!=b[f]:raise ValueError('Unmatched seed43 '+f)
                                delta=b['mean_nll']-a['mean_nll']
                                atomic_json(LIVE/'replication.json',dict(delta_nll=delta,confirmed=delta<0,control=a,candidate=b))
                                state['status']='complete_transfer_candidate' if delta<0 else 'complete_seed43_negative'
        atomic_json(STATE,state);return state

def main():
    command=sys.argv[1]
    if command=='check':check()
    elif command=='receipt':receipt(sys.argv[2])
    elif command=='run-once':
        s=tick();print(json.dumps(dict(status=s['status'],jobs={k:{f:v.get(f) for f in ('job_id','status','slurm_state')} for k,v in s['jobs'].items()})))
    elif command=='status':print(json.dumps(read_json(STATE),indent=2))
    else:raise ValueError(command)
if __name__=='__main__':main()
