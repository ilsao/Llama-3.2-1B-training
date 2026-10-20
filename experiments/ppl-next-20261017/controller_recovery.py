#!/usr/bin/env python3
"""Metadata-only controller overlay; the sealed training source is untouched.

Repairs only the explicitly authorized C submission, using its original budget.
Automatic ticks never retry sbatch. One uncertain job cannot hide other costs.
"""
import copy,datetime as dt,getpass,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
LIVE=Path(__file__).resolve().parent
SOURCE=LIVE/'source'
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('factorial_base',SOURCE/'slurm/factorial_controller.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.STATE=LIVE/'controller-state-recovery.json'
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,read_json,file_digest,locked
ORIGINAL_RECONCILE=base.reconcile
ORIGINAL_CHECK=base.check
ORIGINAL_RUN=subprocess.run

def bounded_run(command,*args,**kwargs):
    # A slow controller RPC is not evidence that a submission failed.
    if command and command[0]=='sbatch':kwargs['timeout']=180
    return ORIGINAL_RUN(command,*args,**kwargs)

def checked_query(command):
    r=ORIGINAL_RUN(command,capture_output=True,text=True,timeout=45)
    if r.returncode:raise RuntimeError('Scheduler query failed; keep reservation: '+r.stderr.strip())
    return dict(command=command,stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)

def discover(item):
    """Successful live/accounting queries plus exact-tag startup receipts."""
    start=(dt.datetime.fromisoformat(item['submission_started_utc'])-dt.timedelta(minutes=2)).astimezone().strftime('%Y-%m-%dT%H:%M:%S')
    queue=checked_query(['squeue','--noheader','--user',getpass.getuser(),'--format=%i|%j|%T|%k'])
    acct=checked_query(['sacct','-X','-n','-P','--user',getpass.getuser(),'-S',start,'--format=JobIDRaw,JobName%100,State,Submit,Start,ElapsedRaw'])
    name='pf17-C-seed42';matches=set();ambiguous=set()
    for row in queue['stdout'].splitlines():
        fields=[x.strip() for x in row.split('|')]
        if len(fields)>=4 and (fields[1]==name or fields[3]==item['submission_tag']):
            if fields[3]==item['submission_tag']:matches.add(fields[0])
            else:ambiguous.add(fields[0])
    for row in acct['stdout'].splitlines():
        fields=[x.strip() for x in row.split('|')]
        if len(fields)>=2 and fields[1]==name and fields[0] not in matches:ambiguous.add(fields[0])
    receipts=[]
    for path in (DEFAULT_ROOT/'logs').glob('job-*/allocation.txt'):
        text=path.read_text();tag=re.search(r'\bComment=(\S+)',text)
        if tag and tag[1]==item['submission_tag']:
            job=path.parent.name[4:];matches.add(job);ambiguous.discard(job)
            receipts.append(dict(path=str(path),sha256=file_digest(path)))
    for job in list(ambiguous):
        q=checked_query(['scontrol','show','job',job,'-o'])
        tag=re.search(r'\bComment=(\S+)',q['stdout'])
        if tag and tag[1]==item['submission_tag']:matches.add(job);ambiguous.discard(job)
    if any(not x.isdigit() for x in matches|ambiguous):raise ValueError('Invalid job identity')
    return dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),queue=queue,accounting=acct,
                startup_receipts=receipts,matches=sorted(matches),ambiguous=sorted(ambiguous))

def reconcile_independently(state):
    for key,item in state['jobs'].items():
        try:
            # The original routine writes its supplied state. Use the whole state
            # but hide other jobs only from iteration, never from persistence.
            reconcile_one(state,key,item)
            item.pop('recovery_query_error',None)
        except (subprocess.TimeoutExpired,RuntimeError) as error:
            item['recovery_query_error']=str(error)
        except ValueError as error:
            if not item.get('job_id') and 'Recovery found' in str(error):
                item['recovery_query_error']=str(error)
            else:raise
        atomic_json(base.STATE,state)

def reconcile_one(state,key,item):
    if item['status'] in ('complete','failed'):return
    ledger=ExperimentLedger(DEFAULT_ROOT)
    if not item.get('job_id'):
        job=base.find_submission_job(item,DEFAULT_ROOT)
        if job not in read_json(ledger.path)['jobs']:ledger.bind(item['reservation'],job)
        item.update(job_id=job,status='submitted');atomic_json(base.STATE,state)
    job=item['job_id']
    q=checked_query(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State'])
    rows=[x.split('|') for x in q['stdout'].splitlines() if x.startswith(job+'|')]
    if not rows:return
    status=rows[0][1].split()[0].rstrip('+');item['slurm_state']=status
    if status not in TERMINAL:return
    item['accounting']=ledger.reconcile(job);item['status']='failed'
    if status=='COMPLETED':
        if item['kind']=='train':base.endpoint(key);read_json(LIVE/'receipts'/f'{key}.json')
        elif key=='gate':
            gate=read_json(DEFAULT_ROOT/'logs'/f'job-{job}'/'ppl-gate.json')
            if not gate['passed'] or gate['seconds']>780:raise ValueError('Gate failed')
        elif read_json(LIVE/'confirmation.json')['status']!='complete':raise ValueError('Confirmation incomplete')
        item['status']='complete'

def check():
    ORIGINAL_CHECK()
    seal=read_json(LIVE/'controller-recovery-seal.json')
    if file_digest(__file__)!=seal['sha256']:raise ValueError('Recovery controller changed')

def repair_c():
    """Explicit one-time repair; persist attempt before calling sbatch --hold."""
    with locked(LIVE/'.controller.lock',blocking=False):
        check();state=read_json(base.STATE);item=state['jobs']['C-seed42']
        if item.get('job_id'):raise ValueError('C already bound; do not submit')
        audit=discover(item);atomic_json(LIVE/'C-recovery-discovery.json',audit)
        if audit['ambiguous'] or len(audit['matches'])>1:raise ValueError('Ambiguous existing jobs; do not submit')
        ledger=ExperimentLedger(DEFAULT_ROOT)
        if audit['matches']:
            job=audit['matches'][0];ledger.bind(item['reservation'],job)
            item.update(job_id=job,status='submitted');atomic_json(base.STATE,state);return state
        if item.get('manual_repair_attempt'):raise ValueError('One repair already attempted; recover it, never repeat')
        run='ppl-next-20261017-C-seed42'
        if (DEFAULT_ROOT/'runs/experiments'/run).exists():raise ValueError('Unexpected initialized C run')
        original=copy.deepcopy(item);tag=item['reservation']
        existing=read_json(ledger.path)['jobs'][tag]
        if existing['run_id']!=run or existing['reserved_h200_hours']!=16 or existing['requested_seconds']!=7200:raise ValueError('Reservation changed')
        item['submission_history']=[original]
        item['manual_repair_attempt']=dict(authorization='User: 修復',time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),mode='one held submission after zero observed jobs; original reservation retained')
        command=[original['submission_command'][0],'--hold',*original['submission_command'][1:]]
        item['repair_submission_command']=command;atomic_json(base.STATE,state)
        with ledger.transaction() as data:
            data['jobs'][tag]['submission_history']=[original]
            data['jobs'][tag]['manual_repair_attempt']=item['manual_repair_attempt']
        env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),
             'FACTORIAL_CONTROLLER':str(SOURCE/'slurm/factorial_controller.py')}
        try:r=bounded_run(command,env=env,capture_output=True,text=True)
        except (OSError,subprocess.TimeoutExpired) as error:
            item['repair_submission_error']=str(error);atomic_json(base.STATE,state);return state
        item['repair_submission_receipt']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        job=r.stdout.strip().split(';')[0]
        if r.returncode==0 and job.isdigit():
            item['returned_job_id']=job;item['repair_held_job_id']=job
            with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=job
        atomic_json(base.STATE,state)
        if r.returncode or not job.isdigit():raise ValueError('Repair not acknowledged; reservation retained')
        # A held job consumes no GPUs; explicitly verify unique accepted identity.
        audit=discover(item);atomic_json(LIVE/'C-recovery-held-discovery.json',audit)
        if audit['matches']!=[job] or audit['ambiguous']:raise ValueError('Leave replacement held until duplicate identities resolved')
        ledger.bind(tag,job);item.update(job_id=job,status='held_verified')
        atomic_json(base.STATE,state)
        return state

def release_c():
    with locked(LIVE/'.controller.lock',blocking=False):
        check();state=read_json(base.STATE);item=state['jobs']['C-seed42'];job=item.get('job_id')
        if not job or item.get('repair_held_job_id')!=job:raise ValueError('No verified held replacement')
        audit=discover(item)
        if audit['matches']!=[job] or audit['ambiguous']:raise ValueError('C identity is not unique')
        q=checked_query(['scontrol','show','job',job,'-o'])
        if 'JobState=PENDING' not in q['stdout'] or 'Reason=JobHeldUser' not in q['stdout']:raise ValueError('Job is not held by user')
        item['release_intent_utc']=dt.datetime.now(dt.timezone.utc).isoformat();atomic_json(base.STATE,state)
        result=checked_query(['scontrol','release',job]);item['release_receipt']=result
        item.update(status='submitted',recovery='one held replacement released; original configuration and reservation preserved')
        atomic_json(base.STATE,state)
        with ExperimentLedger(DEFAULT_ROOT).transaction() as data:data['jobs'][job].update(recovery=item['recovery'],manual_repair_attempt=item['manual_repair_attempt'],submission_history=item['submission_history'])
        return state

def recovered_tick():
    with locked(LIVE/'.controller.lock',blocking=False):
        state=read_json(base.STATE) if base.STATE.exists() else dict(jobs={},status='ready')
        reconcile_independently(state)
        if (LIVE/'recovery-disabled').exists():state['status']='disabled'
        elif any(j['status']=='failed' for j in state['jobs'].values()):state['status']='stopped_on_failure'
        else:
            check()
            done=lambda key:state['jobs'].get(key,{}).get('status')=='complete'
            base.submit(state,'gate','gate');state['status']='gate'
            if done('gate'):
                for arm in 'AB':base.submit(state,arm+'-seed42')
                state['status']='wave1'
            if all(done(a+'-seed42') for a in 'AB'):
                for arm in 'CD':base.submit(state,arm+'-seed42')
                state['status']='wave2'
            if all(done(a+'-seed42') for a in 'ABCD'):
                result=base.selection(state);state['status']='complete_no_promotion'
                if result['qualified']:
                    base.submit(state,'confirm','confirm');state['status']='confirmation'
                    if done('confirm'):
                        conf=read_json(LIVE/'confirmation.json')
                        state['status']='complete_confirmation_negative'
                        if conf['delta_nll']<0 and conf['paired_document_ci95'][1]<0:
                            for arm in ('A',result['winner']):base.submit(state,arm+'-seed43')
                            state['status']='seed43'
                            if all(done(a+'-seed43') for a in ('A',result['winner'])):
                                a=read_json(LIVE/'receipts'/'A-seed43.json');b=read_json(LIVE/'receipts'/f"{result['winner']}-seed43.json")
                                for f in ('initialization','provenance','source_digest','environment_digest'):
                                    if a[f]!=b[f]:raise ValueError('Unmatched seed43 '+f)
                                delta=b['mean_nll']-a['mean_nll']
                                atomic_json(LIVE/'replication.json',dict(delta_nll=delta,confirmed=delta<0,control=a,candidate=b))
                                state['status']='complete_transfer_candidate' if delta<0 else 'complete_seed43_negative'
        atomic_json(base.STATE,state);return state

def main():
    base.reconcile=reconcile_independently;base.check=check
    subprocess.run=bounded_run
    command=sys.argv[1]
    try:
        if command=='repair-c':state=repair_c()
        elif command=='release-c':state=release_c()
        elif command=='run-once':state=recovered_tick()
        elif command=='status':state=read_json(base.STATE)
        elif command=='check':check();return
        else:raise ValueError(command)
    except BlockingIOError:
        print(json.dumps(dict(status='another_controller_tick_active')));return
    print(json.dumps(dict(status=state['status'],jobs={k:{f:v.get(f) for f in ('job_id','status','slurm_state','recovery_query_error')} for k,v in state['jobs'].items()})))
if __name__=='__main__':main()
