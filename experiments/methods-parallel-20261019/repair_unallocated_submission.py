"""One-shot submission repair for never-allocated held job 572036 only.

This is not a training retry: require zero allocation and no run directory;
preserve the cancelled receipt, same identity/config, and original 16H cap.
"""
import copy,datetime,importlib.util,re,subprocess,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('methods_parallel',HERE.parents[1]/'slurm/methods_parallel.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
from training_state import ExperimentLedger,atomic_json,read_json,load_config,digest,file_digest,locked,slurm_record

OLD='572036';KEY='D1-seed42'

def checked(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    if p.returncode:raise RuntimeError(f'{args[0]} failed: {p.stderr.strip()}')
    return p.stdout

def verify_never_allocated(snapshot,state,ledger,root):
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:]*)=(\S+)',snapshot))
    expected={'JobId':OLD,'JobState':'PENDING','Reason':'JobHeldUser','RunTime':'00:00:00',
              'AllocTRES':'(null)','StartTime':'Unknown','Restarts':'0','Comment':state['submission_tag']}
    if any(fields.get(k)!=v for k,v in expected.items()):raise ValueError('Not the exact never-allocated held job')
    if state.get('returned_job_id')!=OLD or state.get('key')!=KEY or state.get('phase')!='bound_held':
        raise ValueError('Unexpected original submission identity')
    previous={k:v for k,v in ledger['jobs'].items() if v['run_id']==state['run_id']}
    if set(previous)!={OLD}:raise ValueError('Additional submission already exists')
    if ledger['runs'][state['run_id']]['config_digest']!=state['config_digest']:raise ValueError('Changed config')
    if (root/'runs/experiments'/state['run_id']).exists():raise ValueError('Run directory already exists')

def main():
    matrix=cli.verify()
    row=next(x for x in matrix['jobs'] if x['key']==KEY)
    authorization=read_json(HERE/'authorization.json')
    if authorization['user_instruction']!='開始實驗' or authorization['max_new_h200_hours']!=128:
        raise ValueError('Missing original eight-job launch authorization')
    repair=HERE/'submission-repair';intent=repair/'repair-intent.json'
    with locked(cli.RECORDS/'.submission.lock',blocking=False):
        if intent.exists():raise ValueError('One-shot recovery already attempted; inspect existing intent, never resubmit')
        path=cli.RECORDS/(KEY+'.json');old=read_json(path)
        cfg=load_config(HERE/row['config'])
        if old['config_digest']!=digest(cfg):raise ValueError('Configuration changed')
        ledger=ExperimentLedger(cli.DEFAULT_ROOT)
        snapshot=checked(['scontrol','show','job',OLD,'-o'])
        verify_never_allocated(snapshot,old,read_json(ledger.path),cli.DEFAULT_ROOT)
        queued=checked(['squeue','--noheader','--user','asciibase64','--format=%i|%k'])
        matches=[line.split('|')[0].strip() for line in queued.splitlines() if line.split('|')[-1].strip()==old['submission_tag']]
        if matches!=[OLD]:raise ValueError('Submission tag is not unique')
        atomic_json(intent,dict(reason='Slurm release failed twice; never-allocated submission repair under user start instruction',
            original=copy.deepcopy(old),snapshot=snapshot,queue=queued,checkpoint_training_load=False,
            old_seal=old['seal_sha256'],new_seal=file_digest(HERE/'execution-seal.json'),
            original_authorization_sha256=file_digest(HERE/'authorization.json'),
            only_allowed_cancel=OLD,maximum_new_submissions=1))
        cancelled=checked(['scancel',OLD]);atomic_json(repair/'cancel-receipt.json',dict(job_id=OLD,stdout=cancelled))
        measured=None
        for attempt in range(10):
            try:measured=slurm_record(OLD,cli.DEFAULT_ROOT);break
            except ValueError:
                if attempt==9:raise
                time.sleep(1)
        if not measured['state'].startswith('CANCELLED') or measured['elapsed_seconds'] or measured['gpu_count']:
            raise ValueError('Cancellation did not confirm zero allocation; stop without replacement')
        ledger.reconcile(OLD);atomic_json(repair/'cancel-accounting.json',measured)
        if (cli.DEFAULT_ROOT/'runs/experiments'/old['run_id']).exists():raise ValueError('Run unexpectedly initialized')
        tag='mp-'+uuid.uuid4().hex
        state=dict(key=KEY,run_id=old['run_id'],reservation=tag,submission_tag=tag,
            submission_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            config_digest=digest(cfg),seal_sha256=file_digest(HERE/'execution-seal.json'),
            requested_seconds=7200,gpu_count=8,cpu_count=8,batch_budget_h200_hours=128,
            phase='repair_intent',prior_submission_attempts=[old],cancelled_unallocated_job=OLD,
            accounting_continuation_only=True,training_checkpoint_load=False)
        atomic_json(path,state)
        # resume=True is ONLY the existing ledger API for cumulative identity/cost.
        # Actual worker command has no --resume and fresh random weights are required.
        ledger.reserve(tag,old['run_id'],cfg,7200,8,True)
        with ledger.transaction() as data:
            data['jobs'][tag].update(accounting_continuation_only=True,training_checkpoint_load=False,
                                    cancelled_unallocated_job=OLD,submission_repair_intent=str(intent))
        state.update(phase='reserved',submission_command=cli.command(row,tag));atomic_json(path,state)
        try:
            result=subprocess.run(state['submission_command'],capture_output=True,text=True,timeout=45)
            state['submission_receipt']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            atomic_json(path,state)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):raise RuntimeError('Uncertain submission; do not resubmit')
            state.update(returned_job_id=job,phase='accepted');atomic_json(path,state)
            ledger.bind(tag,job);state['phase']='submitted';atomic_json(path,state)
            print(__import__('json').dumps(state),flush=True)
        except (OSError,subprocess.TimeoutExpired,ValueError,RuntimeError) as error:
            state.update(error=str(error),recovery_required=True);atomic_json(path,state);raise

if __name__=='__main__':main()
