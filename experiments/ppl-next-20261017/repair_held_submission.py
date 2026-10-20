"""Explicit one-shot repair for Slurm's failure to release held job 561910."""
import copy,datetime as dt,importlib.util,json,os,subprocess,sys,uuid
from pathlib import Path
LIVE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('repair',LIVE/'controller_recovery.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
from training_state import ExperimentLedger,DEFAULT_ROOT,locked,read_json,atomic_json,load_config

def main():
    with locked(LIVE/'.controller.lock',blocking=False):
        r.check();state=read_json(r.base.STATE);old=state['jobs']['C-seed42'];job=old.get('job_id')
        if job!='561910' or old.get('repair_held_job_id')!=job:raise ValueError('Not the reviewed held job')
        if (LIVE/'held-release-repair-intent.json').exists():raise ValueError('Already attempted; recover existing intent')
        discovery=r.discover(old)
        if discovery['matches']!=[job] or discovery['ambiguous']:raise ValueError('C identity not unique')
        q=r.checked_query(['scontrol','show','job',job,'-o'])
        if not all(x in q['stdout'] for x in ['JobState=PENDING','Reason=JobHeldUser','RunTime=00:00:00','AllocTRES=(null)']):raise ValueError('Held job has unexpected allocation')
        atomic_json(LIVE/'held-release-repair-intent.json',dict(authorization='User: 修復',held_job=job,discovery=discovery,held_snapshot=q,original_state=old))
        result=r.checked_query(['scancel',job]);atomic_json(LIVE/'held-cancel-receipt.json',result)
        ledger=ExperimentLedger(DEFAULT_ROOT);measured=ledger.reconcile(job)
        if not measured['state'].startswith('CANCELLED') or measured['elapsed_seconds']!=0 or measured['gpu_count']!=0:raise ValueError('Cancellation not confirmed unallocated')
        atomic_json(LIVE/'held-cancel-accounting.json',measured)
        run='ppl-next-20261017-C-seed42';cfg=load_config(old['config_path'])
        if (DEFAULT_ROOT/'runs/experiments'/run).exists():raise ValueError('C training unexpectedly initialized')
        tag=run+'-'+uuid.uuid4().hex
        # Continue ONLY cumulative resource accounting for the same logical run.
        # The training command below has no --resume and loads no checkpoint.
        ledger.reserve(tag,run,cfg,7200,8,True)
        command=[('--comment='+tag if x.startswith('--comment=') else x) for x in old['submission_command']]
        item=dict(kind='train',status='submission_uncertain',reservation=tag,submission_tag=tag,
                  config_path=old['config_path'],submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  submission_command=command,prior_submission_attempts=[copy.deepcopy(old)],
                  repair_authorization='User: 修復; unallocated held job could not be released, cancelled at zero GPU-hours; fresh training under original run/config/16H cap',
                  checkpoint_training_load=False)
        state['jobs']['C-seed42']=item;atomic_json(r.base.STATE,state)
        with ledger.transaction() as data:data['jobs'][tag].update(item,study='ppl-next-20261017',accounting_continuation_only=True)
        env={**os.environ,'TRAIN_SOURCE_ROOT':str(r.SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),
             'FACTORIAL_CONTROLLER':str(r.SOURCE/'slurm/factorial_controller.py')}
        try:res=r.bounded_run(command,env=env,capture_output=True,text=True)
        except (OSError,subprocess.TimeoutExpired) as error:
            item['submission_error']=str(error);atomic_json(r.base.STATE,state);return
        item['submission_receipt']=dict(returncode=res.returncode,stdout=res.stdout,stderr=res.stderr)
        newjob=res.stdout.strip().split(';')[0]
        if res.returncode==0 and newjob.isdigit():
            with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=newjob
            item['returned_job_id']=newjob;ledger.bind(tag,newjob);item.update(job_id=newjob,status='submitted')
        atomic_json(r.base.STATE,state)
        print(json.dumps(dict(cancelled_unallocated_job=job,new_job=item.get('job_id'),status=item['status'],same_logical_run=run)))
if __name__=='__main__':main()
