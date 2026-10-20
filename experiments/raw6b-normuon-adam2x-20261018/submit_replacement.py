"""Submit exactly the user-approved startup replacement; never auto retry."""
from pathlib import Path
import os,sys,json,subprocess,uuid,datetime
LIVE=Path(__file__).resolve().parent;SOURCE=LIVE/'source-v2';REPO=LIVE.parents[1]
sys.path.insert(0,str(SOURCE))
from training_state import read_json,atomic_json,ExperimentLedger,DEFAULT_ROOT,load_config,locked
from ppl_contracts import STUDY
from slurm.fourb_controller import check
STATE=LIVE/'replacement-state.json'

def main():
    with locked(LIVE/'.controller.lock'):
        if STATE.exists():
            print(json.dumps(read_json(STATE)));return
        check();cfg=load_config(LIVE/'config.json');ledger=ExperimentLedger(DEFAULT_ROOT)
        approval=read_json(LIVE/'recovery-authorization.json')
        if approval.get('approved') is not True:raise ValueError('Missing explicit replacement approval')
        before=read_json(ledger.path)
        prior=[(k,v) for k,v in before['jobs'].items() if v['run_id']==STUDY]
        if [k for k,_ in prior]!=['566121']:raise ValueError('An allocation/reservation already exists; recover it, never resubmit')
        tag=STUDY+'-replacement-'+uuid.uuid4().hex
        ledger.reserve(tag,STUDY,cfg,35940,8,False,fresh_retry=True)
        command=['sbatch','--parsable','--comment='+tag,'--exclude='+','.join(read_json(LIVE/'plan.json')['excluded_nodes']),str(REPO/'slurm/raw6b-normuon-adam2x-recovery.sbatch'),STUDY,str(LIVE/'config.json')]
        state=dict(status='submission_uncertain',job=dict(reservation=tag,submission_tag=tag,submission_command=command,submission_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        atomic_json(STATE,state)
        with ledger.transaction() as data:data['jobs'][tag].update(state['job'],source_root=str(SOURCE),authorization='Explicit user 好: one startup replacement after566121,35940sec,8H200/8CPU; same formal17 and80H cumulative; no checkpoint load/further retry')
        env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),'FOURB_CONTROLLER':str(SOURCE/'slurm/fourb_controller.py')}
        try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
        except (OSError,subprocess.TimeoutExpired) as error:
            state['error']=str(error);atomic_json(STATE,state);print(json.dumps(state));return
        state['submission_receipt']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        job=r.stdout.strip().split(';')[0]
        if r.returncode==0 and job.isdigit():
            with ledger.transaction() as data:data['jobs'][tag]['returned_job_id']=job
            ledger.bind(tag,job);state['job'].update(job_id=job,status='submitted');state['status']='submitted'
        atomic_json(STATE,state);atomic_json(LIVE/'ledger-latest.json',read_json(ledger.path));print(json.dumps(state))

if __name__=='__main__':main()
