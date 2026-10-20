"""Single approved formal20 allocation; durable intent precedes sbatch."""
import argparse,json,os,re,subprocess,time,uuid
from bootstrap import HERE,REPO,SOURCE
from contracts import RUN,TARGET,CONFIG_DIGEST,SECONDS,config
from training_state import DEFAULT_ROOT,ExperimentLedger,atomic_json,read_json,file_digest,digest,locked
SCRIPT=REPO/'slurm/kd6b-c2-earlydecay-20261020.sbatch'
RECORD=HERE/'submission.json'
PREFIXES=('methods-parallel-20261019','kd-acceleration-','kd-pipeline-',
 'kd3b-online-pipeline-','kd3b-io-validation-','kd3b-io8-validation-',
 'kd3b-hidden8-','kd6b-lossless-','kd6b-lossless2-','kd6b-r5-',
 'critical-batch-20261020','kd6b-c2-earlydecay-20261020')

def inventory():
    prior=read_json(HERE.parent/'critical-batch-20261020-recovery2/source-seal.json')['sha256']
    inherited={n:h for n,h in prior.items() if not n.startswith('experiments/critical-batch-20261020')
               and not n.startswith('slurm/critical-batch-20261020')}
    for n,h in inherited.items():
        if file_digest(REPO/n)!=h:raise ValueError('Sealed dependency changed: '+n)
    paths=list(HERE.glob('*.py'))+[HERE/n for n in ('config.json','authorization.json','protocol.md','ledger-scope.json','static-validation.json','wandb-step-evidence.json','proposal.json')]+[SCRIPT,REPO/'experiments/lr-transfer-horizons-20261015/calibration-result.json']
    return {**inherited,**{str(p.relative_to(REPO)):file_digest(p) for p in sorted(paths)}}

def verify_source_seal():
    config()
    assert inventory()==read_json(HERE/'source-seal.json')['sha256'],'Sealed source changed'

def reserve_entries(data,reservation):
    auth=read_json(HERE/'authorization.json');scope=auth['scope']
    assert RUN not in data['runs'] and reservation not in data['jobs'],'No retry/resume'
    assert sorted(k for k,r in data['runs'].items() if r['category']=='full')==scope['formal_ids']
    assert len(scope['formal_ids'])==19
    assert sorted(data['runs'])==scope['run_ids'],'Run identities changed since review'
    total=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values())
    study=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j['run_id'].startswith(PREFIXES))
    assert total+72<=scope['shared_cap_h200_hours']+1e-8
    assert study+72<=scope['methods_cap_h200_hours']+1e-8
    before=dict(run_count=len(data['runs']),job_count=len(data['jobs']),
                jobs_digest=digest(data['jobs']),runs_digest=digest(data['runs']),shared_total=total,study_total=study)
    data['runs'][RUN]=dict(category='full',formal_run_number=20,config_digest=CONFIG_DIGEST,
        max_h200_hours=72,created=time.time(),study='methods-parallel-20261019',
        authorization_file=str(HERE/'authorization.json'),authorization='User approved reviewed6B/C2/token-bound1841 decay/9h72H proposal')
    data['jobs'][reservation]=dict(run_id=RUN,reserved_h200_hours=72,gpu_count=8,cpu_count=8,
        requested_seconds=SECONDS,status='reserved',source_root=str(SOURCE),
        authorization_file=str(HERE/'authorization.json'),no_automatic_retry=True)
    assert digest({k:v for k,v in data['runs'].items() if k!=RUN})==before['runs_digest']
    assert digest({k:v for k,v in data['jobs'].items() if k!=reservation})==before['jobs_digest']
    return before

def submit():
    verify_source_seal()
    assert read_json(HERE/'static-validation.json')['passed']
    with locked(HERE/'.submit.lock',blocking=False):
        assert not RECORD.exists(),'Submission previously attempted; never retry automatically'
        assert not (DEFAULT_ROOT/'runs/experiments'/RUN).exists(),'Existing output artifacts'
        reservation='kd6bc2-'+uuid.uuid4().hex
        state=dict(run_id=RUN,reservation=reservation,configuration_digest=CONFIG_DIGEST,phase='intent',timestamp=time.time(),requested_seconds=SECONDS)
        atomic_json(RECORD,state)
        ledger=ExperimentLedger(DEFAULT_ROOT)
        with ledger.transaction() as data:prior=reserve_entries(data,reservation)
        state.update(phase='reserved',prior=prior);atomic_json(RECORD,state)
        (DEFAULT_ROOT/'logs').mkdir(parents=True,exist_ok=True)
        cmd=['sbatch','--parsable','--comment='+reservation,str(SCRIPT),RUN]
        state.update(phase='submitting',command=cmd);atomic_json(RECORD,state)
        try:
            result=subprocess.run(cmd,env={**os.environ,'CBS_CONFIG_PATH':str(HERE/'config.json')},capture_output=True,text=True,timeout=45)
            job=result.stdout.strip().split(';')[0]
            if result.returncode or not re.fullmatch(r'\d+',job):
                raise RuntimeError('Submission uncertain; retain reservation: '+result.stderr)
            state.update(phase='accepted',job_id=job);atomic_json(RECORD,state)
            ledger.bind(reservation,job);state['phase']='submitted';atomic_json(RECORD,state)
            print(json.dumps(state,indent=2))
        except Exception as exc:
            state.update(error=repr(exc),recovery_required=True);atomic_json(RECORD,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','check','submit','status']);a=p.parse_args()
    if a.action=='seal':
        assert not RECORD.exists();config();atomic_json(HERE/'source-seal.json',dict(configuration_digest=CONFIG_DIGEST,sha256=inventory()))
    elif a.action=='check':verify_source_seal();print('Formal20 source/config seal verified')
    elif a.action=='submit':submit()
    else:print(json.dumps(read_json(RECORD),indent=2))
