"""Explicitly authorized, one-allocation eager Dion retry; no training resume."""
import argparse,copy,datetime,importlib.util,json,re,subprocess,sys,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SOURCE=HERE/'source-dion-retry1';RECORD=HERE/'dion-retry1';SCRIPT=REPO/'slurm/methods-dion-retry.sbatch'
spec=importlib.util.spec_from_file_location('original_launcher',REPO/'slurm/methods_parallel.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from training_state import ExperimentLedger,atomic_json,read_json,file_digest,digest,load_config,locked
from submission import find_submission_job
OLD={'D1-seed42':'572102','D025-seed42':'572113'}
SECONDS=7080

def inventory():
 paths=[p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
 paths += [Path(__file__).resolve(),SCRIPT,HERE/'configs/D1-seed42.json',HERE/'configs/D025-seed42.json']
 return {str(p.relative_to(REPO)):file_digest(p) for p in sorted(paths)}

def verify():
 matrix=base.verify();seal=read_json(RECORD/'execution-seal.json')
 if seal['sha256']!=inventory():raise ValueError('Retry source seal changed')
 if seal['original_execution_seal_sha256']!=file_digest(HERE/'execution-seal.json'):
  raise ValueError('Original source seal changed')
 return [r for r in matrix['jobs'] if r['key'] in OLD]

def command(row,tag):
 return ['sbatch','--parsable','--comment='+tag,'--job-name=mp-'+row['key'],
         '--export=ALL,TRAIN_SOURCE_ROOT='+str(SOURCE)+',TORCH_COMPILE_DISABLE=1',
         str(SCRIPT),row['run_id'],row['key']]

def validate_ready(row,data,state):
 oldjob=OLD[row['key']]
 if state.get('returned_job_id')!=oldjob:raise ValueError('Retry already attempted or unexpected latest job')
 if (base.DEFAULT_ROOT/'runs/experiments'/row['run_id']).exists():raise ValueError('Training artifacts already exist; no fresh retry')
 prior={k:v for k,v in data['jobs'].items() if v['run_id']==row['run_id']}
 expected={'572036','572065','572102'} if row['key']=='D1-seed42' else {'572067','572113'}
 if set(prior)!=expected or any('h200_hours' not in j for j in prior.values()):
  raise ValueError('Unexpected or unreconciled allocation history')
 if prior[oldjob]['state']!='FAILED':raise ValueError('Original gate failure is not terminal')
 cfg=load_config(HERE/row['config'])
 if data['runs'][row['run_id']]['config_digest']!=digest(cfg):raise ValueError('Configuration changed')
 if sum(j['h200_hours'] for j in prior.values())+8*SECONDS/3600>16:raise ValueError('Original 16H cap exceeded')
 return cfg

def submit():
 rows=verify();auth=read_json(RECORD/'authorization.json')
 if auth.get('user_instruction')!='重試失敗的' or auth['execution_seal_sha256']!=file_digest(RECORD/'execution-seal.json'):
  raise ValueError('Retry authorization does not match sealed execution')
 ledger=ExperimentLedger(base.DEFAULT_ROOT)
 with locked(base.RECORDS/'.submission.lock',blocking=False):
  if (RECORD/'submission-started.json').exists():raise ValueError('Already attempted; recover existing submissions, no new retries')
  data=read_json(ledger.path)
  configs={r['key']:validate_ready(r,data,read_json(base.RECORDS/(r['key']+'.json'))) for r in rows}
  study_ids={r['run_id'] for r in read_json(HERE/'matrix.json')['jobs']}
  existing=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in data['jobs'].values() if j['run_id'] in study_ids)
  if existing+2*8*SECONDS/3600>128:raise ValueError('Study cap exceeded')
  atomic_json(RECORD/'submission-started.json',dict(authorization_sha256=file_digest(RECORD/'authorization.json'),
              maximum_study_h200_hours=existing+2*8*SECONDS/3600,keys=list(OLD)))
  for row in rows:
   cfg=configs[row['key']];path=base.RECORDS/(row['key']+'.json');old=read_json(path)
   tag='mp-'+uuid.uuid4().hex
   state=dict(key=row['key'],run_id=row['run_id'],reservation=tag,submission_tag=tag,
    submission_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    config_digest=digest(cfg),seal_sha256=file_digest(RECORD/'execution-seal.json'),source_root=str(SOURCE),
    requested_seconds=SECONDS,gpu_count=8,cpu_count=8,phase='retry_intent',prior_submission_attempts=[copy.deepcopy(old)],
    retry_authorization_sha256=file_digest(RECORD/'authorization.json'),accounting_continuation_only=True,
    training_checkpoint_load=False,automatic_retry=False,submission_command=command(row,tag))
   atomic_json(path,state)
   # Preserve costs and identity through the ledger API; training has no --resume.
   ledger.reserve(tag,row['run_id'],cfg,SECONDS,8,True)
   with ledger.transaction() as data:
    data['jobs'][tag].update(accounting_continuation_only=True,training_checkpoint_load=False,
         retry_for=OLD[row['key']],authorization=str(RECORD/'authorization.json'),source_root=str(SOURCE))
   state['phase']='reserved';atomic_json(path,state)
   try:
    try:
     result=subprocess.run(state['submission_command'],capture_output=True,text=True,timeout=45)
     state['submission_receipt']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
     atomic_json(path,state);job=result.stdout.strip().split(';')[0]
     if result.returncode or not re.fullmatch(r'\d+',job):raise RuntimeError('Uncertain sbatch result')
    except (subprocess.TimeoutExpired,RuntimeError) as error:
     state.update(submission_error=str(error),recovery_required=True);atomic_json(path,state)
     # Discovery only: never issue sbatch twice after uncertain acknowledgement.
     job=find_submission_job(state,base.DEFAULT_ROOT)
    state.update(returned_job_id=job,phase='accepted');atomic_json(path,state)
    ledger.bind(tag,job)
    state.update(phase='submitted',recovery_required=False);atomic_json(path,state)
    print(json.dumps(dict(key=row['key'],job_id=job,requested_seconds=SECONDS,
       previous_jobs=old.get('returned_job_id'),source_root=str(SOURCE))),flush=True)
   except (OSError,subprocess.TimeoutExpired,RuntimeError,ValueError) as error:
    state.update(error=str(error),recovery_required=True);atomic_json(path,state);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','check','submit']);p.add_argument('--key');args=p.parse_args()
 if args.action=='seal':
  RECORD.mkdir(exist_ok=True)
  atomic_json(RECORD/'execution-seal.json',dict(original_execution_seal_sha256=file_digest(HERE/'execution-seal.json'),sha256=inventory()))
  print('Sealed isolated retry source; no allocation')
 elif args.action=='check':
  rows=verify()
  if args.key:rows=[r for r in rows if r['key']==args.key]
  if not rows:raise ValueError('Not a Dion retry key')
  print(json.dumps(dict(keys=[r['key'] for r in rows],source_root=str(SOURCE),seconds=SECONDS,
     torch_compile_disable=True,training_checkpoint_load=False)))
 else:submit()
