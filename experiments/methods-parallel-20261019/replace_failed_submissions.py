"""Review/submit one bounded replacement for six recorded pre-script failures.

Submission requires explicit replacement-authorization.json tied to this plan.
Existing successful jobs and sealed training source are never modified.
"""
import argparse,copy,datetime,importlib.util,json,re,subprocess,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('methods_parallel',HERE.parents[1]/'slurm/methods_parallel.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
from training_state import ExperimentLedger,atomic_json,read_json,digest,file_digest,load_config,locked
OLD={'D1-seed42':'572065','D025-seed42':'572067','K1-seed42':'572070',
     'F1-seed42':'572071','F2-seed42':'572072','C1-seed42':'572073'}
SECONDS=7140
PLAN=HERE/'startup-failures/replacement-plan.json'
AUTH=HERE/'startup-failures/replacement-authorization.json'

def plan(keys=None):
    matrix=cli.verify();ledger=ExperimentLedger(cli.DEFAULT_ROOT);data=read_json(ledger.path)
    rows=[r for r in matrix['jobs'] if r['key'] in OLD and (keys is None or r['key'] in keys)]
    items=[]
    for row in rows:
        oldjob=OLD[row['key']];record=cli.RECORDS/(row['key']+'.json');state=read_json(record)
        if state.get('returned_job_id')!=oldjob or state.get('phase')!='failed_before_script':
            raise ValueError('Original failed submission was already changed')
        previous={k:v for k,v in data['jobs'].items() if v['run_id']==row['run_id']}
        expected={oldjob,'572036'} if row['key']=='D1-seed42' else {oldjob}
        if set(previous)!=expected:raise ValueError('Additional allocation or reservation exists')
        for job,item in previous.items():
            if 'h200_hours' not in item:raise ValueError('Unreconciled old allocation')
            if job=='572036':
                if not item['state'].startswith('CANCELLED') or item['h200_hours']!=0:raise ValueError('Held job changed')
            elif item['state']!='FAILED' or item['exit_code']!='0:53' or item['nodes']!='25a-hgpn001' or not 1<=item['elapsed_seconds']<=6:
                raise ValueError('Not a reviewed node-startup failure')
        if (cli.DEFAULT_ROOT/'runs/experiments'/row['run_id']).exists():raise ValueError('Run initialization already exists')
        cfg=load_config(HERE/row['config'])
        if state['config_digest']!=digest(cfg):raise ValueError('Configuration changed')
        cost=sum(v['h200_hours'] for v in previous.values())
        if cost+SECONDS*8/3600>16:raise ValueError('Original logical run budget exceeded')
        command=cli.command(row,'<unique-ledger-reservation>')
        command[1:1]=['--time=01:59:00','--exclude=25a-hgpn001']
        items.append(dict(key=row['key'],run_id=row['run_id'],original_job_id=oldjob,
            config_digest=digest(cfg),previous_h200_hours=cost,requested_seconds=SECONDS,
            gpu_count=8,cpu_count=8,replacement_max_h200_hours=SECONDS*8/3600,
            cumulative_max_h200_hours=cost+SECONDS*8/3600,command=command,
            training_checkpoint_load=False,max_replacement_allocations=1))
    return dict(study=HERE.name,seal_sha256=file_digest(HERE/'execution-seal.json'),
        replacement_script_sha256=file_digest(Path(__file__).resolve()),
        reason='Six jobs terminated by signal 53 on node 25a-hgpn001 before any script records or run directory',
        unchanged_active_jobs={'A1-seed42':'572068','S1-seed42':'572069'},
        original_study_cap_h200_hours=128,
        study_max_h200_hours_with_replacements=32+sum(x['cumulative_max_h200_hours'] for x in items),
        new_allocation_cap_h200_hours=sum(x['replacement_max_h200_hours'] for x in items),
        source_and_config_unchanged=True,automatic_followup_retry=False,jobs=items)

def submit():
    proposed=read_json(PLAN);authorization=read_json(AUTH)
    if authorization.get('approved') is not True or authorization.get('plan_sha256')!=file_digest(PLAN) or not authorization.get('user_instruction'):
        raise ValueError('Explicit approval tied to this reviewed replacement plan is required')
    ledger=ExperimentLedger(cli.DEFAULT_ROOT)
    with locked(cli.RECORDS/'.submission.lock',blocking=False):
        # Continue only jobs whose approved replacement has never been submitted.
        # Accepted/recovered jobs remain bound and are never submitted again.
        data=read_json(ledger.path);pending=[];already=[]
        for item in proposed['jobs']:
            state=read_json(cli.RECORDS/(item['key']+'.json'))
            if state.get('replacement_authorization_sha256')==file_digest(AUTH):
                job=state.get('returned_job_id')
                if not job or job not in data['jobs']:raise ValueError('Recover uncertain prior submission before continuing')
                bound=data['jobs'][job]
                expected=[x.replace('<unique-ledger-reservation>',state['submission_tag']) for x in item['command']]
                if (bound['run_id']!=item['run_id'] or bound.get('replacement_for')!=item['original_job_id']
                    or bound['requested_seconds']!=SECONDS or state['submission_command']!=expected):
                    raise ValueError('Accepted replacement differs from approval')
                already.append(dict(key=item['key'],job_id=job));continue
            pending.append(item['key'])
        current=plan(pending)
        reviewed=[x for x in proposed['jobs'] if x['key'] in pending]
        if current['jobs']!=reviewed or current['seal_sha256']!=proposed['seal_sha256']:
            raise ValueError('Reviewed replacement configurations or source changed')
        receipt=HERE/'startup-failures/replacement-started.json'
        history=read_json(receipt).get('continuations',[]) if receipt.exists() else []
        history.append(dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            already_submitted=already,pending_keys=pending,submitter_sha256=file_digest(Path(__file__).resolve())))
        atomic_json(receipt,dict(plan_sha256=file_digest(PLAN),authorization=authorization,continuations=history))
        matrix=read_json(HERE/'matrix.json')
        for item in current['jobs']:
            row=next(r for r in matrix['jobs'] if r['key']==item['key'])
            record=cli.RECORDS/(row['key']+'.json');old=read_json(record)
            tag='mp-'+uuid.uuid4().hex
            command=cli.command(row,tag);command[1:1]=['--time=01:59:00','--exclude=25a-hgpn001']
            state=dict(key=row['key'],run_id=row['run_id'],reservation=tag,submission_tag=tag,
                submission_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                config_digest=item['config_digest'],seal_sha256=current['seal_sha256'],
                requested_seconds=SECONDS,gpu_count=8,cpu_count=8,phase='replacement_intent',
                submission_command=command,prior_submission_attempts=[copy.deepcopy(old)],
                replacement_authorization_sha256=file_digest(AUTH),accounting_continuation_only=True,
                training_checkpoint_load=False,automatic_retry=False)
            atomic_json(record,state)
            # Ledger continuation preserves original identity and all spent costs.
            # Training command remains fresh and has no --resume argument.
            ledger.reserve(tag,row['run_id'],load_config(HERE/row['config']),SECONDS,8,True)
            with ledger.transaction() as data:
                data['jobs'][tag].update(accounting_continuation_only=True,training_checkpoint_load=False,
                    replacement_for=item['original_job_id'],replacement_authorization=str(AUTH))
            state['phase']='reserved';atomic_json(record,state)
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=45)
                state['submission_receipt']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
                atomic_json(record,state);job=result.stdout.strip().split(';')[0]
                if result.returncode or not re.fullmatch(r'\d+',job):raise RuntimeError('Uncertain submission; retain reservation')
                state.update(returned_job_id=job,phase='accepted');atomic_json(record,state)
                ledger.bind(tag,job);state['phase']='submitted';atomic_json(record,state)
                print(json.dumps(state),flush=True)
            except (OSError,subprocess.TimeoutExpired,ValueError,RuntimeError) as error:
                state.update(error=str(error),recovery_required=True);atomic_json(record,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','submit']);args=p.parse_args()
    if args.action=='plan':
        value=plan();atomic_json(PLAN,value);print(json.dumps(value,indent=2))
    else:submit()
