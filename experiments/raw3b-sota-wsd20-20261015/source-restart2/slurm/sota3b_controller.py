#!/usr/bin/env python3
"""One fresh formal11: SOTA raw3B with20% WSD tails."""
import datetime as dt
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import uuid
SOURCE=Path(__file__).resolve().parents[1];LIVE=SOURCE.parent
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,digest,file_digest,locked,load_config,read_json
from slurm.submission import find_submission_job
from wsd20_contracts import authorized,RUN_ID
STATE=LIVE/'restart2-controller-state.json';MARKER='# raw3b-sota-wsd20-20261015-controller';PREP='dolma-raw-baseline-20261008';CONFIG=LIVE/'config.json'


def owner():
    p=read_json(LIVE/'restart2-submission-owner.json')
    return p['hostname']==socket.gethostname().split('.')[0] and p['script']==str(Path(__file__).resolve()) and p['sha256']==file_digest(__file__)


def check():
    recovery_seal=read_json(LIVE/'restart2-source-seal.json')
    for name in ('restart2-authorization.json','restart2-protocol.md'):
        if file_digest(LIVE/name)!=recovery_seal['recovery_inputs'][name]:raise ValueError('Recovery approval/protocol changed')
    approval=read_json(LIVE/'restart2-authorization.json')
    if approval['run_id']!=RUN_ID or approval['maximum_restart_seconds']!=21540 or type(approval.get('approved')) is not bool:raise ValueError('Wrong restart approval')
    for f,h in read_json(LIVE/'restart2-source-seal.json')['files'].items():
        if file_digest(SOURCE/f)!=h:raise ValueError('Sealed source changed: '+f)
    plan=read_json(LIVE/'plan.json')
    if file_digest(LIVE/'plan.json')!=read_json(LIVE/'restart2-source-seal.json')['plan_sha256']:
        raise ValueError('Plan changed')
    for f,h in plan['immutable_inputs'].items():
        if file_digest(f)!=h:raise ValueError('Immutable input changed: '+f)
    if not authorized(RUN_ID,load_config(CONFIG)):raise ValueError('Exact formal11 authorization changed')
    ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    full={k for k,v in ledger['runs'].items() if v['category']=='full'}
    if full-{RUN_ID}!=set(plan['previous_formal_runs']):raise ValueError('Historical formal identities changed')
    for run,expected in plan['historical_run_digests'].items():
        if ledger['runs'].get(run,{}).get('config_digest')!=expected:raise ValueError('Historical run lost/changed')
    for job,cost in plan['historical_job_costs'].items():
        if job not in ledger['jobs'] or (cost is not None and ledger['jobs'][job].get('h200_hours')!=cost):
            raise ValueError('Historical allocation/cost lost or changed')
    progress=read_json(DEFAULT_ROOT/'preprocessed'/PREP/'raw/progress.json')
    prefix=read_json(LIVE/'raw-prefix.json')
    if progress['train_files'][:len(prefix['files'])]!=prefix['files']:raise ValueError('Raw consumed prefix changed')
    if sum(f['rows'] for f in prefix['files'])<prefix['required_rows']:raise ValueError('Raw3B buffer incomplete')


def training_check(run_id,path):
    cfg=load_config(path);s=read_json(DEFAULT_ROOT/'runs/experiments'/run_id/'summary.json')
    if (not authorized(run_id,cfg) or s['configuration']!=cfg or s['global_step']!=22889 or s['attempted_updates']!=22889 or s['total_tokens_seen']!=3000107008 or
        s['budget_tokens_charged']!=3000107008 or s['checkpoint_loaded_for_training'] or s['initialization']['pretrained_weights_loaded'] or
        s['stop_reason']!='training_token_target' or s['export_global_step']!=22889 or not math.isfinite(s['last_evaluation']['eval/perplexity'])):
        raise ValueError('Incomplete fresh full-SOTA3B run; no retry/resume')
    return s


def receipt(run_id,path):
    check();s=training_check(run_id,path);run=DEFAULT_ROOT/'runs/experiments'/RUN_ID;full=read_json(run/'full-evaluation.json');control=Path(read_json(LIVE/'plan.json')['control_run'])
    if s['data_provenance']!=read_json(control/'data-provenance.json'):raise ValueError('Two3B jobs use different data/eval provenance')
    if s['initialization']['sample_signature']!=read_json(control/'initialization.json')['sample_signature']:raise ValueError('Matched random initialization differs')
    if full['eval/documents']!=50000 or full['eval/prediction_tokens']!=76930208 or full['global_step']!=22889 or not math.isfinite(full['eval/perplexity']):raise ValueError('Incomplete full evaluation')
    groups=read_json(run/'optimizer-groups.json');old_groups=read_json(control/'optimizer-groups.json')
    for key in ('parameter_names','parameter_counts','tied_aliases','muon_configuration'):
        if groups[key]!=old_groups[key]:raise ValueError('SOTA optimizer groups changed')
    baseline=read_json(control/'full-evaluation.json')['eval/perplexity']
    comparison=dict(control_run=control.name,control_full_perplexity=baseline,
        candidate_full_perplexity=full['eval/perplexity'],
        absolute_ppl_change=full['eval/perplexity']-baseline,
        relative_ppl_change_percent=(full['eval/perplexity']/baseline-1)*100,
        improved=full['eval/perplexity']<baseline)
    atomic_json(LIVE/'final-comparison.json',comparison)
    job=os.environ['SLURM_JOB_ID']
    for rank in range(8):
        if read_json(DEFAULT_ROOT/'logs'/('job-'+job)/f'worker-rank-{rank}-exit.json')['exit_code']!=0:raise ValueError('Rank failure')
    atomic_json(LIVE/'receipt.json',dict(job_id=job,run_id=run_id,tokens=s['total_tokens_seen'],subset_perplexity=s['last_evaluation']['eval/perplexity'],full_perplexity=full['eval/perplexity'],checkpoint=s['checkpoint'],export=s['export'],wandb_url=s['wandb_url']))


def submit(s):
    if not owner():raise ValueError('Wrong submission owner')
    check()
    if 'job' in s or (LIVE/'disabled').exists():return
    if read_json(LIVE/'restart2-authorization.json').get('approved') is not True:raise ValueError('Minute-based restart awaits explicit user approval')
    from wsd20_contracts import fresh_restart_authorized
    old_ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    previous=[j for j in old_ledger['jobs'].values() if j['run_id']==RUN_ID]
    if not fresh_restart_authorized(RUN_ID,load_config(CONFIG),previous,DEFAULT_ROOT/'runs/experiments'/RUN_ID):raise ValueError('Approved fresh restart no longer eligible')
    tag=RUN_ID+'-'+uuid.uuid4().hex;ledger=ExperimentLedger(DEFAULT_ROOT);cfg=load_config(CONFIG)
    ledger.reserve(tag,RUN_ID,cfg,21540,8,False,fresh_retry=True)
    command=['sbatch','--parsable','--time=359','--comment='+tag,'--exclude='+','.join(read_json(LIVE/'plan.json')['excluded_nodes']),str(SOURCE/'slurm/sota3b.sbatch'),RUN_ID,str(CONFIG),PREP]
    j=dict(status='submission_uncertain',reservation=tag,submission_tag=tag,submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),submission_command=command)
    s['job']=j;atomic_json(STATE,s)
    with ledger.transaction() as d:d['jobs'][tag].update(submission_tag=tag,submission_started_utc=j['submission_started_utc'],submission_command=command,source_root=str(SOURCE),allocation_kind=RUN_ID,authorization='Explicitly approved whole-minute21540s correction after zero-token failures553561/553577;formal11 and all costs preserved,combined<=21600s/48H,no checkpoint load or further automatic retry')
    run_path=DEFAULT_ROOT/'runs/experiments'/RUN_ID
    archive=DEFAULT_ROOT/'runs/failed-startups'/RUN_ID/'job-553561'
    if run_path.exists():
        approval=read_json(LIVE/'restart2-authorization.json')
        if {p.name:file_digest(p) for p in run_path.iterdir()}!=approval['failed_metadata_files']:raise ValueError('Failed startup metadata changed')
        if archive.exists():raise ValueError('Refuse to overwrite failed-startup archive')
        archive.parent.mkdir(parents=True,exist_ok=True)
        run_path.rename(archive)
        atomic_json(LIVE/'failed-startup-archive.json',dict(failed_job='553561',archive=str(archive),files=approval['failed_metadata_files']))
    env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT),'SOTA3B_CONTROLLER':str(Path(__file__).resolve())}
    try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired) as e:j['submission_error']=str(e);atomic_json(STATE,s);return
    j['submission_receipt']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);job=r.stdout.strip().split(';')[0]
    if r.returncode==0 and job.isdigit():
        with ledger.transaction() as d:d['jobs'][tag]['returned_job_id']=job
        ledger.bind(tag,job);j.update(job_id=job,returned_job_id=job,status='submitted')
    atomic_json(STATE,s)


def report(s):
    atomic_json(STATE,s)
    (LIVE/'report.md').write_text('# SOTA raw3B with20% WSD tails\n\n'+json.dumps(s,indent=2)+'\n')


def run_once():
    if not owner():return
    try:
        with locked(LIVE/'.controller.lock',blocking=False):
            s=read_json(STATE) if STATE.exists() else dict(stage='training_restart2')
            try:
                check()
                if s.get('blocked'):return
                if 'job' not in s:submit(s);report(s);return
                j=s['job']
                if j['status']=='reconciled':retire();return
                ledger=ExperimentLedger(DEFAULT_ROOT)
                if not j.get('job_id'):
                    try:job=find_submission_job(j,DEFAULT_ROOT)
                    except (ValueError,RuntimeError) as error:
                        j['recovery_deferred']=str(error);report(s);return
                    if j['reservation'] in read_json(ledger.path)['jobs']:ledger.bind(j['reservation'],job)
                    j.update(job_id=job,status='submitted')
                r=subprocess.run(['sacct','-X','-j',j['job_id'],'-n','-P','-o','JobIDRaw,State'],capture_output=True,text=True,timeout=30,check=True)
                rows=[x.split('|')[1].split()[0].rstrip('+') for x in r.stdout.splitlines() if x.split('|')[0]==j['job_id']];j['slurm_state']=rows[0] if rows else 'UNKNOWN'
                if j['slurm_state'] in TERMINAL:
                    j.update(accounting=ledger.reconcile(j['job_id']),status='reconciled')
                    if j['slurm_state']=='COMPLETED' and (LIVE/'receipt.json').exists():s.update(stage='complete',result=read_json(LIVE/'receipt.json'))
                    else:s.update(stage='incomplete_no_retry',blocked=True)
                report(s)
                if j['status']=='reconciled':retire()
            except subprocess.TimeoutExpired as e:s['scheduler_deferred']=str(e);report(s)
            except Exception as e:s.update(blocked=True,error=str(e));report(s);raise
    except BlockingIOError:return


def retire():
    r=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if r.returncode not in (0,1):raise ValueError('Cannot read cron')
    rows=[x for x in r.stdout.splitlines() if not x.rstrip().endswith(MARKER)]
    if len(rows)!=len(r.stdout.splitlines()):
        subprocess.run(['crontab','-'],input='\n'.join(rows)+'\n',text=True,check=True)


def install():
    check()
    if not owner():raise ValueError('Wrong owner')
    r=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if r.returncode not in (0,1):raise ValueError('Cannot read cron')
    rows=[x for x in r.stdout.splitlines() if not x.rstrip().endswith(MARKER)]
    log=DEFAULT_ROOT/'runs'/RUN_ID/'controller.log';log.parent.mkdir(parents=True,exist_ok=True)
    (DEFAULT_ROOT/'logs').mkdir(parents=True,exist_ok=True)
    rows.append('* * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    subprocess.run(['crontab','-'],input='\n'.join(rows)+'\n',text=True,check=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='check':check()
    elif action=='install':install()
    elif action=='run-once':run_once()
    elif action=='training-check':training_check(sys.argv[2],sys.argv[3])
    elif action=='receipt':receipt(sys.argv[2],sys.argv[3])
    else:raise ValueError('Unknown action')
