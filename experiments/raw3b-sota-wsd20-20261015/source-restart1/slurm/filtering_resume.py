#!/usr/bin/env python3
"""Metadata-only submission owner for the authorized select -> pack -> verify recovery."""
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_state import PROJECT,DEFAULT_ROOT,atomic_json,read_json,locked,file_digest,TERMINAL
from slurm.submit_preprocessing import submit,recover

PREP='dolma-dclm10-005'
STAGES=('select','pack','verify')
MARKER='# codex-filtering-batch-reader-v1'
SOURCES=('filtering_batch_reader.py','filtering_reader.py','slurm/filtering_entry.py',
         'slurm/filtering_batch_checks.py','slurm/filtering_batch_fixture.py',
         'slurm/prepare.sbatch','slurm/common.sh','slurm/filtering_resume.py',
         'filtering_parallel.py','filtering_resources.py','slurm/filtering_parallel_checks.py',
         'slurm/submit_preprocessing.py','packing_parallel.py','verification_parallel.py',
         'slurm/packing_entry.py','slurm/packing_checks.py','tests/test_packing_parallel.py')


def timestamp():return dt.datetime.now(dt.timezone.utc).isoformat()
def directory(root):return root/'preprocessed'/PREP/'filtering-access/batch-reader-v1'
def hashes():return {p:file_digest(PROJECT/p) for p in SOURCES}


def shared_upper(root):
    training=read_json(root/'experiment-ledger.json')
    pre=read_json(root/'preprocessing-ledger.json')
    state=read_json(PROJECT/'research-state.yaml')
    reservations={}
    for item in state.get('submission_intents',[]):
        if (item.get('ledger')=='preprocessing' and item.get('status')!='not_dispatched'
                and item.get('job_id') not in pre['jobs']):
            reservations[item['tag']]=item['reserved_h200_hours']
    return training['total_h200_hours_reserved_or_measured']+pre['total_h200_hours']+sum(reservations.values())


def validate_seal(root,state):
    if hashes()!=state['source_hashes']:raise ValueError('Recovery source changed; no further automatic submissions')
    for path,expected in state['immutable_metadata_sha256'].items():
        if file_digest(Path(path))!=expected:raise ValueError('Original preprocessing manifest identity changed')


def tick(root,state):
    path=directory(root)/'controller.json'
    def save():state['updated_utc']=timestamp();atomic_json(path,state)
    # Recover a submission made before a controller crash. Never dispatch twice.
    if state.get('pending'):
        pending=state['pending']; intents=read_json(PROJECT/'research-state.yaml')['submission_intents']
        matches=[i for i in intents if i.get('prep_id')==PREP and i.get('stage')==pending['stage']
                 and i.get('submission_started_utc','')>=pending['started_utc'] and i.get('status')!='not_dispatched']
        if len(matches)!=1:raise ValueError('Uncertain submission requires recovery; keep reservation')
        item=matches[0];job=item.get('job_id') or recover(item['tag'],root,PROJECT/'research-state.yaml')
        state['jobs'].append(dict(job_id=job,stage=pending['stage'],status='submitted'))
        state['pending']=None;save()
    for job in state['jobs']:
        if job['status']=='reconciled':continue
        result=subprocess.run(['sacct','-X','-j',job['job_id'],'--noheader','--parsable2','--format=State%30'],
                              capture_output=True,text=True,check=True,timeout=20)
        status=result.stdout.strip().split('|')[0].split()[0] if result.stdout.strip() else 'UNKNOWN'
        if status not in TERMINAL:
            state['active_job']=dict(job_id=job['job_id'],stage=job['stage'],state=status);save();return
        subprocess.run([sys.executable,str(PROJECT/'slurm/accounting.py'),job['job_id'],'--root',str(root)],
                       check=True,capture_output=True,text=True,timeout=30)
        measured=read_json(root/'preprocessing-ledger.json')['jobs'][job['job_id']]
        exit_path=root/'logs'/('job-'+job['job_id'])/'exit-code.txt'
        exit_code=int(exit_path.read_text()) if exit_path.exists() else None
        job.update(status='reconciled',state=status,exit_code=exit_code,h200_hours=measured['h200_hours'])
        state['active_job']=None;save()
        if status!='COMPLETED' and exit_code!=75:
            state.update(status='blocked',reason=f"{job['stage']} {job['job_id']} failed (exit {exit_code}); no blind retry")
            save();return
    if state['status']!='active':save();return
    validate_seal(root,state)
    prepath=root/'preprocessed'/PREP
    stage=next((s for s in STAGES if not (prepath/s/'manifest.json').exists()),None)
    if stage is None:state.update(status='complete',reason='select, pack and full verify completed');save();return
    if shared_upper(root)+16>192:raise ValueError('Insufficient shared budget for another 16-hour preprocessing reservation')
    cpus=8;extra=[]
    if stage=='select' and state.get('training_workers_override') in (11,15):
        selected=read_json(prepath/'select/state.json')
        configuration=read_json(prepath/'configuration.json')['config']
        if configuration['max_training_tokens']-selected['train_tokens']>=2:
            cpus=state['training_workers_override']+1;extra=['--training-workers',str(cpus-1)]
    if stage=='select' and state.get('evaluation_workers_override')==11:
        if cpus>12:raise ValueError('Evaluation override requires a12CPU allocation')
        cpus=12;extra+=['--evaluation-workers','11']
    state['pending']=dict(stage=stage,started_utc=timestamp(),allocated_cpus=cpus,extra=extra);save()
    job=submit(stage,PREP,extra,root,PROJECT/'research-state.yaml',minutes=960,cpus=cpus,
               after_job=state['jobs'][-1]['job_id'] if state['jobs'] else '518703')
    state['jobs'].append(dict(job_id=job,stage=stage,status='submitted',allocated_cpus=cpus));state['pending']=None;save()
    print(f'Filtering {stage} submitted as {job}',flush=True)


def cron(root,enabled):
    result=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if result.returncode not in (0,1):raise RuntimeError(result.stderr)
    lines=[line for line in result.stdout.splitlines() if MARKER not in line]
    if enabled:
        lines.append(f'* * * * * /usr/bin/python3 {PROJECT}/slurm/filtering_resume.py tick --root {root} >> {directory(root)}/controller.log 2>&1 {MARKER}')
    subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('install','tick','status','disable'))
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to(Path('/work/asciibase64')):raise ValueError('Require /work storage')
    target=directory(root);target.mkdir(exist_ok=True);path=target/'controller.json'
    if args.action=='status':print(json.dumps(read_json(path),indent=2));return
    with locked(target/'controller.lock',blocking=False):
        if args.action=='install':
            if path.exists():raise ValueError('Controller already installed; use tick or status')
            paused=read_json(target/'paused-prefix.json')
            state=dict(status='active',jobs=[],pending=None,installed_utc=timestamp(),source_hashes=hashes(),
                       immutable_metadata_sha256={p:h for p,h in paused['metadata_sha256'].items() if not p.endswith('/select/state.json')},
                       original_checkpoint_sha256=paused['metadata_sha256'][str(root/'preprocessed'/PREP/'select/state.json')],
                       shared_h200_hours_limit=192,preparation_id=PREP)
            if file_digest(root/'preprocessed'/PREP/'select/state.json')!=state['original_checkpoint_sha256']:
                raise ValueError('Paused checkpoint changed before installation')
            validate_seal(root,state);atomic_json(path,state);cron(root,True)
        else:state=read_json(path)
        if args.action=='disable':state['status']='disabled';atomic_json(path,state)
        try:tick(root,state)
        except Exception as error:
            state.update(status='blocked',reason=str(error),updated_utc=timestamp());atomic_json(path,state);print(str(error),file=sys.stderr)
            if not any(j['status']!='reconciled' for j in state['jobs']) and not state.get('pending'):cron(root,False)
            raise
        if state['status']!='active' and all(j['status']=='reconciled' for j in state['jobs']) and not state.get('pending'):
            cron(root,False)
        print(json.dumps({'status':state['status'],'active_job':state.get('active_job'),'reason':state.get('reason')}))


if __name__=='__main__':main()
