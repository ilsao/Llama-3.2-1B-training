"""Prepare a reviewable correction; never installs cron or submits Slurm jobs."""
from pathlib import Path
import json
import socket
import sys

SOURCE=Path(__file__).resolve().parent
LIVE=SOURCE.parent
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,atomic_json,file_digest,read_json


def prepare():
    old=read_json(LIVE/'controller-state.json')
    failed=old['jobs']['validation']
    if failed['status']!='reconciled' or failed['accounting']['state']!='FAILED':
        raise ValueError('Reconcile the original failed validation first')
    revised_state=LIVE/'controller-state-r1.json'
    if revised_state.exists() and any(name not in ('setup','validation-original') for name in read_json(revised_state)['jobs']):
        raise ValueError('Cannot reseal after corrected allocation submission')
    tests=LIVE/'tests-r1';tests.mkdir(exist_ok=True)
    for p in (LIVE/'tests').glob('test_*.py'):
        text=p.read_text().replace("LIVE/'source","LIVE/'source-r1")
        if p.name=='test_controller.py':
            text=text.replace("atomic_json(live/'plan.json',dict(excluded_nodes=[]))",
                "atomic_json(live/'plan.json',dict(excluded_nodes=[]))\n            atomic_json(live/'validation-r1-authorization.json',dict(approved=True))")
        (tests/p.name).write_text(text)
    # Never replace an approved decision with an unapproved default.
    approval=LIVE/'validation-r1-authorization.json'
    if not approval.exists():
        atomic_json(approval,dict(approved=False,run_id='batch-normuon-1b-20261016-validation-r1',
            gpu_count=8,cpu_count=8,max_seconds=780,original_validation_seconds=93,
            combined_validation_seconds=873,campaign_cap_h200_hours=146.5,
            fresh_correctness_validation=True,automatic_retry=False))
    if not revised_state.exists():
        atomic_json(revised_state,dict(stage='awaiting_corrected_validation_approval',jobs={
            'setup':old['jobs']['setup'],'validation-original':failed},blocked_lanes={}))
    validation=read_json(LIVE/'configs/validation.json')
    validation['max_h200_hours']=780*8/3600
    validation['sweep_stage']='validation-r1'
    atomic_json(LIVE/'configs/validation-r1.json',validation)
    ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    plan=read_json(LIVE/'plan.json')
    plan.update(historical_jobs={k:v.get('h200_hours') for k,v in ledger['jobs'].items()},
                historical_run_digests={k:v['config_digest'] for k,v in ledger['runs'].items()},
                corrected_validation_seconds=780,prior_validation_seconds=93,
                correction_protocol=str(LIVE/'validation-r1-protocol.md'))
    atomic_json(LIVE/'plan-r1.json',plan)
    controller=SOURCE/'slurm/batch1b_controller.py'
    atomic_json(LIVE/'submission-owner-r1.json',dict(hostname=socket.gethostname().split('.')[0],
                script=str(controller),sha256=file_digest(controller)))
    original_seal=read_json(LIVE/'source-seal.json')
    inputs={name:file_digest(name) for name in original_seal['inputs']}
    for path in [LIVE/'plan-r1.json',LIVE/'submission-owner-r1.json',LIVE/'validation-r1-protocol.md',
                 approval,LIVE/'configs/validation-r1.json',*tests.glob('*.py')]:
        inputs[str(path)]=file_digest(path)
    files={str(p.relative_to(SOURCE)):file_digest(p) for p in SOURCE.rglob('*') if p.is_file()
           and '__pycache__' not in p.parts and p.suffix in ('.py','.txt','.json','.sh','.sbatch')}
    atomic_json(LIVE/'source-seal-r1.json',dict(files=files,inputs=inputs))
    journal=DEFAULT_ROOT/'runs/experiments/batch-normuon-1b-20261016-validation/token-journal.json'
    atomic_json(LIVE/'failure-554398.json',dict(accounting=failed['accounting'],
        token_journal=read_json(journal),original_source=str(LIVE/'source'),
        maximum_observed_postupdate_weight_difference=4.699453711509705e-6,
        original_tolerance=dict(rtol=1e-5,atol=1e-7),no_endpoints_launched=True))
    print('Prepared correction; approved=',read_json(approval)['approved'],'no submissions performed')


if __name__=='__main__':prepare()
