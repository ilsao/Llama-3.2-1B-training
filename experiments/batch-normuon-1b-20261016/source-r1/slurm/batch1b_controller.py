#!/usr/bin/env python3
"""Sole submission owner for the user-approved nine-endpoint 1B study."""
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

SOURCE = Path(__file__).resolve().parents[1]
LIVE = SOURCE.parent
sys.path.insert(0, str(SOURCE))
from training_state import (DEFAULT_ROOT, ExperimentLedger, TERMINAL, atomic_json,
                            digest, file_digest, load_config, locked, read_json)
from batch1b_contracts import STUDY, TARGET, Q, PREP, count_updates, check_fork
from slurm.submission import find_submission_job

STATE = LIVE/'controller-state-r1.json'
VALIDATION = 'validation-r1'
MARKER = '# '+STUDY+'-controller'
PACKAGE = DEFAULT_ROOT/'environments'/STUDY/'muonium-0.2.0'


def owner():
    record = read_json(LIVE/'submission-owner-r1.json')
    return (record['hostname'] == socket.gethostname().split('.')[0] and
            record['script'] == str(Path(__file__).resolve()) and
            record['sha256'] == file_digest(__file__))


def check():
    seal = read_json(LIVE/'source-seal-r1.json')
    for name, sha in seal['files'].items():
        if file_digest(SOURCE/name) != sha:
            raise ValueError('Source seal changed: '+name)
    for name, sha in seal['inputs'].items():
        if file_digest(name) != sha:
            raise ValueError('Configuration/protocol/input changed: '+name)
    plan = read_json(LIVE/'plan-r1.json')
    ledger = read_json(DEFAULT_ROOT/'experiment-ledger.json')
    if not set(plan['historical_jobs']) <= set(ledger['jobs']):
        raise ValueError('Historical allocations disappeared')
    for name, sha in plan['historical_run_digests'].items():
        if ledger['runs'].get(name, {}).get('config_digest') != sha:
            raise ValueError('Historical logical run changed: '+name)
    for job, cost in plan['historical_jobs'].items():
        if cost is not None and ledger['jobs'][job].get('h200_hours') != cost:
            raise ValueError('Historical allocation cost changed: '+job)
    prefix = read_json(LIVE/'raw-prefix.json')
    progress = read_json(DEFAULT_ROOT/'preprocessed'/PREP/'raw/progress.json')
    if progress['train_files'][:len(prefix['files'])] != prefix['files']:
        raise ValueError('Immutable raw prefix changed')
    if sum(f['rows'] for f in prefix['files']) < TARGET//8192:
        raise ValueError('Insufficient published raw prefix')
    if not owner() and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Wrong host/hash submission owner')
    if (LIVE/'setup-result.json').exists():
        for name, sha in read_json(LIVE/'setup-result.json')['source_hashes'].items():
            if file_digest(PACKAGE/name) != sha:
                raise ValueError('Pinned muonium source changed')


def receipt(run_id, configuration):
    check()
    cfg = load_config(configuration)
    name = run_id.removeprefix(STUDY+'-')
    if read_json(LIVE/'configs'/f'{name}.json') != cfg or run_id != STUDY+'-'+name:
        raise ValueError('Unregistered endpoint configuration')
    run = DEFAULT_ROOT/'runs/experiments'/run_id
    summary = read_json(run/'summary.json')
    inherited = cfg.get('fork_snapshot_tokens', 0)
    expected_step = inherited//Q + count_updates(cfg, inherited)
    if (summary['configuration'] != cfg or summary['stop_reason'] != 'training_token_target' or
        summary['global_step'] != expected_step or summary['attempted_updates'] != expected_step or
        summary['total_tokens_seen'] != TARGET or summary['budget_tokens_charged'] != TARGET or
        summary['cursor'] != TARGET//8192 or
        summary['checkpoint_loaded_for_training'] != bool(inherited) or
        summary['initialization']['pretrained_weights_loaded'] or
        summary['export_global_step'] != expected_step or summary['export_tokens_seen'] != TARGET or
        not read_json(run/'export.json')['reload_verified']):
        raise ValueError('Incomplete or unmatched endpoint; no retry/resume')
    measured = summary['last_evaluation']
    if (measured['eval/documents'] != 1000 or measured['eval/prediction_tokens'] != 1479302 or
        not math.isfinite(measured['eval/perplexity'])):
        raise ValueError('Evaluation subset/token mismatch')
    for rank in range(8):
        if read_json(Path(os.environ['JOB_RECORD'])/f'worker-rank-{rank}-exit.json')['exit_code'] != 0:
            raise ValueError('Training rank did not complete')
    atomic_json(LIVE/'receipts'/f'{name}.json', dict(
        run_id=run_id, job_id=os.environ['SLURM_JOB_ID'], tokens=TARGET,
        global_step=expected_step, inherited_tokens=inherited,
        mean_nll=measured['eval/nll']/measured['eval/prediction_tokens'],
        perplexity=measured['eval/perplexity'], wandb_url=summary['wandb_url'],
        checkpoint=summary['checkpoint'], export=summary['export'],
        initialization_signature=summary['initialization']['sample_signature'],
        provenance_digest=digest(summary['data_provenance']),
        optimizer_groups=read_json(run/'optimizer-groups.json')))


def setup_receipt():
    from importlib.metadata import distributions
    package = [d for d in distributions(path=[str(PACKAGE)]) if d.metadata['Name'] == 'muonium']
    if len(package) != 1 or package[0].version != '0.2.0':
        raise ValueError('Pinned package installation missing')
    atomic_json(LIVE/'setup-result.json', dict(job_id=os.environ['SLURM_JOB_ID'],
        target=str(PACKAGE), version='0.2.0', source_hashes={
            str(p.relative_to(PACKAGE)):file_digest(p) for p in PACKAGE.rglob('*.py')}))


def submit(state, name, seconds, gpus, kind):
    check()
    if name in state['jobs'] or (LIVE/'disabled').exists():
        return
    authorization=read_json(LIVE/'validation-r1-authorization.json')
    if authorization.get('approved') is not True:
        raise ValueError('Corrected validation requires explicit approval; no automatic retry')
    if name==VALIDATION:
        old=read_json(LIVE/'controller-state.json')['jobs']['validation']['accounting']
        if (seconds!=780 or gpus!=8 or old['state']!='FAILED' or
            old['elapsed_seconds']+seconds>900):
            raise ValueError('Corrected validation exceeds original combined15-minute/2H allowance')
    ledger = ExperimentLedger(DEFAULT_ROOT)
    cfg_path = LIVE/'configs'/f'{name}.json'
    cfg = load_config(cfg_path)
    run_id = STUDY+'-'+name
    tag = run_id+'-'+uuid.uuid4().hex
    data = read_json(ledger.path)
    committed = sum(j.get('h200_hours', j['reserved_h200_hours'])
                    for j in data['jobs'].values() if j['run_id'].startswith(STUDY+'-'))
    if committed + seconds*gpus/3600 > 146.5 + 1e-9:
        raise ValueError('Study allocation cap exceeded')
    ledger.reserve(tag, run_id, cfg, seconds, gpus, False)
    script = 'batch1b-setup.sbatch' if kind == 'setup' else 'batch1b.sbatch'
    command = ['sbatch', '--parsable', '--job-name=b1b-'+name, '--time='+str(seconds//60),
               '--comment='+tag, '--exclude='+','.join(read_json(LIVE/'plan.json')['excluded_nodes']),
               str(SOURCE/'slurm'/script), run_id, str(cfg_path), kind]
    job = dict(status='submission_uncertain', reservation=tag, submission_tag=tag,
               submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               submission_command=command, kind=kind)
    state['jobs'][name] = job
    atomic_json(STATE, state)
    with ledger.transaction() as data:
        data['jobs'][tag].update(submission_tag=tag,
            submission_started_utc=job['submission_started_utc'], submission_command=command,
            source_root=str(SOURCE), study=STUDY,
            authorization='User approved complete nine-endpoint1B plan; bounded independent pilots and planned full-state forks only')
    env = {**os.environ, 'TRAIN_SOURCE_ROOT':str(SOURCE), 'DATA_ROOT':str(DEFAULT_ROOT),
           'BATCH1B_CONTROLLER':str(Path(__file__).resolve()), 'MUONIUM_TARGET':str(PACKAGE)}
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        job['submission_error'] = str(error)
        atomic_json(STATE, state)
        return
    job['submission_receipt'] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    job_id = result.stdout.strip().split(';')[0]
    if result.returncode == 0 and job_id.isdigit():
        with ledger.transaction() as data:
            data['jobs'][tag]['returned_job_id'] = job_id
        ledger.bind(tag, job_id)
        job.update(job_id=job_id, returned_job_id=job_id, status='submitted')
    atomic_json(STATE, state)


def reconcile(state):
    ledger = ExperimentLedger(DEFAULT_ROOT)
    for name, job in state['jobs'].items():
        if job['status'] == 'reconciled':
            continue
        if not job.get('job_id'):
            try:
                job_id = find_submission_job(job, DEFAULT_ROOT)
            except (ValueError, RuntimeError) as error:
                job['recovery_deferred'] = str(error)
                continue
            if job['reservation'] in read_json(ledger.path)['jobs']:
                ledger.bind(job['reservation'], job_id)
            job.update(job_id=job_id, status='submitted')
        result = subprocess.run(['sacct','-X','-j',job['job_id'],'-n','-P','-o','JobIDRaw,State'],
                                capture_output=True, text=True, timeout=30, check=True)
        rows = [r.split('|')[1].split()[0].rstrip('+') for r in result.stdout.splitlines()
                if r.split('|')[0] == job['job_id']]
        job['slurm_state'] = rows[0] if rows else 'UNKNOWN'
        if job['slurm_state'] in TERMINAL:
            job.update(accounting=ledger.reconcile(job['job_id']), status='reconciled')
        if job['kind'] not in ('setup', 'validation'):
                path = LIVE/'receipts'/f'{name}.json'
                job['completed_endpoint'] = job['slurm_state'] == 'COMPLETED' and path.exists()


def report(state):
    results = {p.stem:read_json(p) for p in (LIVE/'receipts').glob('*.json')}
    if 'control' in results:
        control = results['control']
        for name, row in results.items():
            if (row['initialization_signature'] != control['initialization_signature'] or
                row['provenance_digest'] != control['provenance_digest']):
                row['invalid_comparison'] = 'Initialization or evaluation/data provenance differs'
            row['delta_mean_nll_vs_control'] = row['mean_nll'] - control['mean_nll']
            row['delta_perplexity_vs_control'] = row['perplexity'] - control['perplexity']
    pairs = {}
    for key in ('50','predecay','70'):
        a, b = 'bs-'+key+'-fixed', 'bs-'+key+'-scaled'
        if a in results and b in results:
            pairs[key] = dict(delta_nll_scaled_minus_fixed=results[b]['mean_nll']-results[a]['mean_nll'],
                              delta_ppl_scaled_minus_fixed=results[b]['perplexity']-results[a]['perplexity'])
    atomic_json(LIVE/'results.json', dict(endpoints=results, paired_beta2=pairs,
                note='seed42 preliminary ranking; latest table LR is extrapolated to20%-tail resplit'))
    atomic_json(STATE, state)
    lines = ['# 1B batch / NorMuon / LR / beta2 study', '', 'Stage: '+state['stage'], '',
             '| Candidate | Job | State |', '|---|---|---|']
    for name, job in state['jobs'].items():
        lines.append(f"| {name} | {job.get('job_id','uncertain')} | {job.get('slurm_state',job['status'])} |")
    for name, reason in state.get('blocked_lanes', {}).items():
        lines.append(f'\n{name}: {reason}')
    if results:
        lines += ['', '| Endpoint | PPL | mean NLL | W&B |', '|---|---:|---:|---|']
        for name, row in results.items():
            lines.append(f"| {name} | {row['perplexity']:.8f} | {row['mean_nll']:.8f} | [run]({row['wandb_url']}) |")
    (LIVE/'report.md').write_text('\n'.join(lines)+'\n')


def run_once():
    if not owner():
        raise ValueError('Wrong submission owner')
    try:
        with locked(LIVE/'.controller.lock', blocking=False):
            state = read_json(STATE) if STATE.exists() else dict(stage='setup', jobs={}, blocked_lanes={})
            try:
                reconcile(state)  # Even disabled/failed seals retain accounting.
                check()
                if (LIVE/'disabled').exists():
                    state['stage'] = 'disabled_accounting_only'
                elif 'setup' not in state['jobs']:
                    submit(state, 'setup', 1800, 1, 'setup')
                elif state['jobs']['setup']['status'] == 'reconciled':
                    if VALIDATION not in state['jobs']:
                        submit(state, VALIDATION, 780, 8, 'validation')
                        state['stage'] = 'validation'
                    elif state['jobs'][VALIDATION]['status'] == 'reconciled':
                        validation = read_json(LIVE/'validation-result-r1.json') if (LIVE/'validation-result-r1.json').exists() else {}
                        if not validation.get('native_pass') or state['jobs'][VALIDATION]['slurm_state'] != 'COMPLETED':
                            state.update(stage='validation_failed_no_retry', blocked=True)
                        else:
                            state['stage'] = 'training_and_snapshot_forks'
                            for name in ('control', 'lr001', 'normuon'):
                                if name == 'normuon' and not validation.get('normuon_pass'):
                                    state['blocked_lanes'][name] = validation.get('normuon_error', 'NorMuon compatibility validation failed')
                                else:
                                    submit(state, name, 7200, 8, 'normuon' if name == 'normuon' else 'train')
                            control = state['jobs'].get('control', {})
                            for key in ('50','predecay','70'):
                                from batch1b_contracts import SWITCHES
                                tokens = SWITCHES[key]*Q
                                snapshot = DEFAULT_ROOT/'runs/experiments'/(STUDY+'-control')/'snapshots'/f'{tokens}.json'
                                for suffix in ('fixed','scaled'):
                                    name = 'bs-'+key+'-'+suffix
                                    if snapshot.exists():
                                        manifest = read_json(snapshot)
                                        if manifest['tokens'] != tokens:
                                            raise ValueError('Unexpected published checkpoint boundary')
                                        check_fork(read_json(LIVE/'configs/control.json'), read_json(LIVE/'configs'/f'{name}.json'), tokens)
                                        submit(state, name, 7200, 8, 'fork')
                                    elif control.get('status') == 'reconciled':
                                        state['blocked_lanes'][name] = 'Control ended without required snapshot; no retry'
                            expected = set(read_json(LIVE/'plan.json')['endpoints'])
                            finished = {name for name, job in state['jobs'].items() if job['status'] == 'reconciled'}
                            if expected <= finished | set(state['blocked_lanes']):
                                complete = all(state['jobs'].get(n,{}).get('completed_endpoint') for n in expected)
                                state['stage'] = 'complete' if complete else 'finished_with_incomplete_lanes'
                report(state)
                if state['stage'] in ('complete','finished_with_incomplete_lanes','validation_failed_no_retry'):
                    retire()
            except Exception as error:
                state.update(error=str(error), submission_blocked=True)
                report(state)
                raise
    except BlockingIOError:
        return


def retire():
    result = subprocess.run(['crontab','-l'], capture_output=True, text=True)
    if result.returncode not in (0,1):
        raise ValueError('Cannot read existing cron')
    rows = [line for line in result.stdout.splitlines() if not line.rstrip().endswith(MARKER)]
    if len(rows) != len(result.stdout.splitlines()):
        subprocess.run(['crontab','-'], input='\n'.join(rows)+'\n', text=True, check=True)


def install():
    check()
    if read_json(LIVE/'validation-r1-authorization.json').get('approved') is not True:
        raise ValueError('Await explicit approval for exactly one corrected validation')
    result = subprocess.run(['crontab','-l'], capture_output=True, text=True)
    if result.returncode not in (0,1):
        raise ValueError('Cannot read existing cron')
    rows = [line for line in result.stdout.splitlines() if not line.rstrip().endswith(MARKER)]
    log = DEFAULT_ROOT/'runs'/STUDY/'controller.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    (DEFAULT_ROOT/'logs').mkdir(parents=True, exist_ok=True)
    rows.append('* * * * * '+shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'run-once'])+
                ' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    subprocess.run(['crontab','-'], input='\n'.join(rows)+'\n', text=True, check=True)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'check': check()
    elif action == 'install': install()
    elif action == 'run-once': run_once()
    elif action == 'receipt': receipt(sys.argv[2], sys.argv[3])
    elif action == 'setup-receipt': setup_receipt()
    else: raise ValueError('Unknown action')
