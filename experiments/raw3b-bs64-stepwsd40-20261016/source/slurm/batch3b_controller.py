#!/usr/bin/env python3
"""Sole owner of the explicitly approved fresh formal13 allocation."""
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
                            digest, file_digest, locked, load_config, read_json)
from batch3b_contracts import STUDY as RUN_ID, PREP, TARGET, Q, PLANNED_UPDATES, count_updates
from batch3b_authorization import authorized
from slurm.submission import find_submission_job

CONFIG = LIVE / 'config.json'
STATE = LIVE / 'controller-state.json'
MARKER = '# ' + RUN_ID + '-controller'
SECONDS = 21540


def owner():
    row = read_json(LIVE / 'submission-owner.json')
    return (row['hostname'] == socket.gethostname().split('.')[0]
            and row['script'] == str(Path(__file__).resolve())
            and row['sha256'] == file_digest(__file__))


def check():
    seal = read_json(LIVE / 'source-seal.json')
    for name, sha in seal['files'].items():
        if file_digest(SOURCE / name) != sha:
            raise ValueError('Sealed source changed: ' + name)
    for name, sha in seal['inputs'].items():
        if file_digest(name) != sha:
            raise ValueError('Sealed configuration/protocol changed: ' + name)
    cfg = load_config(CONFIG)
    if not authorized(RUN_ID, cfg):
        raise ValueError('Exact formal13 authorization changed')
    plan = read_json(LIVE / 'plan.json')
    ledger = read_json(DEFAULT_ROOT / 'experiment-ledger.json')
    full = {k for k, v in ledger['runs'].items() if v['category'] == 'full'}
    if full - {RUN_ID} != set(plan['previous_formal_runs']):
        raise ValueError('Historical formal identities changed')
    for run, sha in plan['historical_run_digests'].items():
        if ledger['runs'].get(run, {}).get('config_digest') != sha:
            raise ValueError('Historical logical run lost or changed')
    for job, cost in plan['historical_job_costs'].items():
        if job not in ledger['jobs'] or (cost is not None and ledger['jobs'][job].get('h200_hours') != cost):
            raise ValueError('Historical allocation/cost lost or changed')
    prefix = read_json(LIVE / 'raw-prefix.json')
    progress = read_json(DEFAULT_ROOT / 'preprocessed' / PREP / 'raw/progress.json')
    if (progress['train_files'][:len(prefix['files'])] != prefix['files']
            or sum(f['rows'] for f in prefix['files']) < TARGET // 8192):
        raise ValueError('Immutable raw3B input prefix changed or incomplete')


def training_check(run_id, path):
    cfg = load_config(path)
    run = DEFAULT_ROOT / 'runs/experiments' / run_id
    s = read_json(run / 'summary.json')
    expected = count_updates(cfg)
    if (not authorized(run_id, cfg) or s['configuration'] != cfg
            or expected != PLANNED_UPDATES or s['global_step'] != expected
            or s['attempted_updates'] != expected or s['cursor'] != TARGET // 8192
            or s['total_tokens_seen'] != TARGET or s['budget_tokens_charged'] != TARGET
            or s['checkpoint_loaded_for_training'] or s['initialization']['pretrained_weights_loaded']
            or s['stop_reason'] != 'training_token_target' or s['export_global_step'] != expected
            or s['export_tokens_seen'] != TARGET or not read_json(run / 'export.json')['reload_verified']
            or s['last_evaluation']['eval/documents'] != 1000
            or s['last_evaluation']['eval/prediction_tokens'] != 1479302
            or not math.isfinite(s['last_evaluation']['eval/perplexity'])):
        raise ValueError('Incomplete fresh3B endpoint; no automatic retry or resume')
    return s


def receipt(run_id, path):
    check()
    s = training_check(run_id, path)
    run = DEFAULT_ROOT / 'runs/experiments' / RUN_ID
    full = read_json(run / 'full-evaluation.json')
    if (full['eval/documents'] != 50000 or full['eval/prediction_tokens'] != 76930208
            or full['global_step'] != PLANNED_UPDATES
            or full['training_global_step'] != PLANNED_UPDATES
            or full['export'] != s['export'] or not math.isfinite(full['eval/perplexity'])):
        raise ValueError('Full final evaluation incomplete or wrong checkpoint')
    groups = read_json(run / 'optimizer-groups.json')
    from slurm.audit_step_schedule import audit
    schedule_audit = audit(s['configuration'], os.environ['SLURM_JOB_ID'])
    comparisons = {}
    for name in ('raw3b-bs64-wsd40-20261016', 'raw3b-sota-wsd20-20261015', 'raw3b-sota-20261015'):
        control = DEFAULT_ROOT / 'runs/experiments' / name
        if s['data_provenance'] != read_json(control / 'data-provenance.json'):
            raise ValueError('Different3B data/evaluation provenance')
        if s['initialization']['sample_signature'] != read_json(control / 'initialization.json')['sample_signature']:
            raise ValueError('Different random initialization')
        previous = read_json(control / 'optimizer-groups.json')
        for key in ('parameter_names', 'parameter_counts', 'tied_aliases', 'muon_configuration'):
            if groups[key] != previous[key]:
                raise ValueError('Unexpected optimizer routing/settings change')
        baseline = read_json(control / 'full-evaluation.json')['eval/perplexity']
        comparisons[name] = dict(control_full_perplexity=baseline,
            candidate_full_perplexity=full['eval/perplexity'],
            delta_perplexity=full['eval/perplexity'] - baseline,
            improved=full['eval/perplexity'] < baseline)
    job = os.environ['SLURM_JOB_ID']
    for rank in range(8):
        if read_json(DEFAULT_ROOT / 'logs' / ('job-' + job) / f'worker-rank-{rank}-exit.json')['exit_code'] != 0:
            raise ValueError('Training worker failed')
    atomic_json(LIVE / 'final-comparison.json', comparisons)
    atomic_json(LIVE / 'receipt.json', dict(job_id=job, run_id=run_id, tokens=TARGET,
        global_step=PLANNED_UPDATES, subset_perplexity=s['last_evaluation']['eval/perplexity'],
        full_perplexity=full['eval/perplexity'], checkpoint=s['checkpoint'],
        export=s['export'], wandb_url=s['wandb_url'], schedule_audit=schedule_audit))


def submit(state):
    if not owner():
        raise ValueError('Wrong submission owner')
    check()
    if 'job' in state or (LIVE / 'disabled').exists():
        return
    if read_json(LIVE / 'authorization.json').get('approved') is not True:
        raise ValueError('The exact formal13 plan requires explicit approval')
    ledger = ExperimentLedger(DEFAULT_ROOT)
    previous = [(k, v) for k, v in read_json(ledger.path)['jobs'].items() if v['run_id'] == RUN_ID]
    if previous:
        # Recover even a crash between reservation and controller-state write.
        if len(previous) != 1:
            raise ValueError('Single allocation only; never retry')
        key, item = previous[0]
        job = dict(status='submission_uncertain', reservation=key,
                   submission_tag=item.get('submission_tag', key))
        for field in ('job_id', 'returned_job_id', 'submission_started_utc', 'submission_command'):
            if field in item:
                job[field] = item[field]
        state['job'] = job
        atomic_json(STATE, state)
        return
    tag = RUN_ID + '-' + uuid.uuid4().hex
    cfg = load_config(CONFIG)
    ledger.reserve(tag, RUN_ID, cfg, SECONDS, 8, False)
    command = ['sbatch', '--parsable', '--time=359', '--comment=' + tag,
               '--exclude=' + ','.join(read_json(LIVE / 'plan.json')['excluded_nodes']),
               str(SOURCE / 'slurm/bs64-3b.sbatch'), RUN_ID, str(CONFIG), PREP]
    job = dict(status='submission_uncertain', reservation=tag, submission_tag=tag,
               submission_started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               submission_command=command)
    state['job'] = job
    atomic_json(STATE, state)
    with ledger.transaction() as data:
        data['jobs'][tag].update(submission_tag=tag,
            submission_started_utc=job['submission_started_utc'], submission_command=command,
            source_root=str(SOURCE), allocation_kind=RUN_ID,
            authorization='User requested 幫我用 steps 後40% 跑一次3B:formal13,exact sealed config,one8H200/8CPU allocation<=6h/48H,fresh random weights,no retry/resume/cleanup/HF upload')
    env = {**os.environ, 'TRAIN_SOURCE_ROOT': str(SOURCE), 'DATA_ROOT': str(DEFAULT_ROOT),
           'BATCH3B_CONTROLLER': str(Path(__file__).resolve())}
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
    job = state.get('job')
    if not job or job['status'] == 'reconciled':
        return
    ledger = ExperimentLedger(DEFAULT_ROOT)
    if not job.get('job_id'):
        try:
            job_id = find_submission_job(job, DEFAULT_ROOT)
        except (ValueError, RuntimeError) as error:
            job['recovery_deferred'] = str(error)
            return
        if job['reservation'] in read_json(ledger.path)['jobs']:
            ledger.bind(job['reservation'], job_id)
        job.update(job_id=job_id, status='submitted')
    result = subprocess.run(['sacct', '-X', '-j', job['job_id'], '-n', '-P', '-o', 'JobIDRaw,State'],
                            capture_output=True, text=True, timeout=30, check=True)
    rows = [r.split('|')[1].split()[0].rstrip('+') for r in result.stdout.splitlines()
            if r.split('|')[0] == job['job_id']]
    job['slurm_state'] = rows[0] if rows else 'UNKNOWN'
    if job['slurm_state'] in TERMINAL:
        job.update(accounting=ledger.reconcile(job['job_id']), status='reconciled')
        if job['slurm_state'] == 'COMPLETED' and (LIVE / 'receipt.json').exists():
            state.update(stage='complete', result=read_json(LIVE / 'receipt.json'))
        else:
            state.update(stage='incomplete_no_retry', blocked=True)


def report(state):
    atomic_json(STATE, state)
    (LIVE / 'report.md').write_text('# Fresh3B BS64 /40% of actual updates per phase\n\n' + json.dumps(state, indent=2) + '\n')


def run_once():
    if not owner():
        raise ValueError('Wrong submission owner')
    try:
        with locked(LIVE / '.controller.lock', blocking=False):
            state = read_json(STATE) if STATE.exists() else dict(stage='prepared')
            try:
                reconcile(state)  # Account even if disabled or the source seal changed.
                check()
                if not state.get('blocked') and not (LIVE / 'disabled').exists() and 'job' not in state:
                    submit(state)
                    state['stage'] = 'submitted_or_uncertain'
                report(state)
                if state.get('job', {}).get('status') == 'reconciled':
                    retire()
            except subprocess.TimeoutExpired as error:
                state['scheduler_deferred'] = str(error)
                report(state)
            except Exception as error:
                state.update(blocked=True, error=str(error))
                report(state)
                raise
    except BlockingIOError:
        return


def retire():
    result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise ValueError('Cannot read cron')
    rows = [r for r in result.stdout.splitlines() if not r.rstrip().endswith(MARKER)]
    if len(rows) != len(result.stdout.splitlines()):
        subprocess.run(['crontab', '-'], input='\n'.join(rows) + '\n', text=True, check=True)


def install():
    check()
    if not owner():
        raise ValueError('Wrong submission owner')
    result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise ValueError('Cannot read cron')
    rows = [r for r in result.stdout.splitlines() if not r.rstrip().endswith(MARKER)]
    log = DEFAULT_ROOT / 'runs' / RUN_ID / 'controller.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    (DEFAULT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    rows.append('* * * * * ' + shlex.join(['/usr/bin/python3', str(Path(__file__).resolve()), 'run-once'])
                + ' >> ' + shlex.quote(str(log)) + ' 2>&1 ' + MARKER)
    subprocess.run(['crontab', '-'], input='\n'.join(rows) + '\n', text=True, check=True)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'check': check()
    elif action == 'install': install()
    elif action == 'run-once': run_once()
    elif action == 'training-check': training_check(sys.argv[2], sys.argv[3])
    elif action == 'receipt': receipt(sys.argv[2], sys.argv[3])
    else: raise ValueError('Unknown action')
