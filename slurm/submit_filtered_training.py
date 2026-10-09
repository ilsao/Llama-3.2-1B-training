#!/usr/bin/env python3
"""Single submission owner for random-init filtered 2B training; status never submits."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import (DEFAULT_ROOT, PROJECT, TERMINAL, ExperimentLedger,
                            atomic_json, digest, file_digest, load_config, locked, read_json)
from slurm.submit_training import mark_uncertain, recover_submission

RUN_ID = 'muon-wsd-filtered-2b-20261009'
PREP_ID = 'dolma-dclm10-005'
CONFIG = PROJECT / 'configs/train-muon-wsd-filtered-2b.json'
DIRECTORY = DEFAULT_ROOT / 'runs/filtered' / RUN_ID
EXPERIMENT = DEFAULT_ROOT / 'runs/experiments' / RUN_ID
MARKER = '# codex-filtered-training-accounting-' + RUN_ID
PREPARATION_FINGERPRINT = '3b38ab98b5a62f3620a2ae2a5adb8d366baf129604725d330e6ff131425b289a'
SOURCES = ('train.py', 'training_state.py', 'dataset.py', 'preparation_parallel.py',
           'optimizers.py', 'attention_backend.py', 'requirements-training.txt',
           'slurm/common.sh', 'slurm/training-common.sh', 'slurm/training_entry.py',
           'slurm/full_evaluation.py', 'slurm/filtered-train.sbatch',
           'slurm/submit_filtered_training.py', 'configs/train-muon-wsd-filtered-2b.json')


def validate_recipe(cfg, best):
    allowed = {'target_training_tokens', 'max_training_tokens', 'max_steps', 'scheduler_steps',
               'max_h200_hours', 'fixed_horizon', 'eval_steps', 'stop_margin_seconds',
               'final_evaluation_reserve_seconds', 'sweep_id', 'sweep_purpose', 'sweep_stage', 'parent_runs'}
    if set(cfg) - set(best) != {'target_training_tokens'} or set(best) - set(cfg):
        raise ValueError('Filtered recipe does not derive from the complete best configuration')
    if any(cfg[k] != best[k] for k in best if k not in allowed):
        raise ValueError('Winning optimizer/model/batch/evaluation configuration changed')
    expected = dict(target_training_tokens=2_000_000_000, max_training_tokens=2_000_027_648,
                    max_steps=15259, scheduler_steps=15259, max_h200_hours=64,
                    fixed_horizon=True, world_size=8, eval_steps=400,
                    stop_margin_seconds=300, final_evaluation_reserve_seconds=1800)
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('Filtered token/schedule/resource contract changed')
    width = cfg['world_size'] * cfg['microbatch_size'] * cfg['gradient_accumulation_steps']
    if cfg['max_steps'] * width * cfg['sequence_length'] != cfg['max_training_tokens']:
        raise ValueError('Whole-batch token cap is inconsistent')


def training_result(summary):
    cfg = summary['configuration']
    tokens = summary['total_tokens_seen']
    return (summary['stop_reason'] == 'training_token_target' and
            summary.get('training_token_target_reached') is True and
            cfg['target_training_tokens'] <= tokens <= cfg['max_training_tokens'] and
            summary['budget_tokens_charged'] == tokens and
            summary['initialization']['pretrained_weights_loaded'] is False and
            summary['data_provenance']['preparation_id'] == PREP_ID and
            summary['data_provenance']['fingerprint'] == PREPARATION_FINGERPRINT)


def cron(install):
    old = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if old.returncode not in (0, 1):
        raise RuntimeError('Cannot read accounting schedule')
    lines = [x for x in old.stdout.splitlines() if MARKER not in x]
    if install:
        command = shlex.join(['/usr/bin/python3', str(Path(__file__).resolve()), 'status'])
        lines.append(f'* * * * * {command} >> {shlex.quote(str(DIRECTORY / "accounting.log"))} 2>&1 {MARKER}')
    subprocess.run(['crontab', '-'], input='\n'.join(lines) + '\n', text=True, check=True)


def submit(test_only=False):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with locked(DIRECTORY / '.submission.lock', blocking=False):
        if (DIRECTORY / 'launch.json').exists():
            raise ValueError('Existing submission; use status/recovery, never another training initialization')
        cfg = load_config(CONFIG)
        best_path = DEFAULT_ROOT / 'runs/experiments/baseline-muon-wsd-raw-20261009/summary.json'
        best = read_json(best_path)
        if best['export_global_step'] != 13717:
            raise ValueError('The reference best checkpoint changed')
        validate_recipe(cfg, best['configuration'])
        prep = DEFAULT_ROOT / 'preprocessed' / PREP_ID
        manifests = {name: read_json(prep / name / 'manifest.json') for name in ('select', 'pack', 'verify')}
        if any(m['fingerprint'] != PREPARATION_FINGERPRINT for m in manifests.values()):
            raise ValueError('Filtered preparation identity changed')
        packed = manifests['pack']['details']; verified = manifests['verify']['details']
        if packed['training_tokens'] != 6_000_000_000 or verified['training']['input_tokens'] != 6_000_000_000:
            raise ValueError('The final verified 6B training dataset is not ready')
        if verified['evaluation_documents'] != 50000 or verified['subset_documents'] != 10000:
            raise ValueError('Reserved evaluation membership changed')
        from slurm.auto_optimizer_sweep import environment_identity
        environment = environment_identity()
        if environment['packages_digest'] != cfg['sweep_environment_digest']:
            raise ValueError('Native Muon/FA3 environment changed')
        from slurm.filtering_resume import shared_upper
        upper = shared_upper(DEFAULT_ROOT)
        if upper + 64 > 192:
            raise ValueError('Shared compute budget cannot reserve 64 H200-hours')
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
        subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *SOURCES], check=True)
        remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', 'main'], text=True).split()[0]
        if branch != 'main' or head != remote:
            raise ValueError('Push the tested main commit before training submission')
        DEFAULT_ROOT.joinpath('logs').mkdir(exist_ok=True)
        tag = 'filtered-2b-' + uuid.uuid4().hex
        frozen = CONFIG if test_only else DIRECTORY / 'config.json'
        command = ['sbatch', '--parsable', '--comment=' + tag]
        if test_only:
            command.append('--test-only')
        command += [str(PROJECT / 'slurm/filtered-train.sbatch'), RUN_ID, PREP_ID, str(frozen)]
        launch_env = {**os.environ, 'TRAIN_SOURCE_ROOT': str(PROJECT), 'DATA_ROOT': str(DEFAULT_ROOT)}
        if test_only:
            result = subprocess.run(command, env=launch_env, capture_output=True, text=True, check=True)
            print(result.stdout + result.stderr)
            return
        if EXPERIMENT.exists():
            raise ValueError('Fresh training output directory already exists')
        ledger = ExperimentLedger(DEFAULT_ROOT)
        ledger.reserve(tag, RUN_ID, cfg, 28800, 8, False)
        atomic_json(frozen, cfg)
        sealed = {str(PROJECT / name): file_digest(PROJECT / name) for name in SOURCES}
        sealed.update({str(prep / name / 'manifest.json'): file_digest(prep / name / 'manifest.json') for name in ('select', 'pack', 'verify')})
        launch = dict(status='submission_uncertain', run_id=RUN_ID, preparation_id=PREP_ID,
                      configuration=str(frozen), reservation=tag, source_commit=head,
                      source_hashes=sealed, environment=environment, config_digest=digest(cfg),
                      best_recipe_sha256=file_digest(best_path), shared_upper_before_reservation=upper,
                      requested_seconds=28800, gpus=8, cpus=8, random_initialization=True,
                      checkpoint_initialization=False, submission_command=command,
                      started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(DIRECTORY / 'launch.json', launch)
        with ledger.transaction() as data:
            data['jobs'][tag].update(submission_tag=tag, submission_started_utc=launch['started_utc'],
                                    submission_command=command, source_root=str(PROJECT),
                                    allocation_kind='filtered_training_and_full_evaluation')
        try:
            result = subprocess.run(command, env=launch_env, capture_output=True, text=True)
        except OSError:
            ledger.cancel_reservation(tag)
            launch['status'] = 'submission_not_executed'
            atomic_json(DIRECTORY / 'launch.json', launch)
            raise
        launch['submission_result'] = dict(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        atomic_json(DIRECTORY / 'launch.json', launch)
        job = result.stdout.strip().split(';')[0]
        if result.returncode or not re.fullmatch(r'\d+', job):
            mark_uncertain(ledger, tag)
            raise ValueError('Uncertain submission: retain reservation and recover; do not retry')
        ledger.bind(tag, job)
        launch.update(status='submitted', job_id=job)
        atomic_json(DIRECTORY / 'launch.json', launch)
        cron(True)
        print(json.dumps(dict(job_id=job, run_id=RUN_ID, source_commit=head,
                             token_target=cfg['target_training_tokens'], hard_token_cap=cfg['max_training_tokens'])))


def worker_check():
    from train import require_compute
    require_compute()
    launch = read_json(DIRECTORY / 'launch.json')
    for name, sha in launch['source_hashes'].items():
        if file_digest(Path(name)) != sha:
            raise ValueError('Sealed training input changed: ' + name)
    allocation = ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'], RUN_ID)
    if allocation['config_digest'] != launch['config_digest'] or allocation['previous_h200_hours'] != 0:
        raise ValueError('Fresh logical-run reservation changed')
    if digest(read_json(launch['configuration'])) != launch['config_digest']:
        raise ValueError('Frozen training configuration changed')
    print('Verified fresh filtered allocation, source seals and immutable preparation', flush=True)


def training_check():
    from train import require_compute
    require_compute()
    codes = [read_json(Path(os.environ['JOB_RECORD']) / f'worker-rank-{i}-exit.json')['exit_code'] for i in range(8)]
    if codes != [0] * 8 or not training_result(read_json(EXPERIMENT / 'summary.json')):
        raise ValueError('Training stopped without completing the processed-token target')
    print('Confirmed processed-token target reached; starting full reserved-set evaluation', flush=True)


def status():
    with locked(DIRECTORY / '.submission.lock', blocking=False):
        state = read_json(DIRECTORY / 'launch.json')
        if state['status'] == 'submission_not_executed':
            print(json.dumps(state)); return
        if not state.get('job_id'):
            state['job_id'] = recover_submission(ExperimentLedger(DEFAULT_ROOT), state['reservation'])
            atomic_json(DIRECTORY / 'launch.json', state)
        if state['status'] != 'reconciled':
            result = subprocess.run(['sacct', '-X', '-j', state['job_id'], '--noheader', '--parsable2',
                                     '--format=JobIDRaw,State%30'], capture_output=True, text=True, check=True, timeout=20)
            rows = [row.split('|') for row in result.stdout.splitlines()]
            slurm = next((r[1].split()[0].rstrip('+') for r in rows if r[0] == state['job_id']), 'UNKNOWN')
            state['slurm_state'] = slurm
            if slurm in TERMINAL:
                state['accounting'] = ExperimentLedger(DEFAULT_ROOT).reconcile(state['job_id'])
                summary_path = EXPERIMENT / 'summary.json'
                summary = read_json(summary_path) if summary_path.exists() else None
                full_path = EXPERIMENT / 'full-evaluation.json'
                full = read_json(full_path) if full_path.exists() else None
                state.update(status='reconciled', training_target_complete=bool(summary and training_result(summary)),
                             full_evaluation=full, outcome='complete' if slurm == 'COMPLETED' and summary and
                             training_result(summary) and full and full['eval/documents'] == 50000 else 'failed_or_incomplete')
                cron(False)
            atomic_json(DIRECTORY / 'launch.json', state)
        print(json.dumps({k: state.get(k) for k in ('job_id', 'status', 'slurm_state', 'outcome', 'training_target_complete')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status', 'worker-check', 'training-check'))
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'submit': submit(args.test_only)
    elif args.action == 'status': status()
    elif args.action == 'worker-check': worker_check()
    else: training_check()
