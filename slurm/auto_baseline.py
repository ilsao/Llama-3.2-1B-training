#!/usr/bin/env python3
"""Deterministic, metadata-only login-node controller for one bounded baseline campaign."""
from __future__ import annotations
import argparse
import datetime as dt
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import (PROJECT, ExperimentLedger, atomic_json, digest, file_digest,
                            identifier, load_config, locked, read_json, slurm_record, storage_root)
from slurm.submission import find_submission_job

MARKER = '# llama-1b-autoresearch-heartbeat'
DATA_STAGES = ('clean', 'score', 'select', 'pack', 'verify')


def source_hashes():
    files = ('slurm/auto_baseline.py', 'slurm/auto-baseline.sbatch', 'slurm/fa3_setup.py',
             'slurm/full_evaluation.py', 'slurm/common.sh', 'slurm/training-common.sh',
             'slurm/submit_preprocessing.py', 'slurm/submission.py', 'slurm/accounting.py',
             'train.py', 'training_state.py', 'attention_backend.py', 'requirements-training.txt', 'requirements-fa3-build.txt',
             'dataset.py', 'preparation_parallel.py', 'requirements-data.txt',
             'tests/fa3_runtime.py', 'tests/training_runtime.py', 'raw_baseline.py')
    return {name: file_digest(PROJECT/name) for name in files}


def now():
    return dt.datetime.now(dt.timezone.utc)


def load_campaign(path):
    cfg = read_json(path)
    identifier(cfg['campaign_id']); identifier(cfg['preparation_id'])
    storage_root(cfg['root'])
    if cfg['max_h200_hours'] != 16 or cfg['max_allocation_seconds'] != 7200:
        raise ValueError('This baseline campaign must retain the 16-hour/7200-second caps')
    if cfg['learning_rates'] != [3e-4, 6e-4] or cfg['lr_minutes'] != 15:
        raise ValueError('Use the approved two-candidate, 15-minute LR stage')
    for key in ('allocation_cutoff', 'delivery_deadline'):
        if dt.datetime.fromisoformat(cfg[key]).tzinfo is None:
            raise ValueError('Deadline must include timezone')
    return cfg


def directory(cfg):
    return Path(cfg['root']) / 'runs/auto-baseline' / cfg['campaign_id']


def save(cfg, state):
    state['updated_utc'] = now().isoformat()
    atomic_json(directory(cfg) / 'state.json', state)


def read_state(cfg):
    state = read_json(directory(cfg) / 'state.json')
    if state['config_digest'] != digest(cfg):
        raise ValueError('Campaign configuration changed; restore original configuration')
    return state


def budget(jobs):
    charged = [j for j in jobs if j.get('kind') == 'training' and j.get('status') != 'not_dispatched']
    seconds = sum(j.get('elapsed_seconds', j['requested_seconds']) for j in charged)
    hours = sum(j.get('h200_hours', j['gpu_count'] * j['requested_seconds']/3600) for j in charged)
    return {'allocation_seconds': seconds, 'h200_hours': hours}


def check_budget(jobs, seconds, gpus):
    spent = budget(jobs)
    if seconds <= 0 or not 1 <= gpus <= 8:
        raise ValueError('Invalid allocation')
    if spent['allocation_seconds'] + seconds > 7200 or spent['h200_hours'] + gpus*seconds/3600 > 16+1e-10:
        raise ValueError('Campaign allocation would exceed 120 minutes or 16 H200-hours')


def shared_cost(cfg):
    root = Path(cfg['root'])
    costs = {}
    for filename in ('preprocessing-ledger.json', 'experiment-ledger.json'):
        path = root / filename
        if path.exists():
            for key, item in read_json(path)['jobs'].items():
                costs[key] = item.get('h200_hours', item.get('gpu_hours', item.get('reserved_h200_hours', 0)))
    research = read_json(PROJECT / 'research-state.yaml')
    for item in research.get('submission_intents', []):
        key = item.get('job_id', item['tag'])
        if key not in costs and item.get('status') != 'not_dispatched':
            costs[key] = item.get('reserved_h200_hours', 0)
    return sum(costs.values())


def terminal(job):
    return job.get('status') in {'complete', 'failed', 'not_dispatched'}


def manifest_ready(cfg, stage):
    prep = Path(cfg['root']) / 'preprocessed' / cfg['preparation_id']
    path = prep / stage / 'manifest.json'
    if not path.exists():
        return False
    record = read_json(path)
    identity = read_json(prep / 'configuration.json')
    if record['fingerprint'] != identity['fingerprint'] or record['stage'] != stage:
        raise ValueError('Dataset manifest identity mismatch')
    previous = {'clean': 'split', 'score': 'clean', 'select': 'score', 'pack': 'select', 'verify': 'pack'}[stage]
    if record['parent_sha256'] != file_digest(prep / previous / 'manifest.json'):
        raise ValueError('Dataset parent lineage mismatch')
    # Metadata only: large artifact checksums are verified by Slurm dataset jobs.
    for item in record.get('artifacts', []):
        if Path(item['path']).stat().st_size != item['size']:
            raise ValueError('Dataset artifact size changed')
    return True


def reconcile(cfg, state):
    root = Path(cfg['root']); ledger = ExperimentLedger(root)
    # Adopt wrapper intents after a crash between data submission and campaign save.
    research = read_json(PROJECT / 'research-state.yaml')
    known = {j['tag'] for j in state['jobs']}
    preledger = read_json(root / 'preprocessing-ledger.json')['jobs'] if (root / 'preprocessing-ledger.json').exists() else {}
    for item in research.get('submission_intents', []):
        if (item.get('prep_id') == cfg['preparation_id'] and item.get('status') != 'not_dispatched'
                and item['tag'] not in known and item.get('job_id') not in preledger):
            state['jobs'].append({'phase': 'data-'+item['stage'], 'kind': 'data', 'tag': item['tag'],
                                 'submission_tag': item.get('submission_tag', item['tag']),
                                 'submission_started_utc': item.get('submission_started_utc', item['timestamp']),
                                 'returned_job_id': item.get('returned_job_id'), 'job_id': item.get('job_id'),
                                 'status': 'submitted' if item.get('job_id') else 'submission_uncertain',
                                 'gpu_count': 1, 'requested_seconds': int(item.get('reserved_h200_hours',24)*3600)})
            save(cfg,state)
    for job in state['jobs']:
        if terminal(job):
            continue
        if job['status'] == 'planned':
            raise RuntimeError('Controller stopped before dispatch; inspect planned intent before releasing reservation')
        if job['status'] == 'submission_uncertain':
            job_id = find_submission_job(job, root)
            job['job_id'] = job_id
            if job['kind'] in ('training', 'setup', 'preprocessing'):
                jobs = read_json(ledger.path)['jobs']
                if job['tag'] in jobs:
                    ledger.bind(job['tag'], job_id)
            else:
                from slurm.submit_preprocessing import recover
                recover(job['tag'], root, PROJECT / 'research-state.yaml')
            job['status'] = 'submitted'; save(cfg, state)
        try:
            measured = slurm_record(job['job_id'], root)
        except (ValueError, subprocess.SubprocessError, OSError):
            # Running, absent or delayed accounting: reservation remains charged.
            continue
        if job['kind'] in ('training', 'setup', 'preprocessing'):
            measured = ledger.reconcile(job['job_id'])
        else:
            subprocess.run([sys.executable, str(PROJECT/'slurm/accounting.py'), job['job_id'],
                            '--root', str(root)], check=True, capture_output=True, text=True, timeout=30)
        job.update(measured)
        exit_path = root / 'logs' / ('job-' + job['job_id']) / 'exit-code.txt'
        job['exit_code'] = int(exit_path.read_text().strip()) if exit_path.exists() else None
        job['status'] = 'complete' if measured['state'] == 'COMPLETED' and job['exit_code'] == 0 else 'failed'
        save(cfg, state)


def phase_job(state, phase):
    return next((j for j in reversed(state['jobs']) if j['phase'] == phase), None)


def phase_done(state, phase):
    job = phase_job(state, phase)
    return job is not None and job['status'] == 'complete'


def submit_worker(cfg, state, phase, train_cfg, minutes, gpus, resume=False):
    seconds = minutes*60
    kind = 'setup' if phase == 'setup' else 'preprocessing' if phase == 'raw-prepare' else 'training'
    if kind == 'training':
        check_budget(state['jobs'], seconds, gpus)
    if shared_cost(cfg) + gpus*seconds/3600 > cfg['shared_h200_hours_limit']:
        raise ValueError('Shared preprocessing/training reservation would exceed conservative aggregate budget')
    root = Path(cfg['root']); ledger = ExperimentLedger(root)
    run_id = cfg['campaign_id'] + '-' + phase
    config_path = directory(cfg) / (phase + '.json')
    if resume:
        train_cfg = load_config(config_path)
    else:
        atomic_json(config_path, train_cfg)
        load_config(config_path)
    tag = 'auto-' + uuid.uuid4().hex
    job = {'phase': phase, 'kind': kind, 'tag': tag, 'submission_tag': tag,
           'submission_started_utc': now().isoformat(), 'run_id': run_id,
           'requested_seconds': seconds, 'gpu_count': gpus,
           'config_path': str(config_path), 'status': 'planned'}
    state['jobs'].append(job); save(cfg, state)
    try:
        ledger.reserve(tag, run_id, train_cfg, seconds, gpus, resume)
    except Exception:
        job['status'] = 'not_dispatched'; save(cfg, state); raise
    command = ['sbatch', '--parsable', '--cpus-per-task=8', f'--gpus-per-node={gpus}',
               f'--time={minutes}', f'--comment={tag}', f'--job-name=auto-{phase}',
               f'--output={root}/logs/%x-%j.out', f'--error={root}/logs/%x-%j.err',
               str(PROJECT/'slurm/auto-baseline.sbatch'), run_id, phase,
               cfg.get('raw_preparation_id', cfg['preparation_id']), str(config_path), state['config_path'],
               'resume' if resume else 'new']
    job['submission_command'] = command; job['status'] = 'submission_uncertain'; save(cfg, state)
    with ledger.transaction() as data:
        data['jobs'][tag].update(submission_tag=tag, submission_started_utc=job['submission_started_utc'], allocation_kind=kind)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except OSError:
        ledger.cancel_reservation(tag); job['status'] = 'not_dispatched'; save(cfg, state); raise
    job_id = result.stdout.strip().split(';')[0]
    if result.returncode or not re.fullmatch(r'\d+', job_id):
        raise RuntimeError('Submission outcome uncertain; retain reservation and recover, never resubmit')
    # Persist returned ID before ledger binding; restart can complete binding.
    job['returned_job_id'] = job_id; save(cfg, state)
    ledger.bind(tag, job_id)
    job.update(job_id=job_id, status='submitted'); save(cfg, state)


def data_advance(cfg, state):
    if cfg.get('raw_preparation_id'):
        raw_job = phase_job(state, 'raw-prepare')
        if raw_job and not terminal(raw_job): return
        if raw_job and raw_job['status'] == 'complete': return
        if raw_job and raw_job.get('exit_code') != 75:
            raise ValueError('Raw baseline producer failed; inspect before retry')
        remaining = int((dt.datetime.fromisoformat(cfg['allocation_cutoff'])-now()).total_seconds())
        minutes = min(cfg['raw_prepare_minutes'], remaining//60-5)
        if minutes < 5: raise ValueError('Insufficient raw preparation allocation time')
        value = base_config(cfg, 'raw-prepare', 1, minutes)
        value['max_h200_hours'] = cfg['raw_max_h200_hours']
        if raw_job:
            value = load_config(raw_job['config_path'])
            used = sum(j.get('elapsed_seconds',j['requested_seconds']) for j in state['jobs'] if j['phase']=='raw-prepare')
            minutes = min(minutes,int((value['max_h200_hours']*3600-used)//60))
            if minutes < 5: raise ValueError('Raw preparation cumulative allocation exhausted')
        submit_worker(cfg,state,'raw-prepare',value,minutes,1,resume=raw_job is not None)
        return
    root = Path(cfg['root'])
    for stage in DATA_STAGES:
        job = phase_job(state, 'data-' + stage)
        if job and not terminal(job):
            return
        ready = manifest_ready(cfg, stage)
        if ready and (not job or job['status'] == 'complete'):
            continue
        if ready and job and job['status'] != 'complete':
            raise ValueError('Dataset manifest exists but producer did not successfully terminate')
        if job and job['status'] == 'failed' and job['exit_code'] != 75:
            raise ValueError('Data stage failed: ' + stage)
        remaining = int((dt.datetime.fromisoformat(cfg['allocation_cutoff']) - now()).total_seconds())
        minutes = min(1440, remaining//60 - 130)
        if minutes < 20:
            raise ValueError('Insufficient time for preprocessing and baseline before cutoff')
        if shared_cost(cfg) + minutes/60 > cfg['shared_h200_hours_limit']:
            raise ValueError('Shared budget insufficient for preprocessing')
        from slurm.submit_preprocessing import submit
        job_id = submit(stage, cfg['preparation_id'], [], root, PROJECT/'research-state.yaml', minutes=minutes)
        intent = next(x for x in reversed(read_json(PROJECT/'research-state.yaml')['submission_intents'])
                      if x.get('job_id') == job_id)
        state['jobs'].append({'phase': 'data-'+stage, 'kind': 'data', 'tag': intent['tag'],
                              'job_id': job_id, 'status': 'submitted', 'gpu_count': 1,
                              'requested_seconds': minutes*60})
        save(cfg, state)
        return


def base_config(cfg, phase, gpus, minutes):
    value = read_json(PROJECT/'configs/train-pilot.json')
    value.update(world_size=gpus, gradient_accumulation_steps=16//gpus,
                 attention_backend='flash_attention_3', checkpoint_interval_seconds=900,
                 final_evaluation_reserve_seconds=0, campaign_id=cfg['campaign_id'],
                 sweep_stage='validation' if phase.startswith('probe') or phase=='setup' else 'ultrashort' if phase.startswith('lr-') else 'baseline',
                 max_h200_hours=gpus*minutes/60, stop_margin_seconds=60 if phase.startswith('probe') else 180,
                 max_steps=32, warmup_steps=2, eval_steps=3, eval_documents=100,
                 category='validation', wandb_mode='offline')
    if phase in ('setup','raw-prepare'):
        value.update(gradient_accumulation_steps=1)
    return value


def probe_measurements(cfg):
    run = Path(cfg['root'])/'runs/experiments'/(cfg['campaign_id']+'-probe8')
    summary = read_json(run/'summary.json')
    prep = Path(cfg['root'])/'preprocessed'/cfg.get('raw_preparation_id',cfg['preparation_id'])
    if cfg.get('raw_preparation_id'):
        from raw_baseline import packed_details
        packed = packed_details(prep)
    else:
        packed = read_json(prep/'pack/manifest.json')['details']
    evaluation = summary['last_evaluation']
    subset_seconds = max(1, summary['evaluation_seconds'])*10*1.5
    full_seconds = math.ceil(max(1, summary['evaluation_seconds'])*
                             packed['evaluation_prediction_tokens']/evaluation['eval/prediction_tokens']*1.5 + 300)
    return summary, packed, subset_seconds, max(600, full_seconds)


def training_advance(cfg, state):
    environment = Path(cfg['environment'])
    setup = phase_job(state, 'setup')
    if not setup:
        submit_worker(cfg, state, 'setup', base_config(cfg, 'setup', 1, cfg['setup_minutes']), cfg['setup_minutes'], 1)
        return
    if not terminal(setup):
        return
    if setup['status'] != 'complete' or not (environment/'fa3-build.json').exists():
        raise ValueError('FA3 installation failed; no automatic fallback/reinstall')
    if cfg.get('raw_preparation_id'):
        ready = Path(cfg['root'])/'preprocessed'/cfg['raw_preparation_id']/'raw/ready.json'
        if not ready.exists(): return
        sealed = read_json(ready)
        identity = ready.parents[1]/'configuration.json'
        if sealed['identity_sha256'] != file_digest(identity):
            raise ValueError('Raw baseline identity seal changed')
    elif not manifest_ready(cfg, 'verify'):
        return
    for phase, gpus in (('probe4', 4), ('probe8', 8)):
        job = phase_job(state, phase)
        if not job:
            submit_worker(cfg, state, phase, base_config(cfg, phase, gpus, cfg['probe_minutes']), cfg['probe_minutes'], gpus)
            return
        if not terminal(job):
            return
        if job['status'] != 'complete':
            raise ValueError('GPU validation failed: ' + phase)
    root = Path(cfg['root'])
    probe8 = root/'runs/experiments'/(cfg['campaign_id']+'-probe8')
    if not read_json(probe8/'export.json')['reload_verified']:
        raise ValueError('Full-model HF reload validation missing')
    record8 = root/'logs'/('job-'+phase_job(state,'probe8')['job_id'])/'fa3-check.json'
    runtime8 = root/'runs/runtime-validation'/(cfg['campaign_id']+'-probe8')/'result.json'
    if not read_json(record8)['passed'] or not read_json(runtime8)['passed']:
        raise ValueError('FA3 kernel or resume correctness checks missing')
    measurements, packed, subset_seconds, full_seconds = probe_measurements(cfg)
    step_seconds = max(.001, measurements['step_seconds'])
    if 'scaling' not in state:
        runs=[root/'runs/experiments'/(cfg['campaign_id']+'-'+p) for p in ('probe4','probe8')]
        summaries=[read_json(r/'summary.json') for r in runs]
        ignored={'world_size','gradient_accumulation_steps','max_h200_hours'}
        recipes=[{k:v for k,v in s['configuration'].items() if k not in ignored} for s in summaries]
        if recipes[0]!=recipes[1] or summaries[0]['data_provenance']!=summaries[1]['data_provenance']:
            raise ValueError('Scaling recipes/data do not match')
        values=[]
        for run in runs:
            values.append({m['train/global_step']:m for m in
                           [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
                           if 'train/step_seconds' in m})
        steps=sorted(step for step in values[0].keys() & values[1].keys()
                     if step>summaries[0]['configuration']['warmup_steps'])[-20:]
        if len(steps)<20 or any(values[0][i]['train/step_input_tokens']!=values[1][i]['train/step_input_tokens'] for i in steps):
            raise ValueError('Scaling lacks 20 matched post-warmup updates')
        throughput=[sum(v[i]['train/step_input_tokens'] for i in steps)/sum(v[i]['train/step_seconds'] for i in steps) for v in values]
        state['scaling']={'global_batch':16,'matched_steps':steps,'tokens_per_second':throughput,
                          'speedup':throughput[1]/throughput[0],
                          'allocation_h200_hours':[phase_job(state,p)['h200_hours'] for p in ('probe4','probe8')],
                          'selected_gpus':8,'reason':'Two-hour baseline deadline; matched-batch cost recorded'}
        save(cfg,state)
    checkpoint_seconds = measurements.get('checkpoint_seconds', 0)
    candidate_steps = int((cfg['lr_minutes']*60-240-11*(subset_seconds+checkpoint_seconds))/step_seconds)
    skip = cfg.get('skip_lr_search',False) or full_seconds > 1800 or candidate_steps < 10
    cutoff_seconds = int((dt.datetime.fromisoformat(cfg['allocation_cutoff'])-now()).total_seconds())
    if cutoff_seconds < (cfg['baseline_minutes'] + 2*cfg['lr_minutes'] + 5)*60:
        skip = True
    state['skip_search'] = skip
    if not skip:
        for phase, lr in zip(('lr-low','lr-high'), cfg['learning_rates']):
            job = phase_job(state, phase)
            if not job:
                value = base_config(cfg, phase, 8, cfg['lr_minutes'])
                steps = min(candidate_steps, 45776, packed['training_sequence_count']//16)
                value.update(category='pilot', wandb_mode='online', learning_rate=lr,
                             max_steps=steps, warmup_steps=max(1, math.ceil(steps*.02)),
                             eval_steps=max(1, steps//10), eval_documents=cfg['evaluation_documents'])
                submit_worker(cfg, state, phase, value, cfg['lr_minutes'], 8)
                return
            if not terminal(job):
                return
    baseline = phase_job(state, 'baseline')
    if baseline:
        if not terminal(baseline):
            return
        result_path = root/'runs/experiments'/baseline['run_id']/'full-evaluation.json'
        if baseline['status']=='complete' and result_path.exists():
            result = read_json(result_path)
            spent = budget(state['jobs'])
            state['status'] = 'achieved' if result['eval/perplexity'] < 20 and spent['h200_hours'] <= 16 and spent['allocation_seconds']<=7200 else 'not_achieved'
            state['result'] = result
        elif baseline['exit_code']==75:
            if cfg.get('raw_preparation_id'):
                progress = read_json(root/'preprocessed'/cfg['raw_preparation_id']/'raw/progress.json')
                summary_path = root/'runs/experiments'/baseline['run_id']/'summary.json'
                summary = read_json(summary_path) if summary_path.exists() else {}
                available = sum(i['rows'] for i in progress['train_files'])
                if summary.get('stop_reason')=='data_unavailable' and progress['status']!='complete' and available < summary['cursor']+1024:
                    return
            remaining_phase = baseline['phase_limit_seconds'] - sum(j.get('elapsed_seconds', j['requested_seconds']) for j in state['jobs'] if j['phase']=='baseline')
            minutes = min(remaining_phase//60, int((7200-budget(state['jobs'])['allocation_seconds'])//60), cutoff_seconds//60)
            if minutes > 5 and (root/'runs/experiments'/baseline['run_id']/'latest.json').exists():
                value = load_config(baseline['config_path'])
                if minutes*60 > value['final_evaluation_reserve_seconds'] + 180:
                    submit_worker(cfg,state,'baseline',value,minutes,8,resume=True)
                    state['jobs'][-1]['phase_limit_seconds'] = baseline['phase_limit_seconds']; save(cfg,state)
                    return
            state.update(status='not_achieved', reason='Baseline stopped without complete full evaluation; remaining budget cannot support resume')
        else:
            state.update(status='not_achieved', reason='Baseline failed; no automatic budget extension')
        save(cfg,state)
        return
    spent = budget(state['jobs'])
    minutes = min(int((7200-spent['allocation_seconds'])//60), int((16-spent['h200_hours'])*60/8), cutoff_seconds//60-1)
    minutes = min(minutes, 110 if skip else cfg['baseline_minutes'])
    lr = 3e-4
    if not skip:
        records = []
        for phase in ('lr-low','lr-high'):
            j = phase_job(state,phase)
            if j['status']=='complete' or j.get('exit_code')==75:
                run = root/'runs/experiments'/j['run_id']
                if (run/'summary.json').exists():
                    summary = read_json(run/'summary.json')
                    if summary['stop_reason']!='nonfinite_or_skipped_update':
                        metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
                        records.append((summary['configuration']['learning_rate'], {m['eval/total_training_tokens_seen']:m for m in metrics if 'eval/nll' in m}))
        if len(records)==2:
            common = records[0][1].keys() & records[1][1].keys()
            if common:
                point = max(common)
                ranked = sorted((m[point]['eval/nll']/m[point]['eval/prediction_tokens'],rate) for rate,m in records)
                lr = min(r[1] for r in ranked) if abs(ranked[0][0]-ranked[1][0])<=.01 else ranked[0][1]
                state['selection']={'token_point':point,'nll_and_lr':ranked,'learning_rate':lr}
        elif len(records)==1:
            lr=records[0][0]
            state['selection']={'learning_rate':lr,'reason':'Only one candidate produced finite held-out results'}
        elif all(phase_job(state,p)['status']=='failed' for p in ('lr-low','lr-high')):
            raise ValueError('Both LR candidates failed; do not relaunch an unvalidated baseline')
    steps = int((minutes*60-full_seconds-300-11*(subset_seconds+checkpoint_seconds))/step_seconds*.9)
    steps = min(steps,45776,packed['training_sequence_count']//16)
    if steps < 10:
        raise ValueError('Insufficient budget/time for training plus full reserved-set evaluation')
    value=base_config(cfg,'baseline',8,minutes)
    value.update(category='full',wandb_mode='online',learning_rate=lr,max_steps=steps,
                 warmup_steps=max(1,math.ceil(steps*.02)),eval_steps=max(1,steps//10),
                 eval_documents=cfg['evaluation_documents'], final_evaluation_reserve_seconds=full_seconds)
    submit_worker(cfg,state,'baseline',value,minutes,8)
    state['jobs'][-1]['phase_limit_seconds']=minutes*60
    state['full_evaluation_reserve_seconds']=full_seconds; save(cfg,state)


def cron_update(cfg, config_path, enabled):
    result=subprocess.run(['crontab','-l'],capture_output=True,text=True,timeout=10)
    if result.returncode and 'no crontab' not in result.stderr.lower():
        raise RuntimeError('Cannot inspect crontab')
    lines=[line for line in result.stdout.splitlines() if not line.rstrip().endswith(MARKER)]
    if enabled:
        log=directory(cfg)/'cron.log'
        lines.append('* * * * * /usr/bin/python3 '+shlex.quote(str(PROJECT/'slurm/auto_baseline.py'))+' tick --config '+shlex.quote(str(config_path))+' >> '+shlex.quote(str(log))+' 2>&1 '+MARKER)
    desired='\n'.join(lines)+ ('\n' if lines else '')
    subprocess.run(['crontab','-'],input=desired,text=True,check=True,timeout=10)
    verified=subprocess.check_output(['crontab','-l'],text=True,timeout=10)
    if verified!=desired:
        raise RuntimeError('Crontab readback differs from requested schedule')


def write_report(cfg,state):
    summary={key:state.get(key) for key in ('status','reason','updated_utc','selection','scaling','result','last_error')}
    summary.update(budget=budget(state['jobs']), jobs=[{key:j.get(key) for key in ('phase','job_id','status','exit_code','h200_hours','elapsed_seconds')} for j in state['jobs']])
    summary['artifacts'] = []
    for job in state['jobs']:
        if job.get('run_id'):
            run = Path(cfg['root'])/'runs/experiments'/job['run_id']
            if (run/'summary.json').exists():
                record = read_json(run/'summary.json')
                summary['artifacts'].append({'run_id':job['run_id'], 'directory':str(run),
                    **{k:record.get(k) for k in ('export','export_global_step','wandb_url','total_tokens_seen','last_evaluation')}})
    atomic_json(directory(cfg)/'report.json',summary)
    lines=['# Automatic baseline status','',str(state['status']),'',json.dumps(summary,ensure_ascii=False,indent=2)]
    (directory(cfg)/'progress.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('install','tick','status','disable','guard'))
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/auto-baseline.json')
    parser.add_argument('--job')
    args=parser.parse_args(); cfg=load_campaign(args.config)
    path=directory(cfg)
    if args.action=='guard':
        state=read_state(cfg)
        job=next((j for j in state['jobs'] if j.get('job_id',j.get('returned_job_id'))==args.job),None)
        if job is None or state['status'] not in ('active','disabled'):
            raise ValueError('Job is not registered in an active campaign')
        if state.get('source_hashes') != source_hashes():
            raise ValueError('Campaign source changed; do not execute a queued job with changed code')
        if job['kind']=='training':
            spent=budget(state['jobs'])
            if spent['h200_hours']>16+1e-10 or spent['allocation_seconds']>7200:
                raise ValueError('Campaign budget exceeded before worker startup')
        print(int(dt.datetime.fromisoformat(cfg['allocation_cutoff']).timestamp())); return
    if args.action=='status':
        state=read_state(cfg); print(json.dumps({'directory':str(path),'status':state['status'],'budget':budget(state['jobs']),'jobs':state['jobs'],'reason':state.get('reason'),'last_error':state.get('last_error')},indent=2)); return
    path.mkdir(parents=True,exist_ok=True)
    with locked(path/'controller.lock',blocking=False):
        if args.action=='install':
            if not (path/'state.json').exists():
                research=read_json(PROJECT/'research-state.yaml')
                preledger=read_json(Path(cfg['root'])/'preprocessing-ledger.json')['jobs']
                state={'status':'active','config_path':str(args.config.resolve()),'config_digest':digest(cfg),'jobs':[],'installed_utc':now().isoformat(),'source_hashes':source_hashes()}
                for item in research.get('submission_intents',[]):
                    if item.get('prep_id')==cfg['preparation_id'] and item.get('status')!='not_dispatched' and item.get('job_id') not in preledger:
                        state['jobs'].append({'phase':'data-'+item['stage'],'kind':'data','tag':item['tag'],'submission_tag':item.get('submission_tag',item['tag']),
                                              'submission_started_utc':item.get('submission_started_utc',item['timestamp']),
                                              'job_id':item.get('job_id'),'status':'submitted' if item.get('job_id') else 'submission_uncertain',
                                              'gpu_count':1,'requested_seconds':int(item.get('reserved_h200_hours',24)*3600)})
                save(cfg,state)
            state=read_state(cfg)
            if state['status']!='active':
                raise ValueError('Cannot silently restart a stopped/finished campaign')
            cron_update(cfg,args.config.resolve(),True)
            (PROJECT/'.heartbeat-disabled').touch()
            print('Installed deterministic campaign; existing data job adopted. Run tick/status to inspect.'); return
        state=read_state(cfg)
        if args.action=='disable':
            state['status']='disabled'; save(cfg,state)
            if all(terminal(j) for j in state['jobs']):
                cron_update(cfg,args.config.resolve(),False)
            write_report(cfg,state); return
        try:
            reconcile(cfg,state)
            if state.get('source_hashes') != source_hashes():
                raise ValueError('Campaign source changed; restore pinned source before continuation')
            last = phase_job(state,'baseline')
            completed = last is not None and last['status']=='complete' and (Path(cfg['root'])/'runs/experiments'/last['run_id']/'full-evaluation.json').exists()
            if completed and state['status']=='active':
                training_advance(cfg,state)
            elif state['status']!='active':
                pass
            elif now()>=dt.datetime.fromisoformat(cfg['allocation_cutoff']):
                state.update(status='blocked',reason='Allocation cutoff reached; no new jobs authorized')
            else:
                data_advance(cfg,state)
                training_advance(cfg,state)
            state.pop('last_error',None)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, BlockingIOError) as error:
            state['last_error']=type(error).__name__+': '+str(error)
        except (ValueError,RuntimeError,KeyError) as error:
            # Uncertain submissions must remain recoverable on later bounded ticks.
            uncertain=any(j['status']=='submission_uncertain' for j in state['jobs'])
            state['last_error']=type(error).__name__+': '+str(error)
            if not uncertain:
                state.update(status='blocked',reason=state['last_error'])
        save(cfg,state); write_report(cfg,state)
        if state['status'] in ('achieved','not_achieved','blocked','disabled') and all(terminal(j) for j in state['jobs']):
            cron_update(cfg,args.config.resolve(),False)
        print(json.dumps({'status':state['status'],'budget':budget(state['jobs']),'last_error':state.get('last_error')}))


if __name__=='__main__':
    try:
        main()
    except BlockingIOError:
        print('Controller already active; skipped')
