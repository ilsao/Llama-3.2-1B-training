"""Two bounded matched-global-batch validations; metadata-only on login node."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import (DEFAULT_ROOT, PROJECT, TERMINAL, ExperimentLedger,
                            atomic_json, file_digest, load_config, locked, read_json,
                            window_indices)
from slurm.submit_training import mark_uncertain, recover_submission
from slurm.performance_report import measurements

CAMPAIGN = 'accumulation-ab-20261013'
DIRECTORY = PROJECT / 'experiments' / CAMPAIGN
CONFIGS = PROJECT / 'configs' / CAMPAIGN
SOURCE = PROJECT / 'experiments/performance-ab-20261010/source-b'
ARMS = {'on': (1, 2), 'off': (2, 1)}


def check():
    seals = [
        (SOURCE, PROJECT / 'experiments/performance-ab-20261010/source-b-seal.json'),
        (PROJECT, DIRECTORY / 'execution-seal.json')]
    for base, seal in seals:
        for name, expected in read_json(seal).items():
            if file_digest(base / name) != expected:
                raise ValueError('Sealed source/configuration changed: ' + name)
    configs = [load_config(CONFIGS / (arm + '.json')) for arm in ARMS]
    common = []
    for cfg, (microbatch, accumulation) in zip(configs, ARMS.values()):
        if (cfg['category'] != 'validation' or cfg['world_size'] != 8 or
                cfg['microbatch_size'] != microbatch or
                cfg['gradient_accumulation_steps'] != accumulation or
                cfg['max_h200_hours'] != 2 or
                cfg['max_training_tokens'] != 15_728_640 or
                cfg['target_training_tokens'] != 15_728_640 or
                cfg['eval_steps'] != 12 or cfg['export_on_completion']):
            raise ValueError('Require exactly the bounded accumulation comparison')
        common.append({k: v for k, v in cfg.items() if k not in
                       ('microbatch_size', 'gradient_accumulation_steps')})
    if common[0] != common[1]:
        raise ValueError('Recipes may differ only in microbatch/accumulation')
    # Rank ownership changes, but each optimizer update sees identical sequences.
    for step in range(120):
        for microbatch, accumulation in ARMS.values():
            indices = [i for rank in range(8) for batch in
                       window_indices(step * 16, 1920, 8, rank, microbatch, accumulation)
                       for i in batch]
            if sorted(indices) != list(range(step * 16, (step + 1) * 16)):
                raise ValueError('Global sequence membership differs')
    return configs


def submit(arm):
    configs = check()
    path = DIRECTORY / (arm + '-launch.json')
    if path.exists():
        raise ValueError('Existing submission; recover/reconcile, never retry automatically')
    node = None
    if arm == 'off':
        previous = read_json(DIRECTORY / 'on-launch.json')
        if previous.get('slurm_state') != 'COMPLETED' or previous['status'] != 'reconciled':
            raise ValueError('Require completed and reconciled accumulation-on arm')
        summary = read_json(DEFAULT_ROOT / 'runs/experiments' / previous['run_id'] / 'summary.json')
        rows = read_rows(DEFAULT_ROOT / 'runs/experiments' / previous['run_id'] / 'timing-steps.jsonl')
        measurements(summary, rows)
        node = previous['accounting']['nodes']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', node):
            raise ValueError('Require a single verified node for the second arm')
    cfg = configs[list(ARMS).index(arm)]
    config = CONFIGS / (arm + '.json')
    run_id = CAMPAIGN + '-' + arm
    tag = 'accumulation-' + uuid.uuid4().hex
    command = ['sbatch', '--parsable', '--comment=' + tag,
               '--job-name=accum-' + arm]
    if node:
        command.append('--nodelist=' + node)
    command += [str(PROJECT / 'slurm/accumulation-ab.sbatch'), run_id, str(config)]
    ledger = ExperimentLedger(DEFAULT_ROOT)
    DEFAULT_ROOT.joinpath('logs').mkdir(parents=True, exist_ok=True)
    ledger.reserve(tag, run_id, cfg, 900, 8, False)
    state = dict(arm=arm, run_id=run_id, reservation=tag,
                 source_root=str(SOURCE), configuration=str(config),
                 configuration_sha256=file_digest(config),
                 status='submission_uncertain', submission_command=command,
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(path, state)
    with ledger.transaction() as data:
        data['jobs'][tag].update(submission_tag=tag, submission_command=command,
            submission_started_utc=state['started_utc'],
            allocation_kind='gradient_accumulation_validation')
    launcher = PROJECT / 'slurm/accumulation_entry.py'
    env = {**os.environ, 'DATA_ROOT': str(DEFAULT_ROOT),
           'TRAIN_SOURCE_ROOT': str(SOURCE),
           'PERFORMANCE_LAUNCHER': str(launcher),
           'PERFORMANCE_LAUNCHER_SHA256': file_digest(launcher),
           'ACCUMULATION_CONTROLLER': str(PROJECT / 'slurm/accumulation_ab.py')}
    try:
        receipt = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    except OSError:
        ledger.cancel_reservation(tag)
        state['status'] = 'submission_not_executed'
        atomic_json(path, state)
        raise
    except subprocess.TimeoutExpired:
        mark_uncertain(ledger, tag)
        raise
    state['submission_result'] = dict(stdout=receipt.stdout, stderr=receipt.stderr,
                                      returncode=receipt.returncode)
    atomic_json(path, state)
    job = receipt.stdout.strip().split(';')[0]
    if receipt.returncode or not re.fullmatch(r'\d+', job):
        mark_uncertain(ledger, tag)
        raise ValueError('Uncertain submission; retain reservation and recover, do not retry')
    with ledger.transaction() as data:
        data['jobs'][tag]['returned_job_id'] = job
    ledger.bind(tag, job)
    state.update(status='submitted', job_id=job)
    atomic_json(path, state)
    print('Submitted', arm, job, 'maximum 2 H200-hours')


def status():
    for path in sorted(DIRECTORY.glob('*-launch.json')):
        state = read_json(path)
        if state['status'] == 'submission_not_executed':
            print(state['arm'], state['status'])
            continue
        if not state.get('job_id'):
            state['job_id'] = recover_submission(ExperimentLedger(DEFAULT_ROOT), state['reservation'])
            state['status'] = 'submitted'
        if state['status'] != 'reconciled':
            output = subprocess.run(['sacct', '-X', '-j', state['job_id'], '--noheader',
                '--parsable2', '--format=JobIDRaw,State%30'], capture_output=True,
                text=True, check=True, timeout=20).stdout
            rows = [row.split('|') for row in output.splitlines()]
            state['slurm_state'] = next((r[1].split()[0].rstrip('+') for r in rows
                                        if r[0] == state['job_id']), 'UNKNOWN')
            if state['slurm_state'] in TERMINAL:
                state['accounting'] = ExperimentLedger(DEFAULT_ROOT).reconcile(state['job_id'])
                state['status'] = 'reconciled'
            atomic_json(path, state)
        print(state['arm'], state['job_id'], state.get('slurm_state'), state['status'])


def read_rows(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def report():
    check()
    output, summaries, metrics, groups = {}, {}, {}, {}
    for arm, (microbatch, accumulation) in ARMS.items():
        launch = read_json(DIRECTORY / (arm + '-launch.json'))
        if launch.get('slurm_state') != 'COMPLETED' or launch['status'] != 'reconciled':
            raise ValueError('Require completed and reconciled allocations')
        path = DEFAULT_ROOT / 'runs/experiments' / launch['run_id']
        summary = read_json(path / 'summary.json')
        if summary['configuration'] != load_config(CONFIGS / (arm + '.json')):
            raise ValueError('Resolved configuration differs from sealed config')
        if summary['initialization']['pretrained_weights_loaded']:
            raise ValueError('Require fresh random weights')
        value = measurements(summary, read_rows(path / 'timing-steps.jsonl'))
        value.update(microbatch_size=microbatch, gradient_accumulation_steps=accumulation,
                     job_id=launch['job_id'], accounting=launch['accounting'],
                     wandb_url=summary['wandb_url'])
        observer_paths = sorted((DEFAULT_ROOT / 'logs' / ('job-' + launch['job_id'])).glob(
            'accumulation-observer-rank-*.json'))
        if len(observer_paths) != 8:
            raise ValueError('Missing rank observations')
        observers = [read_json(p) for p in observer_paths]
        for observer in observers:
            obs = observer['training_graph_observations']
            warm = [r for r in obs if r['step'] >= 20]
            if (len(obs) != 120 or {r['step'] for r in obs} != set(range(1, 121)) or
                    any(r['microbatch_size'] != microbatch for r in obs) or
                    warm[0]['unique_graphs'] <= 0 or
                    len({r['unique_graphs'] for r in warm}) != 1):
                raise ValueError('Incomplete geometry or recompilation during measurement')
        value['peak_allocated_gib'] = max(v['peak_allocated_bytes'] for v in observers) / 2**30
        value['peak_reserved_gib'] = max(v['peak_reserved_bytes'] for v in observers) / 2**30
        metric_rows = [r for r in read_rows(path / 'metrics.jsonl') if 'train/loss' in r]
        if (len(metric_rows) != 120 or
                {r['train/global_step'] for r in metric_rows} != set(range(1, 121)) or
                any(r['train/step_input_tokens'] != 131072 for r in metric_rows)):
            raise ValueError('Missing metrics or incorrect input token accounting')
        metrics[arm] = sorted(metric_rows, key=lambda r: r['train/global_step'])
        groups[arm] = read_json(path / 'optimizer-groups.json')
        output[arm], summaries[arm] = value, summary
    a, b = summaries['on'], summaries['off']
    for key in ('code_digest', 'environment_digest', 'cursor'):
        if a[key] != b[key]:
            raise ValueError('Different experiment identity: ' + key)
    if a['initialization']['sample_signature'] != b['initialization']['sample_signature']:
        raise ValueError('Different initial weights')
    for key in ('fingerprint', 'evaluation_indices_sha256', 'pack_settings'):
        if a['data_provenance'].get(key) != b['data_provenance'].get(key):
            raise ValueError('Different data identity: ' + key)
    for key in ('parameter_names', 'parameter_counts', 'tied_aliases'):
        if groups['on'][key] != groups['off'][key]:
            raise ValueError('Different optimizer parameter routing')
    for on, off in zip(metrics['on'], metrics['off']):
        for key in ('train/step_prediction_tokens', 'train/total_tokens_seen',
                    'train/learning_rate', 'train/lr_muon'):
            if on[key] != off[key]:
                raise ValueError('Different batch masks/LR/tokens: ' + key)
    speedup = output['off']['end_to_end_tokens_per_second'] / output['on']['end_to_end_tokens_per_second']
    quality = abs(output['off']['perplexity'] / output['on']['perplexity'] - 1)
    first_loss = abs(metrics['off'][0]['train/loss'] / metrics['on'][0]['train/loss'] - 1)
    first_norm = abs(metrics['off'][0]['train/grad_norm'] / metrics['on'][0]['train/grad_norm'] - 1)
    loss_diffs = [abs(off['train/loss'] / on['train/loss'] - 1)
                  for on, off in zip(metrics['on'], metrics['off'])]
    consistent = quality <= .01 and first_loss <= .005 and first_norm <= .03
    winner = ('off' if speedup >= 1.05 else 'on' if speedup <= 1 / 1.05 else 'inconclusive')
    output.update(off_vs_on_speedup=speedup, relative_perplexity_difference=quality,
                  first_step_relative_loss_difference=first_loss,
                  first_step_relative_grad_norm_difference=first_norm,
                  mean_relative_training_loss_difference=statistics.mean(loss_diffs),
                  numerical_consistency_passed=consistent,
                  recommendation=winner if consistent else 'numerical_mismatch',
                  total_h200_hours=sum(v['accounting']['h200_hours'] for v in
                                       (output['on'], output['off'])),
                  limitations='One short trial per arm, within 170-update LR warmup; scalar loss/gradient norm and held-out PPL checks, not full-gradient equivalence or final-quality evidence.')
    atomic_json(DIRECTORY / 'results.json', output)
    lines = ['# 梯度累積開關比較', '',
        '| 梯度累積 | 每 GPU batch | 累積次數 | tokens/s | median update 秒 | 峰值 allocated GiB | PPL |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        v = output[arm]
        lines.append(f"| {arm} | {v['microbatch_size']} | {v['gradient_accumulation_steps']} | "
                     f"{v['end_to_end_tokens_per_second']:.0f} | {v['median_update_seconds']:.4f} | "
                     f"{v['peak_allocated_gib']:.2f} | {v['perplexity']:.6f} |")
    lines += ['', f'不累積 / 累積吞吐比：{speedup:.4f}×；PPL 相對差 {quality:.4%}。',
              f'數值檢查：{consistent}；建議：{output["recommendation"]}。',
              f'兩組實際 allocation：{output["total_h200_hours"]:.6f} H200-hours。',
              '同一節點、8 H200 / 8 CPU、8192 context、global batch 16、相同初始化與每 update 的 global sequences。',
              '21–120 updates 的完整 update 計時含資料、token journal、training、logging，排除 evaluation；另記錄 compute throughput。',
              '獨立冷編譯 cache，前20 updates排除；allocation成本仍包含暖機、編譯、評估與checkpoint。',
              '峰值顯存包含 initialization/evaluation/checkpoint；每 rank 原始觀測留在 job logs。',
              '固定1000 reserved documents，未評估完整OJ。120 updates仍在170 warmup內，只能比較效能及短期數值一致性。']
    (DIRECTORY / 'results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: output[k] for k in ('off_vs_on_speedup', 'relative_perplexity_difference',
          'numerical_consistency_passed', 'recommendation', 'total_h200_hours')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'on', 'off', 'status', 'report'))
    args = parser.parse_args()
    if args.action == 'check':
        check()
        print('Source/config seals and all 120 global batch memberships verified')
    else:
        with locked(DIRECTORY / '.controller.lock', blocking=False):
            if args.action in ARMS:
                submit(args.action)
            elif args.action == 'status':
                status()
            else:
                report()
