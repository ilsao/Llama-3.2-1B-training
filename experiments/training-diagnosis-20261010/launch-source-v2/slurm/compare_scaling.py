#!/usr/bin/env python3
"""Compare completed, reconciled 4/8-H200 scaling runs without loading a model."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import DEFAULT_ROOT, identifier, read_json, storage_root


def select_gpu_count(speedup, cost_ratio, stable_updates):
    if (stable_updates >= 20 and math.isfinite(speedup) and math.isfinite(cost_ratio)
            and speedup >= 1.8 and 0 < cost_ratio <= 1.25):
        return 8
    return 4


def compare(root, run4, run8):
    paths = [Path(root) / 'runs/experiments' / identifier(run) for run in (run4, run8)]
    configs = [read_json(path / 'resolved-config.json') for path in paths]
    summaries = [read_json(path / 'summary.json') for path in paths]
    identities = [read_json(path / 'execution-identity.json') for path in paths]
    ignored = {'world_size', 'gradient_accumulation_steps'}
    if ({k: v for k, v in configs[0].items() if k not in ignored}
            != {k: v for k, v in configs[1].items() if k not in ignored}):
        raise ValueError('Scaling runs must use the same recipe, seed and token/step budgets')
    for cfg, count in zip(configs, (4, 8)):
        if (cfg['world_size'] != count or cfg['category'] != 'scaling'
                or cfg['world_size'] * cfg['microbatch_size'] * cfg['gradient_accumulation_steps'] != 16):
            raise ValueError('Compare 4/8 GPU scaling runs with global batch 16')
    if (summaries[0]['data_provenance'] != summaries[1]['data_provenance']
            or any(identities[0][key] != identities[1][key] for key in ('source_hashes', 'environment'))):
        raise ValueError('Scaling data, code or environment changed')
    hours, tokens = [], []
    for path, summary in zip(paths, summaries):
        accounting = read_json(path / 'resource-accounting.json')
        if not accounting['all_jobs_reconciled'] or accounting['budget_exceeded']:
            raise ValueError('Reconcile every scaling allocation within budget before comparing')
        for job in accounting['jobs']:
            if job['state'].split()[0] == 'COMPLETED':
                continue
            exit_path = Path(root) / 'logs' / ('job-' + str(job.get('job_id', ''))) / 'exit-code.txt'
            if (job['state'].split()[0] != 'FAILED' or summary['stop_reason'] != 'wall_time_or_signal'
                    or not exit_path.is_file() or exit_path.read_text().strip() != '75'):
                raise ValueError('Scaling needs successful completion or a confirmed checkpointed exit 75')
        if summary['stop_reason'] == 'nonfinite_or_skipped_update':
            raise ValueError('Nonfinite/skipped updates invalidate scaling')
        hours.append(accounting['h200_hours_measured'])
        tokens.append(summary['total_tokens_seen'])
    if min(hours + tokens) <= 0 or not all(math.isfinite(value) for value in hours):
        raise ValueError('Scaling needs measured, positive training tokens and allocation costs')
    measurements = []
    for path in paths:
        steps = {}
        with (path / 'metrics.jsonl').open() as stream:
            for line in stream:
                metric = json.loads(line)
                if 'train/step_input_tokens' in metric and 'train/step_seconds' in metric:
                    steps[metric['train/global_step']] = metric
        measurements.append(steps)
    warmup = configs[0]['warmup_steps']
    common = sorted(step for step in measurements[0].keys() & measurements[1].keys() if step > warmup)
    stable = common[-20:]
    if any(measurements[0][step]['train/step_input_tokens'] != measurements[1][step]['train/step_input_tokens']
           for step in stable):
        raise ValueError('Matched scaling updates processed different non-padding token counts')
    throughput = []
    for measured in measurements:
        duration = sum(measured[step]['train/step_seconds'] for step in stable)
        inputs = sum(measured[step]['train/step_input_tokens'] for step in stable)
        throughput.append(inputs / duration if duration > 0 else 0)
    speedup = throughput[1] / throughput[0] if throughput[0] > 0 else 0
    costs = [hours[i] / tokens[i] for i in range(2)]
    ratio = costs[1] / costs[0]
    selected = select_gpu_count(speedup, ratio, len(stable))
    return {'runs': [run4, run8], 'selected_gpus': selected, 'stable_updates': len(stable),
            'matched_update_steps': stable, 'tokens_per_second': throughput, 'speedup': speedup,
            'allocation_h200_hours': hours, 'training_tokens': tokens,
            'h200_hours_per_token': costs, 'cost_ratio': ratio,
            'thresholds': {'minimum_speedup': 1.8, 'maximum_cost_ratio': 1.25, 'minimum_updates': 20},
            'reason': '8 GPU meets both thresholds' if selected == 8 else 'Keep 4 GPU: insufficient updates or thresholds unmet'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run4')
    parser.add_argument('run8')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(compare(storage_root(args.root), args.run4, args.run8), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
