"""Read small run metadata and summarize exactly two matched performance arms."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import DEFAULT_ROOT, PROJECT, atomic_json, read_json

DIRECTORY = PROJECT/'experiments/performance-ab-20261010'


def measurements(summary, rows):
    if (summary['global_step'] != 120 or summary['attempted_updates'] != 120 or
            summary['total_tokens_seen'] != 15728640 or summary['budget_tokens_charged'] != 15728640
            or summary['stop_reason'] != 'training_token_target'):
        raise ValueError('Incomplete or mismatched benchmark endpoint')
    if len(rows) != 120 or {row['step'] for row in rows} != set(range(1,121)):
        raise ValueError('Missing, duplicate, or repeated training updates')
    warm = [r for r in rows if r['step'] > 20]
    update_seconds = sum(r['update_seconds'] for r in warm)
    compute_seconds = sum(r['compute_seconds'] for r in warm)
    median = statistics.median(r['update_seconds'] for r in warm)
    cold_estimate = max(0, sum(r['update_seconds'] for r in rows if r['step'] <= 20)-20*median)
    return dict(measured_updates=100, nonpadding_tokens=100*131072,
        end_to_end_tokens_per_second=100*131072/update_seconds,
        compute_tokens_per_second=100*131072/compute_seconds,
        median_update_seconds=median, p95_update_seconds=sorted(r['update_seconds'] for r in warm)[94],
        mean_optimizer_seconds=statistics.mean(r['optimizer_seconds'] for r in warm),
        cold_start_excess_seconds_estimate=cold_estimate,
        perplexity=summary['last_evaluation']['eval/perplexity'])


def compare(a, b):
    ac, bc = dict(a['configuration']), dict(b['configuration'])
    bc.pop('performance', None)
    if ac != bc:
        raise ValueError('Training recipe differs between original and optimized arms')
    if a['initialization']['sample_signature'] != b['initialization']['sample_signature']:
        raise ValueError('Initial random weights differ')
    for field in ('fingerprint','evaluation_indices_sha256','pack_settings'):
        if a['data_provenance'].get(field) != b['data_provenance'].get(field):
            raise ValueError('Data/evaluation provenance differs: '+field)


def main():
    output = {}
    summaries = {}
    for stage in ('A','B'):
        launch = read_json(DIRECTORY/(stage+'-launch.json'))
        if launch.get('slurm_state') != 'COMPLETED' or launch.get('status') != 'reconciled':
            raise ValueError('Require completed and reconciled allocations')
        path = DEFAULT_ROOT/'runs/experiments'/launch['run_id']
        summary = read_json(path/'summary.json')
        if not read_json(path/'export.json')['reload_verified']:
            raise ValueError('HF reload verification incomplete')
        rows = [json.loads(row) for row in (path/'timing-steps.jsonl').read_text().splitlines()]
        value = measurements(summary, rows)
        value.update(job_id=launch['job_id'], accounting=launch['accounting'],
                     wandb_url=summary['wandb_url'], performance=summary.get('performance', {}))
        observer_paths = sorted((DEFAULT_ROOT/'logs'/('job-'+launch['job_id'])).glob('performance-observer-rank-*.json'))
        if len(observer_paths) != 8:
            raise ValueError('Missing rank observations')
        observers = [read_json(p) for p in observer_paths]
        value['peak_allocated_bytes'] = max(v['peak_allocated_bytes'] for v in observers)
        value['peak_reserved_bytes'] = max(v['peak_reserved_bytes'] for v in observers)
        value['rank_compiler_time_seconds_by_category'] = [v['compiler_time_seconds_by_category'] for v in observers]
        if stage == 'B':
            for observer in observers:
                observations = observer['training_graph_observations']
                warm = [r for r in observations if r['step'] >= 20]
                if len(observations) != 120 or not warm or warm[0]['unique_graphs'] <= 0:
                    raise ValueError('Compiled graph evidence missing')
                if len({r['unique_graphs'] for r in warm}) != 1:
                    raise ValueError('Graphs recompiled during measurement interval')
        output[stage] = value
        summaries[stage] = summary
    compare(summaries['A'], summaries['B'])
    a_groups = read_json(DEFAULT_ROOT/'runs/experiments'/'performance-ab-20261010-a'/'optimizer-groups.json')
    b_groups = read_json(DEFAULT_ROOT/'runs/experiments'/'performance-ab-20261010-b'/'optimizer-groups.json')
    for key in ('parameter_names','parameter_counts','tied_aliases'):
        if a_groups[key] != b_groups[key]: raise ValueError('Optimizer parameter routing differs')
    a, b = output['A'], output['B']
    speedup = b['end_to_end_tokens_per_second']/a['end_to_end_tokens_per_second']
    quality_difference = abs(b['perplexity']/a['perplexity']-1)
    savings = a['median_update_seconds']-b['median_update_seconds']
    overhead = max(0,b['cold_start_excess_seconds_estimate']-a['cold_start_excess_seconds_estimate'])
    output.update(end_to_end_speedup=speedup, relative_perplexity_difference=quality_difference,
        recommended=quality_difference <= .01 and speedup >= 1.05,
        compile_break_even_updates_estimate=overhead/savings if savings>0 else None)
    for stage, value in (('A',a),('B',b)):
        s = summaries[stage]
        t = s['timing_totals']
        value['projected_2b_allocation_seconds_estimate'] = (
            t['initialization_max_seconds']+value['cold_start_excess_seconds_estimate']+
            15259*value['median_update_seconds']+
            t['evaluation_seconds']+t['checkpoint_seconds']+
            max(0, value['accounting']['elapsed_seconds']-
                sum(t.get(k,0) for k in ('initialization_max_seconds','training_seconds','data_seconds',
                    'journal_seconds','logging_seconds','evaluation_seconds','checkpoint_seconds'))))
    atomic_json(DIRECTORY/'results.json',output)
    lines=['# 原版與整合優化版實測','', '| 版本 | 端到端 tokens/s | median update 秒 | PPL |',
           '|---|---:|---:|---:|']
    for stage in ('A','B'):
        v=output[stage]
        lines.append(f"| {stage} | {v['end_to_end_tokens_per_second']:.1f} | {v['median_update_seconds']:.4f} | {v['perplexity']:.6f} |")
    lines += ['',f"端到端加速：{speedup:.3f}×；PPL 相對差異：{quality_difference:.3%}。",
              '2B 投影與 cold-start excess 是估計；compiler categories 為巢狀計時，不互相加總。',
              'Optimizer GPU event 時間和原版 host wall 時間口徑不同；主要比較完整 update 時間。',
              '評估範圍：固定 1000 reserved documents，非完整 OJ。']
    (DIRECTORY/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__': main()
