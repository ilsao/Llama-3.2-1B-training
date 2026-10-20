#!/usr/bin/env python3
"""Summarize completed validation metadata; no model/data loading or submission."""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/work/asciibase64/Llama-3.2-1B-training')
OUT = ROOT/'runs/experiments/kd3b-io8-validation-20261019/repair1'


def read(path):
    return json.loads(path.read_text())


def main():
    producer, consumer = [read(OUT/role/'results.json') for role in ('producer', 'consumer')]
    assert producer['complete'] and consumer['complete']
    old = read(HERE.parent/'kd3b-tokens-20261019/validation-results.json')
    old_rows = {(r['mode'], r['global_batch']): r for r in old['rows']}
    rows = []
    for p, c in zip(producer['records'], consumer['records']):
        assert p['segment'] == c['segment'] and p['timed_updates'] == c['timed_updates'] == 6
        mode, batch = c['mode'], c['global_batch']
        accum = batch//8
        write_seconds = max(sum(m['write_seconds'] for m in rank[2*accum:]) for rank in p['rank_records'])
        compute_seconds = max(sum(m['compute_and_d2h_seconds'] for m in rank[2*accum:]) for rank in p['rank_records'])
        window_seconds = [max(rank['windows'][i]['seconds'] for rank in c['rank_records']) for i in range(2, 8)]
        reference = old_rows[(mode, batch)]
        row = dict(mode=mode, global_batch=batch,
            teacher_produce_write_tokens_per_second=p['production_tokens_per_second'],
            student_resident_tokens_per_second=c['resident_tokens_per_second'],
            student_disk_tokens_per_second=c['disk_tokens_per_second'],
            producer_over_resident_student=p['production_tokens_per_second']/c['resident_tokens_per_second'],
            disk_slowdown=c['disk_slowdown'],
            six_timed_global_window_seconds=window_seconds,
            slowest_timed_student_tokens_per_second=batch*8192/max(window_seconds),
            payload_bytes_per_update=p['payload_bytes_per_update'],
            producer_write_GB_per_second=6*p['payload_bytes_per_update']/write_seconds/1e9,
            producer_compute_copy_seconds_slowest_rank=compute_seconds,
            producer_write_seconds_slowest_rank=write_seconds,
            teacher_scaling_vs_four_gpu=p['production_tokens_per_second']/reference['producer_tokens_per_second'],
            student_scaling_vs_four_gpu=c['disk_tokens_per_second']/reference['student_disk_tokens_per_second'])
        if mode == 'hidden':
            row['student_speed_ratio_vs_historical_online8'] = c['disk_tokens_per_second']/old['historical_online8gpu'][str(batch)]['median']
        rows.append(row)
    indexed = {(r['mode'], r['global_batch']): r for r in rows}
    plan = read(HERE.parent/'kd3b-tokens-20261019/schedule-plan.json')
    # Include the three truncated BS32 windows at the slower BS16 rate.
    token_counts = {16: (plan['batch_counts']['16']*16+plan['batch_counts']['32']*32)*8192,
                   64: plan['batch_counts']['64']*64*8192}
    assert sum(token_counts.values()) == plan['endpoint_inputs']
    estimates = {}
    for mode in ('hidden', 'logits'):
        seconds = sum(tokens/indexed[(mode, batch)]['student_disk_tokens_per_second'] for batch, tokens in token_counts.items())
        overhead = old['projected_data_journal_logging_seconds']
        estimates[mode] = dict(training_loop_hours=seconds/3600,
            plus_historical_host_overhead_hours=(seconds+overhead)/3600,
            both_eight_gpu_roles_allocated_for_training_loop_h200_hours=16*seconds/3600,
            caveat='Short validation extrapolation; excludes startup/eval/export/safety margin. Historical host overhead may change with the cache implementation. Teacher time includes backpressure until its job can exit.')
    accounting = dict(original=read(HERE/'validation-accounting.json'), replacement=read(HERE/'repair-accounting.json'))
    cost = sum(j['h200_hours'] for group in accounting.values() for j in group.values())
    ledger = read(ROOT/'experiment-ledger.json')
    prefixes = ('methods-parallel-20261019', 'kd-acceleration-', 'kd-pipeline-',
                'kd3b-online-pipeline-', 'kd3b-io-validation-', 'kd3b-io8-validation-')
    study_cost = sum(j.get('h200_hours', j['reserved_h200_hours']) for j in ledger['jobs'].values() if j['run_id'].startswith(prefixes))
    result = dict(complete=True, geometry=dict(teacher_gpus=8, student_gpus=8,
        teacher_cpus=8, student_cpus=8, different_nodes=True), rows=rows,
        projections_3b=estimates, original_and_replacement_accounting=accounting,
        validation_pair_cumulative_h200_hours=cost, study_reserved_or_measured_h200_hours=study_cost,
        input_reservations=dict(teacher=producer['processed_input_reservation'], student=consumer['processed_input_reservation']),
        full_training_submitted=False,
        uncertainty='Six timed windows per format/batch; no long-duration storage guarantee. Historical online8 and four+four runs used different nodes/windows; ratios are planning evidence, not controlled scaling claims.')
    (HERE/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'original_and_replacement_accounting'}, indent=2))


if __name__ == '__main__':
    main()
