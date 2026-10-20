#!/usr/bin/env python3
"""Read only small scalar run metadata; no model/data loads on the login node."""
import datetime
import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
WORK=Path('/work/asciibase64/Llama-3.2-1B-training')
RUN=WORK/'runs/experiments/kd3b-hidden8-20261019-seed42'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
summary=read(RUN/'summary.json')
accounting=read(HERE.parent/'kd3b-hidden8-20261019/accounting.json')
progress=read(RUN/'transport/producer/progress.json')
timings=[json.loads(s) for s in (RUN/'timing-steps.jsonl').read_text().splitlines()]
metrics=[]
for line in (RUN/'metrics.jsonl').open():
    x=json.loads(line)
    if 'train/global_batch_sequences' in x and 'train/tokens_per_second' in x:
        metrics.append({k:v for k,v in x.items() if k in (
            'train/global_batch_sequences','train/tokens_per_second','train/step_seconds',
            'train/peak_allocated_bytes','train/peak_reserved_bytes',
            'train/hidden_wait_seconds_cumulative_max_rank','train/hidden_read_seconds_cumulative_max_rank')})
totals={k:sum(r[k] for r in timings) for k in ('update_seconds','compute_seconds','optimizer_seconds')}
measured={}
for width in (16,64):
    rows=[m for m in metrics if m['train/global_batch_sequences']==width]
    measured[str(width)]={k:statistics.median(r[k] for r in rows) for k in rows[0] if all(k in r for r in rows)}
student_hours=accounting['575377']['h200_hours']
teacher_hours=accounting['575376']['h200_hours']
ratio=6e9/summary['total_tokens_seen']
local=summary['teacher_accounting']
evidence=dict(
    generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    lossless_definition='mathematical equivalence; BF16 operation-order rounding permitted by user',
    baseline=dict(tokens=summary['total_tokens_seen'],updates=summary['global_step'],
        teacher_h200_hours=teacher_hours,student_h200_hours=student_hours,
        combined_h200_hours=teacher_hours+student_hours,
        projected_6b_h200_hours=(teacher_hours+student_hours)*ratio,
        projected_6b_student_only_h200_hours=student_hours*ratio,
        required_cost_speedup=(teacher_hours+student_hours)*ratio/64,
        fixed_1k_ppl=summary['last_evaluation']['eval/perplexity'],
        endpoint_10k_ppl=read(HERE.parent/'kd3b-hidden8-20261019/status.json')['endpoint_10k_perplexity']),
    student=dict(timing_sums=totals,timing_totals=summary['timing_totals'],
        optimizer_fraction_of_compute=totals['optimizer_seconds']/totals['compute_seconds'],
        host_fraction_of_update=(totals['update_seconds']-totals['compute_seconds'])/totals['update_seconds'],
        maximum_logged_exposed_read_wait_seconds=max(m.get('train/hidden_wait_seconds_cumulative_max_rank',0) for m in metrics),
        rank0_read_seconds=local['local_read_seconds'],rank0_exposed_wait_seconds=local['local_exposed_wait_seconds'],
        logged_medians=measured,note='Optimizer and exposed wait are nested inside compute; not additive with it.'),
    teacher=dict(progress,rank_scope='service=max of cumulative rank sums; local_write/backpressure=rank0; do not subtract mixed scopes'),
    transport=dict(bytes_per_prediction_position=3072*2,
        bytes_for_6b_inputs=6_000_000_000*(8191/8192)*3072*2,
        read_plus_write_bytes_for_6b=2*6_000_000_000*(8191/8192)*3072*2,
        tokens_buffered_in_1tb=1e12/((8191/8192)*3072*2)),
    targets=dict(minimum_tokens_per_second_8gpu=6e9/(64*3600/8),
        minimum_tokens_per_second_16gpu=6e9/(64*3600/16),
        required_8gpu_rate_with_10pct_runtime_margin_30min_overhead_and_validation=1.1*6e9/(64*3600/8-1800-1200),
        note='Rate is token-weighted harmonic across BS16 and BS64. Full-run cost counts all allocated teacher+student GPUs.'),
    baseline_sources={str(p):sha(p) for p in [RUN/'summary.json',RUN/'timing-steps.jsonl',RUN/'transport/producer/progress.json',HERE.parent/'kd3b-hidden8-20261019/accounting.json']})
candidate=WORK/'runs/experiments/kd6b-lossless-validation-20261020'
for f in ('small-gradient-checks.json','full-model-gradient-check.json','timings.json','results.json'):
    if (candidate/f).exists(): evidence.setdefault('validation',{})[f]=read(candidate/f)
if (HERE/'accounting.json').exists():
    evidence['validation_accounting']=read(HERE/'accounting.json')
    validation_seconds=evidence['validation_accounting']['h200_hours']*3600/8
    evidence['targets']['required_8gpu_rate_using_actual_validation_cost']=1.1*6e9/(64*3600/8-1800-validation_seconds)
if (candidate/'results.json').exists():
    best=[r for r in read(candidate/'results.json')['records'] if r['arm']=='no_checkpoint_fused']
    def cost_without(stage):
        return 8/3600*sum(3e9/(8192*r['global_batch'])*(r['seconds_per_update']-r['stage_mean_ms_rank0'][stage]/1000) for r in best)
    evidence['optimistic_sensitivity_not_measured']=dict(
        note='Subtract rank0 event time from max-rank wall time; hypothetical free stage, not a measured speedup.',
        full_6b_training_h200_hours_if_optimizer_free=cost_without('optimizer_ms'),
        full_6b_training_h200_hours_if_head_kd_free=cost_without('head_kd_ms'))
(HERE/'analysis.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(baseline=evidence['baseline'],targets=evidence['targets'],validation_files=list(evidence.get('validation',{}))),indent=2))
