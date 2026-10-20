# Lossless KD performance validation, 2026-10-20

The user requests bottleneck analysis and a way to fit 6B-token distillation
within 64 H200-hours, and explicitly defines lossless as mathematical equivalence
with BF16 rounding differences permitted. This is a bounded validation of that
request, not approval to launch a new formal 6B training run.

One allocation at most: 8 H200, 8 CPUs, one launcher/eight ranks, 20 minutes,
2.6666666667 H200-hours including startup/compilation/checks. No retries, resume,
requeue, trained checkpoint initialization/export, dependency changes or cleanup.
Keep all prior formal identities and costs. Submit only through launch.py after
the current 3B pair is terminal/reconciled; reserve in the shared ledger and
require measured/reserved methods-study cost plus this allocation <=191.419H.
This is a validation category, not an additional formal training run.

Use sealed 3B recipe/source, pinned raw preparation, random residual-half seed42
1B student, official pretrained 3B teacher, context8192, actual intradoc spans.
Teacher and student share the same eight GPUs. All four arms use serial teacher
backbone execution followed by the student; do not confuse their timings with
the historical logits-prefetch pipeline. Teacher final hidden stays on-device.
No disk spool and no extra GPU allocation for the teacher.

Paired arms: checkpoint+eager (control), no-checkpoint+eager, no-checkpoint+fused
CE/KL, checkpoint+fused. Preserve full128256 vocabulary, T1, .75hard CE+.25
KL(teacher||student), ignore_index=-100, tied head, original optimizers/NS5 and
global batch16/64 with micro1 and accumulation2/8. The prototype fuses only
softmax, CE, KL and dlogits; projection and head-gradient GEMMs remain the same.
FP32 normalization/reductions, BF16 GEMMs, no gradient filtering/quantization/
top-k/teacher replacement. It is mathematically equivalent, not bitwise exact.

First test synthetic FP32/BF16 loss and gradients, lambda0/.25/1, partial/all
ignored labels, nonmultiple chunk sizes and differing teacher/student widths.
Compare against both sealed precomputed gradients and autograd. Then compare
full8192-token,128256-vocabulary student gradients using real canonical document
spans for checkpoint+eager versus no-checkpoint+fused, before timings. Gate:
synthetic FP32 relative L2<2e-5, BF16<.01, full-model relative L2<.01 and absolute
normalized loss difference<.001. These tolerances are numerical screening, not
proof of long-run trajectory equality. A nonzero-LR multistep optimizer/heldout
gate remains necessary before a production change.

Each arm: three warmups plus15 measured updates at each batch size. Use original
AdamW/NorMuon computations with LR/decay LR zero for timing only; optimizer
states warm up, weights remain the same. Reinitialize optimizer state per arm.
Record max-rank wall time, rank0 CUDA stage events, peak memory, and every real
teacher/student input including repeats/warmups/gradient checks. Separate role
token counters, each capped80M. Synthetic checks are recorded separately.

Timing includes on-device teacher, student body/head/backward, DDP, clipping and
optimizers; excludes durable token journal/disk dataset loading. Forecast6B as
3B/rate_BS16+3B/rate_BS64 (harmonic token weighting), add10% runtime margin and
1800s startup/evaluation/checkpoint margin. Count all8GPUs throughout, and show
both full-run hours and hours including the entire2.667H validation allocation.
Final endpoint rounding, 6B schedule and full-run authorization are separate;
do not inherit another run's permission to exceed6,000,000,000 inputs.

Evidence limitations: small repeated canonical windows, random weights, one
seed/node and short measurements. Speedups require actual GPU results; AST and
shell checks alone do not validate CUDA kernels. OOM/numerical failure stops
the allocation; preserve logs/costs and do not automatically resubmit.
