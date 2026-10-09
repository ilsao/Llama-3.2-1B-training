# Native Muon × AdamW × scheduler sweep

Implementation lives in the isolated `sweep/muon` worktree. Native PyTorch
2.9.1+cu126 and FA3 are installed in
`/work/asciibase64/Llama-3.2-1B-training/environments/training-muon-torch291-fa3-94e22c9`.
Dependency job 516510 completed, costing 0.333333 H200-hours. GPU runtime checks are the first automated allocations; installation alone does
not prove kernel/optimizer compatibility. The original baseline and its environment stay
frozen. The user authorized the dedicated automatic controller and Slurm launch.

## Why this design

Baseline completed 8,903 updates / 1,166,934,016 actual input tokens. Best export
at step 8,848 has full 50,000-document PPL **25.1114**, compared with fixed-subset
PPL 25.0172. It did not reach PPL <20. Subset PPL fell from 38.90 at 0.387B tokens,
to 27.71 at 0.760B and 25.02 at 1.160B. This supports further recipe experiments,
not a guarantee that increasing LR will reach the target. The cosine schedule
finished at step 7,621; the final 14.4% of updates ran at its floor. New schedules
finish on the final planned update. WSD is never pruned before finishing cooldown.
Training compute took 81.57 minutes out of the 96.05-minute allocation; 1,000-doc
evaluation took about 3.4 seconds each. Instrumentation measures initialization,
data preparation/wait, journal, logging, training, optimizer, evaluation and saving.
The required throughput metric still excludes evaluation and checkpointing.

## Matrix and invariants

There are **18 candidates**, prepared in `configs/optimizer-sweep/`:

| Optimizer | Peak LR | Hidden matrix WD | Schedulers |
| --- | --- | --- | --- |
| AdamW | 0.0003 / 0.0006 / 0.0012 | 0.1 | linear / cosine / WSD |
| native Muon | 0.02 / 0.04 / 0.08 | 0.1 / 0.05 / 0.025 | linear / cosine / WSD |

Muon uses `torch.optim.Muon` exclusively: `adjust_lr_fn=original`, momentum .95,
Nesterov, 5 Newton–Schulz iterations, eps 1e-7 and native coefficients
(3.4445, -4.775, 2.0315). Its LR×WD is .002 across candidates; this controls decay
within the Muon LR search, **not between AdamW and Muon**. These are recipe
comparisons, not a pure optimizer-only ablation. Tied embeddings/output head,
norms and biases use auxiliary AdamW LR .0003 with no decay; unknown non-Linear
matrices stay on AdamW with WD .1. Identity deduplication assigns every trainable
parameter once. Required `train/learning_rate` is the Muon recipe's auxiliary
`adamw_no_decay` group; all group LRs and decay factors are logged separately.

All runs start from random weights, seed 42, original Llama 3.2 1B architecture,
8192 context, BF16 autocast with FP32 parameter/optimizer storage, FA3 and matched
global batch 16 (8 ranks × microbatch 1 × accumulation 2). Canonical raw preparation
`dolma-raw-baseline-20261008` supplies an identical training prefix and fixed 1,000
original reserved evaluation documents. Native AdamW controls use the same new
environment as Muon; historical baseline curves are supporting evidence only.

Warmup = max(50, steps/50); all schedules end at 10% peak LR. Linear/cosine decay
from the end of warmup; WSD stays at peak until its final 20% linear-decay updates.
Both optimizers share the multiplier. Evaluation occurs at each 10% and the end;
PPL is exp(global NLL / global prediction tokens). Finite loss/global gradients
are required, and nonfinite evaluation also records a failed numerical run.

## Calibrate and freeze before execution

The checked-in 100-step configurations are **templates, not measured budgets**.
The submission wrapper rejects them until a stage is frozen from calibration.
No torch/model/data-processing checks run on the login node.

The campaign first submits four calibrations sequentially: AdamW4, AdamW8,
Muon4, Muon8. Each requests at most 10 minutes and 8 CPUs. They test FA3
GQA/causal/padding/8192/backward, native optimizer routing, token-weighted DDP,
and exact optimizer/scheduler/RNG replay for all three schedules. Full-size
calibration checkpoints step 1, resumes in the same allocation, and completes
30 updates; timings exclude the first 5. Node 25a-hgpn062 is excluded. Failed
or incomplete calibration evidence stops new submissions before the next job.
After all four finish, verified timing, hardware, source, initialization,
preparation and environment evidence are frozen in calibration.json under
`/work/.../runs/optimizer-sweep/optimizer-scheduler-raw-20261008/`.

Preparation freezes the same horizon for every candidate in a stage using:

`floor_to_multiple_50(0.8 × (stage_seconds − overhead_reserve) / slowest_8GPU_update_seconds)`

Use measured p95 full-update duration, including data/journal/logging overhead.
Overhead includes maximum initialization plus 15 seconds Slurm startup allowance,
10 evaluations, final checkpoint, and a stopping margin of at least 60 seconds,
increased to checkpoint+evaluation+30 seconds when necessary. Clamp to the current
published raw buffer and the 6B-token limit. Reject budgets below 100 updates.
Configuration, source hashes, environment digest, evaluation identity and horizon
are immutable. A completed stage cannot be rerun under the same IDs.

## Fully automatic campaign

From the isolated worktree:

```bash
python3 slurm/auto_optimizer_sweep.py launch
python3 slurm/auto_optimizer_sweep.py status
python3 slurm/auto_optimizer_sweep.py disable
```

Launch freezes source and configurations, installs a separate minute cron while
preserving existing entries, persists the submission intent and shared-ledger
reservation, then submits the first calibration. It is idempotent for an active
campaign and never resets a terminal campaign or reuses a run ID.
One job runs at a time; each tick recovers uncertain submissions and reconciles
terminal allocations before submitting the next. Every job uses one Slurm launcher
and one CPU thread per rank. No model/data work happens in the login controller.

Successful calibration automatically starts all 18 initial candidates, then
promotes eligible survivors to short ranking. The recorded ambiguity criteria
below determine whether the two medium finalists are necessary. Each stage
restarts from the same random seed and canonical prefix. Failed/incomplete sweep
candidates are excluded without retry. Calibration failure, changed frozen
identity, or insufficient capacity for 100 complete updates stops new jobs.
Disabling preserves the minute tick for receipt recovery and resource accounting.

All state, configs, reports and controller.log are under
`/work/asciibase64/Llama-3.2-1B-training/runs/optimizer-sweep/optimizer-scheduler-raw-20261008/`.
Completion writes summary.json and best-config.json with ranking, W&B links and
actual costs. The best-config file describes the winning pilot; it is not an
automatically approved configuration for a longer final training run.
Do not activate the older stage-scoped cron while this campaign owns submissions.

Each candidate's entire allocation cap is 5 minutes initial / 15 minutes short /
20 minutes medium. Finish matched tokens early when possible. **16 H200-hours /
7200 seconds are per sbatch per the user's latest override**; previous baseline
spending is not deducted from new job allocations. The shared ledger still records
all pilots, validation, failed jobs and continuations. Original 6B processed-input
and 64 H200-hour logical-run ceilings and the three full-run limit remain.

Initial pruning is within each optimizer/LR recipe: remove only the worst of
three completed schedulers if it trails the same best scheduler by at least .10
NLL/token at both 90% and 100% matched tokens and both endpoint rankings agree.
Keep ambiguous candidates, normally 12 and at most 18. Incomplete and nonfinite
runs are recorded and excluded rather than silently retried or promoted.

Short ranking automatically selects the lowest final NLL AdamW and Muon across
LRs/schedulers. Medium runs only if their ranking reverses between 90%/100%,
or their final NLL gap is below .05. Otherwise the campaign finishes after short.
There is no automatic formal-budget run or HF upload.

Interruptions retain a complete checkpoint, both optimizer/scheduler states, RNG,
cursor, W&B identity, prefix receipt and durable high-water token charges. For a
manual same-stage continuation, reconcile the job and use `submit_training.py
resume RUN_ID --prep-id dolma-raw-baseline-20261008 --config FROZEN_CONFIG --minutes
STAGE_MINUTES`. Keep geometry/config/code/environment identical and reconcile
that allocation before rerunning the stage tick. Replayed inputs still count;
exceeding the frozen token allowance leaves the candidate incomplete. No automatic
retry after numerical failure is implemented.

Pilots save only at completion/interruption, retain all checkpoints, and skip HF
export. No autonomous artifact cleanup or HF upload occurs. Typical sweep cost is
12 + 24 + optional 5.333 = **41.333 H200-hours**; ambiguous short-stage survivors
can raise this to **53.333**. Four calibration reservations add at most **4**;
actual costs reconcile from Slurm. Each stage reports measured and outstanding
costs. Pilot rankings do not prove full 50,000-document PPL <20; final training
and full evaluation require their own explicit configuration.

## Validation status

Lightweight scheduler/config/budget/controller/recovery tests and end-to-end
campaign simulations run on the login node without torch/model imports. Runtime
checks execute only in the calibration sbatch jobs. See validation.json for the
recorded static checks and campaign.json for actual submission/runtime status.
