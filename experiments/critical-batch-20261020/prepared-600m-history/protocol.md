# Critical batch size experiments — implementation prepared, not submitted

## Question and reference

Does adapting the online KD student's global batch to gradient noise scale
improve held-out NLL at matched input tokens, or attain equal NLL in less
allocated H200 time than the R5 fixed16→64 recipe?

Reference code: [optim_dynamics.py, pinned commit7696fdb](https://github.com/yukicoder0509/training-llama-3-2/blob/7696fdbca9de69c8ca76d9aa63d2752b267a5115/llama_train/optim_dynamics.py).
Its `GradNoiseScale` uses two independent gradients at the same weights,
G²=g1·g2, trΣ=b[(||g1||²+||g2||²)/2−g1·g2]. It smooths neither internally.
Source provenance/checksum is in `reference-provenance.json`.
Theory: [McCandlish et al., An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162).
B_simple is a useful noise proxy, not an exact Muon/AdamW critical batch or a
quality guarantee. Curvature and nonlinear optimizer preconditioning are absent.
Raw gradient group estimates are logged for global, hidden matrices, and
embedding/norms; only the global statistic controls the batch in this experiment.

## Experimental arms

One seed42, random residual-half student, official pretrained3B teacher.
All arms preserve R5 full-vocabulary KD(.75CE+.25KL,T1), dataset/split/order,
8192 intradoc context, microbatch2,8GPU/8CPU, optimizer groups/formula, clipping,
table LRs with AdamW×2, and original6B **token-clock** phases/warmup/decay.
No LR scaling, early2.4B decay, optimizer EMA correction, or WD rescaling is added.
Changing batch inherently changes update frequency, per-token momentum memory
and ordinary per-update weight decay; that is part of this policy comparison,
not a claim that these effective dynamics stay constant.

| Arm | Batch rule | Purpose |
|---|---|---|
| Existing R5 |16→64 at3,000,107,008 inputs | Historical reference, no extra cost |
| C0 shadow |Same16→64; estimate but do not act | Matched estimator/communication control |
| C1 CBS-half |Nearest power-of-two batch to0.5×smoothed noise scale | Conservative adaptive policy |
| C2 CBS-one |Nearest power-of-two batch to1.0×smoothed noise scale | More aggressive policy |

Physical global batch choices16/32/64/128, accumulation1/2/4/8. Event boundaries
can shorten the current update to16/32/48/…/112 sequences without losing a row.
Controller targets use full batch choices, not these boundary-shortened widths.
Initial batch16; no decisions during phase warmup; EMA beta.9 over signal/noise
separately; at least8 valid observations; guarded20% hysteresis; at most2x change
per observation. Decreases are allowed. Invalid samples hold state; negative
individual signal estimates are smoothed rather than clamped into a huge batch.
If smoothed signal is nonpositive, keep the current batch.

Probe every16,777,216 input tokens, aligned identically for all arms, about35
observations in the short screen. Probe samples are regular consumed training
windows at identical student weights, before clipping, without optimizer steps
between rank-local gradients. All backward passes in a probe window use
`DDP.no_sync`, followed by exactly one manual chunked average of each gradient.
There are **no additional teacher/student forward or backward passes**, no
evaluation documents used for probes, and no extra processed tokens.
The additional overhead is norms/reductions and lost DDP overlap on probe steps;
it is unmeasured until the GPU comparison. `probe_sync_seconds` includes the
necessary gradient reduction and must not be called pure incremental overhead.

## Estimator with unequal label counts

Let n_r be local valid prediction tokens,N=Σn_r,K=8, and g_r its mean gradient.
The current objective creates h_r=K n_r g_r/N before synchronization.
Compute L=Σ(n_r/N)||g_r||²=(N/K²)Σ||h_r||²/n_r and
M=||Σh_r/K||². Then

    trΣ_hat = N(L−M)/(K−1)
    G²_hat  = M−trΣ_hat/N
    B_simple_valid_target_tokens = EMA(trΣ_hat)/EMA(G²_hat)
    B_sequence_equivalent = B_simple / EMA(valid_targets/global_sequences)

Norms include tied parameters once; mean-gradient norms are **not summed across
replicated ranks**; whole-matrix optimizer owners do not define the probe's
parameter selection. FP64 scalar norms and bounded FP32 communication scratch.
All ranks get the same decisions. The estimator reduces to the reference pair
formula forK=2 with equal-size local batches.

Tokens within documents and nearby canonical sequences are correlated. Therefore
the per-token independence assumption is approximate; conversion to sequence
equivalents is a proxy. The reference's fresh disjoint probes better isolate
independent sampling but add teacher/student compute and charged tokens. This
implementation favors training-window reuse; validate usefulness empirically.

## Stages and decision rules

1. **Runtime correctness**: one8rank/10min toy validation (≤1.333333H). It compares
   ordinary DDP to probe averaging with unequal counts, checks noise moments,
   and checks that the next ordinary DDP update still synchronizes. No full model
   is loaded by this gate. It does not establish full-model memory/throughput.
2. **Short ranking**: C0/C1/C2 each independent≤60min/8H; hard endpoint
   **599,916,544 input tokens**, matching an existing R5 evaluation at global
   update4577, PPL25.63620525395. Each starts fresh, using the original6B LR
   schedule prefix rather than compressing a6B decay into0.6B. The three jobs
   can start in parallel after correctness validation. Allocation cap is a
   stopping limit, not a guarantee all arms reach the common endpoint.
3. **Ranking persistence, proposed only**: best adaptive arm vs shadow control,
   fresh3,000,107,008 input runs,≤5h/40H each. This is where the old fixed batch
   transitions, and where late-training CBS growth matters. Requires a new
   reviewed resource scope; no long-run submission/configuration is authorized.
4. **Full6B, proposed only**: promote only after persistence evidence; write a
   new exact endpoint/resource proposal, retain10k evaluation, and compare
   against R5's12.3989218778. Do not inherit formal19's rounded-token or72H
   exceptions. Early2.4B decay is a separate subsequent factorial experiment.

Short allocation maxima total24H plus1.333333H validation. These **do not fit**
the remaining prior study allowance (~7.29H before separately charged HF upload).
No jobs have been submitted. `authorization.json` is deliberately unapproved;
the submitter requires exact config/source digests plus shared and campaign
caps and reserves them atomically. Do not bypass it with direct `sbatch`.
Old formal identities/costs remain; no retry/resume/requeue/cleanup/upload.

Compare fixed1k NLL at every common input-token evaluation boundary. The endpoint
also gets additional9k/combined10k. Use final fixed10k NLL for ranking, and
training-only throughput plus full allocated H200-hours for cost. A wall-stopped
arm is not an endpoint winner; use the largest shared measured token milestone.
Do not infer full6B noninferiority or statistical significance from a single
seed/short prefix. Prune clear instability or consistently worse NLL, not merely
training loss. Promote only if adaptive NLL is no worse across late matched
milestones and its compute tradeoff is useful; if neither adaptive arm improves
the observed quality/cost frontier, retain fixed batches.

## Evaluation, checkpointing and logging

Evaluation uses the canonical fixed1k documents at identical token milestones
across arms. A conservative gap derived from the minimum possible adaptive
update count ensures evaluation at least every10% of updates even when batches
shrink. Short runs have90 common evaluation events; include their overhead in
the allocation budget and separate it from train-step throughput.

Complete checkpoint events:599,916,544;1,199,964,160;1,800,011,776;
**2,399,928,320** (old W&B_step1841);3,000,107,008;5,000,003,584 input tokens,
when inside the current run, plus final/wall-stop. States include all8 owner
optimizers, AdamW, schedulers, RNG, token journals, data cursor and CBS EMA/state.
The implementation saves branch states but does not authorize checkpoint loads.

`train/global_step` always means actual optimizer updates. W&B's internal
`_step` remains a logging record index; compare runs on input tokens, since
update counts intentionally differ. `critical-batch.jsonl` records all probes
with parameter-space moments, controller state, token window and probe timing.
Required training metrics retain their original R5 semantics.

## Commands (not yet executed on GPUs)

```bash
python3 -m unittest discover -s experiments/critical-batch-20261020 -p 'test_*.py' -v
python3 experiments/critical-batch-20261020/launch.py check
python3 experiments/critical-batch-20261020/launch.py status
# Only after exact new resource approval is recorded:
python3 experiments/critical-batch-20261020/launch.py submit --config experiments/critical-batch-20261020/configs/gpu-gate.json
# After gpu-gate passed; these three are independent and may be submitted together:
python3 experiments/critical-batch-20261020/launch.py submit --config experiments/critical-batch-20261020/configs/c0-shadow.json
python3 experiments/critical-batch-20261020/launch.py submit --config experiments/critical-batch-20261020/configs/c1-cbs-half.json
python3 experiments/critical-batch-20261020/launch.py submit --config experiments/critical-batch-20261020/configs/c2-cbs-one.json
```

CPU unit tests and static source/config/shell checks are completed. CUDA/DDP toy
gate, model memory, real probe overhead and PPL effects remain unrun.
