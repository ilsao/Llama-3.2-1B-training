# AGENTS.md

## Approved 6B C2 token-bound early-decay run (2026-10-20)

The user approved “好，注意到 decay 綁定的是 tokens 喔” after reviewing
`kd6b-c2-earlydecay-20261020-seed42`. Exactly one8H200/8CPU/32400second allocation,72H
inclusive, formal20; preserve all19 prior formal identities and every cost.
Digest `6aec35618cbc60da734f7d9de459e8b831a09a80638328baf0c28ec53f6856c3`. Fresh residual-half seed42,
no training checkpoint load. Use6B table LRs with AdamWx2, R5 KD, C2 adaptive
batch16/32/64/128. First decay is bound to2399928320 cumulative nonpadding
input tokens (old R5 W&B_step1841/local18310), never update/logging steps.
Keep5B phase split and second-phase schedule. Hard rounded endpoint6000082944
inputs per role; teacher accounting separate. This exact approval supersedes
nominal6B,64H,three-full-run/120min gates only for this allocation. New methods
cap350.65416666666664H; shared cap1545.525H,
including all existing costs. Rank-private cache and compiled/DDP/data-reader
gates inside this allocation; save complete states and verify export. No
automatic retry/resume/requeue, cleanup, HF upload or further allocation.
See experiments/kd6b-c2-earlydecay-20261020/authorization.json and protocol.md.

## Approved second critical-batch startup replacements (2026-10-20)

The user explicitly replied “允許，兩組再各補交一次” after reviewing the
JOB_RECORD export failure and recovery2 entrypoint regression tests. Authorize
exactly one further8H200/8CPU/7080second allocation for each existing logical
C0-shadow/C1-cbs-half run, same numerical config digests, sealed recovery2 source.
Preserve C0 jobs587009(99s) and587032(9s), C1 jobs587010(107s) and587033(12s),
all costs and original131072 pending charged inputs per role. The latest failed
attempts processed no new training inputs. New attempt endpoints1,000,079,360;
logical per-role input cap1,000,210,432 unchanged. Fresh seed42, no checkpoint load.
Use unique startup-replacement2 paths, exported JOB_RECORD plus worker fallback,
rank-private caches and compiled/DDP gates inside each replacement allocation.
Maximum campaign48.0288888888889H remains within49.3333333333333H, including
all prior failures/gate and C2's full reserved cost; each logical arm stays<=16H.
This exception applies only to these two additional allocations. No further
automatic retry/resume/requeue, promotion, cleanup or upload. Preserve all19
formal identities and the running C2 source/config. See recovery2 authorization.

## Approved critical-batch startup replacements (2026-10-20)

The user explicitly replied "yes" to the reviewed C0/C1 replacement proposal.
Authorize exactly one further8H200/8CPU/7080second allocation per logical run
critical-batch-20261020-c0-shadow and critical-batch-20261020-c1-cbs-half,
using the original numerical config digests and the sealed recovery1 source.
Preserve failed jobs587009(99s,.22H) and587010(107s,.2377777777778H), and
all131072 charged but unconfirmed inputs per role in each failed attempt.
Fresh random seed42 student, no training checkpoint load. New attempt endpoint
1,000,079,360 inputs; cumulative logical input cap per role1,000,210,432.
Use unique startup-replacement1 artifact paths and rank-private compiler caches;
run compiled CUDA and DDP checks inside each allocation before training.
Including gate, failures, both replacements and C2's full reserved allocation,
maximum47.9822222222222H remains within49.3333333333333H. Each logical arm's
allocated cost remains within16H; keep all19 existing formal identities.
This supersedes the single-allocation restriction only for these two expressly
approved attempts. No further automatic retry/resume/requeue, promotion,
cleanup or HF upload. See experiments/critical-batch-20261020-recovery1/
authorization.json and plan.json. Do not modify the running C2 source/config.

## Approved critical-batch 1B screen (2026-10-20)

The user explicitly requested “初篩 1B tokens 吧，開始。” after reviewing C0/C1/C2.
Authorize exactly one8H200/8CPU/600second correctness gate and one8H200/8CPU/
7200second allocation per C0 shadow, C1 CBS-half, C2 CBS-one, all fresh seed42.
Runs use the critical-batch-20261020 prefix and exact configuration digests in
experiments/critical-batch-20261020/authorization.json. Hard student endpoint
1,000,079,360 nonpadding inputs; teacher accounting remains separate. Preserve
R5 KD/math/token-clock LR schedule; only the batch policy differs by arm.
Gate must pass before the three independent pilots may run concurrently
(up to24H200 total,8H200/8CPU per job). No new formal/full run is authorized;
preserve all19 formal identities and every prior allocation/cost.
The new scoped campaign cap is49.3333333333333H200-hours inclusive. This approval
supersedes the prior methods-study ceiling only for these four allocations;
new methods-study cap290.2230555555556H and shared-ledger cap
1485.0938888888888H including all existing1435.7605555555556H. No automatic
retry/resume/requeue, training checkpoint load, promotion to3B/6B, cleanup,
or HF upload. Save complete branch/final checkpoints and verify export.

## Approved R5 online KD 6B run (2026-10-20)

The user replied "好，開始訓練" after reviewing the detailed R5 6B proposal,
including the 9h/72H resource proposal, rounded token endpoint and study cap.
Authorize exactly one allocation for `kd6b-r5-20261020-seed42`:8 H200/8 CPUs,
32400 seconds,72 H200-hours inclusive. Record formal19 and preserve all18 prior
formal identities and all costs. For this exact run, this approval supersedes
the64H,120-minute/three-full-run and nominal6B gates; the scoped study cap is
248.1786111111111H including176.1786111111111H already spent.
Configuration digest `3080c174a87a531814a8e26b5caf122b66806046c51c57cf9b58329278ecc24c`.
Fresh residual-half seed42 random1B student, official pretrained3B teacher;
same8GPUs for both roles. Full-vocabulary KD,T1,teacher weight.25,intradoc mask,
micro2,globalBS16->64,original event-aligned6B schedule and table LRs/AdamWx2.
R5 fused CE/KL,local compile,checkpointing disabled,whole-matrix NorMuon owners.
Hard endpoint6,000,082,944 inputs/28,619updates; separate teacher accounting.
Read-only reuse of verified canonical6B tail overlay is authorized; perform
actual8rank TailDataset preflight before model initialization. Save all owned
optimizer states and verify export inside allocation. No trained checkpoint
load,automatic retry/resume/requeue,old-artifact cleanup,HF upload or extra run.
See experiments/kd6b-r5-20261020/authorization.json and protocol.md.

## Approved 8-teacher + 8-student hidden-state 3B run (2026-10-19)

The user replied "好，開始訓練" after the reviewed8+8 validation and6h-per-role,
96H200-hour proposal. Authorize exactly one producer and one consumer allocation
for `kd3b-hidden8-20261019-seed42`, each8H200/8CPU/21600seconds, one combined
logical full run, formal18. Preserve all17 prior formal identities and all costs.
This scoped approval supersedes the64H logical ceiling, three-full-run/120minute
gates, and128H methods-study cap; the new study cap is191.419H including the
historical95.4186111111H. Both roles together may use16H200 concurrently.
Fresh random residual-half seed42 student, no checkpoint load; pretrained official
Llama3.2-3B teacher, BF16 hidden-state transport, original3B-token KD recipe.
Hard student endpoint3,000,107,008 inputs/14,312 updates, separate teacher journal.
Keep cache below1TB; only consumed newly generated payloads/READY may be deleted.
No automatic retry/resume/requeue, old-artifact cleanup or HF upload.
See experiments/kd3b-hidden8-20261019/authorization.json and protocol.md.

## Approved second 6B startup replacement (2026-10-18)

The user explicitly replied "可" after reviewing the second startup failure and
rank0-only provenance fix. Authorize exactly one further allocation for
raw6b-normuon-adam2x-20261018, same digest and formal17:8H200/8CPU,9h58m,
35880seconds. Preserve jobs566121(5s) and567063(97s), cumulative0.22666666666666666
H200-hours; all allocations remain within80H200-hours. Use source-v3 and execute
the real TailDataset8rank preflight before model initialization. Existing
metadata-only run directory may be reused only with the approved exact file
hashes and no initialization/token journal/weights. Existing validated data-tail
overlay may be reused. Start fresh random weights; no training checkpoint load.
This supersedes the single-replacement restriction only for this expressly
approved attempt; no further automatic retries. See recovery2-authorization.json.

## Approved one-time 6B startup replacement (2026-10-18)

The user explicitly replied "好" to the reviewed replacement request after job
566121 failed in5seconds before training. Authorize exactly one replacement
allocation for raw6b-normuon-adam2x-20261018, same configuration digest and
formal17 identity:8H200/8CPU,9h59m,35940seconds. Preserve failed job566121 and
its0.011111111111111112 H200-hour cost; cumulative cap remains80H200-hours.
Use source-v2 with original pinned data environment for tail preparation,
original requirements-data.txt and canonical Preparation fingerprint checks.
Fresh random initialization; no checkpoint load or further automatic retry.
This explicit replacement supersedes the original single-allocation restriction
only for this one startup repair. See recovery-authorization.json.

## Approved 6B SOTA AdamW LR x2 run (2026-10-18)

The user approved "可以，開始訓練" after reviewing the complete configuration.
Run `raw6b-normuon-adam2x-20261018`, digest `92b99288eb9afdfba6749bdd03058a42d27279f0b83765ec9424cb67da8c5f9b`.
One allocation only: 8 H200 / 8 CPUs, 10 hours / 80 H200-hours inclusive.
Category full, formal17; preserve all16 prior formal identities and all costs.
For this exact run, approval supersedes the three-formal-run/120-minute gates,
64 H200-hour cap and nominal6B token cap. Rounded-up hard endpoint is
6,000,082,944 nonpadding inputs /28,619 updates. Fresh residual-half seed42;
intradoc document-span causal mask,NS5 NorMuon, ordinary WD;6B lookup matrix
LR .00425961126463665 and AdamW lookup x2 .000255576675878199.
BS16->64 at3,000,107,008 inputs;phase split5,000,003,584 inputs;second peak .1x;
170 BS16-equivalent token warmup units per phase;token-clock40% tail with
4:1:1 ramp/hold/final and .2 plateau. Fixed1k + additional9k evaluation.
Complete the canonical tail in an isolated overlay within this allocation;
retain every original shard and all existing nonpadding tokens/document spans.
Bound W&B completion waits, preserve local logging. No training checkpoint
load, automatic retry/resume/requeue, cleanup, HF upload or additional run.
See experiments/raw6b-normuon-adam2x-20261018/authorization.json and config.json.

## Approved 4B last30-percent step-decay run (2026-10-18)

The user requested a fresh 4B-token training run using the current recipe except
last30% step decay, and explicitly reiterated the correct attention mask.
Run `raw4b-normuon-adam2x-step30-20261018`, configuration digest
`10dbaeef65171439aa652dad1ded13c5cd6deeac0f041e31f6af0e499747e2ad`.
Category full, formal16; preserve all15 prior formal identities and all costs.
One8H200/8CPU allocation,7h/56H inclusive, as in the immediately preceding run.
For this exact run the new request supersedes the three-formal-run/120-minute
gates. Fresh residual-half seed42; no checkpoint load or automatic retry/resume.
Preserve intradoc document-span causal mask, NS5 NorMuon, AdamW LRx2, ordinary
WD, canonical raw data, all19080 batch windows,4,000,055,296input endpoint,
BS16-to-64 at2,000,027,648inputs, phase split3,333,292,032inputs, token warmup
170 BS16-equivalent units per phase, second peak0.1x, fixed1k+additional9k eval.
Only the tail changes to last30% of actual optimizer updates per phase, with
20% initial ramp/5% hold/5% final ramp and original0.2 plateau ratio. Decay
starts after completed updates12463 and18697 in phases of17804 and1276updates.
No cleanup, HF upload or additional run. See this run's authorization.json,
config.json and schedule-plan.json under experiments/.


## Approved 4B AdamW LR x2 run (2026-10-17)

The user explicitly approved "允許，開始訓練" and then "重新嘗試" the reviewed
`raw4b-normuon-adam2x-20261017` configuration, digest
`3f12b71e61585347c61a17ca4e369e2a1abdc8eb41c3402685102ae4b4d9db2f`.
One allocation only, 8 H200 / 8 CPUs, at most 7 hours / 56 H200-hours inclusive.
Category full, formal run 15; preserve all 14 prior formal identities and costs.
For this exact configuration only, this approval supersedes the three-formal-run
and 120-minute gates. Start random residual-half seed42 weights, no checkpoint
load, using the prior raw4B intradoc NorMuon recipe and 4B table matrix LR;
double only the auxiliary AdamW peak LR to 0.00029973682960834946. Ordinary WD.
Stop at 4,000,055,296 non-padding inputs / 19,080 updates; original token-clock
phases and BS16-to-64 at 2,000,027,648 inputs. Evaluate fixed 1k documents every
1900 updates and at endpoint, then canonical additional 9k in this allocation.
No automatic retry/resume/requeue, cleanup, upload, or further formal run.
See experiments/raw4b-normuon-adam2x-20261017/authorization.json and config.json.


## Approved intradoc NorMuon 4B run (2026-10-17)

The user explicitly approved ("允許，提交") the reviewed 4B configuration:
`raw4b-intradoc-normuon-20261017`, configuration digest
`ed7d3b6317bd308ce76d1bcecc5db4c3332d8db2568ad56cdee1118bb0ab19bf`.
Use one allocation of eight H200s/eight CPUs up to seven hours/56 H200-hours,
including preflight, initialization, evaluation, checkpointing, export and idle
time. Record category `full`, formal run fourteen; preserve all thirteen prior
formal identities and every historical/failed allocation cost. For this exact
run only, this approval supersedes the three-formal-run and120-minute gates.
Start from random residual-half weights, seed42, on `dolma-raw-baseline-20261008`.
Use the 1B-winning document-span causal mask and NorMuon normalization after NS5,
with the 3B recipe otherwise preserved and the4B table LRs .004995613826805825
and .00014986841480417473. Stop at4,000,055,296 non-padding inputs/19,080 updates;
BS16→64 at2,000,027,648 inputs; phase split3,333,292,032 inputs, second peak .1×,
170 BS16-equivalent warmup units and40% tail per phase. No training checkpoint
load, automatic retry/resume, other formal-run exception, cleanup or HF upload.
See `experiments/raw4b-intradoc-normuon-20261017/protocol.md`.

## Approved BE60P20 LR-scaled 2B run exception (2026-10-10)

After reviewing the concrete configuration and the automatic review rejection,
the user explicitly approved ("可以") this fourth formal run, with eight H200s,
eight CPUs, and a single allocation up to four hours /32 H200-hours.
This exception applies only to `be60p20-lrscaled-2b-20261010` and the configuration
in `configs/be60p20-lrscaled-2b-20261010.json`, digest
`c2ec561c2bae73988d292fa9fe1bc7093265a3c1a6f401930dceedea40da46bf`.
Record it as category `full`, formal run four, preserving all three prior runs
and every historical/failed allocation cost. For this exact run only, this
approval supersedes the three-formal-run and120-minute submission gates.
Start from random weights on parent raw preparation `dolma-raw-baseline-20261008`.
Scale both BE60P20 peak LRs by `(2000027648/399900672)**(-0.32)` using the requested
arXiv:2409.19913v3 method. Keep phase ratios,20% plateau,170 warmup updates,
8192 context, batch geometry and WD=0.1. Stop at15259 updates /2,000,027,648
non-padding inputs, the complete batch crossing2B tokens. No training checkpoint
load, automatic retry/resume, other formal-run exception, cleanup or HF upload
is authorized. See `experiments/be60p20-lrscaled-2b-20261010/protocol.md`.

## Filtered 2B-token run authorization (2026-10-09)

The user authorized merging the winning Muon/WSD worktree into main and pushing
main to GitHub, then starting `muon-wsd-filtered-2b-20261009` from random weights
on verified `dolma-dclm10-005`. Use the best step13717 recipe, eight H200/eight
CPUs, extend WSD to15259 updates with170 warmup updates and20% decay, and stop
after the complete batch crossing2B non-padding training input tokens (normally
2,000,027,648). The allocation may last8 hours/64 H200-hours including all
initialization, evaluation and export. This overrides the earlier120-minute
submission limit for this run only. Do not load any checkpoint for training or
automatically resume. Record it as the third formal run in the shared ledger.
Use a new W&B identity; preserve historical budgets and preparation artifacts.

## One-time checkpoint continuation authorization (2026-10-09)

The user explicitly authorized exactly one continuation from the current best
Muon + WSD checkpoint on unfiltered raw data for a one-hour training phase while
filtering verification runs. This exception applies only to
`best-raw-one-hour-20261009`, retaining the original logical run, eight-GPU
geometry, optimizer/scheduler/RNG/data cursor/W&B identity and cumulative token
and compute accounting. Initialization and final checkpoint/evaluation overhead
may use the separate stopping margin. Do not reset the 6B-token or 64-H200-hour
logical-run caps. Subsequent training requests must start from random weights;
this authorization does not permit another checkpoint load for training.

## Scope and objective

These instructions apply to the entire repository. Follow the user's explicit
instructions and the lab specification when implementing or running experiments.

Train a language model **from randomly initialized weights** using the
`meta-llama/Llama-3.2-1B` architecture and the `allenai/dolma3_mix-150B-1025`
dataset. Optimize held-out perplexity through optimizer design and token-efficient
training, using a context window of **8192 tokens**.

- Prefer an AdamW baseline before scaling optimizer experiments, as recommended
  by the assignment.
- Reach the baseline target of perplexity **20 or lower before 2026-10-09**.
  The assignment estimates this is feasible within **16 H200-hours**; this is a
  planning estimate, not a guaranteed result.
- Complete the full lab by **2026-10-20**. Use `Asia/Taipei` for deadline planning.

## nano4 login node: keep heavy work in Slurm jobs

**The current shell is on nano4's login node. Do not execute CPU-heavy,
GPU-heavy, memory-heavy, or large data-processing tasks directly here.**

Use the login node for lightweight work only: editing files, reviewing code,
small metadata queries, Git operations, shell syntax checks, and Slurm job
submission/status checks.

Before any resource-heavy task, create or update an appropriate `.sbatch` script
and execute the task through `sbatch` on allocated compute nodes. This includes:

- Dataset downloads at scale, extraction, global shuffling, split construction,
  tokenization, sequence packing, and cache generation.
- Training, evaluation, throughput benchmarks, and model-loading smoke tests.
- Tests that load a full model, process substantial data, or launch workers.
- Large checkpoint conversion, verification, and transfers, where supported by
  the cluster's compute-node or designated transfer-node policy.
- Expensive dependency builds or installations.

Do not bypass this rule with a background process, `nohup`, `tmux`, a notebook,
direct `python`/`torchrun`, or a multiprocessing pool on the login node. Invoking
`bash job.sbatch` does not allocate resources; submit it with `sbatch`.

Keep batch scripts in `slurm/` unless the repository establishes another layout.
Separate CPU preprocessing and GPU training/evaluation jobs so preprocessing does
not reserve idle GPUs. Each script must:

1. Declare its job name, wall-time limit, CPU count, memory, log paths, and GPU
   allocation when needed. Select account, partition, QoS, GPU request syntax,
   modules, and storage paths from verified nano4 configuration. Do not invent
   cluster settings or assume that any GPU partition supplies H200s.
2. Use `set -euo pipefail`, explicitly enter the repository, and activate the
   documented software environment. Create log directories before submission,
   because Slurm opens its output files before executing the script.
3. Launch work using the cluster-supported `srun`/distributed launcher setup.
   Avoid accidentally launching one entire distributed job per task.
4. Put large data, caches, and checkpoints under `/work/asciibase64/`, with
   unique run directories. Follow the storage rules below and do not overwrite
   another run's artifacts.
5. Record the job ID, allocated GPU type/count, resolved configuration, and
   timestamps. Set checkpointing and a stopping margin before the wall-time cap.

Lightweight commands such as `sinfo`, `squeue`, `scontrol show`, `sacct`, and
`bash -n slurm/example.sbatch` are appropriate on the login node. Do not submit
resource-consuming jobs merely to verify this guidance file. When a requested
implementation needs a heavy check, prepare its batch script and report whether
the check was submitted, completed, or remains unrun.

## Authorized parallel resources

The user authorizes up to **8 H200 GPUs and 8 CPUs total per job** for subsequent
work. This is a per-job limit, not a per-rank CPU allowance. Allocate one Slurm
launcher task on one node; GPU training uses one rank per allocated GPU and one
CPU thread per rank. CPU preprocessing defaults to seven workers plus a
coordinator, with bounded ordered results and one library thread per worker.
Keep canonical split, document selection, packing and resume semantics unchanged.

The existing one-H200 exception for CPU preprocessing remains authorized because
this account has no verified CPU-only partition access. Record these allocations
as preprocessing costs. Do not allocate extra GPUs for CPU-only algorithms.
Compare 4/8-GPU training at matched global batch and charge every validation and
scaling allocation to the shared ledger. GPU count changes require new runs;
resumes retain their original geometry and cumulative budgets.

## Large-file storage

**Save all large files under `/work/asciibase64/`.** Keep source code, small
configuration files, batch scripts, and documentation in this repository.

- Use a project directory such as `/work/asciibase64/Llama-3.2-1B-training/`
  with separate subdirectories for datasets, caches, preprocessed data, and runs.
- Store downloaded datasets, tokenized/packed sequences, large split/index
  artifacts, model weights, checkpoints, optimizer states, and model exports
  there. Store large logs and W&B artifacts there as well.
- Explicitly configure Hugging Face, W&B, and framework caches/output paths to
  use this storage before downloading data or launching jobs; do not let large
  files accumulate in default home-directory caches or the repository.
- Put large temporary/intermediate files under this storage too. Use absolute
  paths in batch scripts/configurations and unique output directories per run.
- Verify the storage is accessible from allocated compute nodes before heavy
  work. Keep large artifacts out of Git and record their paths in run metadata.
- Continue to perform heavy data processing and transfers through the Slurm or
  designated transfer-node workflow described above.

## Hard training and compute limits

- Each run may process **at most 6,000,000,000 non-padding training tokens**.
- Count tokens every time they are processed, including repeated documents or
  sequences, all gradient-accumulation microbatches, and batches whose optimizer
  update is skipped. Do not count only unique tokens or only successful updates.
- Count actual non-padding input tokens across all distributed ranks; do not
  substitute the number of loss-bearing labels or include padding. Evaluation
  tokens are separate from the training-token counter.
- Check the remaining budget before processing the next training batch. Stop or
  reduce the batch to avoid overshooting the cap.
- Each run is limited to **64 H200-hours**, using **at most 8 H200s** at once.
  GPU-hours equal allocated GPU count multiplied by allocation duration in hours;
  account for initialization, evaluation, checkpointing, and idle allocation time.
- For example, 8 H200s for 8 hours or 4 H200s for 16 hours consume 64 H200-hours.
  A 16 H200-hour baseline allocation could be 4 H200s for 4 hours.
- A resumed/requeued continuation remains part of the same logical training run.
  Preserve and accumulate its training-token and GPU-hour totals across jobs.
- The pair is limited to **three 64 H200-hour runs**. Maintain a shared experiment
  ledger with pilot costs, scaled-run costs, and completed/committed full runs.
  Respect the assignment's spirit: do not evade this limit with many 63-hour runs
  or by relabeling continuations. The ladder allowance does not make pilots free;
  record their costs and use the course's interpretation for aggregate accounting.

Use hard token limits in training code and appropriately conservative Slurm
wall-time limits. Reconcile actual resource use from Slurm accounting. Do not
claim a run met the limits based only on estimated throughput or nominal steps.

## Dataset split: before tokenization or packing

Construct the split from the dataset's complete **`train` document split**:

1. Pin and record the dataset revision and canonical document ordering.
2. Shuffle the document split with **seed 42**.
3. Reserve the **last 50,000 documents of that shuffled split** for evaluation.
4. Exclude those documents from every training input.
5. Only then tokenize and pack training and evaluation documents separately.

Use the canonical global shuffle semantics expected by the lab/OJ, such as the
map-style Hugging Face Dataset `shuffle(seed=42)` followed by index selection.
A streaming buffer shuffle, shard-level shuffle, per-worker shuffle, or random
50,000-document sample is not a substitute for this split. If using an optimized
split implementation, verify it produces the same ordered document indices as
the canonical operation. Run split construction in a CPU Slurm job.

Persist split manifests or equivalent reproducible index artifacts and their
checksums. Verify the training/evaluation document indices are disjoint, and
record the dataset revision, row count, seed, splitting implementation/library
version, and preprocessing configuration. Do not mix split artifacts from
different dataset revisions. Never create a sequence crossing the split boundary.

For in-training evaluation, select a fixed, reproducible subset of **at most
10,000 documents (20%)** from the reserved 50,000. Record its indices and use the
same subset across comparable runs. The OJ evaluates the full reserved set.

## Model initialization and experiment design

- Reuse the Llama 3.2 1B architecture configuration and tokenizer as appropriate,
  but instantiate the model with random weights. Do not load pretrained model
  weights to initialize any submitted training run.
- Preserve the Llama 3.2 1B architecture and keep training sequence/context length
  at 8192. Record tokenizer revision,
  special-token handling, document boundaries, packing, padding/loss masks,
  positional configuration, and any architecture configuration changes.
- Pin dependencies and record the code commit, configuration, and random seeds.
  Resume only from checkpoints belonging to the same experiment; a resumed run
  must trace back to its original random initialization.
- Prefer starting with a reproducible AdamW baseline. Then compare Muon or custom update
  rules, momentum, weight decay, learning rates, schedules, and parameter-group
  assignments. Treat effective batch size as a tunable hyperparameter.
- Follow the staged experiment ladder below to progressively prune candidates
  before committing full-budget runs.
- Document global batch size, per-device microbatch size, gradient accumulation,
  GPU count, and actual non-padding tokens per update. Changing GPU count can
  change batch size unless accumulation is adjusted deliberately.
- For mixed optimizers, explicitly list the parameter groups assigned to each
  optimizer. Verify each trainable parameter is assigned exactly once, including
  tied parameters, embeddings/output heads, matrices, biases, and normalization
  parameters. Record learning rate, momentum/betas, epsilon, decay, and Muon
  orthogonalization settings where applicable.
- Compare recipes using matched split/evaluation settings and token budgets.
  Record both perplexity versus tokens and compute/throughput. Do not promote an
  experiment solely because its training loss is lower.

### Staged experiment ladder

| Stage | 單次時間 | 目的 | 候選數建議 |
| ----- | -------: | ---- | ---------: |
| 超短期 | 15–20 min | 排除明顯爛設定 | 15–30 → 5–8 |
| 短期 | 45–60 min | 粗略 ranking | 5–8 → 3–4 |
| 中期 | 2 hr | optimizer × scheduler × data 等 interaction | 3–6 組 → 2–3 |
| 長期 | 4–5 hr | 確認 ranking 是否保持 | 2–3 → 1–2 |
| 最終 | 64 H200-hours | 正式結果 | 1–2 |

前四個階段的「單次時間」是每個候選的 wall-clock duration；最終階段的
64 H200-hours 是每個正式 run 的 GPU-hours 上限，按 GPU 數量 × allocation
時間計算，包含初始化、evaluation、checkpointing 與 idle time。
候選數是建議的篩選目標，須依共享 experiment ledger 的剩餘預算調整。
所有 pilot、scaling trial、正式 run 與續跑成本都要記帳，並遵守上面的
training-token、GPU-hours、同時 GPU 數量及 pair 的正式 run 數量限制。

Save model, optimizer(s), scheduler(s), RNG states, data cursor/sampler state,
global step, W&B run identity, and cumulative token/resource accounting for a
faithful resume. Checkpointing must not reset the run's budgets.

## Required Weights & Biases logging

Log training runs to the W&B project **`lab5-training-llama`**. Include the complete
resolved training configuration and a persistent run link. Define the training
step consistently as an optimizer-update step, and preserve it across resumes.

| Required metric | Meaning at a logging step |
| --- | --- |
| `train/loss` | Average training loss over the preceding `logging_steps` interval; aggregate valid loss tokens across microbatches/ranks correctly. |
| `train/grad_norm` | Gradient norm for the current logged step; document whether it is measured before clipping. |
| `train/learning_rate` | Learning rate used for the current logged step. |
| `train/tokens_per_second` | Actual non-padding training tokens processed during the current step divided by that step's measured duration. |
| `train/total_tokens_seen` | Cumulative non-padding training tokens processed from the start through the logged step, across all ranks and resumed jobs. |
| `eval/perplexity` | Perplexity measured on the fixed held-out evaluation subset. |

Only `train/loss` is an interval average. The other training metrics must reflect
the logged step, not the average of the logging interval. Synchronize/timestamp
step timing appropriately and document its boundaries. Do not silently include
evaluation time in a training step's throughput measurement.

With multiple optimizer groups, retain the required `train/learning_rate` field
with a documented primary group, and log each group's learning rate separately.
Compute a distributed/global gradient norm rather than reporting one rank's
local shard norm. Avoid double-counting replicated parameters.

Evaluate **at least once every 10% of planned training steps**, as well as at the
end of training. For an ordinary fixed-step run, choose a positive integer
`eval_steps` no greater than `floor(planned_training_steps / 10)`; very short
pilots should evaluate each step. Revisit the interval if the planned run length
changes or a wall-time limit shortens the run.

Compute perplexity as `exp(total held-out negative log-likelihood / total valid
prediction tokens)`, aggregating across batches and ranks and excluding ignored
labels. Do not average batch perplexities. Evaluate without gradient updates and
restore training mode afterward. Record the evaluated checkpoint/global step
and the subset/tokenization settings so the result is reproducible.

## Validation and delivery

Before committing significant compute, verify random initialization, exact split
membership, parameter-group coverage, token accounting, padding/loss masking,
resume behavior, W&B metric names/semantics, and resource limits. Use lightweight
static checks on the login node and Slurm jobs for any substantial runtime checks.
Keep validation proportional to the change; do not launch training for a
documentation-only edit.

After training, when performing the requested lab submission:

1. Export the trained model weights, architecture configuration, and tokenizer in
   a Hugging Face-compatible format. Verify loading/evaluation in a Slurm job.
2. Upload weights and tokenizer to the intended Hugging Face repository and
   verify that the OJ can access them. Record the model ID and uploaded revision.
3. Set **`eval_model_id`** in the OJ script to that repository's model ID. Locate
   the actual OJ script rather than inventing a path.
4. Include the resolved training configuration, W&B run link, optimizer and
   parameter-group settings, training tokens, training duration, GPU type/count,
   total H200-hours, split provenance, and final held-out perplexity.

Never commit or print Hugging Face/W&B tokens or other credentials. Use the
existing documented authentication mechanism. Report actual measurements and
completed checks; distinguish estimates, queued jobs, and unrun checks.

## Lab references

- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Llama 3.2 1B architecture and tokenizer](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Dolma training dataset](https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Batch-size and hyperparameter scaling](https://sadhikamalladi.github.io/blog/2024/01/22/SDEs-ScalingRules/)

## Sweep-specific stages and automatic baseline campaign

Select the starting stage by sweep purpose, not by a universal progression.
LR/numerical-stability screening starts at ultrashort (15–20 minutes/candidate);
batch, decay or momentum ranking starts at short (45–60 minutes/candidate);
optimizer × scheduler × data interactions use medium (2 hours/candidate);
ranking persistence uses long (4–5 hours/candidate); final experiments have
the existing 64 H200-hour ceiling. Runtime correctness and matched-batch scaling
are validation allocations, not hyperparameter sweeps. Candidate counts are
suggestions; promotion requires held-out NLL at matched tokens, stability and
compute evidence, and a recorded remaining budget. Never automatically execute
all stages. Record sweep ID, purpose, stage, parent runs and promotion criteria.

For baseline-fa3-20261009 specifically, the user's explicit override is a hard
combined cap of 16 H200-hours and 7200 allocation-seconds across GPU validation,
scaling, both LR candidates, baseline, full evaluation and all continuations.
Use at most 8 H200s and 8 CPUs per job. LR candidates get 15 minutes each; runtime
and scaling get 10 minutes combined; baseline/export/full evaluation get the
remaining allocation (normally 80 minutes). Skip search rather than shortening
it below its stage duration when evaluation or deadline pressure requires it.
Preprocessing and FA3 dependency installation are separately charged shared-ledger
costs; model GPU checks cannot be charged as dependency installation.
The deadline override is 2026-10-09 23:59 Asia/Taipei, with allocation cutoff at
23:30. Success requires full reserved-set PPL <20, not subset-only evidence.
Do not extend this campaign to 64 H200-hours or reset limits under new run IDs.

The deterministic slurm/auto_baseline.py controller is the sole submission owner
after installation. Legacy Codex heartbeat deliveries are status-only. Keep
preprocessing identity immutable; uncertain submissions retain reservations and
are recovered rather than retried. Disabling stops new submissions while terminal
allocations are still reconciled. No autonomous artifact cleanup or HF upload is
authorized by this campaign.

The user's subsequent baseline-first instruction authorizes an unfiltered raw
baseline from the existing downloads, without waiting for clean/score/select.
For this campaign use dolma-raw-baseline-20261008, derived from the already built
complete canonical seed-42 split of dolma-dclm10-005. Preserve all 50,000 reserved
original evaluation documents, the fixed subset, 8192-token packing and masks.
Publish immutable packed shards in canonical training order as an append-only
stream; persist and check the consumed prefix on faithful resumes. Start GPU
validation once the reserved evaluation cache and initial training buffer are
ready, then start AdamW baseline directly, skipping the LR sweep for this campaign.
The existing filtering job may finish, but its completion is no longer a baseline
prerequisite and no further filtered preparation stages are auto-submitted.
Charge the one-H200 CPU producer separately as preprocessing; retain every prior
allocation and the same 16 H200-hour/7200-second combined GPU campaign limits.

## Filtering continuation wall-time override (2026-10-09)

The user explicitly requested pausing the scoring job and restarting filtering
with a16-hour sbatch limit. CPU filtering stages (clean, score, select, pack,
verify) default to16 hours per allocation, retaining1 H200 and8 CPUs total,
seven workers plus coordinator, immutable preparation/split, and faithful
checkpoint continuation. This supersedes the120-minute preprocessing choice
and does not change the optimizer-sweep allocation caps. Reconcile stopped
allocations before submitting the continuation; no artifact cleanup.

## Bounded filtering recovery

For score005 recovery, use the explicit `slurm/filtering_entry.py` execution
overlay and `filtering_reader.py`, recording their hashes separately from the
immutable original preparation identity. Do not modify dataset.py or existing
split/clean manifests to bypass fingerprint checks. The overlay must verify real
archived originals, native bounded selection,1/7-worker official scores and saved
scoring rows before any score continuation writes. Retain at most two cached raw
shards; score/selection order and checkpoint semantics remain canonical.
See experiments/filtering-recovery/protocol.md.

## Latest optimizer/scheduler sweep overrides (2026-10-08)

The user's latest instructions supersede the earlier combined baseline-campaign
budget interpretation for subsequent allocations: **16 H200-hours and 7200 seconds
are per sbatch**, not an amount reduced by baseline spending. Continue recording
all allocations in the shared ledger and preserving cumulative limits for logical
run continuations. This change does not rewrite historical baseline accounting.

For `optimizer-scheduler-raw-20261008`, use the explicitly selected staged limits:
5 minutes/candidate initially, 15 minutes/candidate for short ranking, and only
when needed 20 minutes/candidate for medium ranking. These entire-allocation caps
include initialization, evaluation and checkpointing. Use 8 H200/8 CPU jobs,
matched training tokens and completed scheduler cooldowns. Preserve the complete
18-candidate optimizer/LR/scheduler matrix; use native PyTorch Muon and keep its
LR-times-weight-decay product fixed across its LR candidates. Warmup is at least
50 updates. See `experiments/optimizer-sweep/README.md` for exact configurations,
promotion rules, calibration and stage activation commands.

The user subsequently authorized implementation and launch of the full automatic
sweep. `slurm/auto_optimizer_sweep.py` is its sole submission owner. Its dedicated
minute cron preserves other schedules, submits four sequential 4/8-H200 native
calibrations, then initial and short stages; only the recorded ranking ambiguity
criteria authorize medium. Reconcile before each promotion and stop after ranking,
without automatic formal training, cleanup, or HF upload. Calibration failure,
source/evaluation identity changes, or fewer than 100 updates fitting the initial
allocation stop new submissions. Incomplete/unstable sweep candidates are excluded;
uncertain submissions remain reserved and are recovered, never automatically retried.
Disable stops new submissions while accounting continues. See the README for
launch, status and disable commands. Earlier prepared-only/manual-stage text is
superseded by this explicit user authorization.
