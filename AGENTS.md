# AGENTS.md

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
