# Approved 1B student / 3B tokens with an eight-GPU pretrained teacher

The user replied **「好，開始訓練」** after reviewing the successful8+8 hidden-state
benchmark and the conservative6h-per-role/96H200-hour proposal. This authorizes
one producer and one consumer allocation, each8H200/8CPU on one node, one launcher
and eight ranks. Both allocations belong to `kd3b-hidden8-20261019-seed42`, category
full, formal18. Preserve all17 historical full identities and every prior cost.
The scoped logical-run cap is96H. The methods-study ceiling becomes191.419H,
covering95.4186111111H historical cost plus96H reserved here. No automatic retry,
checkpoint initialization, resume, requeue, HF upload or old-artifact cleanup.

## Recipe and provenance

Use random residual-half seed42 Llama3.2-1B weights, canonical raw Dolma data,
8192 context and document-span causal attention. No training checkpoint load.
The random initialization signature must match the historical seed42 control.
Official pretrained `meta-llama/Llama-3.2-3B` base teacher revision
`13afe5124825b4f3751f836b40dafda64c1ed062`, frozen BF16 weights, manifest
`bf1dec0109c97ad0d00d5714127aca10a6046edbd0b35e0d4c1a55fb2f192507`.
All model files are verified on allocated compute nodes before use.

Objective is .75 hard CE + .25 KL(teacher||student), T1, full128256 vocabulary,
1024-position chunks. Transport BF16 final normalized teacher hidden vectors
for8191 prediction positions, with3072 features. Student loads only the frozen
tied teacher output head, outside its model/DDP/optimizer/checkpoint parameters.
The previously validated custom first-order KD loss remains unchanged.

NorMuon NS5 momentum.95 Nesterov, ordinary external WD.1, peak matrix
LR.005593716406714465. Auxiliary AdamW LR.0003356229844028679, betas(.9,.95),
eps1e-8; original parameter grouping; global clip.5. BS16→64 at1,499,987,968
inputs, phase2 at2,500,067,328, second peak.1x; token-clock170 reference units
warmup per phase,40% tail with4:1:1 ramp/hold/final and.2 plateau.
Hard endpoint3,000,107,008 nonpadding student inputs/14,312 updates. Teacher
inputs are separately reserved and confirmed, including ahead-of-student work.
No repetition or change to document selection, order, tokenizer or split.

## Transport and completion controls

Each teacher rank produces canonical row `key*8+rank`. Every payload has an atomic
READY manifest binding batch SHA256(input IDs, labels, document IDs and padding
mask), row/key/rank, config, preparation fingerprint, teacher manifest, shape and
dtype. Payload is published only after complete write+fsync+rename. Student reads
into two pinned host buffers, validates identity and length, copies to GPU, then
prefetches the next microbatch while executing current forward/backward.
Delete only this run's consumed payload/READY after synchronized backward;
publish an ACK. Preserve all unconsumed/partial data and all historical artifacts.

2048 disk slots per rank bound live payloads to824.533057536GB, with one additional
in-flight microbatch per rank still below1TB. This is a ceiling; training starts
from the first ready batch without prefilling1TB. No extra I/O worker processes;
one I/O thread per consumer rank, one library thread per rank,8CPUs per job.
Producer and consumer each wait for both READY records, including their own
rank0 publication, and validate matching identities/different nodes. Pairing
must finish within8min of each allocation start and leave the conservative
completion margin. An abort is propagated through immutable failure files.
Teacher exits after final publication; student consumes remaining cache.

The six-window benchmark projected4.46h for student training+host overhead.
Each allocation is bounded to6h including startup/idle/I/O/evaluation/export.
Use the earlier of both allocation end times for student stopping. Reserve900s
for final additional9k evaluation and at least300s for final1k/checkpoint/export,
with the existing measured-overhead stopping logic. Early or wall-time partial
results remain explicitly partial; no automatic retry/resume.

Evaluate fixed1k at least every1431 actual updates, at scheduled token events,
and endpoint; then additional canonical9k documents. Export and verify HF reload
locally in the same student allocation. Record W&B project `lab5-training-llama`,
fresh identity, all required metrics and local logs. Throughput includes exposed
hidden I/O, forward/backward/optimizer, excludes evaluation and host data/journal.
Teacher production and allocated compute are separately recorded. Live compute
is an estimate; final GPU-hours use Slurm accounting for both jobs together.

## Validation and submission

Reuse the completed8+8 benchmark evidence, then run stdlib transport/identity/
capacity/budget regressions and AST/shell checks on the login node. Synthetic
GPU gradient checks, attention/document isolation, real-loader first/last rows,
teacher provenance and exact first transported projection checks run inside the
authorized allocations. Student preserves random initialization and optimizer
group coverage checks. No separate validation allocation is introduced.

The controller reserves both48H allocations atomically in the shared ledger,
submits both held, binds their job IDs, then releases them together. Uncertain
submissions retain reservations and IDs for inspection, never automatic retry.
Seal source/config/script/auth before submission and verify on both nodes.
