# 1B student / 3B training-input-token KD proposal and I/O validation

The user explicitly confirmed 1B student architecture and 3B training tokens,
and requested configuration first, a teacher-writing/student-training speed test,
and sufficient time to complete training. The full run is not submitted by this
validation controller. No training checkpoint is loaded.

## Proposed training

- Fresh random residual-half seed42 Llama-3.2-1B; original 8192-token context,
  tied head and embedding, canonical raw Dolma split and intradoc causal mask.
- Frozen official pretrained Llama-3.2-3B base, revision
  `13afe5124825b4f3751f836b40dafda64c1ed062`; full tokenizer equivalence and
  weight manifests already verified, rechecked in each validation role.
- Objective `.75 CE + .25 KL(teacher || student)`, temperature1, full vocabulary,
  no top-k truncation or lossy quantization; BF16 teacher projections.
- NorMuon NS5, momentum .95, Nesterov, ordinary matrix WD .1; peak matrix LR
  .005593716406714465. AdamW beta(.9,.95), eps1e-8, auxiliary peak LR
  .0003356229844028679 (3B-table LR x2), original group coverage/decay rules.
- BS16 ->64 at 1,499,987,968 inputs; phase boundary 2,500,067,328 inputs;
  phase2 peak .1x; 170 BS16-equivalent warmup units per phase; token-clock 40%
  ramp/hold/final tail in 4:1:1 proportions and .2 plateau.
- Endpoint 3,000,107,008 actual nonpadding inputs; exact batch plan will be
  verified before full launch. Fixed1k evaluation <=1431 optimizer updates
  between checks, token knots and endpoint, plus additional9k at endpoint.
- Candidate allocation: 8 H200 /8 CPU, 8h/64 H200-hours inclusive, 900s final
  evaluation reserve and300s stop margin. This is a proposal, subject to measured
  completion-time calculation and recorded authorization; no shorter knowingly
  incomplete allocation. Online8GPU is the existing measured fallback.
- Persistent W&B project lab5-training-llama, required global metrics, faithful
  checkpoint/export, bounded shutdown; no automatic restart, upload or old-file cleanup.

## Authorized validation

Exactly two independent role allocations, producer and consumer, each4 H200,
4 CPU,20min; different nodes enforced. Combined maximum2.6666667 H200-hours,
charged to the existing128H methods study with every prior cost preserved.
No automatic retry. Each role reserves actual repeated inputs before processing,
with a20M-input ceiling; student LR/decay LR0, weights never updated.

Real canonical packed windows, matched document masks and complete128256-vocab
KD loss/backward/optimizer work. Compare full BF16 logits and BF16 final
post-normalization hidden states at global BS16/64. Each segment: one warmup
optimizer window and two timed windows, followed by two GPU-resident student
reference windows. Teacher input generation and write/fsync time are separated
from backpressure; student read/ready wait, compute, and whole-window throughput
are reported. Resident/disk repeated objective equality is checked. A synthetic
hidden/logits full-gradient comparison and real teacher-head reconstruction
checksum check run in the allocation. Throughput is a short measurement, not a
long-duration filesystem guarantee or convergence result.

Persistent jobs communicate via a bounded spool, not one sbatch per batch.
Each rank holds at most two published payloads; each microbatch is one sequence.
Maximum full-logits spool is 16,808,718,336 bytes, below decimal1TB. Partial writes
are fsynced then atomically published; input/label/document digest and teacher
identity must match. Student prefetches one file with one bounded I/O thread,
then acknowledges only after backward. It deletes only this benchmark's consumed
payload/READY files, as explicitly requested for this cache workflow; receipts,
unconsumed failures and every prior artifact remain. Cross-node access and
POSIX_FADV_DONTNEED reduce local page-cache bias; no direct-I/O claim.

## Decision and cost gates

Compare producer service rate with resident student demand and measured consumer
end-to-end rate. A producer advantage alone does not establish system speedup:
splitting8 GPUs into4 teacher+4 student loses half the student replicas. Compare
completion time and total allocated GPU-hours with the existing online8GPU run.
Full logits demand256512 bytes/position, hidden6144, a41.75x difference. Decimal1TB
logits cover only~3.9M positions. Temporary capacity cannot repair a persistent
production deficit.

Before validation, previous study spending is92.9219444444 H200-hours. Maximum
after validation is95.5886111111H; a further64H allocation would exceed128H.
Do not reset the study ledger or classify full3B as a pilot. Existing17 formal
identities remain. Any required full-run exception must be concrete and scoped
to the reviewed configuration/allocation, after measured validation.
