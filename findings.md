# Research Findings

## Research Question
Train random-initialized Llama 3.2 1B at context 8192, optimizing held-out perplexity
under strict token and H200-hour limits. Start with AdamW.

## Current Understanding
No measured baseline perplexity yet. The original download is running as job
511918 on one H200 and eight CPUs, with 1,831/6,081 checkpointed shards
at 2026-10-08 00:42 Asia/Taipei. Storage has approximately 9.9 TB
available after the earlier capacity failure. Complete download import into
`dolma-dclm10-002` remains the prerequisite for production split/filter/packing.
Synthetic preprocessing/resume/retirement and four/eight-GPU distributed correctness
checks passed. These checks do not establish full-model throughput or perplexity.
The lightweight suite now passes 50 tests.

## Lessons and Constraints
- Use sbatch for heavy work. Per-job limit: 8 CPUs total, 8 H200s maximum.
- CPU algorithms use seven workers plus coordinator and only the authorized one-H200 exception.
- Preserve native full-train seed-42 shuffle and its final 50,000 documents for eval.
- Existing original download identity belongs to preserved source commit 274a416.
  Finish it there, then reuse a completed manifest from a new preparation ID.
- Existing eval text depends on complete raw cache. Save an independent archive
  before retirement. Keep full eval tokens and fixed subset provenance.
- Every processed training input token and every allocation must be charged across resumes.
- Cleanup must be confined to this project and cannot remove active references.

## Key Results
No training or scaling claims are justified yet. Jobs 511170, 511509 and 511510 failed
and their allocated costs have been reconciled; see validation-status.json.

## Open Questions
Actual remaining download/preprocessing time, baseline throughput, held-out perplexity,
and measured 4/8-GPU scaling efficiency remain unknown.

## Runtime validation update
Job 511939 passed native/worker equivalence and independent eval original archive
resume checks. It used 0.0152778 H200-hours. Real-corpus verification and training
measurements remain pending.

Job 511949 confirms synthetic sealed data loads identically after intermediate
retirement, including full eval and subset. Training environment is installed and
pip check passed (511940). Production cleanup must still await real-corpus verify.

Four/eight-GPU distributed correctness checks passed (511971 and 511980).
These are synthetic runtime checks; full 1B/context-8192 training and actual
throughput scaling on real sealed data remain unvalidated. Costs are reconciled.

## Heartbeat accounting and submission safety (2026-10-08 00:42 Asia/Taipei)
All eight terminated allocations agree with sacct and the authoritative ledgers:
preprocessing 0.9180556, training/environment/runtime
0.1536111, total 1.0716667 H200-hours.
Job 511918 is still accruing costs (0.4955556 H200-hours observed,
24 H200-hours maximum for this allocation); its observed cost is provisional.
There are 0 committed full runs and no processed production training tokens.

Training submission previously cancelled budget on every nonzero sbatch response.
A lost client reply could therefore discard a live allocation's reservation.
The wrapper now records a unique tag before dispatch, retains uncertain reservations,
saves returned job receipts before binding, and offers bounded recovery using queue,
accounting and saved startup allocation metadata. Zero/multiple matches or query
failure retain the charge and prevent resubmission. Nano4 reports
`AccountingStoreFlags=(null)`, so terminal comment recovery relies on receipts.
Mocked regression tests cover these cases; no extra Slurm job was needed.

No stage advanced because the download has no completion manifest. Baseline <=20
before 2026-10-09 remains an unachieved target; roughly
23.3 hours
remain at this observation, and preprocessing/training time is not yet measured.
