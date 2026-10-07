# Research Findings

## Research Question
Train random-initialized Llama 3.2 1B at context 8192, optimizing held-out perplexity
under strict token and H200-hour limits. Start with AdamW.

## Current Understanding
Download 511918 completed successfully at 2026-10-08 02:39:57 Asia/Taipei:
all 6081 shards and a complete manifest. Its 8810 seconds on one H200 cost
2.4472222 H200-hours, reconciled. Shared reconciled cost is now 3.5188889.
New import/checksum job 512500 is RUNNING on one H200 and 8 CPUs, reusing
completed dolma-dclm10-001 in dolma-dclm10-002 without copying corpus files.
The durable wrapper stored submission intent, reservation, tag and job receipt.
Split/filtering await successful full-checksum import and accounting.
Full 50,000-doc eval raw/tokens/packed preservation and production cleanup remain
pending real split/pack/verify; synthetic archive/resume/retirement and 4/8-GPU
correctness checks passed. No measured baseline perplexity or production training
tokens yet. Latest lightweight suite: 56 passed; no repeated runtime tests this cycle.
Heartbeat continues reporting in the visible thread.

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

## Heartbeat update (2026-10-08 01:05 Asia/Taipei)
Download continues to progress; no new or newly terminated allocation was found.
Eight terminated jobs still match sacct: 1.0716667 H200-hours reconciled.
The active download has 0.8788889 H200-hours observed provisionally; retain its
24-hour allocation ceiling, giving 25.0716667 measured-plus-reserved hours.
There are still zero full runs committed and zero production training tokens.
Shard counts indicate progress, not byte-volume completion or a timing forecast.

The preprocessing shell wrapper previously saved only a post-reply TSV entry and
used no recovery tag. It now delegates to a metadata-only Python helper which
persists the unique intent in central state and a /work receipt before dispatch,
retains a 24-H200-hour reservation for uncertain replies, prevents overlapping
stages for that preparation until recovery/accounting, and binds returned job IDs.
Training and preprocessing share bounded metadata recovery with receipt fallback
for nano4's absent sacct comments. Five additional small regression tests pass;
55 total, shell syntax and diff checks pass. Existing download source and geometry
remain unchanged; no heavy runtime check or production submission was needed.

Approximately 22.9 hours remain before 2026-10-09 Asia/Taipei. Baseline
perplexity <=20 is still an unachieved target; production preprocessing and full
model training have not begun. Advance only after completion, artifact checks,
and accounting. No cleanup is authorized before the independent full eval archive
and production verification seal exist.
