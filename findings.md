# Research Findings

## Research Question
Train random-initialized Llama 3.2 1B at context 8192, optimizing held-out perplexity
under strict token and H200-hour limits. Start with AdamW.

## Current Understanding
New 004 import/checksum 513930 completed exit 0 in 34 seconds. All 6081 artifact
records match completed original 001 download, canonical file order and pinned
revision. Full compute-node checksums passed; corpus files reused without copies.
Cost 0.0094444 H200-hours is reconciled. All 16 terminated allocations agree with
Slurm/ledgers; shared measured total 3.9900000. 004 config/source hashes/environment
versions/process methods match validated source and identity
52f00b18ade1299ea55e96b61dfcb8e7a3a9fa491759d7c8e606f3f3b5ca426b.

Split 513955 FAILED exit 1 after 3606 seconds (one H200/eight CPUs), with zstd
"Data corruption detected" while generating full map-style data. Error stack
omits the shard filename. No canonical split/state/eval archive was produced.
Cost 1.0016667 H200-hours is reconciled; shared measured total 4.9916667 across
17 terminated allocations, all matching Slurm/ledgers. Earlier import checksums
matched recorded downloaded bytes; this did not validate all compressed streams
or independently authenticate upstream LFS hashes. Do not skip records/files.

Protocol 692e6b0 and diagnostic code 64175a7 precede read-only cache-audit 514055,
RUNNING on one H200/eight CPUs with seven spawn workers. It hashes and fully
streams each canonical compressed shard, preserving exact filenames/errors and
an ordered resumable cursor under runs/cache-audit/corpus-audit-001. At 06:46,
500/6081 committed shard checks showed no failure; complete result and failed
shard remain pending. Ten-minute runtime timeout, nine-minute worker deadline,
30-minute allocation (0.5 H200-hour reservation); shared upper 5.4916667.
62 lightweight tests pass; substantial audit is pending, no corpus repair,
production retry, filtering, cleanup or training. Retain all identities/files.
Compare any failed files to pinned upstream metadata and independent decode/new
separate copy before choosing a repair; preserve canonical revision/order/split.

Lossless schema validation 513790 passed heterogeneous fields, verbatim JSON,
native projection order/indices, one/seven-worker full split entrypoint,
archive/resume/downstream/offline contracts and bounded real-source samples
(12 documents across six families). This is not complete-corpus verification.
No production split/eval archive, cleanup, production training tokens, committed
full runs or measured perplexity yet. Prior 59 lightweight tests passed;
no source changes or repeated tests this metadata cycle. Baseline remains an
unachieved target; actual preprocessing completion/training throughput is unknown.

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
Remaining full preprocessing time, complete canonical row count, baseline throughput, held-out perplexity,
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
