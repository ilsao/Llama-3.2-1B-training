# Research Findings

## Research Question
Train random-initialized Llama 3.2 1B at context 8192, optimizing held-out perplexity
under strict token and H200-hour limits. Start with AdamW.

## Current Understanding
Full read-only audit 514055 scanned all 6081 canonical shards. Exactly one
compressed stream failed: data/common_crawl-science_math_and_technology-0015/
shard_00001148.jsonl.zst. The other 6080 streams decoded completely and matched
historical manifest hashes. Recorded download checksum alone had accepted the
corrupt local bytes; the blob filename differed from actual SHA256 only here.
Audit cost 0.0286111 H200-hours is reconciled.

Protocol 1efd4ef preceded code 7889201 and isolated replacement validation 514077.
The exact pinned upstream LFS SHA256 is
c624bddd1c7f810c737b4cb0a75a6d7b6a45c9667ce19a4441b26e22d98e9ec3;
original local SHA256 was
7ec222880366d0dc111b22f369a797ddad2be070026c4f6dde0e8f594ac06564.
A separate new copy matches upstream hash and size. Full Python-zstandard and
Arrow-zstd decoded byte hashes agree (432076050 bytes), and all 47233 JSON
documents passed id/text validation. Original Arrow decoding also failed,
confirming local corruption rather than a Python-only decoder failure.
Old blobs and all historical manifests remain unchanged. Result SHA256:
a9050e57c6ae06615795420232a13bb635ac3c71630ac28488db8e2d7160bc4d.

Replacement job completed in 11 seconds on one H200/eight CPUs, costing
0.0030556 H200-hours. All 19 allocations agree with Slurm and ledgers:
preprocessing 4.8697222222, training/environment/runtime
0.1536111111, shared 5.0233333333 H200-hours.
The earlier isolated-replacement cycle had no outstanding allocation; its 66
lightweight tests passed. Current import allocation is described below.

Authenticated repaired import514094 completed exit0 in39s on1H200/eight CPUs;
cost0.0108333333 H200-hours is reconciled. Completed new005 manifest SHA256
237fcc984b19261dea4bb765ebb279cd4849d9a71119c8421a7c548a4f825d53 preserves6081
canonical file records: only slot1072 uses the authenticated replacement;
6080 remain identical. Source004 manifest, repair result, immutable import request,
configuration/source/environment/spawn identity and completed job event match.
Full corpus checksums ran on compute node; current cycle checked metadata only.
No corpus copies or old manifest rewrites. All20 terminated allocations match
Slurm/ledgers: preprocessing4.8805555556, training/runtime
0.1536111111, shared5.0341666667 H200-hours.

New005 canonical full split514101 remains RUNNING on one H200/eight CPUs.
Full map-style cache contains 104020922 observed rows. Original source field/order
validation advances from 524 to 805/6081 canonical shards,
covering 42214136 contiguous rows at 2026-10-08T01:40:28.644599+00:00.
Cursor order/names/contiguous ranges match immutable download metadata.
Complete source authentication still precedes native seed42 shuffle/eval archive.
Split manifest and independent evaluation-raw state remain absent; filtering is
not ready. Partial progress does not authenticate complete canonical membership.

stdout is fresh at 2026-10-08T01:40:27.767224+00:00. Elapsed 7135s gives
1.9819444444 H200-hours observed provisionally;20 terminated allocations agree
with Slurm/ledgers, shared reconciled5.0341666667, reservation24, shared upper
29.0341666667. Current005 identity/config/four source hashes/repaired manifest
remain unchanged. Storage reports 9285219717120 available bytes.
No new jobs, source changes, repeated tests, cleanup or training; prior72 tests
remain valid. Continue bounded checks and preserve old identities/artifacts.

The native full-train seed42 shuffle, last50000 eval originals and fixed10000
subset must complete before clean/filtering. Full production raw/tokenized/packed
eval preservation and reference-safe cleanup remain pending. Previous001-004
identities and damaged files are preserved. No production training tokens,
committed full runs or measured perplexity. Baseline<=20 before2026-10-09 remains
unachieved; completion time and training throughput are not measured.

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
