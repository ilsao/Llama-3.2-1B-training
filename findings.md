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

## Data takeover (2026-10-08)

Split514101 completed and reconciled.005 has full canonical104020922rows,
103970922training documents and independent50000eval originals.
Clean514774 is RUNNING on1H200/8CPUs; unchanged fingerprint, no startup error
in single observation. Shared reconciled9.6938888889H200-hours; clean reserved24.
Heartbeat restored to current visible thread for data through verify only.
Stage results/accounting gate promotion; full production verification pending.

## Explicit resume after pause (2026-10-08)

Cancelled514774 final89s charged0.0247222222H200-hours; no clean output yet.
Clean continuation514780 submitted once and PENDING in single snapshot;005
source/config/split identity unchanged. Heartbeat restored every20min/current
thread. Shared reconciled9.7186111111, with reservation33.7186111111H200-hours.

## Heartbeat (2026-10-08 12:51 Asia/Taipei)

Clean514780 now RUNNING on25a-hgpn003 with verified1H200/8CPUs and
unchanged005 startup fingerprint/spawn. stderr empty; clean artifacts absent
in early snapshot. Score not yet ready.22 completed/cancelled/failed
allocations agree with ledgers; shared9.7186111111H200-hours reconciled.
Current-thread heartbeat delivery and completed cycle confirmed.

## Heartbeat (2026-10-08 13:41 Asia/Taipei)

Clean514780 committed scan cursor16141312/104020922 (~15.5%); live
database mtime confirms ongoing work. Cursor is input scan coverage, not final
retained count. Diagnostic periodic Timeout stacks at SQLite transaction do
not show Slurm failure. Score gated on clean completion/accounting. Shared
reconciled9.7186111111H200-hours, active0.8519444444provisional, reservation24.

## Heartbeat (2026-10-08 13:42 Asia/Taipei)

Clean514780 scan advanced from16141312 to17491968/104020922
(16.82%); fresh database, manifest absent. No stall observed.
Score awaits clean success/accounting. Shared actual9.7186111111H200-hours.

## Heartbeat (2026-10-08 13:43 Asia/Taipei)

Clean514780 committed scan18150912/104020922 (17.45%),
advanced from17491968; database fresh, manifest absent. Score remains
completion/accounting gated. Shared actual9.7186111111H200-hours.

## Heartbeat (2026-10-08 13:44 Asia/Taipei)

Clean514780 scan advanced from18150912 to18831872/104020922
(18.10%); database and journal fresh, complete manifest absent.
Score still gated on clean completion/accounting. Shared actual9.7186111111.

## Heartbeat (2026-10-08 14:21 Asia/Taipei)

Clean514780 database575021056bytes and fresh journal show ongoing updates.
Indexed progress query temporarily locked after1sec; not retried. Last18.10%
is from13:44 and is not current. Manifest absent; score remains gated.
Shared measured9.7186111111H200-hours; active1.5213888889provisional.

## Heartbeat (2026-10-08 14:23 Asia/Taipei)

Clean514780 live database grew to582205440bytes with fresh journal; still
RUNNING, no manifest. Readonly indexed progress query unavailable; no live
DB writes or recovery attempted. Previous18.10% remains stale. Score gated
on clean completion/accounting. Shared measured9.7186111111H200-hours.

## Heartbeat (2026-10-08 14:25 Asia/Taipei)

Clean514780 database grew to587530240bytes; fresh journal, no manifest.
Readonly cursor unavailable during live journal; root cause unconfirmed.
Skip repeated cross-node DB queries while journal is present; metadata
shows continuing work. Last18.10% remains stale; score awaits completion.

## Heartbeat (2026-10-08 14:26 Asia/Taipei)

Clean514780 database grew to596209664bytes with fresh journal.
Skipped live-journal cursor query; no current percentage or completion claim.
Score gated on clean success/accounting; shared actual9.7186111111H200-hours.

## Heartbeat (2026-10-08 14:30 Asia/Taipei)

Clean514780 database grew to625307648bytes with fresh journal; RUNNING.
Cursor query skipped; exact percentage unavailable, clean manifest absent.
Score gated on successful completion/artifact checks/accounting.22 allocations
match Slurm; shared measured9.7186111111H200-hours, active1.6733333333provisional.

## Heartbeat (2026-10-08 15:21 Asia/Taipei)

Clean514780 database grew from625307648 to973721600bytes with fresh journal;
RUNNING. No completion manifest; exact cursor lookup skipped. File growth is
activity evidence, not completion percentage. Score awaits success and accounting.
22 allocations match Slurm; shared measured9.7186111111H200-hours,
active2.5266666667provisional within existing24h reservation.

## Heartbeat (2026-10-08 15:23 Asia/Taipei)

Clean514780 database grew from973721600 to988164096bytes; fresh journal,
still RUNNING. Exact cursor query skipped; no completion manifest. Score gated
on clean success/artifact checks/accounting.22 allocations match Slurm; shared
measured9.7186111111H200-hours, active2.5541666667provisional.

## Heartbeat (2026-10-08 15:24 Asia/Taipei)

Clean514780 database grew from988164096 to1001144320bytes; fresh journal,
still RUNNING. Cursor query skipped; no completion manifest. Score awaits
clean success/artifact checks/accounting.22 allocations match Slurm; shared
measured9.7186111111H200-hours, active2.5791666667provisional.

## Heartbeat (2026-10-08 15:26 Asia/Taipei)

Clean514780 database grew from1001144320 to1015570432bytes; fresh journal,
still RUNNING. Cursor skipped; no completion manifest. Score awaits success
and accounting.22 allocations match Slurm; shared actual9.7186111111,
active2.6063888889H200-hours provisional.

## Heartbeat (2026-10-08 15:30 Asia/Taipei)

Clean514780 database grew to1037889536bytes; fresh journal, RUNNING.
Slurm query delayed; one fallback returned live allocation, no further queries.
22 historical ledger records unchanged; full sacct refresh not completed here.
No manifest; score gated. Shared actual9.7186111111, active2.6600000000
H200-hours provisional within existing24h reservation.

## Heartbeat (2026-10-08 15:32 Asia/Taipei)

Clean514780 database grew from1037889536 to1064046592bytes; fresh journal,
RUNNING. Slurm responded normally;22 allocations again match ledgers. Exact
cursor skipped, manifest absent, score gated. Shared actual9.7186111111,
active2.7047222222H200-hours provisional within existing24h reservation.

## Heartbeat (2026-10-08 15:33 Asia/Taipei)

Clean514780 database grew from1064046592 to1074606080bytes; fresh journal,
RUNNING.22 allocations match Slurm/ledgers; cursor skipped, no manifest.
Score gated; shared actual9.7186111111, active2.7300000000H200-hours
provisional within existing24h reservation.

## Heartbeat (2026-10-08 16:21 Asia/Taipei)

Clean514780 database grew from1074606080 to1370673152bytes; fresh journal,
RUNNING.22 allocations match Slurm/ledgers; cursor skipped, no manifest.
Score gated; shared actual9.7186111111, active3.5280555556H200-hours
provisional within existing24h reservation. File size gives no exact percentage/ETA.

## Heartbeat (2026-10-08 16:23 Asia/Taipei)

Clean514780 database grew from1370673152 to1377107968bytes; fresh journal,
RUNNING.22 allocations match Slurm/ledgers; cursor skipped, no manifest.
Score gated; shared actual9.7186111111, active3.5547222222H200-hours
provisional within existing24h reservation.

## Heartbeat (2026-10-08 16:24 Asia/Taipei)

Clean514780 database grew from1377107968 to1382973440bytes; fresh journal,
RUNNING.22 allocations match Slurm/ledgers; cursor skipped, no manifest.
Score gated; shared actual9.7186111111, active3.5825000000H200-hours
provisional within existing24h reservation.

## Heartbeat (2026-10-08 16:26 Asia/Taipei)

Clean514780 database grew from1382973440 to1390231552bytes; RUNNING.
Journal briefly absent then reappeared before optional progress query; SQL
lookup skipped. No manifest; score gated.22 allocations match Slurm/ledgers;
shared actual9.7186111111, active3.6077777778H200-hours provisional.

## Heartbeat (2026-10-08 16:29 Asia/Taipei)

Clean514780 current committed input scan57708800/104020922(55.48%)
read once with journal absent; advanced from old18831872. Scan includes
skipped eval rows, not final retained count. Database1420492800bytes;
no manifest, score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active3.6502777778H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 16:30 Asia/Taipei)

Clean514780 database grew from1420492800 to1437315072bytes; fresh journal,
RUNNING. Cursor skipped; prior55.48% is16:28 input scan, not current.
No manifest; score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active3.6838888889H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 16:33 Asia/Taipei)

Clean514780 input scan advanced from57708800 to61106688/104020922
(58.74%), one indexed readonly query with journal absent. Scan
includes skipped eval rows, not final retained count. Database1459765248bytes;
no manifest, score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active3.7169444444H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 16:34 Asia/Taipei)

Clean514780 database grew from1459765248 to1475809280bytes; fresh journal,
RUNNING. Cursor skipped; prior58.74% is16:32 input scan, not current.
No manifest; score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active3.7494444444H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 16:37 Asia/Taipei)

Clean514780 database grew from1475809280 to1494171648bytes; fresh journal,
RUNNING. Cursor skipped; prior58.74% is16:32 input scan, not current.
No manifest; score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active3.7825000000H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 16:39 Asia/Taipei)

Clean514780 database grew from1494171648 to1513033728bytes; fresh journal,
RUNNING. Cursor skipped; prior58.74% is16:32 input scan, not current.
No manifest; score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active3.8175000000H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 16:41 Asia/Taipei)

Clean514780 database grew from1513033728 to1535148032bytes; fresh journal,
RUNNING. Cursor skipped; prior58.74% is16:32 input scan, not current.
No manifest; score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active3.8638888889H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 17:23 Asia/Taipei)

Clean514780 database grew from1535148032 to1874927616bytes; fresh journal,
RUNNING. Cursor skipped; prior58.74% is16:32 input scan, not current.
No manifest; score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active4.5375000000H200-hours provisional within24h reservation.

## Heartbeat (2026-10-08 17:42 Asia/Taipei)

Clean514780 database grew from1874927616 to1951641600bytes (mtime17:37);
fresh17:41 journal, RUNNING. Cursor skipped; prior58.74% is16:32 scan.
No manifest; score gated.22 allocations match Slurm/ledgers; shared actual
9.7186111111, active4.8672222222H200-hours provisional within24h reservation.

## Concurrent filtering resumed — 2026-10-08T17:13:07.754131+00:00

Clean514780 completed2026-10-08 18:56:42 Asia/Taipei; actual6.125H200-hours already reconciled. Immutable005 clean manifest:103970922 train documents scanned,68250278 exact duplicates and40124 empty removed,35680520 retained, including28586141 Common Crawl documents. Score/select/pack/verify manifests absent. User explicitly authorized filtering in parallel with raw-data sweep. Score517587 submitted for120min,1H200/8CPUs (seven CPU fastText workers plus coordinator). Metadata checks match original source/config/environment/fingerprint, parent and clean artifact size; heavy validation remains on compute. Original split and all50000 reserved originals preserved. Raw sweep controller/preparation unchanged.

2026-10-08T17:20:42.730108+00:00 — User requested stop/restart filtering with16-hour sbatch limit. SentUSR1 then cancelled score517587 while still loading canonical Arrow cache (no scores DB); terminal351sec=0.0975H200-hours reconciled. Submitted continuation517611, same005 fingerprint/source/config/data environment,960minutes,1H200/8CPUs, durable afterany517587 dependency prevents overlap. Filtering helper defaults and score/prepare batch headers now16hours; optimizer-sweep caps/controller unchanged.12 metadata submission tests and shell syntax passed.

2026-10-08T18:08:27.581958+00:00 — Progress inspection: score517611 FAILED at01:45:21 Asia/Taipei,1507sec,exit139/SIGSEGV,actual0.418611H200-hours reconciled. Scores SQLite51802112bytes, no journal; one bounded readonly primary-key query gives committed last_row5574559 (source cursor, not scored-doc count). Periodic traces show Dataset random selection/Arrow table and numpy cumsum; root cause unconfirmed. No blind retry or artifact changes. Optimizer native calibrations all passed; initial15/18 completed,16th active; best completed Muon0.04/WSD subsetPPL621.48 vsAdamW0.0012/WSD1131.06 at matched13107200tokens, early pilot evidence only.

2026-10-08T18:26:23.473469+00:00 — Filtering repair c43345b:16-hour score517834 RUNNING. Canonical index3004shards/104020922rows complete. Runtime checks passed for12 independent originals across6sources, native cross-shard order/duplicates/text projection, exact1/7-worker official classifier outputs and16 saved scores. Same original005 core/config/environment fingerprint; overlay hash receipts agree. Resumed after committed last_row5574559; new scores141568, total2438912/28586141. Initial score throughput2334documents/sec; ETA excludes ranking and later stages. No split/clean/raw-sweep changes or cleanup.

2026-10-09T03:22:15.143118+00:00 — Score517834 complete at03:45:55 Asia/Taipei,28586141documents,5300sec=1.472222H200-hours reconciled; scored ranking index/manifest identities match immutable005 and validated bounded reader. Continued authorized filtering via select/tokenization job518703,16h,1H200/8CPU, unchanged6B whole-document budget and all50000 reserved originals. Raw optimizer campaign complete:18initial+18short allcomplete,31.374444H200-hours inclcalibration, no medium(clear ranking). Best short Muon0.02/WSD subsetPPL55.4453 vsAdamW0.0003/WSD140.8943 at104857600trainingtokens. Formal final training/full reserved-set result remain unrun.

## Filtering tokenization reader repair (2026-10-09)

The slow path repeatedly reopened approximately511MB raw IPC shards in canonical
shuffle order, with9/10diagnostic main-thread samples in native Dataset.from_file
and Arrow read_all. Increasing GPU count would not accelerate this CPU reader.
Graceful select pause preserved81733docs/126905498tokens/117shards; actual cost
3.9813888889H200-hours was reconciled before continuation519327.

The replacement uses IPC record-batch offsets, lazy sealed sidecars and at most two
mappings. Immutable core, config, tokenizer, manifests and selected prefix remain
unchanged.34stdlib tests passed; compute-native fixtures verified batch boundaries,
corrupt-index rejection, order/duplicates, native shuffle equivalence, exact original
select interruption/resume and packing masks. Real archived six-source equivalence,
1/7official scores, latest16saved scores and latest16saved tokenized documents passed.

Matched1024candidate documents generated exactly1765290tokens in both pipelines.
New complete pipeline43.1747s versus legacy96.6425s is2.2384x faster, including new
lazy indexing. Conservative process+worker RSS upper26.24GiB is separate from Slurm's
approximately200GiB MaxRSS statistic; their disagreement has not been independently
resolved, so do not report the Slurm peak as eliminated. Production throughput and ETA are logged independently, with a
15-minute snapshot; preprocessing/full verification are still ongoing.

## Seven-worker reader/prefetch (2026-10-09)

User-requested worker-local reading+tokenization with one prefetched batch each
passed native correctness/budget/resume/mask/overlap checks. All seven workers use
<=2 outstanding batches, one pinned CPU each; coordinator owns the eighth CPU.
4096matched documents/6094282tokens:20.5996s old coordinator reader vs7.1494s
worker prefetch (2.8813x). Production519502 interval524CPU-seconds/106wall-seconds
uses4.9434cores out of8 (61.79%), compared with13.73% measured for519327.
Initial219.64s production average981447tokens/s is a short observation, not the
15-minute result; remaining training-tokenization estimate1.50h excludes eval,
pack, verify, queue and continuation overhead. Existing prefix and all original
core/config/split identities preserved; no training or upload was dispatched.

## Horizon-scaled WSD research — 2026-10-14

Current evidence: completed BE60P20 at399,900,672 inputs has fixed1,000-document PPL27.4331322294; previous LR-scaled2B endpoint subset PPL17.7475548804. These are historical subset measurements, not new results or full-set claims. The new0.7B/2B campaign tests LR multipliers, clipping, and shared-prefix decay before two final continuations. beta0.32 is a cosine/AdamW transfer heuristic for WSD/Muon, not an established law here. Pure contracts pass; GPU validation and all tuning outcomes remain unrun at protocol creation. Persistent continuity was rejected by automatic review and remains uninstalled; never bypass that rejection. Historical allocation costs and failed/uncertain jobs remain preserved.
