# Bounded canonical-shard filtering recovery

Score517611 failed withSIGSEGV after1507seconds, exit139, during Arrow/global
Dataset random selection; MaxRSS209686720KiB. This observation suggests the
corpus-wide concatenation/selection path, but does not establish a native-library
root cause. Preserve all source/split/clean artifacts and committed scores.

Keep dataset.py, preparation_parallel.py, requirements-data.txt, classifier,
config, Python environment and original005 preparation fingerprint unchanged.
Use an explicit filtering reader execution overlay, recorded by source hashes,
Slurm ID, original fingerprint and split checksum in filtering-access/job-*.json.
This changes access mechanics only: score uses the same official function,
original full text, canonical row indices, cleaned membership, worker ordering,
SQLite batch/cursor atomicity, ranking and ties. No new split or cleanup.

Build a resumable per-shard row-count index from the ordered3004 canonical raw
Arrow cache descriptors. Verify descriptor prefixes/sizes, row count104020922,
and split identity. Retain at most two Dataset shard mappings; project scoring
reads to text before bounded shard-local selection. Record exact committed score
count, current-session rate and remaining-time estimate in score/progress.json.
Use the same reader for future select (whole-document/token-budget/packing rules
remain the original stage_select implementation). Existing full production verify
remains required; bounded reader checks are not a replacement for full checksums
and native global-shuffle validation.

Before resuming real scoring in the16-hour1-H200/8-CPU allocation, fail closed
unless all these runtime checks pass on the compute node:
- Full-field/byte comparison with independently archived original eval samples,
  including at least two from each of the six source families.
- Native Dataset.select equivalence across three real shards, bounds, reversed
  order, duplicates and text-only projection.
- Exact official classifier score equality with1 and7 CPU workers.
- Exact recomputation of the latest16 saved scoring rows under the new reader.

The original core Preparation constructor still verifies source/config/environment
identity. Save validation.json and its identity/checksum with the score manifest.
A failed check submits no additional allocation and performs no scoring writes.
Lightweight tests cover mapping/boundaries/bounded cache/full fields and atomic
score interruption/resume; substantial validation runs only in Slurm.

## Record-batch reader continuation (2026-10-09)

Select518703 was gracefully signaled through its Python Slurm step and exited75.
Its14333 allocation-seconds cost3.9813888889H200-hours; the preserved checkpoint
has81733documents,126905498tokens and117shards. The paused-prefix receipt records
all pre-existing metadata identities and shard descriptors. No artifacts were
removed and original core/config/environment hashes remain unchanged.

`filtering_batch_reader.py` retains the reader interface but maps at most two raw
shards, builds immutable per-shard IPC message offsets and row ranges, and seeks
only to requested record batches. It detaches selected Python records before
closing mappings. Sidecars are sealed with a content digest and original shard,
preparation and split identities. Unsupported messages/dictionaries or inconsistent
ranges fail closed. Sidecars live under filtering-access/batch-reader-v1/indexes.

Select runs archived six-source/native/official-score checks under an overlay-hash
specific receipt directory, leaving all old validation evidence untouched. Before
production selection it runs real-Arrow boundary/corruption tests, interrupted
versus uninterrupted original stage_select fixtures, mask/packing comparisons,
checksums of the saved prefix, and exact regeneration of16saved tokenized documents.
A1024-candidate matched-input benchmark uses the original32-document batches,
seven single-threaded workers, and unchanged tokenizer; new-reader timing includes
lazy index construction. It resumes selection only if complete-pipeline speedup
is at least2x and conservative process/worker RSS upper bound is below160GiB.
Benchmark writes only evidence/fixture paths, never production selection.

The verify overlay also uses the bounded reader. Its `shuffle` calls the pinned
native map-style Dataset.shuffle on a zero-buffer null column with the exact full
row count, seed and cache arguments. Original stage_verify compares every resulting
ordered index to the immutable permutation and performs the unchanged complete
artifact/provenance/ranking/budget/mask checks. It never concatenates full raw text
just to obtain the shuffle indices.

`slurm/filtering_resume.py` owns only this authorized filtering continuation.
Metadata-only minute ticks reconcile terminal jobs, retain uncertain submission
reservations, and submit select -> pack -> verify with16-hour/1H200/8CPU allocations
through the durable preprocessing helper. It checks original metadata and execution
hashes before each dispatch, checks the192H200-hour shared measured-plus-reserved
ceiling, and retries only faithful exit75 checkpoints. Other failures block rather
than retry. Disabling stops dispatch but terminal accounting continues. No training,
cleanup or upload is dispatched. Inspect or disable using the controller's status
or disable action. Production progress includes committed counts, reader timing,
index cost, pipeline rate, process memory and training-only ETA; a15-minute snapshot
separates preliminary benchmark evidence from sustained performance.

## Worker-local read and one-batch prefetch (2026-10-09)

The user explicitly requested independent CPU batches with at most two batches
per worker (processing plus prefetch). Allocation519327 was gracefully stopped
at472451933tokens/305288documents/439shards and reconciled:1849seconds,
0.5136111111H200-hours. Its paused-parallel-prefix receipt preserves the original
committed shard descriptors; the same controller and shared budget continue.

`filtering_parallel.py` starts seven spawn workers and one coordinator within the
existing8CPU Slurm allocation. Each worker is pinned to one allocated CPU, with
one tokenizer/library thread and a single I/O-prefetch thread sharing that CPU.
Worker queues receive32-document canonical batches and at most two outstanding
credits per worker, including finished out-of-order results. A credit returns only
when the coordinator has consumed the complete batch. The prefetch thread reads
one next batch while the main worker tokenizes the current one; a one-slot ready
queue and the credit bound prevent a third batch from accumulating.

Worker readers use prebuilt sealed IPC indexes and ephemeral mappings. At most one
mapping is active in each reader, closed after each shard group; no raw shard cache
is retained between reads. Detached Python documents and tokenized results are
bounded by batch credits. Workers never write indexes, manifests or checkpoints.
All sidecars are prepared/verified before workers launch to avoid concurrent writes.

The adapter supplies PreparedDocument results to the unchanged dataset.stage_select
through runtime load_raw/tokenize_documents overlays. Its independent SQL cursor
must match every original requested row_index batch exactly; each result's position,
source and evaluation slot are checked. Train and evaluation worker sessions are
separate; unread speculative batches are discarded at phase/budget boundaries.
Only the original writer advances committed counters/cursors. Source/config/split,
whole-document skipping, BOS/EOS, loss masks and packing remain unchanged.

38stdlib tests passed. Real-Arrow spawn fixtures compared original and parallel
selection, interrupted/resumed output, tight-budget skips, masks and canonical
order, and measured actual per-worker read/tokenization overlap. All worker credits
stayed<=2. A4096-document matched benchmark on519502 produced identical6094282tokens:
coordinator-reader20.5996s vsworker-prefetch7.1494s,2.8813x speedup. Startup is included;
this is separate from the production rate. Promotion requires>=1.1x speedup,
conservative process RSS upper<160GiB, and<=2credits. Evidence/progress receipts are
now overlay-hash-specific, with the root progress file retaining only the latest
status. New job519502 uses the original16h/1H200/8CPU/200GB allocation geometry.

## Train-only CPU scaling override (2026-10-09)

The user explicitly authorized16CPUs for training tokenization. Nano4 rejects
16CPUs with oneH200: its job submission policy permits at most12CPUs perGPU.
The accountacd115198 is absent from every CPU-only partition AllowAccounts
list; a CPU-only test submission also reports an invalid account/partition.
Use the corrected12CPU allocation with11 workers plus a coordinator, retaining
one thread perworker and two32-document batch credits perworker.
The immutable preparation config and core fingerprints remain unchanged;
filtering_resources.py records the execution-only override in overlay identity.
On the final durable training checkpoint, stop with exit75 and submit an8CPU
evaluation continuation with7 workers. Pack and verify also retain8CPUs.

519502 paused faithfully after2,053,802,665tokens/1,310,570documents/1,913shards
and used1,725allocation seconds (0.4791666667H200-hours).519577 is the corrected
continuation. Its Slurm preflight compares identical canonical inputs using7
and11 independent reader/tokenizer workers, plus the earlier coordinator-reader
baseline, and checks every saved shard and token equivalence before writes.
The initial16CPU submission reply was not saved. Recovery found no tagged job
and an exact test-only request demonstrates the deterministic12CPU rejection.
Keep its16H200-hour reservation conservatively charged, allow only corrected
requests within12CPUs, and block retrying the rejected16CPU request. Later
submissions save stdout/stderr/returncode in their durable intent receipts.

## Evaluation CPU override (2026-10-09, subsequent instruction)

The user subsequently authorized12CPUs for evaluation tokenization too.
This supersedes the8CPU evaluation continuation above. Active training519577
keeps its already loaded11-worker execution; after its faithful exit75 at the
training token cap, filtering_resume submits select with12CPUs and the explicit
--evaluation-workers11 override. Evaluation-only continuation preserves the
training cursor/token counts and uses11 independently reading/tokenizing workers
plus coordinator, with the same two-batch credits and canonical evaluation slots.
The preparation config/core fingerprints remain unchanged. Each new execution
identity records its effective train/evaluation worker counts. The continuation
runs original prefix checks and matched7/11-worker token equivalence gates on
Slurm before writing evaluation artifacts. Packing/verification remain8CPUs.
40lightweight checks passed; the future evaluation runtime gate is not yet run.

## Parallel packing and full verification (2026-10-09)

The user explicitly requested both stages be parallelized. Keep the immutable
dataset.py, preparation_parallel.py, environment, split, clean, score and select
identities intact. slurm/packing_entry.py is the explicit execution overlay for
pack/verify of dolma-dclm10-005, sealed separately by the existing submission
controller. Use one H200 and eight CPUs: seven pinned worker processes plus a
coordinator, one library thread each, at most fourteen outstanding ordered tasks.

Packing plans each original shard boundary using Arrow token-list lengths and
canonical document metadata. It transfers only input ranges and at most one
sequence of carry. Workers reconstruct with the original TrainingPacker, check
their final carry against the plan, and write/fsync/hash uniquely named files.
Only the coordinator appends results in canonical order to the original state.
Uncommitted speculative outputs remain unpublished and are retained; there is
no cleanup or overwrite of committed prefix artifacts. Evaluation uses the
original independent-document windowing and whole-document shard boundaries.

Full verification retains every original check. Workers verify all manifest
artifact checksums; every entry of the native seed-42 shuffle and inverse; every
archived evaluation original and every selected document against the raw corpus;
BOS/EOS, canonical order, budget and split disjointness; and every packed sequence
through independent reconstruction of tokens, masks and document spans. Strict
canonical shuffle positions plus the verified bijection prove training uniqueness
and disjointness with the ordered evaluation suffix. Global quality membership
is checked by indexed SQL anti-joins against every score/winner, using the actual
score/position tie boundary and full counts, including detection of out-of-range
rows. Workers retain no raw shard cache and use at most one active raw mapping.
The coordinator preserves the native global shuffle construction and seals the
original verification report only after all checks pass.

519687 paused with exit75 after 1,962,541,056 packed tokens / 1,832 shards and
used 1,135 allocation-seconds (0.3152777778 H200-hours), reconciled before 519745.
62 lightweight tests passed. Slurm fixtures verified byte-identical Arrow shards
and identical masks/spans/counters for serial, parallel, original-to-parallel
resume and parallel resume. Original and parallel full verification reports
agreed. Independent reconstruction rejected a mask corruption whose checksum
was updated; ranking checks rejected a removed quality winner. The real saved
1,832-shard prefix remains intact, including reconstruction of its final two
shards and equality of its cursor/carry with a full canonical plan.

Matched 34,242,560-token benchmark: serial 19.5738s / 1,749,409 tokens/s; parallel
4.1589s / 8,233,582 tokens/s, 4.7065x speedup, byte-identical output, conservative
RSS upper bound 24,766,447,616 bytes. Plans were precomputed for both paths;
measurements include worker startup and reading/packing/writing/hashing. Full
production throughput and future full-data verification are tracked separately
in packing-access/parallel-v1/{progress,verification-progress}.json and the
repository validation receipt. The sole controller submits verify after pack.

Production packing 519745 completed at 2026-10-09 18:33:18 Asia/Taipei with
exactly 6,000,000,000 training input tokens in 732,422 sequences / 5,584 shards.
Evaluation packing produced 52,515 windows / 410 shards and 76,930,208 prediction
tokens from all 50,000 reserved documents. The original 1,832-shard prefix remains
identical, both cursors reached EOF and carry is empty. The allocation was
reconciled at 0.195 H200-hours, including validation and benchmarking.
Controller submitted full verification 519754 with eight CPUs and one H200;
actual checksum progress records seven worker PIDs and 14 outstanding tasks.
Full real-data verification remains running; its completion has not been claimed.

Verification 519754 failed at 18:40:05 with KeyError `id` in archived-original
provenance: the immutable archive schema contains document_json and provenance
fields, without flattened id/text columns. This was a verifier bug after all
artifact checksums and canonical split checks passed, not a detected corruption.
The verifier now uses the existing full JSON equality check for archived IDs;
selected document ID checks remain unchanged. The native fixture now matches
the immutable archive schema exactly. 62 lightweight tests passed. Controller
was disabled before the fix, immutable metadata checked and overlay resealed
with an audit entry; failed allocation reconciled at 0.1008333333 H200-hours.
Controller submitted verification 519786, which reruns native fixture gates
under the new source identity before full real-data verification.
