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
