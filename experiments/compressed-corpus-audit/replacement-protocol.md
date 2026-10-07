# Isolated pinned shard replacement validation

Audit 514055 completed all 6081 shards: exactly one decode failure, in
data/common_crawl-science_math_and_technology-0015/shard_00001148.jsonl.zst.
Its actual SHA256 is 7ec222880366d0dc111b22f369a797ddad2be070026c4f6dde0e8f594ac06564,
different from the cache blob name c624bddd1c7f810c737b4cb0a75a6d7b6a45c9667ce19a4441b26e22d98e9ec3.
This is the only blob-name/hash discrepancy in all audit records; a cache blob
name is evidence, not independent upstream authentication.

In a bounded Slurm cache-repair job, query HF paths-info for the exact pinned
dataset revision and failed path. Require upstream LFS SHA256/size. Preserve the
bad bytes and historical manifests, then download a new copy into a unique
/work runs/cache-repair directory with a separate HF cache. Authenticate its
full hash/size against pinned upstream metadata. Compare full decoded byte hashes
using Python-zstandard and Arrow compressed input; parse every JSON document and
validate id/text without filtering. Record old/new artifact lineage and result.
Only a completely verified replacement can be promoted into a new preparation's
download manifest. Do not overwrite 001/002/003/004 or change canonical file order,
dataset revision or split semantics. Remaining 6080 artifacts must remain identical.

Allocate one H200/eight CPUs for at most 30 minutes (0.5 H200-hour reservation),
with a ten-minute runtime timeout. Charge even a failed or partial attempt.
On success and accounting, implement/validate an explicit authenticated download
repair import into new preparation dolma-dclm10-005; then rerun full canonical
split/eval preservation. No production split/filter/cleanup before prerequisites.
