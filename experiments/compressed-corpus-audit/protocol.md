# Compressed corpus integrity diagnosis

Split 513955 failed after 3606 seconds on one H200/eight CPUs. The error is
zstd decompression "Data corruption detected" during full map-style generation;
the exception contains no shard filename. No canonical split/eval archive was
produced. Charge 1.0016667 H200-hours and preserve all artifacts/identities.

Submit a bounded read-only cache-audit in Slurm, one H200/eight total CPUs,
seven spawn workers plus coordinator, one library thread each. Use the completed
004 download manifest, verify its identity and paths, then record compressed
SHA256/size and full zstd stream decoding for each canonical shard. Persist
ordered per-file results and resumable cursor under a unique /work run directory.
Record exact failed shard names, byte positions and exception type/message.
Do not parse/filter/tokenize records or replace/delete downloads or raw caches.

Allocate at most 30 minutes (0.5 H200-hour reservation). Runtime timeout is ten
minutes; worker deadline is nine minutes, with resume exit 75 for incomplete
scans. Partial results identify corruption but are not a full integrity seal.
Charge every continuation before resuming the same audit identity/cursor.

After diagnosis, compare failed files with pinned Hub LFS metadata and an
independent decoder or newly downloaded copy in a separate directory. Distinguish
local download corruption, upstream damaged data and decoder-specific behavior
before selecting a repair. Never skip files/documents or silently change the
dataset revision/canonical order, manifests, or existing preparation identity.
Resume/promote production only after evidence and required validation/accounting.
