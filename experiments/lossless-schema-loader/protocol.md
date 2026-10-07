# Lossless schema loader recovery

Split 513322 failed in 28 seconds: the pinned dataset card declares metadata as
string, but actual JSON records contain nested structures. Its allocation is
charged; no split manifest or evaluation archive was produced.

Build a disk-backed map-style HF Dataset from the completed download's canonical
file list, preserving each nonblank JSON document line as a large string plus
validated id/text columns. No filtering, sampling or shuffling occurs in this
adapter. Dataset.from_generator uses contiguous ordered file groups (verified in
pinned datasets 3.6.0 utils/sharding.py); verify every full original document
against the source files before using Dataset.shuffle(seed=42). The resulting
row count and order determine the canonical indices, independent of column schema.
Keep the last 50,000 documents and fixed first 10,000 subset as before.

Archive all reserved original JSON documents losslessly with row/source/shuffle
provenance, independent of raw caches. Keep semantic full-field equality checks
and byte equality of preserved JSON in validation. Retain all previous identities
and files. Changed production sources require dolma-dclm10-004.

Before production, submit one bounded Slurm parallel-check on 1 H200/8 CPUs,
30-minute allocation / 10-minute runtime timeout, reservation 0.5 H200-hours.
Exercise heterogeneous nested metadata/attributes, missing and extra keys;
compare 1/7-worker order and seed-42 indices with native JSON id/text projection;
check full-field round trips, archive/interrupted resume and existing downstream
contracts. Also sample at most two real documents from each source family on the
compute node and compare adapter indices/order with native id/text projection.
Fail closed on any mismatch. Charge the allocation before production promotion.
Do not wait/poll for it in this heartbeat cycle.
