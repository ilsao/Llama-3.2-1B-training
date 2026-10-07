# Native loader recovery protocol

Registered before any recovery validation allocation. CONFIRMATORY hypothesis:
explicit spawn startup for both stdlib multiprocessing (datasets parallel download)
and multiprocess (native Arrow building), before thread pools and Arrow libraries,
avoids the observed no-progress child wait while preserving native document order.
The original job shows suspected deadlock; no exact Python stack confirms the cause.

Validate through Slurm with one H200, eight CPUs total, one thread per worker.
Use eight small compressed JSON shards, a preceding threaded artifact/checksum
phase, and pinned native load_dataset at one and seven workers. Require identical
original fields/order and Dataset.shuffle(seed=42) ordered indices, then exercise
the existing split/archive/resume/retirement checks. Charge failed and successful
allocations. A synthetic pass does not establish real-corpus completion.

Do not change the canonical full train ordering, split, filters, selection, packing,
or dependency versions. If source hashes change, use dolma-dclm10-003 and reuse
only a complete authenticated download. Retain dolma-dclm10-002 metadata and files.
Promote to production split only after validation success and accounting; record
both start methods in resolved metadata. If it stalls again, collect a Python stack
before attributing a root cause or repeating the same configuration.
