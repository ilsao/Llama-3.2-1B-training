# Authenticated new preparation import

The completed, reconciled isolated repair 514077 provides an authenticated
replacement for the only failing shard in audit 514055. Result SHA256 is
a9050e57c6ae06615795420232a13bb635ac3c71630ac28488db8e2d7160bc4d.

Implement an explicit download import into new dolma-dclm10-005. Require source
004 completed manifest/configuration, the replacement run ID and the exact
result checksum. Authenticate source fingerprint, manifest, dataset revision,
validator source hash, successful terminal accounting, pinned upstream hash/size,
both decoded hashes, positive document count and unmodified old artifacts.
Reject different membership, order, filename/source, untrusted paths or a
replacement outside its isolated run. Keep all 6080 other records unchanged.
Full compute-node checksums must validate the new manifest before completion.
Record immutable source/result/audit lineage and changed artifact explicitly.
No corpus copies, old manifest rewrites, dropped documents or revision changes.

Use tiny metadata tests for successful replacement and rejection of tampering,
mixed identities, incomplete decoder results, changed order and unsafe paths.
Then submit one bounded Slurm download/repair import through the durable wrapper:
one H200/eight CPUs, 30-minute allocation (0.5 H200-hour reservation), ten-minute
runtime timeout and the existing stopping margin. Reconcile actual cost after
termination. Only a passed authenticated import permits a later canonical full
split with seed42 and independent full50000 eval preservation. Filtering and
cleanup remain gated on those production artifacts. This is an integrity repair,
not a perplexity experiment or a claim that production split has passed.
