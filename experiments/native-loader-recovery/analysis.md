# Native loader recovery result

CONFIRMATORY synthetic validation: passed (job 512718, exit 0, 2026-10-08
03:43:42–03:44:50 Asia/Taipei). Actual allocation: 68 seconds, one H200,
eight CPUs total; 0.0188888889 H200-hours reconciled. Protocol 294505c preceded
implementation/validation; tested implementation commit 5f1d4e7.

Both stdlib multiprocessing and multiprocess used spawn. Eight compressed JSON
shards loaded after a threaded checksum phase produce identical original fields,
document order and native seed-42 shuffle indices for one/seven workers. All
existing dedup/filter/token/packing/mask/checksum/resume/worker-failure/archive
and offline retirement/reload contracts passed in the same allocation.

Result: /work/asciibase64/Llama-3.2-1B-training/runs/preparation-validation/native-loader-spawn-001/result.json.
This supports using the spawn recipe in a new preparation. It does not establish
the exact Python root cause of cancelled 512532 or verify the real full corpus.
Promote to dolma-dclm10-003 completed-download import, then production split;
retain earlier preparation identities/files and require production verification.
