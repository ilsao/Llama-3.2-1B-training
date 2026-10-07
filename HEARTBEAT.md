# Autoresearch heartbeat

Every 20 minutes, read AGENTS.md, research-state.yaml, findings.md and research-log.md.
Use the existing pinned environment, Slurm submission helpers and experiment ledgers.
This controller runs on nano4's login node: only small metadata reads, edits, Git,
accounting and job submissions are allowed here. All heavy work runs through sbatch.
Never print credentials, use broad deletion, bypass sandbox approvals, or launch
another heartbeat agent. The controller lock prevents overlapping cron deliveries. Cron queues a prompt into
the bound visible Codex thread; each research cycle reports directly in that chat.

## Work on each tick

1. Inspect tracked jobs with squeue/sacct and their small log tails. If still running,
   record progress and perform useful independent code/research work; do not poll/wait.
2. Reconcile terminated allocations using the appropriate preprocessing/training ledger.
3. Advance a stage only after exit 0, required artifact checks, and accounting. Exit 75
   continues the original stage/ID/code/environment. Diagnose other failures first.
4. Before submitting, persist a unique submission intent in research-state.yaml. Use
   a unique Slurm --comment tag for raw sbatch submissions; store returned job ID
   immediately. If submission outcome is uncertain, recover via squeue/sacct and the
   ledger before retrying; never assume a failed client means no job was submitted.
5. Update state, findings and decision log when something changes. Store large outputs
   only under /work/asciibase64/Llama-3.2-1B-training/. Put a concise progress report
   in to_human/. Do not message third-party channels.
6. Stop this cycle after bounded work. Let the next cron tick continue. When the lab
   is complete, disable this project's heartbeat. If compute budgets are exhausted,
   submit no further heavy jobs; record the limiting condition and use lightweight
   analysis/reporting instead. Never reset budgets or change resume geometry.

## Authorized next steps

- Complete download dolma-dclm10-001 with the preserved .download-continuation worktree.
  Its original source hashes match; CPU allocation is now 8 total, with 1 H200.
- On current source, add independent full eval raw preservation and safe artifact
  retirement, then use new preparation dolma-dclm10-002 to import the COMPLETE
  download. Do not import incomplete state or mutate preparation fingerprints.
- Preserve exact seed-42 native global shuffle, global dedup, CC top 10%, whole-document
  selection order, 6B cap and context 8192. Complete split/clean/score/select/pack/verify.
- Save all 50,000 eval original documents, tokenized documents and packed sequences,
  and fixed first 10,000 subset. Verify before project-only cleanup.
- Validate environments and 1/7-worker resume equivalence; run model/resume checks
  and matched-global-batch 4/8 GPU scaling checks. Charge all allocations.
- Prefer reproducible AdamW baseline before optimizer experiments. Protocol commits
  precede experiments; do not stage unrelated user changes. Target perplexity <=20
  before 2026-10-09; report measured results and remaining deadline honestly.

## Native loader recovery

Split 512532 was cancelled and charged after a suspected multiprocessing stall;
the exact root cause remains unconfirmed. Follow the committed
experiments/native-loader-recovery/protocol.md. Configure spawn for both Python
process libraries, validate native compressed JSON with a threaded prelude and
1/7-worker equivalence in Slurm, then reconcile before production promotion.
Source changes require dolma-dclm10-003, reusing only a completed download.
Retain dolma-dclm10-002 identity and files; do not resubmit its changed sources.

Real split 513322 failed and was charged: declared string metadata conflicts with
actual nested JSON. Follow experiments/lossless-schema-loader/protocol.md:
validate the lossless canonical file/line adapter, all original fields, native
shuffle equivalence, complete split entrypoint and bounded real-source samples
in Slurm. Reconcile validation before production promotion to new preparation
dolma-dclm10-004. Preserve 001/002/003; import only the completed 001 download.
After successful split manifest/full eval archive/accounting, submit clean;
never let a failed split trigger filtering or cleanup.

Full generation split 513955 subsequently failed on zstd data corruption without
a shard filename. Follow experiments/compressed-corpus-audit/protocol.md and use
`cache-audit` through the durable wrapper: seven workers plus coordinator,
one H200/eight CPUs, 30-minute ceiling and ten-minute runtime timeout. Save exact
per-shard hashes/decoding failures and an ordered resumable cursor. Exit 75
continues the same audit only after accounting. Read only; do not skip documents,
change canonical revision/order, overwrite previous manifests or delete artifacts.
Locate failed shards, then authenticate against pinned upstream metadata and an
independent decoder/new separate copy before selecting any production repair.

Audit 514055 completed all 6081 shards and identified one corrupt local blob.
Isolated replacement validation 514077 passed pinned upstream LFS hash/size,
both complete decoder byte hashes, and all 47,233 JSON documents. Both jobs are
reconciled. Follow experiments/compressed-corpus-audit/replacement-protocol.md:
The authenticated import is implemented in 9e9cd2a following protocol 4e7054c
(experiments/compressed-corpus-audit/import-protocol.md). Job 514094 imports into
new dolma-dclm10-005, preserving the remaining 6080 artifacts and canonical
ordering. Inspect it once and reconcile after termination; validate the complete
manifest, identity and single replacement lineage before submitting split.
Old blobs and 001/002/003/004 manifests remain immutable. Filtering still requires
the complete independent eval archive. Never resubmit an uncertain/live import.

## Controls

`python3 slurm/heartbeat.py status|install|run-once|disable`
`python3 slurm/heartbeat.py run-once --check-only` validates agent access without jobs.
Logs and last-run metadata live under the project work root's runs/heartbeat/.
Normal ticks use codex queue to continue the bound visible thread. Target UUID and
delivery receipts live in runs/heartbeat/target.json and last-run.json. Install
defaults to CODEX_THREAD_ID; use --thread UUID only to intentionally bind another
user-selected task. Do not launch hidden codex exec research cycles. The manual
--check-only option remains a standalone read-only CLI-access check.
