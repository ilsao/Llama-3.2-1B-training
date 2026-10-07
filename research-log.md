# Research Decision Log

- 2026-10-08 Asia/Taipei: User authorized autoresearch heartbeat and execution of
  the plan. Adopt 20-minute cron-driven Codex cycles with persistent state and
  overlap prevention. Restore full download/filter pipeline because storage expanded;
  retain full eval raw/tokens and implement project-only cleanup after verification.

- Runtime validation 511939 completed: 1/7-worker equivalence, native ordering,
  shuffle, split disjointness, selection, packing/masks, worker failure cursor and
  independent eval original archive/resume passed. Reconciled cost: 55 seconds,
  one H200 = 0.0152778 H200-hours. Added a separate synthetic offline-retirement
  reload scenario; this requires a new validation allocation.

- Validation 511949 passed offline train/full-eval/subset reload after retirement;
  cost 65 seconds / one H200 = 0.0180556 H200-hours. Training environment setup
  511940 passed pip check; cost 200 seconds / one H200 = 0.0555556 H200-hours.
  Both reconciled. Added historical-manifest hashes to production verification
  receipts so cleanup rejects modified provenance. Lightweight suite: 44 passed.

- Four-GPU runtime correctness 511971 passed; actual 20 seconds x four H200s
  = 0.0222222 H200-hours, reconciled. Eight-GPU runtime submitted as 511980.

- Eight-GPU runtime correctness 511980 passed; 32 seconds x eight H200s =
  0.0711111 H200-hours, reconciled. Handing continuation to the installed heartbeat.
  Original download remains running at 1632/6081 recorded shards.

- 2026-10-08T00:42:51.300676+08:00 bounded heartbeat: download 511918 RUNNING,
  1831/6081 checkpointed shards; no completed manifest, no downstream submission.
  Audited eight terminated allocations against sacct: 1.0716667 reconciled
  H200-hours across both ledgers; active download 0.4955556 hours
  observed separately, 24-hour allocation upper bound retained. No new allocation.
  Fixed ambiguous sbatch replies releasing training reservations. Persist unique
  comment/intents and returned receipts, retain uncertain budget, and recover one
  matching allocation without retry. Verified nano4 does not store accounting
  comments; startup allocation receipts provide terminal fallback. 50 lightweight
  tests, shell syntax and diff checks pass. Full model/resume/scaling remain blocked
  on sealed real data. Durable report: to_human/heartbeat-20261008-0043.md.

- 2026-10-08T01:05:51.138250+08:00 bounded heartbeat: download 511918 still RUNNING,
  2827/6081 checkpointed shards, no completion manifest. Audited eight
  terminated jobs; 1.0716667 reconciled H200-hours; active observed
  0.8788889 separate from its 24-hour reservation. No new jobs or stage
  advancement. Preserved four source hashes still match. Fixed preprocessing
  lost-reply safety with durable central intents, /work receipts, unique tags,
  retained reservations and bounded recovery shared with training. 55 lightweight
  tests and shell/diff checks passed. Report: to_human/heartbeat-20261008-0105.md.

- 2026-10-07T17:15:52.812764+00:00: User reported missing visible
  heartbeat outputs. Verified codex queue delivery to the current thread, then
  changed normal ticks from hidden codex exec sessions to visible-thread queue.
  Kept cron interval and project-only controls. Read-only --check-only remains
  available; normal progress is now reported directly in the bound chat.

- 2026-10-08T01:20:07.801786+08:00 visible heartbeat: download 511918 RUNNING, 3389/6081
  shards; complete manifest absent. Eight terminated allocations match Slurm,
  1.0716667 reconciled H200-hours; active download 1.1166667 provisional.
  No new jobs, no production cleanup, no repeated tests. Saved to_human/heartbeat-20261008-0120.md.

- 2026-10-08T01:22:26.038189+08:00 visible heartbeat: download 511918 RUNNING, 3455/6081
  shards; no completion manifest. No ready downstream stage, no new allocation
  or cleanup. 1.0716667 H200-hours reconciled; 1.1552778 active provisional.
  State and to_human/heartbeat-20261008-0122.md saved; no repeated tests or polling.

- 2026-10-08T01:43:22.428841+08:00 bounded visible heartbeat: download 511918 RUNNING, 3936/6081 shards; no completion manifest. Eight terminated allocations match Slurm and both ledgers; 1.0716667 H200-hours reconciled, 1.4636111 active provisional at 01:40:56. Preserved source hashes match. No new allocation, cleanup, polling or repeated tests. Saved to_human/heartbeat-20261008-0143.md.

- Provenance clarification: the original configuration records three source hash entries (dataset.py, requirements-data.txt, third_party/dclm_fasttext.py), all checked and matching this cycle. Earlier state wording counted four; no preparation configuration or source was changed.

- 2026-10-08T02:02:19.084697+08:00 bounded visible heartbeat: download 511918 RUNNING, 4523/6081 shards; completion manifest absent. Eight terminated allocations still match Slurm/ledgers; 1.0716667 H200-hours reconciled, 1.7911111 active provisional at 02:00:35. No new allocations or cleanup, no polling, no repeated tests. Next action remains complete-download import followed by split/filter pipeline. Saved to_human/heartbeat-20261008-0202.md.

- 2026-10-08T02:20:41.545718+08:00 bounded visible heartbeat: download 511918 RUNNING, 5375/6081 shards; completion manifest absent. Eight terminated allocations match Slurm/ledgers; 1.0716667 H200-hours reconciled, 2.1261111 active provisional. No new allocations or cleanup, no polling, no repeated tests. Saved to_human/heartbeat-20261008-0220.md.

- 2026-10-08T02:44:19.253432+08:00 bounded heartbeat milestone: download 511918 COMPLETED exit 0, all 6081 shards and full manifest. Reconciled 8810 seconds / one H200 = 2.4472222 H200-hours, shared total 3.5188889. Verified completed manifest metadata, matching dataset revision, original source hashes, and nine allocations against ledger/accounting. Submitted completed-download import for dolma-dclm10-002 via durable wrapper as job 512500; RUNNING on one H200 and 8 CPUs. Full checksums execute on compute node; corpus files are reused without copying. Keep 24 H200-hour reservation; next split awaits import success/accounting. No cleanup or polling. Saved to_human/heartbeat-20261008-0244.md.

- 2026-10-08T03:02:23.682216+08:00 bounded heartbeat: import/checksum 512500 COMPLETED exit 0, full 6081-artifact reuse manifest. Reconciled 31 seconds x one H200 = 0.0086111 H200-hours; shared total 3.5275000. Ten terminated allocations match ledger/Slurm. Source hashes/config and manifest chain match. Submitted split dolma-dclm10-002 as 512532 through durable intent/reservation/tag/receipt wrapper, now RUNNING on 1 H200 and 8 CPUs. Native full-train cache, seed-42 global shuffle, last-50,000 eval and independent original eval archive run on compute node. Filtering awaits split success/accounting. No cleanup or polling. Saved to_human/heartbeat-20261008-0302.md.

- 2026-10-08T03:28:33.511421+08:00 bounded heartbeat: split 512532 had no log/native Arrow/split state progress for >20 minutes. Slurm step .1 had about 9 CPU seconds; bounded overlap inspection showed parent wait4 on futex-blocked child and another zombie exit-0 child. Suspected multiprocessing deadlock, root cause unconfirmed. Persisted cancellation intent, cancelled exactly 512532, preserved all files, reconciled 1416 seconds x 1 H200 = 0.3933333 H200-hours; shared total 3.9208333, no outstanding allocation. Registered native-loader recovery protocol; no new jobs or polling. Saved to_human/heartbeat-20261008-0328.md.

- 2026-10-08T03:44:11.098175+08:00 bounded recovery heartbeat: no existing active allocations; prior cancellation agrees with Slurm/ledger. Implemented spawn for both process libraries before threaded runtime, identity/log methods, phase logs and 5-minute stack dumps; changed sources require new dolma-dclm10-003. 57 lightweight tests, shell syntax, CLI help and diff checks pass. Committed 5f1d4e7 before submitting registered recovery validation as 512718, RUNNING on 1 H200/8 CPUs. 30-minute allocation reservation 0.5 H200-hours and runtime timeout 600s+30s teardown. Compressed 1/7 native loaders reached dataset save; full validation result pending. No production resubmission/cleanup/polling. Shared reconciled 3.9208333; upper with reservation 4.4208333. Saved to_human/heartbeat-20261008-0344.md.

- 2026-10-08T04:02:00.942851+08:00 bounded heartbeat: recovery 512718 COMPLETED exit 0 in 68s; full registered compressed 1/7 native order/shuffle/archive/resume/offline contracts passed with both start methods spawn. Reconciled 0.0188889 H200-hours; shared 3.9397222, all 12 terminated jobs agree with Slurm/ledgers. Submitted new dolma-dclm10-003 completed-download import as 513010 with durable intent/tag/reservation/receipt; RUNNING on 1 H200/8 CPUs. Retain 24-hour allocation ceiling (27.9397222 shared measured plus reserved upper bound). Preserve 001/002 original identities and files; source hashes changed only for new 003. No cleanup, repeated tests or polling. Saved to_human/heartbeat-20261008-0402.md.

- 2026-10-08T04:21:17.797630+08:00 bounded heartbeat: 003 import/checksum 513010 COMPLETED exit 0, 41 seconds / 1 H200 = 0.0113889 H200-hours reconciled. Shared total 3.9511111; all 13 terminated allocations agree with Slurm/ledgers. Manifest, original download provenance, source hashes/config and spawn identity match. Submitted canonical split on dolma-dclm10-003 as 513322 via durable wrapper, RUNNING on 1 H200/8 CPUs; startup confirms seven native workers and both spawn methods. Keep 24-hour allocation reservation (27.9511111 shared upper bound). Full eval preservation/filtering await split, no cleanup/repeated tests/polling. Saved to_human/heartbeat-20261008-0421.md.

## 2026-10-08 04:49 Asia/Taipei — lossless schema recovery

Split 513322 FAILED exit 1 (28s): nested metadata vs declared string in pinned dataset card. Reconciled 0.0077778 H200-hours; shared measured 3.9588889. No split manifest/archive. Protocol a0594fc precedes implementation fdaa233 and one bounded validation submission 513790. New lossless map-style adapter authenticates all raw JSON fields/order, retains canonical native shuffle and reserved eval originals. 59 lightweight tests pass; heavy heterogeneous/full-entrypoint/real-source checks RUNNING, not yet passed. Reserve 0.5 H200-hours; no polling, production resubmission, training or cleanup this cycle. New prep004 only after validation success and accounting.

- 2026-10-08T05:03:42.762082+08:00 bounded heartbeat: lossless validation 513790 COMPLETED exit 0, 78s/1 H200 = 0.0216667 H200-hours reconciled; heterogeneous original JSON/order/native indices/full split entrypoint/archive/resume/offline checks and 12 real docs across six sources passed. Results/source hashes/protocol chain checked; 15 terminated allocations agree with Slurm/ledgers, shared 3.9805556. Result commit 32534da precedes new 004 completed-download import 513930, observed RUNNING on 1 H200/8 CPUs. Fingerprint/config/source/spawn metadata match. Reserve 24 H200-hours, shared upper 27.9805556. Full split/filtering await import success/accounting. No code changes, repeated tests, polling, cleanup or production training. Saved to_human/heartbeat-20261008-0503.md.

- 2026-10-08T05:22:43.814085+08:00 bounded heartbeat: 004 import 513930 COMPLETED exit0, all 6081 artifacts fully authenticated on compute node and reused without copies. Reconciled 34s/1 H200 = 0.0094444 H200-hours; shared 3.9900000, all 16 terminated allocations agree with Slurm/ledgers. Manifest/canonical order/source lineage/config/environment/source hashes match. Submitted full canonical lossless split 513955 through durable wrapper; observed RUNNING at 05:21:31 on 1 H200/8 CPUs, spawn and seven workers, generation log 549963 examples. Reserve 24 H200-hours (27.9900000 shared upper). Full eval archive/filtering remain completion/accounting gated; no code changes/repeated tests/polling/cleanup/training. Saved to_human/heartbeat-20261008-0522.md.

- 2026-10-08T05:41:45.115179+08:00 bounded heartbeat: split 513955 RUNNING, 1159s/1 H200 observed (0.3219444 H200-hours provisional); generation advances to 42793721 examples and stderr fresh at 2026-10-07T21:40:20.380355+00:00. Split/eval archive manifests absent, so no filtering submission. Sixteen terminated allocations match Slurm/ledgers; shared reconciled 3.9900000, upper with active reservation 27.9900000. Current identity/source hashes unchanged. No new jobs, repeated tests, polling, cleanup or production training. Saved to_human/heartbeat-20261008-0541.md.

- 2026-10-08T06:01:34.245705+08:00 bounded heartbeat: split 513955 RUNNING (2361s, 1 H200/8 CPUs), generation 57842721 examples, up from 42793721; fresh stderr at 2026-10-07T22:00:23.594749+00:00. Split/state/full eval archive absent, so filtering not yet ready. Sixteen terminated allocations match Slurm/ledgers, shared reconciled 3.9900000; active provisional 0.6558333, reservation 24, shared upper 27.9900000. Identity/source hashes unchanged; no jobs submitted, polling, tests repeated, cleanup or production training. Saved to_human/heartbeat-20261008-0601.md.

- 2026-10-08T06:21:27.277098+08:00 bounded heartbeat: split 513955 RUNNING, 72380721 generation examples (previous 57842721), fresh stderr, 3560s on 1 H200/8 CPUs = 0.9888889 H200-hours provisional. Split/state/eval archive manifests absent, filtering gated. Sixteen terminated allocations match Slurm/ledgers; shared reconciled 3.9900000, upper with reservation 27.9900000. Source/config identity unchanged. No new jobs, polling, repeated tests, cleanup or production training. Saved to_human/heartbeat-20261008-0621.md.

- 2026-10-08T06:47:50.089364+08:00 bounded heartbeat: split 513955 FAILED exit1 (3606s), zstd Data corruption detected without shard path; no split/eval archive. Reconciled 1.0016667 H200-hours; shared measured 4.9916667, 17 terminated allocations audited. Protocol 692e6b0 then read-only audit code 64175a7, 62 lightweight tests and syntax/AST/CLI/diff pass. Submitted 514055 corpus-audit-001, RUNNING on 1 H200/8 CPUs, 7 spawn workers, 10min timeout/9min worker deadline, 0.5 H200-hour reservation (shared upper5.4916667). At 06:46, 500/6081 ordered shard checks committed with no failure yet. Preserve all corpus/cache files, no skipped documents, repair/retry/filter/cleanup/train or polling. Saved to_human/heartbeat-20261008-0647.md.

- 2026-10-08T07:10:13.148840+08:00 bounded visible heartbeat: audit 514055 COMPLETED all 6081 shards, one local corrupted shard identified; reconciled 103s/1 H200=0.0286111 hours. Protocol 1efd4ef and code 7889201 precede isolated replacement 514077, completed 11s/1 H200=0.0030556 hours. Pinned upstream LFS hash/size, two full decoded byte hashes and all 47233 documents passed; original Arrow decoder also fails. Old blobs/manifests preserved. 66 lightweight tests/syntax/CLI/AST/diff passed. All 19 allocations agree with Slurm/ledgers; shared 5.0233333333 hours, no active/reserved allocation. Next cycle implements authenticated new005 import before canonical split/eval/filtering. One bounded post-submit observation; no polling/duplicate submission/cleanup/training. Saved to_human/heartbeat-20261008-0710.md.
