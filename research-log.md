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

- 2026-10-08T07:24:50.887511+08:00 bounded visible heartbeat: 19 terminated allocations match Slurm/ledgers, no active old jobs; isolated repair result remains pinned. Registered protocol4e7054c and implementation9e9cd2a for authenticated new005 import. 72 lightweight metadata/tiny-file/mocked tests and syntax/CLI/AST/diff pass. Submitted exactly one repaired download import514094 with durable intent/tag/receipt/reservation. Single snapshot RUNNING on1H200/8CPUs,4s observed; new manifest absent. Reserve0.5 H200-hours with30-minute allocation/600s runtime; shared measured5.0233333333, upper5.5233333333. Config/source/spawn fingerprint matches new005; preserve001-004/old corrupted bytes. No polling, split/filtering/cleanup or training this cycle. Report:to_human/heartbeat-20261008-0724.md.

- 2026-10-08T07:42:57.703692+08:00 bounded visible heartbeat: authenticated005 import514094 COMPLETED exit0,39s/1H200=0.0108333333hours reconciled. Full6081 manifest preserves canonical order,6080 identical artifacts,one approved replacement; source/config/environment/spawn/result/request/event lineage checked. All20 terminated allocations match Slurm/ledgers; shared5.0341666667hours. Result commit3de0902 precedes canonical005 split514101, submitted once and observed RUNNING on1H200/8CPUs,10s provisional,7 configured workers. Reserve24hours (shared upper29.0341666667); split/eval archives pending. No source changes or repeated tests; syntax/metadata checks passed. No polling/filtering/cleanup/training or independent heartbeat. Savedto_human/heartbeat-20261008-0742.md.

- 2026-10-08T08:01:08.650441+08:00 bounded visible heartbeat: split514101 RUNNING on1H200/8CPUs,1127s observed; full map-style generation reaches45076522 examples, fresh stderr. Split/state/evaluation-raw absent, so no filtering or other jobs. Current005 identity/source/config/download manifest unchanged;20 terminated allocations match Slurm/ledgers. Reconciled5.0341666667 H200-hours; active0.3130555556 provisional,24-hour reservation retained,shared upper29.0341666667. Lightweight storage metadata records10119828045824 available bytes. No polling/source changes/repeated tests/cleanup/training. Savedto_human/heartbeat-20261008-0801.md.

- 2026-10-08T08:21:11.480942+08:00 bounded visible heartbeat: split514101 RUNNING on1H200/8CPUs,2330s observed; full map-style generation74415313 examples, up from45076522; stderr fresh. Complete split/state/evaluation-raw absent, so downstream stages not ready and no jobs submitted. Current005 identity/source/config/repaired manifest unchanged;20 terminated allocations agree with Slurm/ledgers. Reconciled5.0341666667 H200-hours; active0.6472222222 provisional,24-hour reservation retained,shared upper29.0341666667. Storage metadata:9689817481216 available bytes. No polling/source changes/repeated tests/cleanup/training. Savedto_human/heartbeat-20261008-0821.md.

- 2026-10-08T08:41:13.673833+08:00 bounded visible heartbeat: split514101 RUNNING on1H200/8CPUs,3531s observed; full map-style generation102970649 examples, up from74415313; stderr fresh. Split/state/evaluation-raw absent, so no downstream jobs submitted. Current005 identity/source/config/repaired manifest unchanged;20 terminated allocations agree with Slurm/ledgers. Reconciled5.0341666667 H200-hours; active0.9808333333 provisional,24-hour reservation retained,shared upper29.0341666667. Storage metadata:9295654879232 available bytes. No polling/source changes/repeated tests/cleanup/training. Savedto_human/heartbeat-20261008-0841.md.

- 2026-10-08T09:02:09.000995+08:00 bounded visible heartbeat: split514101 RUNNING on1H200/8CPUs,4730s observed; map-style cache generation completed with104020922 observed rows. Source lineage now251/6081 shards,13833131 contiguous rows checked; cursor/order metadata matches canonical manifest, fresh stdout. Periodic300s faulthandler Timeout stack dump is diagnostic, not Slurm failure. Native shuffle/eval archive/complete split pending, no downstream job. Current005 identity/config/source/repaired manifest unchanged;20 terminated allocations agree with Slurm/ledgers. Reconciled5.0341666667 H200-hours; active1.3138888889 provisional,24 reservation retained,shared upper29.0341666667. No polling/source changes/repeated tests/cleanup/training. Savedto_human/heartbeat-20261008-0902.md.

- 2026-10-08T09:21:42.632833+08:00 bounded visible heartbeat: split514101 RUNNING on1H200/8CPUs,5934s observed; source lineage524/6081 shards, up from251, covers28344326 rows. Native rowcount remains104020922; cursor/order/contiguity match canonical manifest, stdout fresh. Complete native shuffle/split/eval archive pending; no downstream job.20 terminated allocations match Slurm/ledgers; shared5.0341666667 H200-hours reconciled,active1.6483333333 provisional,24-hour reservation retained,upper29.0341666667. Identity/config/source/repaired manifest unchanged; no polling/source changes/repeated tests/cleanup/training. Savedto_human/heartbeat-20261008-0921.md.

- 2026-10-08T09:41:36.618985+08:00 bounded visible heartbeat: split514101 RUNNING on1H200/8CPUs,7135s observed; source lineage805/6081 shards, up from524, covers42214136 rows. Native rowcount remains104020922; cursor/order/contiguity match canonical manifest, stdout fresh. Complete native shuffle/split/eval archive pending; no downstream job.20 terminated allocations match Slurm/ledgers; shared5.0341666667 H200-hours reconciled,active1.9819444444 provisional,24-hour reservation retained,upper29.0341666667. Identity/config/source/repaired manifest unchanged; no polling/source changes/repeated tests/cleanup/training. Savedto_human/heartbeat-20261008-0941.md.

- 2026-10-08T04:44:07.077519+00:00 user-authorized data takeover: queue empty, old heartbeat has no cron entry and old thread is not loaded. Split514101 COMPLETED exit0; accounted16775s/1H200=4.6597222222hours. Canonical6081 ranges cover104020922rows; eval archive196files/50000rows complete; identity/parent manifest/small eval indices/archive integrity checked. Shared reconciled9.6938888889H200-hours. Proceed clean005; restore visible-thread heartbeat scoped to DATA through verify, no training/cleanup.

- 2026-10-08T04:45:24.710968+00:00 takeover submitted clean514774 once, observed RUNNING on25a-hgpn001/1H200/8CPUs/200G, fingerprint unchanged, startup stderr empty. Restored20min heartbeat to current thread; old thread remains notLoaded. Shared reconciled9.6938888889, with24h clean reservation33.6938888889. Data through verify only; reportto_human/heartbeat-20261008-1245.md.

- 2026-10-08T04:46:17.596297+00:00 user requested pause. Disabled heartbeat cron and created stop flag. scancel514774 accepted; Slurm CANCELLED, allocation initially COMPLETING,89s observed. No clean directory yet; complete split preserved. No automatic stage/resubmission allowed until explicit resume.

- 2026-10-08T04:48:35.452682+00:00 explicit user resume: queue empty; cancelled clean514774 final89s/1H200=0.0247222222hours accounted; no clean directory had been produced. Shared reconciled9.7186111111H200-hours.005 split/config/source identity and eval archive50000rows unchanged. Resume clean005 and restore visible-thread20min continuity, data-only through verify.

- 2026-10-08T04:49:33.864750+00:00 resume submitted clean514780 once, single snapshot PENDING;1GPU/8CPUs/200G/24h requested, runtime H200 verification pending. Heartbeat restored20min/current thread. Shared actual9.7186111111, with reservation33.7186111111H200-hours. Reportto_human/heartbeat-20261008-1249.md; data-only through verify.

- 2026-10-08T04:51:21.411478+00:00 bounded heartbeat: clean514780 RUNNING on25a-hgpn003/1H200/8CPUs;59s observed, startup fingerprint/spawn match005, stderr0bytes, clean directory/manifest absent. No downstream submission.22 terminated jobs agree with Slurm/ledgers; shared reconciled9.7186111111, active0.0163888889provisional,24h reservation/shared upper33.7186111111. Correct-thread cron delivery and execution verified; no polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1251.md.

- 2026-10-08T05:41:42.453075+00:00 bounded heartbeat: clean514780 RUNNING,3067s observed, committed scan16141312/104020922 (~15.5%), fresh378576896-byte database, complete manifest absent. Periodic faulthandler Timeout stacks at database transaction are diagnostic; source/config/split unchanged.22 accounted allocations match Slurm/ledgers; shared9.7186111111 reconciled, active0.8519444444provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1341.md.

- 2026-10-08T05:42:48.635676+00:00 bounded heartbeat: clean514780 RUNNING; scan17491968/104020922(16.82%) advanced from16141312; fresh database, no manifest.22 reconciled allocations match Slurm/ledgers; source/config/split unchanged. Shared9.7186111111reconciled, active0.8830555556provisional, reservation24/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1342.md.

- 2026-10-08T05:43:52.460564+00:00 bounded heartbeat: clean514780 RUNNING; committed scan18150912/104020922(17.45%) from17491968, database fresh, complete manifest absent.22 accounted jobs match Slurm/ledgers; identity unchanged. Shared9.7186111111reconciled, active0.9011111111provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1343.md.

- 2026-10-08T05:44:54.478792+00:00 bounded heartbeat: clean514780 RUNNING; scan18831872/104020922(18.10%) from18150912; fresh database/live journal, no manifest.22 accounted jobs match Slurm/ledgers, identity unchanged. Shared9.7186111111reconciled, active0.9188888889provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1344.md.

- 2026-10-08T06:21:56.173085+00:00 bounded heartbeat: clean514780 RUNNING,5477s; database575021056bytes vs previous407818240, fresh mtime/journal, manifest absent. Indexed cursor query locked after1sec, not retried; previous18831872/18.10% retained as stale observation.22 accounted allocations match Slurm/ledgers, identity unchanged. Shared9.7186111111reconciled, active1.5213888889provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1421.md.

- 2026-10-08T06:23:32.460755+00:00 bounded heartbeat: clean514780 RUNNING,5602s; database582205440bytes vs575021056, fresh journal; manifest absent. Single readonly indexed query returned attempt to write a readonly database; not retried or repaired, exact cursor unavailable; previous18.10% retained as stale13:44 observation.22 accounted jobs match Slurm/ledgers; identity unchanged. Shared9.7186111111reconciled, active1.5561111111provisional, reservation24/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1423.md.

- 2026-10-08T06:25:12.480851+00:00 bounded heartbeat: clean514780 RUNNING,5697s; database587530240bytes vs582205440, fresh journal, no manifest. One readonly indexed query unavailable; no retry/write/recovery. Future ticks skip cross-node SQLite lookup while active journal exists; last18.10% explicitly stale.22 accounted allocations match Slurm/ledgers; identity unchanged. Shared9.7186111111reconciled, active1.5825000000provisional, reservation24/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1425.md.

- 2026-10-08T06:26:32.956041+00:00 bounded heartbeat: clean514780 RUNNING,5797s; database596209664bytes from587530240, fresh journal; cursor query skipped, manifest absent.22 accounted jobs match Slurm/ledgers; identity unchanged. Shared9.7186111111reconciled, active1.6102777778provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1426.md.

- 2026-10-08T06:30:23.925501+00:00 bounded heartbeat: clean514780 RUNNING,6024s; database625307648bytes from596209664, fresh journal; cursor query skipped, no manifest.22 accounted jobs match Slurm/ledgers; source/config/split unchanged. Shared9.7186111111reconciled, active1.6733333333provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1430.md.

- 2026-10-08T07:21:30.041145+00:00 bounded heartbeat: clean514780 RUNNING,9096s; database973721600bytes from625307648, fresh journal; cursor skipped, manifest absent.22 accounted jobs match Slurm/ledgers; source/config/split unchanged. Shared9.7186111111reconciled, active2.5266666667provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Corrected stale next_expected_local_tick to next nominal20min boundary (not a delivery guarantee). Reportto_human/heartbeat-20261008-1521.md.

- 2026-10-08T07:23:03.225299+00:00 bounded heartbeat: clean514780 RUNNING,9195s; database988164096bytes from973721600, fresh journal; cursor skipped, no manifest.22 accounted jobs match Slurm/ledgers; source/config/split unchanged. Shared9.7186111111reconciled, active2.5541666667provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1523.md.

- 2026-10-08T07:24:29.069085+00:00 bounded heartbeat: clean514780 RUNNING,9285s; database1001144320bytes from988164096, fresh journal; cursor skipped, no manifest.22 accounted jobs match Slurm/ledgers; source/config/split unchanged. Shared9.7186111111reconciled, active2.5791666667provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1524.md.

- 2026-10-08T07:26:10.197295+00:00 bounded heartbeat: clean514780 RUNNING,9383s; database1015570432bytes from1001144320, fresh journal; cursor skipped, no manifest.22 accounted jobs match Slurm/ledgers; source/config/split unchanged. Shared9.7186111111reconciled, active2.6063888889provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1526.md.

- 2026-10-08T07:30:17.956635+00:00 bounded heartbeat: initial Slurm snapshot unavailable; one fallback eventually returned clean514780 RUNNING,9576s,1H200/8CPUs. Database1037889536bytes from1015570432, fresh journal; no cursor/manifest. Historical22-job ledger unchanged, full sacct refresh not completed this cycle. Source/config/split unchanged. Shared9.7186111111reconciled, active2.6600000000provisional,24h reservation/shared upper33.7186111111. No downstream/duplicate submissions, training or cleanup. Reportto_human/heartbeat-20261008-1530.md.

- 2026-10-08T07:32:08.102913+00:00 bounded heartbeat: clean514780 RUNNING,9737s; database1064046592bytes from1037889536, fresh journal; cursor skipped, no manifest. Short-timeout Slurm snapshot returned normally;22 accounted allocations match ledgers. Source/config/split unchanged. Shared9.7186111111reconciled, active2.7047222222provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1532.md.

- 2026-10-08T07:33:39.191001+00:00 bounded heartbeat: clean514780 RUNNING,9828s; database1074606080bytes from1064046592, fresh journal; cursor skipped, no manifest.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active2.7300000000provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1533.md.

- 2026-10-08T08:21:35.324772+00:00 bounded heartbeat: clean514780 RUNNING,12701s; database1370673152bytes from1074606080, fresh journal; cursor skipped, no manifest.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.5280555556provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1621.md.

- 2026-10-08T08:23:10.198061+00:00 bounded heartbeat: clean514780 RUNNING,12797s; database1377107968bytes from1370673152, fresh journal; cursor skipped, no manifest.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.5547222222provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1623.md.

- 2026-10-08T08:24:48.109319+00:00 bounded heartbeat: clean514780 RUNNING,12897s; database1382973440bytes from1377107968, fresh journal; cursor skipped, no manifest.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.5825000000provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1624.md.

- 2026-10-08T08:26:59.887279+00:00 bounded heartbeat: clean514780 RUNNING,12988s; database1390231552bytes from1382973440, freshmtime. Journal reappeared before optional readonly cursor query, skipped without SQL; no manifest.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.6077777778provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1626.md.

- 2026-10-08T08:29:13.112542+00:00 bounded heartbeat: clean514780 RUNNING,13141s; single readonly indexed query during absent journal gives input scan57708800/104020922(55.48%) from old18831872; database1420492800bytes, no manifest.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.6502777778provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1629.md.

- 2026-10-08T08:30:58.377934+00:00 bounded heartbeat: clean514780 RUNNING,13262s; database1437315072bytes from1420492800, fresh journal; cursor skipped, no manifest. Prior55.48% retains16:28 timestamp.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.6838888889provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1630.md.

- 2026-10-08T08:33:01.152174+00:00 bounded heartbeat: clean514780 RUNNING,13381s; single readonly indexed query during absent journal gives input scan61106688/104020922(58.74%) from57708800; database1459765248bytes, no manifest.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.7169444444provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1633.md.

- 2026-10-08T08:34:54.640118+00:00 bounded heartbeat: clean514780 RUNNING,13498s; database1475809280bytes from1459765248, fresh journal; cursor skipped, no manifest. Prior58.74% retains16:32 timestamp.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.7494444444provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1634.md.

- 2026-10-08T08:37:03.538498+00:00 bounded heartbeat: clean514780 RUNNING,13617s; database1494171648bytes from1475809280, fresh journal; cursor skipped, no manifest. Prior58.74% retains16:32 timestamp; current diagnostic trace at input iterator.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.7825000000provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1637.md.

- 2026-10-08T08:39:01.037493+00:00 bounded heartbeat: clean514780 RUNNING,13743s; database1513033728bytes from1494171648, fresh journal; cursor skipped, no manifest. Prior58.74% retains16:32 timestamp.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.8175000000provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1639.md.

- 2026-10-08T08:41:49.118437+00:00 bounded heartbeat: clean514780 RUNNING,13910s; database1535148032bytes from1513033728, fresh journal; cursor skipped, no manifest. Prior58.74% retains16:32 timestamp.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active3.8638888889provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1641.md.

- 2026-10-08T09:23:20.896134+00:00 bounded heartbeat: clean514780 RUNNING,16335s; database1874927616bytes from1535148032, fresh journal; cursor skipped, no manifest. Prior58.74% retains16:32 timestamp.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active4.5375000000provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1723.md.

- 2026-10-08T09:42:07.727954+00:00 bounded heartbeat: clean514780 RUNNING,17522s; database1951641600bytes from1874927616,mtime2026-10-08T09:37:58.287595+00:00, fresh journal; cursor skipped, no manifest. Prior58.74% retains16:32 timestamp.22 accounted allocations match one Slurm snapshot; source/config/split unchanged. Shared9.7186111111reconciled, active4.8672222222provisional,24h reservation/shared upper33.7186111111. No new jobs/polling/repeated tests/cleanup/training. Reportto_human/heartbeat-20261008-1742.md.

2026-10-08T17:13:07.754131+00:00 — User authorized concurrent filtering continuation. Updated stale514780 state from live Slurm/accounting and completed clean manifest; submitted durable score005 intent/job517587 with120min limit,1H200 and8CPUs. Next: reconcile terminal scoring; exit75 continues original score, success gates select/pack/verify. No cleanup/upload.

2026-10-08T17:20:42.730108+00:00 — User requested stop/restart filtering with16-hour sbatch limit. SentUSR1 then cancelled score517587 while still loading canonical Arrow cache (no scores DB); terminal351sec=0.0975H200-hours reconciled. Submitted continuation517611, same005 fingerprint/source/config/data environment,960minutes,1H200/8CPUs, durable afterany517587 dependency prevents overlap. Filtering helper defaults and score/prepare batch headers now16hours; optimizer-sweep caps/controller unchanged.12 metadata submission tests and shell syntax passed.

2026-10-08T18:08:27.581958+00:00 — Progress inspection: score517611 FAILED at01:45:21 Asia/Taipei,1507sec,exit139/SIGSEGV,actual0.418611H200-hours reconciled. Scores SQLite51802112bytes, no journal; one bounded readonly primary-key query gives committed last_row5574559 (source cursor, not scored-doc count). Periodic traces show Dataset random selection/Arrow table and numpy cumsum; root cause unconfirmed. No blind retry or artifact changes. Optimizer native calibrations all passed; initial15/18 completed,16th active; best completed Muon0.04/WSD subsetPPL621.48 vsAdamW0.0012/WSD1131.06 at matched13107200tokens, early pilot evidence only.

2026-10-08T18:26:23.473469+00:00 — Filtering repair c43345b:16-hour score517834 RUNNING. Canonical index3004shards/104020922rows complete. Runtime checks passed for12 independent originals across6sources, native cross-shard order/duplicates/text projection, exact1/7-worker official classifier outputs and16 saved scores. Same original005 core/config/environment fingerprint; overlay hash receipts agree. Resumed after committed last_row5574559; new scores141568, total2438912/28586141. Initial score throughput2334documents/sec; ETA excludes ranking and later stages. No split/clean/raw-sweep changes or cleanup.

2026-10-09T03:22:15.143118+00:00 — Score517834 complete at03:45:55 Asia/Taipei,28586141documents,5300sec=1.472222H200-hours reconciled; scored ranking index/manifest identities match immutable005 and validated bounded reader. Continued authorized filtering via select/tokenization job518703,16h,1H200/8CPU, unchanged6B whole-document budget and all50000 reserved originals. Raw optimizer campaign complete:18initial+18short allcomplete,31.374444H200-hours inclcalibration, no medium(clear ranking). Best short Muon0.02/WSD subsetPPL55.4453 vsAdamW0.0003/WSD140.8943 at104857600trainingtokens. Formal final training/full reserved-set result remain unrun.

2026-10-09T07:30:39.526230+00:00 — Implemented seekable record-batch filtering recovery. Gracefully stopped/reconciled518703:14333s,3.9813888889H200-hours,81733docs/126905498tokens/117shards preserved.34lightweight tests and shell syntax passed; immutable core/config/environment hashes unchanged. Installed metadata-only bounded filtering controller; continuation519327 running16h/1H200/8CPU/200GB, preflight and2x/160GiB gate before production writes. Original validation/manifests kept; pack/full verify will follow via durable submission and shared-budget checks.

2026-10-09T07:33:16.163049+00:00 — Recovery519327 passed all runtime gates; production selection resumed.1024matched candidates/1765290tokens: batch43.1747s (including35.9645s reader/lazy index), legacy96.6425s,2.2384x complete-pipeline speedup, byte-identical outputs. Conservative process+worker RSS upper28.1738GB (distinct from Slurm cgroup memory/pagecache statistic). Real IPC/corruption/ordered duplicates/interrupted select/packing checks passed;16saved documents regenerated exactly, prefix hashes verified.15-minute production measurement pending.

2026-10-09T07:48:10.947330+00:00 — Recovery519327 sustained15-minute observation completed: 904.57s, 164649823new committed tokens, 182019.55tokens/s, 20.53x versus old observed overall rate. Remaining training-tokenization ETA 8.71h excludes evaluation tokenization, pack, verify, queue and future resume overhead. Recent Slurm CPU difference67CPU-sec/61wall-sec=1.098cores/8=13.73%; reader remains coordinator-limited. Prefix unchanged, controller active.

2026-10-09T08:14:47.559970+00:00 — User-requested worker-local read+tokenize+prefetch implemented. Seven pinned workers plus coordinator;32docs/batch,<=2unconsumed credits/worker,oneprefetch slot,zero retained raw shard cache. Original stage_select/code/config/split/439-shard prefix unchanged.519327 graceful stop1849s=.513611H200-hours reconciled;519502 resumed16h/1H200/8CPU/200GB after38stdlib and real worker/order/resume/budget/mask/overlap checks.4096matched docs6094282tokens:20.5996s serial-reader vs7.1494s worker-prefetch,2.8813x. Production metrics separate; controller remains sole filtering submission owner.

2026-10-09T08:16:57.299056+00:00 —519502 production sample:106wall-sec/524CPU-sec=4.9434of8CPUs,61.79% utilization. Seven workers observed; every per-worker maximum2batches. First219.64production seconds averaged981447tokens/s;~1.5h remaining training tokenization excludes downstream. Versioned progress continues; no extra GPU or CPU allocation.

### 2026-10-09 train-tokenization CPU override

User authorized16CPUs fortraining tokenization only. Cluster enforces12CPUs/H200
and this account lacks CPU-only partition access. Paused519502 with exit75,
reconciled0.4791666667H200-hours, and preserved2.053802665Btokens/1913shards.
Submitted corrected12CPU/oneH200 continuation519577 (11workers+coordinator),
with automatic8CPU evaluation continuation; immutable config/core unchanged.
37lightweight tests and shell/Python syntax checks passed. Real worker/output
equivalence and7/11-worker scaling run inside the new Slurm job. Retain the
failed16CPU intent reservation conservatively; see cpu-scaling-validation.json.

### 2026-10-09 evaluation tokenization CPU override

User corrected the evaluation allocation to12CPUs too. Updated controller and
execution overlay to11evaluation workers plus coordinator without interrupting
active519577. Immutable preparation identity/split/order unchanged; the upcoming
evaluation continuation retains12CPUs and will validate actual workers on Slurm.
40lightweight checks passed; runtime evaluation gate remains upcoming.

### 2026-10-09 parallel packing and verification

Implemented explicit overlays for both stages, with seven pinned workers plus
coordinator on eight CPUs. Paused/reconciled 519687 and resumed as 519745, retaining
1.962541056B packed tokens and all 1,832 prefix shards. 62 lightweight checks and
all Slurm equivalence/resume/corruption gates passed. Original and parallel full
fixture verification reports match; matched real-data packing is byte-identical
and 4.7065x faster. Initial production throughput is about 7.6M tokens/s. Full
real-data verification is configured after packing and has not started yet.
See experiments/filtering-recovery/packing-parallel-validation.json for evidence.

### 2026-10-09 explicitly authorized one-hour best-checkpoint continuation

User authorized exactly one best-checkpoint resume on unfiltered data while
verification runs, and prohibited checkpoint initialization for future training
requests. Recorded in both repository guidance files and immutable one-time
authorization. Training519858 uses eight H200/eight CPUs and the same logical
run baseline-muon-wsd-raw-20261009, checkpoint008500-b746a2a7d6b4, optimizer,
WSD8500 horizon (terminal LR floor), RNG, data cursor136000 and W&B44c3ab1a26b94760afba44992342140d.
Preserved parent1,114,112,000 charged/confirmed tokens and13.62 H200-hours;
the continuation raises the bounded logical compute ceiling to23 H200-hours
without resetting the6B-token/64-H200-hour hard caps or committing another full run.
The training phase started19:19:42 Taipei and is bounded by a3600-second timer;
70-minute allocation includes initialization and finalization. Training already
progressed beyond step8726 at about209k input tokens/s, concurrently with519786.
Original resolved configuration/execution/checkpoint/result metadata are archived
under runs/experiments/<run>/continuations/best-raw-one-hour-20261009.

Raw buffer supplementation retains the original preparation, published prefix,
8192-token packing, carry and whole-document flush boundaries. Seven workers
read/tokenize32-document batches with at most two batches each; coordinator
appends unique shards until at least3B raw input tokens are available. Initial
producer519857 failed before publication because OrderedBatches lacks a context
manager; cancelled/reconciled121 seconds (.0336111111 H200-hours). Explicit v2
lifecycle overlay closes workers on early return/errors, preserving original
sealed code; producer519863 resumes the unchanged prefix. Producer allocations
are separately labelled preprocessing in the shared experiment ledger; costs
are counted once. Only metadata accounting cron is installed; no automatic
training resubmission, cleanup or upload. Runtime raw-tokenizer equality gate
precedes any append; final one-hour training result remains pending.

### 2026-10-09 best checkpoint published to Hugging Face

User explicitly authorized publication to asciibase64/Llama-3.2-1B-dolma3_mix.
Published best step13717, 1,797,914,624 cumulative non-padding training tokens,
at revision 0b54c9d826c765cb0e3d75494b00e898d04f882e. All146 exported tensors are exactly
equal to the original best checkpoint. Standard Transformers BF16/SDPA loading,
tied output weights, tokenizer parity and finite8192-token inference passed
in Slurm520037. All15 uploaded files have matching remote Git/LFS checksums;
unauthenticated remote config/tokenizer loading passed. Model card explicitly
labels PPL20.939945 as the fixed1000-document subset, with current full50000
PPL unmeasured. Included license/notice and full training provenance.
First allocation520031 could not initialize CUDA on25a-hgpn001 and stopped
before upload; retained source/authorization/accounting and excluded the node.
Both validation/publication allocations reconciled: 0.09194444444444444 H200-hours.
Parent training accounting remains21.802222222 H200-hours; no new training.

### 2026-10-09 filtered 2B-token training implementation

Merged sweep/muon into main (merge18c4b56), preserving completed filtering
and the latest one-time checkpoint authorization. The new request starts
muon-wsd-filtered-2b-20261009 from random weights, seed42, on verified
dolma-dclm10-005. Best step13717 optimizer/model/batch/evaluation recipe
is preserved; WSD horizon extends to15259 updates with170 warmup and20%
decay. Target2B confirmed input tokens; final whole-batch cap2,000,027,648.
8 H200/eight CPU threads, allocation limit28800 seconds/64 H200-hours,
1800-second full-evaluation reserve and300-second stopping margin.
74 lightweight tests passed, including token accounting, scheduler endpoints,
old allocation caps, no-resume submission, and uncertain submission retention.
Fresh-run source/config/preparation/environment seals precede Slurm training.
Status/accounting owner never resubmits; prematurely stopped runs stay incomplete.
Full50000 evaluation and reload verification occur in the same allocation.

### 2026-10-09 pre-initialization filtered launcher recovery

Main7094ce2 pushed to GitHub. First allocation520098 failed at CLI argument
validation before run directory, model, tokenizer, journal or W&B creation: the
existing validate-export flag was limited to validation configurations. All
eight rank exits2, no training tokens processed. Reconciled12 allocation
seconds/0.026666667 H200-hours; retained source/launch/config evidence.
Allow post-training export verification for formal runs while keeping deliberate
validation pauses restricted. Explicit retry-uninitialized checks terminal
failed accounting and absence of any training initialization; retains the same
logical run, configuration and measured costs. No checkpoint load, no new formal
run, no automatic retry. Remaining allocation rounded down to479 minutes so
cumulative reservation plus measured cost remains below64 H200-hours.
Regression tests cover full-run CLI dispatch, denied checkpoint resume, rejected
retry after initialization, preserved prior cost and reduced wall-time.

## 2026-10-14 — horizon-wsd campaign protocol

User requested sequential LR/clipping/decay tuning with independent sbatches, a shared stable checkpoint, and two final two-sbatch comparisons; clarified0.7B sweep and2B final. Scoped preparation is immutable raw baseline, BE60P20 WSD hold/zero, matched token endpoints. Prepared18 allocations,205.333333H200-hour upper bound, and preserved197.968611 historical training hours plus4 outstanding reserved hours. Six pure contracts, Python/sbatch syntax and sbatch --test-only passed. Protocol/source/config seals created before compute. Persistent20-minute schedule installation was auto-review rejected; no cron installed. See experiments/horizon-wsd-20261014/protocol.md.

2026-10-14 — User explicitly approved “批准，照此協定自動完成” after reviewing18 sbatches/205.333333H200-hour cap, persistent20-minute controller and two final formal runs5/6. Previous automatic-review persistence objection resolved by this explicit reply. Validation537355 is the only allocated new job at authorization time. Approval is bounded to the committed protocol; no retries/cleanup/upload.

2026-10-14 — Validation537355 TIMEOUT on25a-hgpn123 at00:42:53Asia/Taipei;625s/1.388888889H200-hours including25s termination overhead. Reconciled actual charge; its1.333333 logical validation cap was exceeded and remains flagged. No real-model training tokens processed. Native stacks show NCCL communicator bootstrap socketPollConnect/sendToRoot; ss confirms all8 self-connect sockets SYN-SENT. Original source/protocol/config/accounting preserved. User explicitly approved “批准單次修復後繼續整輪”: one additional10min validation withloopback bootstrap; LR0.5 allocation79min offsets new validation/observed overrun,0.7B target unchanged. Auto-review rejected a separate Slurm test-only invocation; it was not executed and is omitted from the recovery workflow.

2026-10-14 — Recovery537466 COMPLETED158sec/0.351111111H200-hours;8-rank clipping/off/globalnorm/checkpointmodeloptimizer/scheduler/RNGchecks passed. Realrandom-init1B20-updates validation completed2,621,440 charged=confirmed non-padding inputs, no skipped updates, identical reference initialization signature, endpoint evaluation. Three independent0.7BLR jobs537499(.5),537500(1),537501(2) submitted onv2 withloopback bootstrap,79/90/90minute caps. Sole20-minutecron verified. Validation physical total1.74H200-hours includesfailed537355; no history/reset/cleanup/upload.

2026-10-13T18:00:02.785724+00:00 — horizon-wsd-20261014 completed lr_ratio ranking: selected 0.5; endpoint NLL 3.126626289, PPL 22.796939. All compared candidates completed/reconciled at matched tokens. Single-seed exploratory transfer interpretation; no significance claim.

2026-10-13T19:20:02.537724+00:00 — horizon-wsd-20261014 completed clipping ranking: selected 0.5; endpoint NLL 3.118394912, PPL 22.610059. All compared candidates completed/reconciled at matched tokens. Single-seed exploratory transfer interpretation; no significance claim.

2026-10-14T02:25:23.710726+00:00 — User-requested repair-and-continue: controller run IDs corrected with sealed source-v2 unchanged; all five decay recovery branches verified past2136 updates with279969792 inherited inputs. CUDA startup failure539630 retained (17seconds/0.037777778H200-hours), node027 excluded, manual replacement539636 confirmed. Original five failed allocations retained. v3 remains sole20-minute cron owner; original205.333333 campaign cap and formal5/6 retained.

2026-10-14T03:20:02.377492+00:00 — horizon-wsd-20261014 completed decay_tail ranking: selected 0.6; endpoint NLL 3.119660760, PPL 22.638698. All compared candidates completed/reconciled at matched tokens. Single-seed exploratory transfer interpretation; no significance claim.

2026-10-14T04:20:23.499055+00:00 — Corrected host-local cron diagnosis: shared v4 path/hash/host owner on25a-lgn04; guarded legacy v1/v2/v3 entries are inert on other login hosts. Failed539819 retained under formal5; fresh179-minute replacement540265 verified random initialization and new training inputs onnode024. Resplit539820 unchanged. Standard cap31.9H200-hours including failure and tail; campaign cap preserved.21 pure checks passed.

2026-10-14T06:38:49.019730+00:00 — User replaced resplit reheating1.673461643 with four LR scales1.0/0.7/0.5/0.3 relative to first phase peak.540582 cancelled/reconciled and preserved. Jobs540742–540745 all verified beyond12715 from exact checkpoint012715-7b7f9aa120ca, same1666580480 inherited inputs, complete optimizer/RNG/cursor restoration and separately sealed fork overlay. First phase peaks remainAdamW9.500173326e-5/Muon0.006333448884. Shared v5 owner on25a-lgn01; historical caps/costs preserved.
