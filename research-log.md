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
