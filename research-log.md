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
