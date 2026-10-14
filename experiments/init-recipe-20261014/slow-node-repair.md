# User-requested slow-node repair, 2026-10-14

The user explicitly requested「修正」after the campaign stopped at
combination_control_failed. Exclude25a-hgpn083 for every subsequent submission.
Slurm jobs542604 and542982 both ran on this node and stopped with
wall_time_or_signal before388759552 inputs; no nonfinite optimizer stop occurred.

Perform exactly one fresh replacement per failed candidate:
- promote-muon13 → promote-muon13-slow-r1: same resolved configuration,
  random seed42,388759552 new inputs,8 H200/8 CPU,60 minutes. Preserve failed
  allocation6.5377777777777775H and120455168 processed inputs.
- combo-baseline → combo-baseline-slow-r1: same resolved configuration,
  random seed42,388759552 new inputs,8 H200/8 CPU,60 minutes. Preserve failed
  allocation6.533333333333333H and123469824 processed inputs.

No failed checkpoint is loaded for training. Both candidates keep the original
logical16H limit: failed allocation plus replacement reservation stays below16H.
Both count prior processing, including repeats, against6B. Every failed artifact,
W&B identity and shared-ledger cost remains. These are operator-authorized repairs;
the controller gains no automatic retry policy. Final32H remains reserved within
the same campaign256H cap.

After the repaired Muon candidate completes, recompute family winners. Reuse the
completed combo-merged only if its settings still match. If winners change, submit
one new60-minute combo-merged-reranked, keep the original result as a candidate,
and validate against repaired baseline. Continue paired-seed verification and the
already authorized single fresh formal7 final2B. A failed seed control or candidate
stops final submission rather than silently falling back to an unverified result.

The numerical training payload, dataset,8192 context,optimizer geometry, LR peaks,
initializers and evaluation identities remain unchanged. Validation for this
repair is dependency-free controller tests and shell syntax checks; actual runs
are submitted through Slurm. No additional GPU validation allocation, cleanup or
HF upload is required or authorized by this repair.
