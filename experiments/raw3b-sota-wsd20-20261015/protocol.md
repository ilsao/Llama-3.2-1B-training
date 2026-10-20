# User-approved formal11: SOTA raw3B with20% WSD tails

The user approved the complete proposed plan via "Implement the plan." This
explicitly authorizes this run only as formal11, one allocation with8H200/8CPUs,
200GB RAM,6h/48H200-hours including startup, evaluation, checkpoint and export.
This supersedes the three-formal-run and120-minute gates for this exact digest
only. Preserve the ten prior formal runs and all historical/failed costs.

Random residual-half initialization seed42; no pretrained or training checkpoint
load. Original SOTA raw3B peak LRs Muon0.005593716406714465 and
AdamW0.00016781149220143396; unchanged parameter groups, epsilon/momentum/betas,
WD0.1 (no decay on embedding/tied head/norm/bias), global clipping0.5,8192 context,
bf16 FA3,activation checkpointing,EMA OFF,8ranks/microbatch1/accumulation2.
Stop at22889updates/3,000,107,008 nonpadding inputs. No repeated documents.

Keep resplit19074, second phase3815updates with0.1 times peak and170 warmup
updates per phase. Retain optimizer/model/data state across the phase boundary.
Each phase has20% tail, divided4:1:1:13.3333%first decay,3.3333%hold at20% of
that phase's peak,3.3333%final decay to zero. Integer phase boundaries, zero based:
phase1(15259,17802,18438,19074),phase2(22126,22635,22762,22889).
Top-level hold_schedule and both phase_schedule entries carry the same ratios.
No LR recalibration or additional recipe changes.

Use the identical canonical unfiltered prefix from dolma-raw-baseline-20261008,
complete seed42 split and original50000 held-out documents. Existing6B data
preparation does not increase this run's token budget. Pin the consumed prefix
manifest without hashing large corpus files on the login node; native consumers
verify shard checksums inside the allocated compute job.

Fixed first1000 original held-out documents every2288updates and final step;
final full50000-document eval on FINAL22889 weights,76930208 prediction tokens.
Verify local HF-format export reload. Compare full PPL against SOTA15.2777345960;
report measured differences without presuming improvement. New persistent W&B
identity in lab5-training-llama; retain original metric semantics and all logs.

Single host/hash-pinned owner submits once, retains uncertain reservations and
recovers job identity without resubmission. Minute cron reconciles accounting
then removes only its own entry after terminal reconciliation. No automatic
retry/resume/requeue, cleanup or HF upload. Stop margin300s, final eval reserve900s.
All large artifacts under /work/asciibase64/Llama-3.2-1B-training/.
