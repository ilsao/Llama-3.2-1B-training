# User-requested formal13:3B with actual-step warmup and40% phase tails

The user explicitly requested "幫我用 steps 後40% 跑一次3B" after reviewing the
formal12 token-clock audit,and selected "Warmup也按steps：每段170次實際optimizer
updates". Retain the previously selected two phases;both warmup and decay now
use actual optimizer updates. This is one NEW random-weight training run,not a
resume or a relabeling of formal12. Record formal13 and preserve all twelve
prior formal identities and every allocation cost.

Resources match the previous3B run:one8H200/8CPU/200GB allocation,at most6h/48H.
Request359whole minutes/21540s/47.866667H,leaving60s terminal-accounting margin.
For this exact run/config digest only,the user's new-run instruction authorizes
the formal-run-count and120minute exception. No automatic retry,resume/requeue,
checkpoint training load,cleanup,HF upload or additional candidate is authorized.

Unchanged recipe:random residual-half initialization seed42,raw preparation
dolma-raw-baseline-20261008,canonical document split,8192 context,bf16 FA3,
activation checkpointing,native Muon+AdamW,EMA OFF,global clip.5,WD.1 with prior
embedding/tied-head/norm/bias exclusions,AdamW(.9,.95),Muon momentum.95.
Table3B peak LRs:Muon.005593716406714465,AdamW.00016781149220143396. Record
unvalidated extrapolation status. No LR batch scaling or beta2 exponentiation.

Preserve formal12's complete data/batch window trajectory,including smaller
batches at its recorded data/phase/evaluation boundaries,for14312 actual
updates and3000107008 nonpadding input tokens. Nominal global BS16 becomes64
at1499987968 inputs through accumulation2 to8 with8ranks/microbatch1. Maintain
optimizer moments,RNG,data cursor and scheduler state at that point.

Keep the5/6-token phase split at2500067328 inputs. This corresponds to13355
actual updates in phase1 and957 in phase2,not a5/6 split of actual updates.
Second peak remains.1 times first. Each phase warms up for170 ACTUAL updates.
Each phase's last40% actual updates use the prior4:1:1 first-ramp/plateau/final-
ramp shape,plateau LR20% of that phase peak. Zero-based completed-update knots:
phase1(8014,11575,12465,13355),phase2(13929,14184,14248,14312).
Integer rounding:phase1 tail5341/13355 updates;phase2 tail383/957 updates.
First decay starts after8014 actual updates at1050411008 inputs,approximately
60% of phase1 updates and35% of total tokens. It must NOT wait until11444
updates/1.5B inputs. Final LR is zero in updates13355 and14312.

The scheduler stores completed_updates and tokens explicitly and refuses any
step/data cursor mismatch. W&B uses actual optimizer-update global_step and
logs per-group LR,scheduler update index,batch,beta2,nonpadding inputs and
throughput. Save expected per-step LR/token/batch table before submission and
audit all actual logged LRs at completion to expose clock errors directly.

Fixed1000 heldout documents/1479302 prediction tokens at the same token
milestones as formal12,plus at most1431 actual updates between evaluations
and final. Verify final local HF export reload and run full50000 documents/
76930208 prediction tokens inside the same allocation. Compare full/subset PPL
against formal12(full15.2117025675),20%-tail3B(full15.9122976658),and prior SOTA
(full15.2777345960). This run also changes second-phase warmup duration versus
formal12;report that explicitly when interpreting differences.

Use a new immutable source snapshot and new persistent W&B identity. Single
host/hash-pinned controller submits once,retains/retrieves uncertain submission
identity,reconciles actual cost even when disabled,and retires only its own
cron after terminal accounting. Lightweight contract/recovery tests run on the
login node;all tensor/model work runs in the one Slurm allocation. Large files
remain under /work/asciibase64/Llama-3.2-1B-training/.
