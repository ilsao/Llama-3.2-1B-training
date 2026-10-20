# Paired lossless quality and acceleration validation, 2026-10-20

The user explicitly approved trying the next optimizations and reiterated
"一樣要求 Lossless。即最終 ppl 不降". Interpret this as no deterioration in
model quality (PPL must not increase), retaining the prior allowance for BF16
operation-order rounding. Do not call a short-prefix check proof of final6B PPL.

One new validation allocation:8H200/8CPU,30min/4H200-hours maximum, no automatic
retry/resume/requeue. All18 formal identities and every historical cost remain.
Prior reconciled study cost173.8897222222H; with this reservation177.8897222222H
is below191.419H. Record category validation. No full6B allocation, checkpoint
training load, HF upload, environment modification or old artifact cleanup.

Six predeclared arms, one seed42, within this allocation:
R0 original checkpoint/eager loss/micro1/replicated optimizers;
R1 previous best (no checkpoint + full-vocab fused CE/KL);
R2 R1 + micro2 (globalBS16 accum1 and globalBS64 accum4);
R3 R1 + compile RMSNorm/MLP/RoPE on teacher and student;
R4 R1 + complete-matrix NorMuon owner updates and FP32 all-gather;
R5 all three new optimizations combined.

Same original random residual-half seed42 weights are captured in GPU memory
before any update and copied back between arms. This is a random initialization
snapshot, never a trained checkpoint. Preserve the official pretrained3B teacher
revision/manifests, full128256-vocabulary .75CE+.25KL(T||S), T1, context8192,
canonical raw preparation/document order, intradoc masks, tied student head,
NorMuon NS5/EMA/WD, AdamW LR2x and original token-clock schedule prefix.

Teacher and student share the same8GPUs; no hidden disk spool. Each forward
processes the canonical inputs and corresponding teacher once. Journal teacher
and student inputs separately before execution, including all repeated arms,
warmup, numerical checks and speed probes. Each role cap180M inputs; if all six
arms run, planned142,344,192 inputs per role. Synthetic matrix optimizer checks
have separate dimensions/update receipts and are not language-model inputs.

Runtime gates: teacher attention/document isolation;12 nonzero-LR synthetic
NorMuon steps comparing whole-matrix owner weights/state against the replicated
implementation, requiring bitwise equality; then real8192/context128256-vocabulary
globalBS16 full-model gradients for every arm versusR0 (relative L2<.01,
absolute normalized loss difference<.001). Numerical failures are recorded and
exclude that arm; runtime/compile/OOM failures stop this allocation, no retry.

Quality phase:128 actual nonzero-LR updates per eligible arm, exact same first
2048 canonical sequences/16,777,216 inputs; globalBS16 throughout, original
schedule prefix (no compressed cooldown). The optimizer geometry uses all model
gradients for clipping and zeroing even when matrix updates are owner-partitioned.
Owner assignment preserves complete matrix shapes and updates each matrix once;
momentum/variance states remain on its owner. AdamW remains replicated.

Evaluate fixed canonical first128 held-out docs every12 updates, and original
fixed1k at update128. Use identical eager evaluation for all arms even when
training was compiled. Save per-document NLL/tokens and W&B links in project
lab5-training-llama; log all required metrics with original semantics. Compare
endpoint PPL toR0 strictly: any increase fails this short-prefix promotion screen.
Also report comparison toR1. This is NOT a statistical guarantee or a final6B
noninferiority result. Never promote on training loss or speed alone.

Throughput: use nonzero-LR BS16 steps4..128, excluding evaluation, metadata and
data loading. After quality evaluation only, use13 distinct canonical BS64
windows (three warmup+ten timed) with LR and decay LR zero. Optimizer operations
and state updates still execute; no resulting weights are exported or reused.
Report max-rank synchronized time, rank0 optimizer events, peak memory and
token-weighted harmonic BS16/BS64 rate. Compilation time is charged and reported
separately. Conservative full6B projection adds10% runtime and1800s overhead;
validation costs are separately charged and must also be deducted if64H includes
development. Do not claim full-run feasibility from a short prefix alone.

Final-PPL guarantee remains outstanding until a matched full-horizon comparison
or a demonstrably identical execution trajectory is established. New long runs
need their own concrete configuration/resource authorization; this approval is
used for the requested bounded tests only.
