# Corrected correctness validation; explicit approval required before submission

Original source,commit0767159,configs,controller state,logs,checkpoints and all
costs remain immutable. Setup554390 completed4seconds/.001111H. Validation
554398 failed after93seconds/.206667H on eightH200; actual token journal
is retained. No research endpoint or checkpoint branch was launched.

Observed failure: recomputing a full BF16 forward/backward after checkpoint
reload produced a next-update weight difference up to4.699453711509705e-6,
exceeding atol1e-7. This alone does not establish whether checkpoint restore
or forward/backward recomputation caused the discrepancy. The correction does
not relax checkpoint tolerance: verify full weights and every optimizer state
tensor EXACTLY after reload,then replay the exact same captured clipped
gradients and require bit-exact next weights and optimizer state. Per-rank RNG
and token-weighted DDP accumulation remain separately tested. Replay without
forward/backward processes no additional input tokens; all actual full-model
passes,including repeated inputs,remain charged by the durable token journal.

Two additional integration defects were found by inspecting the installed
muonium.2.0 source: it drops caller name/weight_decay metadata when rebuilding
groups and stores LR as a scalar Tensor. Restore our metadata after construction,
preserve tensor LR storage in TokenScheduler,keep exact configured peaks,and
convert logged LRs to JSON-safe numbers. Explicit foreach=True makes dispatch
match between CPU-first training construction and GPU validation. Algorithm,
LR,WD and parameter routing remain as approved. Do not change shared torch/FA3.

Source-r1 and its source seal/owner/state are separate; original failed source
is never edited. Same nine endpoint configs and scopes. Exactly ONE corrected
fresh validation allocation is proposed:8H200/8CPU,13minutes/1.733333H maximum,
new identity validation-r1. Prior93s+780s=873s/1.94H stays within original
900s/2H combined validation allowance. All nine training allocations retain
their original120-minute caps; total campaign ceiling stays146.5H.

Do not install the replacement cron or submit corrected validation until the
user explicitly approves validation-r1-authorization.json. No automatic retry
of a failed run is authorized by the original plan. Once the single corrected
validation passes,the already approved nine-endpoint controller proceeds as
before. Any further failure stops;no retry,resume,cleanup or HF upload.
