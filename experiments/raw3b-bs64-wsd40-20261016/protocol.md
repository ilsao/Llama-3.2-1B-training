# User-approved formal12: raw3B, BS16 to64,40% tail per phase

The user selected "每段後40%" and then explicitly approved the complete plan
with "Implement the plan." This authorizes exactly this fresh run,one allocation
with8H200/8CPU,200GB RAM,at most21600 seconds/48H200-hours including startup,
evaluation,checkpointing and export. For this exact run/config digest only,
override the three-formal-run and120-minute gates. Preserve all eleven prior
formal identities and every historical/failed allocation. No automatic retry,
checkpoint resume/requeue,cleanup,HF upload or additional candidate is authorized.

Start from random residual-half weights,seed42,on canonical unfiltered
dolma-raw-baseline-20261008. Keep Llama3.2-1B,8192 context,bf16,FA3,activation
checkpointing,EMA OFF,native Muon+AdamW parameter routing,WD.1 except embeddings,
tied head,norms/biases,global clipping.5,momentum.95,AdamW betas(.9,.95).
Table peak LRs:Muon.005593716406714465,AdamW.00016781149220143396,from the3B row
of lr-transfer-horizons-20261015/calibration-result.json. Record unvalidated
extrapolation status;no LR search or batch-based LR scaling.

Stop at3000107008 nonpadding training inputs,22889 reference units Q=131072.
At11444Q=1499987968 inputs (nearest aligned approximately50% boundary),change
global batch16 to64 through accumulation2 to8;keep8ranks/microbatch1 and all
optimizer moments/RNG/data/scheduler state. First decay starts at that boundary.
Two phases split at19074Q=2500067328 inputs;second peak is.1 times first,with
170Q warmup per phase. Each phase has40% tail,first-ramp:hold:last-ramp4:1:1,
hold LR20% of that phase's peak. Reference-unit knots:
phase1(11444,16530,17802,19074),phase2(21364,22381,22635,22889).
Shorten accumulation at knots,warmup ends,eval milestones and final zero-LR
updates19073/22888. Exactly14312 actual optimizer updates;reference units
are not optimizer steps. Keep the token clock through the phase transition.

Fixed first1000 heldout documents,1479302 prediction tokens,at reference units
floor(k*22889/10),k1..9,and final. Also enforce at most1431 actual updates
between evals. Full final50000 documents/76930208 predictions on FINAL weights,
local HF export reload verification. Use new W&B identity in lab5-training-llama.
Compare both subset and full results against raw3b-sota-wsd20-20261015 and
raw3b-sota-20261015 (full PPL15.9122976658 and15.2777345960 respectively).
Batch and tail change together;do not claim isolated causal attribution.

Single immutable host/hash owner submits once,recovers uncertain identity,
reconciles actual Slurm cost even if disabled,then retires only its own cron.
Use a whole-minute359-minute request (21540s/47.866667H) inside the6h/48H ceiling
to leave60s for terminal overhead. Stop margin300s and final evaluation reserve
900s are inside that allocation. All large files under /work/asciibase64/.

Source is a new isolated snapshot of validated batch-normuon-1b source-r4 plus
the verified full evaluation workflow. Pure checks cover token/LR/geometry and
submission contracts before launching;all model/GPU checks occur in the single
training allocation. Failed or incomplete training stops without another job.
