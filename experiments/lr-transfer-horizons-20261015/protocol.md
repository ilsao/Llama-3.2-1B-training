# True WSD horizon calibration after Phase A

User continuation: 2026-10-15「繼續」after400M valley and ledger anomaly. The earlier aggregate256H gate remains lifted. This bounded stage commits at most62 H200-hours across eleven pilots, not a formal full run. Each job uses8 H200,8 CPUs,one launcher task,one CPU thread/rank. No checkpoint training load, resume/retry, artifact cleanup, upload or additional formal-run exception.

## Accounting recovery before submission

Restored188 allocation records and170 logical runs from138 per-run accounting snapshots and188 per-allocation startup records, with a single batched sacct query for38 missing measurements. Total actual training-ledger allocation cost614.2808333333338H, including failed allocations and all eight formal run identities. Preprocessing ledger remains separate, unchanged, and has no overlapping allocation IDs. Branch inherited cost remains part of each logical run cap; global actual costs count each allocation once. Restoration candidates, original empty ledger, source hashes and tests are in the prior stage's ledger-recovery directory. Cause of original empty ledger remains unverified. New owner checks minimum historical job/run/cost counts before every batch of submissions; an empty ledger cannot be treated as zero-cost history.

## Frozen calibration grid

Same raw canonical split, residual-half random initialization seed42,8192 context,16 global sequences/update,131072 non-padding inputs/update,8 ranks,Muon/AdamW peak ratio1/.03,clipping0.5,WD0.1,170 warmup updates. Complete original single hold-zero WSD:40% first decay,10% hold at20% peak,10% final decay to zero. No resplit and no EMA in WSD targets. End after complete batch crossing nominal horizon; no checkpoint is loaded for training.

-200M: MuonLR[.008285485413204676,.0117174458421985,.01657097082640935,.023434891684397,.0331419416528187],five independent30-minute allocations. Exact200015872 inputs/1526updates each.
-600M: MuonLR[.00585872292109925,.008285485413204676,.0117174458421985,.01657097082640935,.023434891684397],five independent60-minute allocations. Exact600047616 inputs/4578updates each.
-New highestLR.0331419416528187 first gets a15-minute,120061952-token warmup-constant numerical screen using the existing EMA.6 screen code. Only its own200M target waits for successful complete screen; all other WSD targets are independent and submit in parallel. Failed/partial screen blocks that promotion; no automatic retry.

Existing six400031744-token WSD observations are reused with immutable receipt/data identity; best measuredLR.0117174458421985,PPL25.775291014457313; local fittedLR.01274989575236192. This is not yet a1B optimum.

Held-out NLL uses the same fixed1000 reserved documents/1479302 valid prediction tokens. Evaluate at least ten times and at completion. Every receipt must match configuration, random initialization signature, canonical data/eval identity, exact attempted/processed tokens and successful rank exits. Finite training loss is insufficient for promotion.

## Analysis and stopping

At each calibration horizon, measured best must be interior with worse neighbors. Fit a local quadratic to mean NLL versus logLR using winner plus neighbors. Three points always interpolate and their R2 is not independent evidence. If any optimum is at a boundary or curve invalid, output needs_wider_bracket and do not silently extrapolate or automatically expand.

With valid0.2/0.4/0.6B optima, fit logLR*=gamma+beta*log(T/1B). Report leave-one-horizon-out error on these three horizons and every raw observation. Freeze an artifact with0.8B and1–6B predictions explicitly unvalidated. Three horizons/one seed are calibration evidence only. This controller never submits0.8/1B tests or formal2B+ jobs. Freeze the fit before separate0.8B and1B held-out work; LR error<=15% and NLL regret<=.005 remain the prespecified held-out criteria.

This direct-WSD law is a diagnostic control; it does not validate the paper's stable+EMA transfer. Token-relative EMA and scheduler-shape diagnosis remain a separate planned experiment, and may proceed independently after their own numerical runtime verification. Do not select EMA window using the unseen0.8/1B target results.

Source inherits the already validated tensor/optimizer/data payload unchanged from c98136d/f8b2cca. Only new coordination/configuration/analysis code is introduced. Six pure metadata tests plus shell syntax checks passed; no redundant GPU smoke allocation is needed for an unchanged tensor payload. The existing step32 full-update timing guard stops a run if it cannot complete its exact horizon with evaluation/checkpoint stopping margin.
