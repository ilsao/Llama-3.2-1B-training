# User-approved 1B batch / NorMuon / LR / AdamW beta2 study

Authorization: user selected20% tails+5/6 resplit,first decay boundary,
unchanged LR on BS16-to64,Muon.01/AdamW.0003 challenger,and table LR
for NorMuon; approved the complete plan with "Implement the plan."

Nine matched endpoints: control; three batch-switch tails with fixed beta2;
NorMuon+AdamW+Polar Express; LR.01/.0003; three paired beta2-scaled tails.
Each endpoint has1,000,079,360 non-padding inputs,including inherited prefix.
All are bounded research pilots,not extra2B+ formal runs. Each training job
is8H200/8CPU,one launcher,120minutes/16H at most; branch logical cap32H
includes actual parent prefix elapsed through snapshot publication.
Setup1H200/8CPU30minutes/.5H; validation8H200/8CPU15minutes/2H.
Total reservation ceiling146.5H. Physical training excluding verification
is5,267,259,392 inputs. Historical ledgers and every failed cost are retained.

Raw dolma-raw-baseline-20261008 prefix,original canonical seed42 split,
reserved50000 documents and fixed first1000-document evaluation remain
identical. Random residual-half initialization seed42,context8192,bf16 FA3,
clipping.5,optimizer groups/WD=.1 unchanged. No decay on tied embedding/head,
norms or biases. BS16 uses8ranks,microbatch1,accumulation2; BS64 accumulation8.

Latest actual-WSD calibration predicts Muon.008614891542046706 and
AdamW.00025844674626140116 at1B; still unvalidated extrapolation and
originally fitted to single60%-tail WSD,not this20%-tail resplit recipe.
Discard the failed EMA-proxy table as an LR source. Second phase peak is.1
times the first. Both tails have first/hold/final ratios4:1:1,20%-peak hold.
Phase1 ends at833355776 inputs (6358BS16 units),phase2 at1000079360 (7630).
Zero-based scheduler knots: phase1(5086,5934,6146,6358),phase2(7376,7546,7588,7630).
Each warmup spans170BS16 units/22282240 inputs. The scheduler clock is
processed nonpadding tokens,while W&B train/global_step is real successful
optimizer updates and remains inherited across forks.

Control publishes complete immutable checkpoints after units3815/5086/5341,
inputs500039680/666632192/700055552. Switch applies to the NEXT batch.
All ranks save/load full model,two optimizers,schedulers,EMA-empty state,RNG,
data cursor,token high-water journal and lineage. SHA256 verified in Slurm.
Published snapshots may be consumed while parent continues; never read latest
as a branch pointer. New W&B ID per branch,inherit actual prefix accounting,
physical prefix charged once. No reset of momentum,moments,step,RNG or cursor.

At exact token knots/endpoints shorten accumulation to a wholeBS16 multiple.
Scaled arm uses beta2=.95**(actual_global_sequences/16),normally.81450625,
with exact historical-product bias correction. Fixed arm remains.95.
Both arms retain AdamW beta1=.9 and Muon momentum=.95; LR has no BS scaling.
No NorMuon/beta2 interaction sweep is implied.

NorMuon uses pinned muonium.2.0 wheel SHA256,isolated --no-deps target in/work,
existing torch2.9.1/FA3 runtime; incompatibility blocks NorMuon without upgrading
the shared environment. algorithm=normuon,muon_beta2=.95,PE5steps,cautious WD
disabled; auxiliary groups remain existing AdamW. Record package-specific
matrix shape scaling/decay semantics and treat as a combined optimizer change.
Reference:https://github.com/yukicoder0509/training-llama-3-2

Evaluate every<=10% of each branch's planned remaining updates,at common
763BS16-unit intervals,and at completion. Compare final mean held-out NLL,
exp(totalNLL/valid_predictions),not best checkpoint/training loss. Fixed1000
docs/1479302 predictions; historical3B full PPL is context,not a matched1B score.
Write final HF-compatible local exports with reload equality checks,without upload.
Single seed ranking is preliminary; no significance claim.

One host/hash-pinned minute controller owns submissions/reservations/recovery
and reconciliation. Only immutable complete snapshots release dependent tails.
Uncertain submissions stay reserved and are recovered,never retried. Failed
candidate blocks its dependent work and is reported; independent authorized
lanes continue. Disabling stops submissions but retains accounting. No automatic
resume/requeue,artifact cleanup,HF upload or changes to unrelated jobs/cron.
Source is isolated from existing dirty main and running jobs.
