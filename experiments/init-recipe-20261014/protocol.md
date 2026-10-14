# init-recipe-20261014: approved parallel campaign

User approved the plan and explicitly requested implementation. Incremental hard
campaign budget256 H200-hours, including validation, failures and final evaluation.
All historical ledger costs remain. At most8 H200/8 CPU per job; no artificial
concurrent-job limit. All heavy operations run in Slurm. No automatic retries,
cleanup, HF uploads or additional final runs.

One named fresh final `init-recipe-20261014-final-2b` is formal7 and receives the
scoped fourth-hour/32H exception, without relaxing any other formal-run gates.
Architecture/tokenizer,8192 context, canonical seed42 split, fixed1000-document
subset and complete reserved50000-document final evaluation remain fixed.

The21 initial independent allocations are six initializers, two AdamW LR factors,
two Muon LR factors, two clipping policies, two Muon WD values, two batch sizes,
one baseline with three EMA shadows, one filtered-data candidate and three
frozen-checkpoint lower-LR tails. Shared baseline observations are raw-weight NLL;
EMA observations are separate. Initializer/LR top2 per family are freshly trained
at the short horizon, with duplicate baseline promotions shared. All candidates
use completed WSD cooldowns and fixed peak LR values, without short-horizon LR
upscaling. Promotions require finite token-weighted held-out NLL at matched tokens.

Calibration derives the short horizon from the slowest complete update, bounded
by600M and rounded down to262144 tokens. Initial horizon is at most150077440,
rounded down to131072; if initialization overhead makes150M infeasible within20m,
it uses the largest common endpoint with15% throughput margin, minimum100M.
This avoids allocating pilots which cannot complete their cooldown. Warmup is
22282240 tokens. Optional transferred resplit schedule retains12715/15259 phase
ratio; short second-phase warmup is capped at10% of its phase to avoid warmup/decay
overlap. Final2B restores the original22282240-token local warmup.

Combination verification tests replay baseline and merged family winners.
If merged does not beat replay, test baseline-initialization and
baseline-LR/clipping/WD rollback variants. Choose the lowest eligible seed42 NLL,
including EMA variants. Compare the exact chosen raw/EMA variant against baseline
on seed43; use mean paired NLL over42/43. Failed controls stop dependent promotion.
Batch8 is eligible for final only if measured throughput predicts completion
within4h including initialization, evaluation and stopping margin. Otherwise use
baseline. One2B run stops at first full batch crossing2B (2000027648 for B8/B16;
2000158720 for B32), with raw optimizer checkpoint and separately labeled EMA HF
export. Full evaluation reports raw/EMA and selects the lower full-set NLL export.

The three tail scales.05/.1/.2 are relative to original first-phase peaks, from
frozen checkpoint012715-7b7f9aa120ca, never latest. They load the original sealed
numerical source and restore all optimizers, RNG and cursor, rebasing only future
LR. Parent/wrong-tail/four historical siblings cost41.262222H; three55m allocations
bound the tree at63.262222H. Parent processed tokens plus all sibling tails remain
below6B. The shared prefix is physically charged once, inherited logical caps
still include its cost. These tails have their own W&B identities.

Submission is owned by one hostname/path/hash registered controller, protected by
an interprocess lock. Uncertain submissions keep reservations and are recovered,
never retried. Each allocation writes immutable endpoint receipt; no verification
of a prior phase relies on a later overwritten summary. Minimum32H is held for
final before pilot submission. Every allocation is reconciled via sacct.

Output: receipts, ranking.json, ranking.tsv and Chinese report.md in the campaign
results directory; W&B lab5-training-llama; all large artifacts under/work.
