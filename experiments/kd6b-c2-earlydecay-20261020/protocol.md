# Approved fresh 6B C2 KD with token-bound early decay

User approved the reviewed configuration with “好，注意到 decay 綁定的是 tokens 喔”.
Exactly one 8 H200 / 8 total CPU / 32400-second allocation, max72 H200-hours;
formal20 preserves all19 previous identities and every allocation/cost. New
scoped cap is the existing accounted/reserved total plus72; no prior cap is reset.
Fresh residual-half seed42; no training checkpoint load/retry/resume/requeue,
cleanup, upload or additional allocation.

Use nominal6B calibration table row with Muon .00425961126463665 and AdamW
.0001277883379390995 x2. C2 adaptive batch16/32/64/128, noise fraction1,
EMA.9, min8 probes,20% hysteresis, probe every16777216 inputs, max2x change.
No batch-dependent LR scaling and no forced3B batch change.

W&B run58970e70c88c protobuf history confirms _step1841 is an evaluation row:
train/global_step18310, eval/training_tokens2399928320. This run's first decay
starts at that exact cumulative nonpadding input-token boundary. W&B _step,
logging cadence, update count and changing batch cannot move it.

Explicit token boundaries and two phases are in config.json. First tail
4:1:1 proportions rounded to complete131072-input quanta; second phase remains
unchanged. At the decay boundary the first ramp LR equals peak, matching the
existing convention; the next token quantum is strictly lower. At each phase's
last quantum LR is zero. Windows are shortened at events so no event is crossed.
Warmups are170 BS16-equivalent input-token units, independent of actual updates.

All rank-private compiler/CUDA caches, compiled backward/DDP gates, owner-state
serialization and actual8rank canonical-tail reader checks occur inside this
allocation before training. Full model is never loaded on the login node.
Student/teacher journals are separate and cap each role at6000082944 inputs;
82944 rounded-up inputs above nominal6B are explicitly included in approval.

Evaluate fixed1k at conservative <=570-update gaps, ten-percent token milestones and scheduler boundaries;
final additional9k and combined10k NLL-weighted perplexity. Complete branch/final
states include all owned optimizer states and CBS state; export reload is checked
inside allocation. Stop training480seconds before allocation end for final work.
