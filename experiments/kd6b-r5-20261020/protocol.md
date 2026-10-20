# Approved R5 online full-vocabulary KD 6B run

The user replied "好，開始訓練" to the complete reviewed proposal, including
8H200/8CPU,9hours/72H inclusive,rounded6000082944-input endpoint and scoped
study cap248.1786111111111H. Exactly one allocation, formal19. Preserve18 prior
formal identities and all176.1786111111111H of prior study cost. No automatic
retry/resume/requeue,trained-checkpoint load,old-artifact cleanup or HF upload.

The approved configuration is copied unchanged from training-config.proposed.json;
digest3080c174a87a531814a8e26b5caf122b66806046c51c57cf9b58329278ecc24c.
Its descriptive `proposed-full` label is retained to preserve that exact digest;
the authoritative ledger category is full. Formal-run implementation and approval
state are recorded separately. All source and execution switches are sealed.

Fresh random residual-half seed42 Llama3.2-1B; official frozen pretrained3B
teacher, same8GPUs, BF16 teacher and autocast,FP32 student weights. Full128256
vocabulary .75CE+.25KL(T||S),T1,chunk1024,intradoc document-span causal mask.
R5 means no activation checkpointing,micro2,compiled RMSNorm/MLP/RoPE,whole
NorMuon matrix owners with FP32 all-gather,and fused exact-vocabulary CE/KL.
No approximation,teacher replacement,extra seed or optimizer-formula change.

Retain the established6B table LRs and token-clock schedule,including all event
boundaries. 28619 updates; globalBS16/32/48/64 counts22897/3/2/5717. Micro2
uses accumulation1/2/3/4. Last update has LR0 and reaches6000082944 inputs.
Teacher/student journals reserve before forward and count repeats/skipped updates;
no unjournaled real-LM warmup or benchmark pass. Synthetic numerical and serializer
checks occur inside this allocation before random model initialization.

Use canonical raw split/evaluation and read-only6B tail overlay. Rank0 verifies
original metadata/manifests and each touched immutable shard checksum before any
rank reads it; actual8rank first/final-window reader preflight precedes models.
No reshuffle,repeated data,or original artifact modification.

Evaluate fixed1k every2800 updates, at preserved token events and at endpoint;
then additional9k, combining total NLL/prediction tokens for10k. All evaluations
use eager execution. Record complete config and persistent W&B link in
lab5-training-llama; local metrics precede bounded tracking calls. train/loss is
the token-weighted interval KD objective; CE and KL are separately logged.
Other required metrics are current-step values. End-to-end step time includes
data and durable journals,excludes evaluation; also report compute-only timing.

Save all owner optimizer states with parameter names,replicated AdamW once,
per-rank RNG,schedulers,data cursor,token journals and resource/W&B identity.
Checkpoint serialization is checked on synthetic states before training; final
owned states are reloaded and compared, and HF student export reload is verified.
No checkpoint is loaded into a training model. Final or wall-stop checkpoint only.
Stop new training work1200seconds before actual Slurm end (900evaluation+300
stop-margin), preserving the reviewed900/300 configuration. Preserve incomplete
outcomes honestly; no automatic continuation. Do not claim6B quality noninferiority
from the prior128-step screen or claim this72H allocation satisfies64H.
