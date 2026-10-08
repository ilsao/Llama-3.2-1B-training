# Automatic FA3 baseline, 2026-10-09

This campaign follows the user's hard combined 16 H200-hour and 7200
allocation-second budget. At most 8 H200/8 CPUs per job, including failed,
validation, scaling, pilot, baseline and evaluation allocations. Preprocessing
and FA3 builds are separately charged to shared ledgers, not free.

LR sweep: 3e-4 vs 6e-4, ultrashort stage, 15 minutes each, same seed42/canonical
packed prefix and fixed first1000 held-out documents. Select by global NLL at
largest shared evaluated token point; <=.01 NLL difference favors lower LR.
Skip search for estimated full-eval reserve >30min, insufficient pilot runtime,
or deadline pressure. Default3e-4. No Muon or other sweeps before baseline.

AdamW fused, betas(.9,.95), eps1e-8, decay.1 on matrices, no decay on tied
embeddings/norm/bias; clipping1.0 preclip logged. BF16 compute/FP32 master and
optimizer states. Standard Llama3.2 1B random initialization, context8192,
microbatch1, 8GPU accumulation2/global16. Cosine2% warmup/floor.1. Evaluate at
least every10% planned updates plus end. Actual input tokens <=6B/run.

4/8 probes have matched global16 and32 planned updates (two warmup updates).
8GPU is the deadline choice; record scaling and charge both allocations. 8GPU
probe also checks FA3 GQA/causality/padding/backward/8192, distributed loss and
RNG/optimizer resume, and full-model checkpoint/resume/standard HF reload.

FA3 is the official Hopper source at94e22c906678e5483fa0e9e24d8e787bc2c0ed4c,
BF16/head64 with backward/varlen/GQA. Pinned current Python3.11/torch2.6.0/cu124/
transformers4.51.3/accelerate1.6.0. Build in separate work environment under
Slurm with cuda/12.6 and compiler threads bounded to8 CPUs. No fallback.

Data identity is dolma-dclm10-005, unchanged native seed42 split, global CC top10%,
whole-document6B selection,8192 packing and independent full50000 evaluation.
Allocation cutoff2026-10-09 23:30+08:00, delivery deadline23:59+08:00. Full-set
PPL<20 is required; never claim success from the fixed subset alone.

## Baseline-first raw-data override

The user requested training on the already downloaded dataset immediately rather
than waiting for quality filtering. The campaign now builds a separate
`dolma-raw-baseline-20261008` preparation from the sealed complete seed-42 split
of `dolma-dclm10-005`. It skips text cleaning, deduplication and quality ranking;
this is explicitly an unfiltered AdamW baseline. The existing filtered artifacts
and allocation are retained and accounted for.

`raw_baseline.py` runs in the `raw-prepare` Slurm phase with seven CPU workers and
one coordinator. It first authenticates the canonical split indices and packs the
50,000 original reserved evaluation documents with the existing evaluation
window/loss rules. It then publishes an initial 1,024-sequence training buffer
and a fixed input/tokenizer/packing identity. GPU checks can start at that point,
while the producer continues publishing immutable training shards in canonical
shuffle order, up to six billion non-padding input tokens. No full filtering or
full training-corpus packing completion is required before launch.

Consumers verify shard checksums when mapping a shard, reject changes to the
published prefix, and save consumed-prefix receipts with checkpoints. Producer
resumes restore its committed document cursor and exact packer carry. Training
resumes retain weights, optimizer, scheduler, RNG, cursor, W&B identity and token
journal. A stalled producer causes a synchronized safe checkpoint/stop instead
of silently substituting or repeating data.

This override skips LR search: matched 4/8-GPU checks run first, then the baseline
starts at AdamW LR 3e-4 with the remaining allocation (normally 110 minutes).
Full reserved-set evaluation remains required for the PPL target. All GPU checks,
training and full evaluation still share the original 16 H200-hour/7,200-second
campaign cap. CPU preparation is separately charged in the shared experiment
ledger with allocation_kind=preprocessing.

The first 4-GPU allocation completed 14 finite optimizer updates (1,835,008
input tokens) and saved a checkpoint/export before its conservative wall stop.
Torchrun converted the intended child exit 75 into launcher exit 1. Its FAILED
allocation and 239 seconds / 0.265556 H200-hours remain charged. The controller
accepts this partial validation only with the completed summary, checkpoint,
export and actual FA3 kernel calls; scaling uses at least 12 common post-warmup
updates. The 8-GPU FA3/padding/backward/resume/export checks remain prerequisites.
`slurm/training_entry.py` now reports rank stop codes explicitly and maps only a
checkpointed 75 to launcher success. Validation jobs save at the configured
checkpoint interval, deliberate pause and end, avoiding a full 1B checkpoint at
every improved validation loss; baseline best-checkpoint behavior is unchanged.

The eight-H200 checks passed FA3 forward/backward/masking and distributed
weights/optimizer/scheduler/RNG correctness. A full 1B update and checkpoint also
completed, with the same random-initialization signature as the 4-GPU model.
The subsequent full-model resume exposed a JSON metadata mismatch: numeric
`id2label` keys become strings when saved. Initialization metadata is now compared
in its persisted JSON representation, retaining strict random-weight signature
and architecture checks. The failed allocations remain charged; they are not
resumed with changed training code.

Baseline now proves its own full-model resume inside the same allocation: one
optimizer update, an end evaluation and faithful checkpoint, then a fresh
launcher restores that logical run and continues. FA3 and tiny correctness
receipts are reused with their real original job IDs. The 4/8 comparison has only
one matched startup update; its limitations are recorded and it is not reported
as steady throughput. A conservative planning estimate uses the observed 4-GPU
cold/warm ratio. Baseline runs until the wall-time/token stopping margin, with a
separately pinned `scheduler_steps` horizon so an estimated step count cannot
cause it to stop early. Actual costs and the same original campaign caps remain
in force. Final full-set evaluation reloads the exported HF model.
