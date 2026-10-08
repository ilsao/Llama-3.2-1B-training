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
