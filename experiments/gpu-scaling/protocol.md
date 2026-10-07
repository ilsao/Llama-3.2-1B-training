# Matched-batch GPU scaling protocol

CONFIRMATORY comparison, registered before allocating scaling jobs.
Use configs/train-scaling-4.json and configs/train-scaling-8.json, the same sealed
preparation/evaluation subset, seeds and effective global batch 16. Per-device
microbatch 1; accumulation 4 at four GPUs and 2 at eight GPUs. Run model/resume
validation first. At most ten allocation minutes per scaling candidate; include all
allocation time in reconciled H200-hours. Compare at least twenty matched stable
post-warmup updates. Promote eight GPUs only with >=1.8x step throughput and
<=1.25x allocated H200-hour cost per input token, without correctness failures.
Use slurm/compare_scaling.py; incomplete/failed measurements do not justify promotion.
These are scaling pilots, not full-budget runs; charge every allocation to the ledger.
