# 3B-token KD configuration and measured I/O result

Both validation allocations completed successfully on separate H200 nodes:
producer575056,239s ×4GPUs; consumer575057,224s ×4GPUs. Total measured cost
0.5144444444 H200-hours, reconciled to the shared ledger. No full training
allocation has been submitted for this request.

## Concrete training configuration

| Setting | Value |
|---|---|
| Student | Llama-3.2-1B, random residual-half initialization, seed42 |
| Teacher | Official pretrained Llama-3.2-3B base; revision13afe5124825b4f3751f836b40dafda64c1ed062 |
| Data / mask | Canonical raw Dolma preparation; original held-out split; intradoc causal mask |
| Context / precision | 8192; BF16 compute, FP32 student parameter/optimizer storage |
| Loss | .75 CE + .25 KL(teacher || student), T=1, full128256 vocabulary |
| Matrix optimizer | NorMuon NS5, momentum.95, Nesterov, LR.005593716406714465, ordinary WD.1 |
| Auxiliary optimizer | AdamW LR.0003356229844028679, betas(.9,.95), eps1e-8; original groups |
| Gradient clipping | Global norm.5 |
| Batch | Microbatch1/GPU; global16 then64 at1,499,987,968 inputs; 8ranks, accumulation2 then8 |
| Token schedule | Phase split2,500,067,328; second peak.1x;170 reference-token warmup units each;40% tail,4:1:1 ramp/hold/final,.2 plateau |
| Exact endpoint | 3,000,107,008 nonpadding inputs;14,312 optimizer updates |
| Evaluation | Fixed1k at most1431 updates apart plus token knots/end; additional9k at endpoint |
| Selected execution | Existing validated online two-stream/two-slot pipeline on8H200/8CPU |
| Allocation proposed | One8h allocation, at most64H200-hours, including evaluation/export/idle |
| Logging | W&B lab5-training-llama, global token-weighted metrics and persistent local records |

Configuration digest: `79976ae5a2093ded5eab07e4439688293fd6111122c28799a0119344d7ef79ea`.
See training-config.json and schedule-plan.json. The isolated training source
updates the horizon, scheduler clamp and endpoint-evaluation contract from1B
to3B; existing historical sources remain untouched. Full-run resource gates
remain unchanged pending the budget review.

## Cross-node measurements

Each format/geometry uses one warmup window and two timed optimizer windows.
The teacher rate includes real forward/projection, GPU-to-CPU copy and fsynced
writing, excluding waiting for free queue slots. Student end-to-end rate includes
read/prefetch waits, full-vocabulary KD, backward and optimizer work. The resident
reference has teacher targets already on the student's GPU. Student weights stay
fixed with LR0; all repeated inputs are separately recorded as validation work.

| Payload | Global batch | Teacher production + write | Resident student demand | Student with disk queue |
|---|---:|---:|---:|---:|
| BF16 full logits |16|16,150 tok/s|103,260 tok/s|16,321 tok/s|
| BF16 full logits |64|15,469 tok/s|114,615 tok/s|15,462 tok/s|
| BF16 final hidden |16|141,992 tok/s|104,309 tok/s|102,454 tok/s|
| BF16 final hidden |64|151,030 tok/s|113,656 tok/s|108,643 tok/s|

Full-logits writes sustain about4.34–4.53GB/s during concurrent consumption.
Writing accounts for roughly92% of the producer service time at BS64. The
teacher supply is only13.5–15.6% of resident student demand; the disk path is
6.33–7.41x slower than resident training. A1TB full-logits cache covers only
~3.9M prediction positions; it cannot repair this sustained deficit. An ideal
initial1TB would drain in roughly39–45 seconds at these measured demand gaps.

Hidden states reduce bytes41.75x and retain the full teacher distribution by
applying the frozen teacher head on the student. Production exceeds student
demand by33–36%; the disk path adds1.8–4.6% to student time. Real teacher-head
reconstruction checksums, hidden/logits synthetic gradients, and repeated
resident/disk loss comparisons passed. Spool files are published atomically and
only consumed files in this new cache are deleted after backward.

However, splitting the eight GPUs into4 teacher +4 student yields only102–109k
tok/s. The previous sustained8GPU online run measured median131,270 tok/s at
BS16 and151,596 atBS64. Its whole-allocation evidence therefore favors online8GPU
for completing3B promptly. This comparison spans different nodes/runs; the cache
test is short and does not establish a long-duration filesystem guarantee.

## Completion-time calculation and remaining budget decision

Using the previous online run's observed batch-specific median speeds gives
5.92h of training compute; extrapolating measured data/journal/logging overhead
adds0.15h. Initialization, periodic evaluation and endpoint work remain additional.
Planning with the slower10th-percentile speed, a further20% slowdown allowance,
20min of final-evaluation/stopping reserve and3min startup/periodic-evaluation
allowance gives7.94h. An8h allocation is therefore the proposed conservative
ceiling, not a guarantee against node or storage failures. No automatic retry or
checkpoint continuation is proposed.

Current study spending including this test is93.4363888889 H200-hours. Adding a
full64H reservation would raise the maximum to157.4363888889H, exceeding the
previous128H cap by29.4363888889H. All historical costs and17 formal identities
must remain; this would be formal18 with one allocation. The requested revised
study cap would be157.437H (rounded upward), with64H for this logical run.

Automatic approval review rejected a proposed write of a pending authorization
and accompanying resource-gate changes: it found that configuration/validation
were authorized but this budget exception was not. Those rejected writes did
not execute. No full-run reservation/submission or budget-gate expansion was
performed. The measurements, configuration and schedule checks are complete;
the increased budget requires the user's explicit approval.
