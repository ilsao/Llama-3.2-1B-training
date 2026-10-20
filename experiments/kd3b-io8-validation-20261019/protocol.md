# Eight teacher GPUs plus eight student GPUs: cross-node validation

The user explicitly proposed retaining8 student GPUs and also allocating8 teacher
GPUs. This validation implements that requested geometry: two concurrent jobs,
each8 H200 and8 CPU, on different nodes. The earlier4+4 measurement and all its
costs remain intact. No full-run budget exception, full-run allocation, checkpoint
continuation or automatic retry is authorized or performed by this controller.

Each validation job has a20min allocation ceiling. Combined maximum cost is
5.3333333333 H200-hours. Before submission the methods study has spent
93.4363888889H, so the maximum including both reservations is98.7697222222H,
within the existing128H cap. Preserve all historical jobs in the shared ledger.

Teacher: frozen official pretrained Llama-3.2-3B base, revision
13afe5124825b4f3751f836b40dafda64c1ed062, manifest SHA256
bf1dec0109c97ad0d00d5714127aca10a6046edbd0b35e0d4c1a55fb2f192507.
Student: fresh random residual-half seed42 Llama-3.2-1B; no pretrained/checkpoint
initialization. Context8192, canonical raw Dolma rows and intradoc causal mask.
Full-vocabulary .75 CE +.25 KL(teacher||student), temperature1, BF16 teacher
targets. Student optimizer work is real but LR and decay LR are zero; weights
stay fixed. GPU/data/model operations run only inside the Slurm allocations.

Compare final post-normalization BF16 hidden states and full BF16 logits at
global BS16 and64. Microbatch1,8 student ranks, accumulation2 or8. For each
format/batch geometry run2 warmup and6 timed optimizer windows, then6 resident
reference windows. Teacher reserves10,485,760 input tokens in total; student
reserves18,350,080 including repeated resident windows. Actual nonpadding counts
are recorded before work and bounded by20M per role. No convergence claim.

Use the previously validated bounded cross-node filesystem protocol: one
producer and consumer per rank, two disk slots per rank, one CPU I/O prefetch
thread per student rank, one library thread per rank. Maximum full-logits disk
payload33,617,436,672 bytes, below decimal1TB. Fsync partial files, atomically
publish READY; verify row/token/label/document and teacher identity. Acknowledge
and delete only this validation's consumed payload after backward. Preserve
receipts, failed/unconsumed payloads and all historical artifacts. Different
nodes plus POSIX_FADV_DONTNEED reduce page-cache bias; no direct-I/O guarantee.

Each GPU runs synthetic hidden-vs-logits gradient equality and teacher attention
boundary tests. Each real hidden stream checks the frozen teacher-head projection
checksum after transport; resident repeats check identical objective values.
Report producer forward/projection/copy/write service excluding backpressure;
student end-to-end time including exposed I/O wait; resident student demand;
and all actual allocation costs. Geometry is now8+8, not a fixed total of8 GPUs.

Decision: determine whether teacher production exceeds8GPU student demand,
whether hidden transport or full-logits transport scales, and estimate3B-token
completion time. A claimed speedup over historical online8GPU remains an
estimate across different nodes/runs. Teacher and student GPU-hours must both
count, including queue backpressure idle time. A faster wall-clock configuration
does not automatically fit the existing64H full-run proposal. Prepare any
revised full-run budget for review after measurements; do not reuse the pending
online8GPU approval for a different configuration.
