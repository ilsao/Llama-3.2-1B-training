# User approved the359-minute correction

The user explicitly replied "可以" to submitting this prepared correction once.
This supersedes the pending-approval wording below for this exact21540-second,
8H200/8CPU allocation only. Keep formal11, failures553561/553577 and all costs.
No further automatic retry/resume or checkpoint training load is authorized.

# Prepared correction:359-minute fresh restart, pending approval

This correction is PREPARED ONLY and does not authorize another allocation.
User approval is required before submission. The previous authorized attempt
job553577 failed at0seconds/0H because Slurm rounded05:59:40 up to06:00:00;
existing budget checks correctly stopped before training. Job553561 remains
charged20seconds/0.044444444444444446H. Keep both failed allocations and formal11.

Prepared request uses exactly359minutes/21540seconds,8H200/8CPUs,200G,original
training configuration and random initialization, no checkpoint training load.
Combined allocation21560seconds/47.9111111111H stays below original6h/48H cap.
No automatic further retry. Submission requires approved=true in the exact
sealed restart2-authorization.json. Only these two failed zero-token startup
jobs qualify; a third attempt invalidates the eligibility predicate. Source,
failed startup archive, all ledgers and original approval records are preserved.
sbatch --test-only validated the minute-based request without allocating GPUs.
