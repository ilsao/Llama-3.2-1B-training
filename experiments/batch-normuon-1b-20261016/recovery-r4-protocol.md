# Fix and continue: preserve all validation gates and the approved nine endpoints

Latest user instruction:修正並繼續,after the recorded NorMuon timeout.
This revokes the previous stop after the one corrected validation. It does not
authorize unbounded retries,additional research candidates,formal runs,cleanup
or HF upload. Preserve both failed source snapshots and all allocation costs.

The original approved plan (user:Implement the plan) contains exactly nine
endpoints and146.5H reservation ceiling. Original control/NorMuon/LR/fork
configurations remain byte-for-byte unchanged. Training is still8H200/8CPU,
one independent sbatch per candidate,at most120minutes. Use119-minute requests
in this recovery to leave room for terminal-accounting overhead within the
same campaign ceiling. No research endpoint has yet been submitted.

Automatic review rejected releasing multiple8GPU training jobs before NorMuon
validation passed. This recovery preserves the complete validation gate:
NO research candidate is submitted before BOTH native validation and NorMuon
validation have passed. The submitted whitelist remains the original nine
endpoints plus exactly one new bounded correctness allocation.

Reuse the thirteen successful native checks durably published by554476:
full weights/two optimizers/scheduler/RNG bit-exact restoration,identical-gradient
BS64 update replay,variable-beta2 math,DDP normalization and fixed eval identity.
Before reuse,verify native numerical payload equivalence by AST/file hashes.
The old allocation TIMEOUT remains a failure;never relabel it COMPLETED.

For NorMuon,disable only the package torch.compile wrapper through the official
torch2.9.1 TORCH_COMPILE_DISABLE=1 environment switch. Keep muonium.2.0,
NorMuon,PE5steps,foreach=True,momentum/betas/LRs/WD and parameter groups unchanged.
No package patch,optimizer reset,shared environment upgrade or pretrained load.

Run one NEW single-H200/8CPU correctness job,15minutes/.25H requested,
with an internal120second stopping margin. Test a full randomly initialized
Llama1B at8192 context,CPU-first optimizer construction,four finite updates,
all NorMuon momentum/variance state,bit-exact restore and identical-gradient
next update,RNG restore,finite heldout NLL,and zero Dynamo compiled graphs.
Record stage timestamps and errors BEFORE any process collective so a primary
failure cannot be hidden by a later collective timeout. Training entrypoints
also write per-rank exceptions before failing the distributed launcher.

Validation inputs65536 are separately journaled. The smoke checkpoint is never
used to initialize a research candidate. All three roots of the research DAG
still start from the same random initialization. Six scheduled full-state forks
are the only authorized research checkpoint loads.

Past measured study cost is1.981111H (including.001111H setup). Nine119minute
allocations reserve142.8H;one new single-GPU15minute check reserves.25H.
Worst reservation total145.031111H,below the unchanged146.5H ceiling.
Record actual Slurm duration including termination overhead;do not erase
the validation-r1 individual overrun flag or uncertain token reservation.

After both gates pass,the sole sealed owner submits control/NorMuon/LR roots
in parallel,and releases each of the six independent sbatches only when its
exact immutable shared checkpoint is complete. Do not retry failed research
candidates or automatically retry a failed new correctness job.

Recovery-r3: single-GPU check555093 failed before any training input due to unindexed cuda device. Normalize only this check to the current indexed CUDA device;preserve research and native-validated code. User instruction 修正並繼續 remains the authorization;all costs and the146.5H total ceiling remain.

Recovery-r4:555108 exposed Accelerate moving the group LR scalar to CUDA. A local Muon subclass restores only this scalar to CPU before each step,including after checkpoint load,for torch2.9 foreach compatibility. Package source and NorMuon/PE math are unchanged. Preserve32seconds of failed single-GPU cost and its pending input reservation.
