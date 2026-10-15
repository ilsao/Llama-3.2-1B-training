# Validation passed; original experiment DAG launched

The user requested 修正並繼續 after the previous stop. Current sealed source is source-r4 (commit2fdc230);14 lightweight tests pass. Native numerical payload equivalence is verified against the13 successful checks in554476.

Single-H200/8CPU validation555117 COMPLETED successfully. It verified full random Llama1B initialization,8192 context,NorMuon+AdamW+Polar Express updates,complete state/RNG restoration and identical-gradient replay. Eager execution avoids the package compile wrapper;only the LR scalar is returned to CPU before steps for torch2.9 foreach compatibility.

Both validation gates passed. The sole source-r4 controller submitted control555139,LR.01 555140 and NorMuon555141 as independent8H200/8CPU119minute sbatches. The six BS64 fixed/scaled-beta2 branches require their exact complete control snapshots and will each use an independent sbatch. No failed research endpoint is automatically retried. The aggregate ceiling remains146.5H200-hours;historical failed allocations and pending token reservations remain recorded.

Failures retained:554398(93s,8GPU),554476(798s,8GPU;13 native checks passed,then timeout),555093(25s,1GPU;unindexed CUDA device before training),555108(32s,1GPU;LR scalar device mismatch at first NorMuon update). Setup554390 used4s,1GPU. No cleanup,HF upload or changes to unrelated3B training.

Live machine-readable state:controller-state-r4.json. Live human-readable status and endpoint comparisons:report.md and results.json.

NorMuon validation completed four updates /65536 inputs,bit-exact complete model/optimizer restoration and identical-gradient replay,RNG restoration,finite heldout NLL and zero Dynamo compiled graphs. Result:normuon-result-r4.json. The validation checkpoint is not used by research candidates.

All three roots are RUNNING and have completed real optimizer updates;W&B is online in lab5-training-llama. All initialization signatures match32b43fc6e5c34a787fc02db6674dde1afc341123e39d48abb60b5d84e932a05d,and pretrained_weights_loaded is false. Timestamped initial progress and run links:startup-evidence-r4.json. No final1B-token result is available yet.
