# Stopped after the user-approved single corrected validation

Code commits:0767159,30d2c10,a44eef0. Twelve lightweight checks passed.

Validation554476 passed thirteen native/checkpoint checks,including bit-exact full model/optimizer state restoration and same-gradient BS64 update replay. NorMuon did not complete;NCCL collective timeout occurred. Root cause remains unconfirmed.

Slurm requested780seconds;actual798seconds including18seconds termination overhead. The individual validation-r1 logical limit is flagged exceeded in the preserved resource ledger. With failed validation554398(93seconds),combined891seconds/1.98H stays within original900seconds/2H validation allowance. Setup554390 cost4single-GPU seconds/.001111H. No historical cost was removed or reset.

All nine research endpoints remain unsubmitted. The controller retired its own cron after the second failure,following explicit user approval condition:再次失敗即停止. No automatic retry,resume,cleanup,HF upload or changes to the unrelated running3B job.
