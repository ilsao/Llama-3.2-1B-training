Updated 2026-10-20T00:58:02.421323+08:00

Run kd3b-hidden8-20261019-seed42

Jobs: {"575376": {"state": "COMPLETED", "elapsed_seconds": 17073, "start": "2026-10-19T19:58:26", "end": "2026-10-20T00:42:59", "node": "25a-hgpn027"}, "575377": {"state": "COMPLETED", "elapsed_seconds": 17879, "start": "2026-10-19T19:58:26", "end": "2026-10-20T00:56:25", "node": "25a-hgpn035"}}

Student: {"train/global_step": 14312, "train/total_tokens_seen": 3000107008, "train/loss": 2.699357360939576, "train/tokens_per_second": 168061.20753123297, "train/grad_norm": 0.4269091784954071, "timestamp": 1792428880.7648728}

Teacher: {"local_backpressure_seconds": 4961.3963598444825, "local_write_seconds": 4547.680946129025, "produced_updates": 14312, "rank_payloads": 45778, "service_seconds_max_rank": 10319.6357421875, "service_tokens_per_second": 290718.304691252, "teacher_inputs": 3000107008, "timestamp": 1792428177.9885077, "wall_seconds": 17010.820650815964}

Budget: {"measured_h200_hours": 173.0897222222222, "reserved_or_measured_h200_hours": 173.0897222222222, "study_cap_h200_hours": 191.419}

Endpoint10k PPL: 13.946899272688054

W&B: https://wandb.ai/cerulean-labs/lab5-training-llama/runs/82e6b67d45e6444ba2dcdb77dc387282
