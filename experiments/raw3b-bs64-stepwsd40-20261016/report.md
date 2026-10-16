# Fresh3B BS64 /40% of actual updates per phase

{
  "job": {
    "job_id": "558510",
    "reservation": "raw3b-bs64-stepwsd40-20261016-4d4b52fe067e409d9f5831eae1948a59",
    "returned_job_id": "558510",
    "slurm_state": "RUNNING",
    "status": "submitted",
    "submission_command": [
      "sbatch",
      "--parsable",
      "--time=359",
      "--comment=raw3b-bs64-stepwsd40-20261016-4d4b52fe067e409d9f5831eae1948a59",
      "--exclude=25a-hgpn001,25a-hgpn027,25a-hgpn031,25a-hgpn039,25a-hgpn062,25a-hgpn071,25a-hgpn083,25a-hgpn151",
      "/home/asciibase64/Llama-3.2-1B-training/experiments/raw3b-bs64-stepwsd40-20261016/source/slurm/bs64-3b.sbatch",
      "raw3b-bs64-stepwsd40-20261016",
      "/home/asciibase64/Llama-3.2-1B-training/experiments/raw3b-bs64-stepwsd40-20261016/config.json",
      "dolma-raw-baseline-20261008"
    ],
    "submission_receipt": {
      "returncode": 0,
      "stderr": "",
      "stdout": "558510\n"
    },
    "submission_started_utc": "2026-10-16T08:30:06.395746+00:00",
    "submission_tag": "raw3b-bs64-stepwsd40-20261016-4d4b52fe067e409d9f5831eae1948a59"
  },
  "stage": "submitted_or_uncertain"
}
