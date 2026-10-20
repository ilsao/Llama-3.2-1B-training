#!/bin/bash
#SBATCH --job-name=muon-baseline
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --exclude=25a-hgpn062
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?Submit through submit_muon_baseline.py}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
PREPARATION_ID=${2:?Supply canonical preparation ID}
TRAIN_CONFIG=${3:?Supply frozen baseline configuration}
[[ $(wc -l < "$JOB_RECORD/gpus.txt") == 8 ]] || exit 1
cp "$TRAIN_CONFIG" "$JOB_RECORD/baseline-config.json"
launch() {
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
        --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
        --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
        --num_cpu_threads_per_process 1 "$@"
}
launch "$REPO/slurm/training_entry.py" --config "$TRAIN_CONFIG" \
    --prep-id "$PREPARATION_ID" --run-id "$RUN_ID"
# The entry wrapper translates a faithful wall-time stop to launcher success.
# Validate the original per-rank status before evaluating an exported checkpoint.
python3 - "$JOB_RECORD" <<'PY'
import json, sys
from pathlib import Path
codes = [json.loads((Path(sys.argv[1])/f'worker-rank-{i}-exit.json').read_text())['exit_code'] for i in range(8)]
if len(set(codes)) != 1 or codes[0] not in (0, 75):
    raise SystemExit('Ranks did not finish training safely')
PY
SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 60))
[[ "$SECONDS_LEFT" -gt 0 ]] || { echo 'No time remains for full evaluation'; exit 124; }
timeout --signal=TERM --kill-after=15s "$SECONDS_LEFT" \
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --num_cpu_threads_per_process 1 "$REPO/slurm/full_evaluation.py" \
    "$TRAIN_CONFIG" "$RUN_ID" "$PREPARATION_ID"
