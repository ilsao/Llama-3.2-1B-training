#!/bin/bash
#SBATCH --job-name=sweep-calibration
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=00:10:00
#SBATCH --signal=B:USR1@60
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?Submit through submit_training.py}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
PREPARATION_ID=${3:?Supply canonical preparation ID}
TRAIN_CONFIG=${4:?Supply calibration configuration}
GPU_COUNT=$(wc -l < "$JOB_RECORD/gpus.txt")
launch() {
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
        --num_processes "$GPU_COUNT" --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --num_cpu_threads_per_process 1 "$@"
}
srun --ntasks=1 --kill-on-bad-exit=1 python "$REPO/tests/fa3_runtime.py"
for SCHEDULER in linear cosine wsd; do
    launch "$REPO/tests/training_runtime.py" --run-id "$RUN_ID" \
        --optimizer "${RUNTIME_OPTIMIZER:?}" --scheduler "$SCHEDULER"
done
launch "$REPO/slurm/training_entry.py" --config "$TRAIN_CONFIG" --prep-id "$PREPARATION_ID" \
    --run-id "$RUN_ID" --stop-after-step 1
python - "$DATA_ROOT" "$RUN_ID" <<'PY'
import sys
from pathlib import Path
from training_state import atomic_json, read_json
path = Path(sys.argv[1]) / 'runs/experiments' / sys.argv[2]
summary = read_json(path / 'summary.json')
if summary['global_step'] != 1 or summary['stop_reason'] != 'validation_pause':
    raise SystemExit('Full-size validation did not checkpoint at step 1')
atomic_json(path / 'calibration-pause.json', {k: summary[k] for k in ('global_step', 'cursor', 'wandb_id')})
PY
launch "$REPO/slurm/training_entry.py" --config "$TRAIN_CONFIG" --prep-id "$PREPARATION_ID" \
    --run-id "$RUN_ID" --resume
