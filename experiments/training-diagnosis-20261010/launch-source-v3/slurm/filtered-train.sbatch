#!/bin/bash
#SBATCH --job-name=muon-filtered-2b
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
export TRAIN_ENV_PREFIX=/work/asciibase64/Llama-3.2-1B-training/environments/training-muon-torch291-fa3-94e22c9
source "${TRAIN_SOURCE_ROOT:?Submit through submit_filtered_training.py}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
PREPARATION_ID=${2:?Supply the verified filtered preparation}
TRAIN_CONFIG=${3:?Supply the frozen training configuration}
python -u "$REPO/slurm/submit_filtered_training.py" worker-check
cp "$TRAIN_CONFIG" "$JOB_RECORD/filtered-config.json"
launch() {
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
        --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
        --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
        --num_cpu_threads_per_process 1 "$@"
}
# No --resume: this request must start from random weights.
launch "$REPO/slurm/training_entry.py" --config "$TRAIN_CONFIG" \
    --prep-id "$PREPARATION_ID" --run-id "$RUN_ID" --validate-export
# A graceful wall-time stop is not completion of this token-target request.
python -u "$REPO/slurm/submit_filtered_training.py" training-check
SECONDS_LEFT=$((TRAIN_END_EPOCH - $(date +%s) - 60))
[[ "$SECONDS_LEFT" -gt 0 ]] || { echo 'No remaining time for full evaluation'; exit 124; }
timeout --signal=TERM --kill-after=15s "$SECONDS_LEFT" \
    srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --num_cpu_threads_per_process 1 "$REPO/slurm/full_evaluation.py" \
    "$TRAIN_CONFIG" "$RUN_ID" "$PREPARATION_ID"
