#!/bin/bash
#SBATCH --job-name=optimizer-sweep
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=00:05:00
#SBATCH --signal=B:USR1@60
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?Submit through the sweep controller}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
PREPARATION_ID=${2:?Supply canonical preparation ID}
TRAIN_CONFIG=${3:?Supply frozen stage configuration}
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --num_cpu_threads_per_process 1 "$REPO/slurm/training_entry.py" \
    --config "$TRAIN_CONFIG" --prep-id "$PREPARATION_ID" --run-id "$RUN_ID"
