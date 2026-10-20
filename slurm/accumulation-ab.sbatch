#!/bin/bash
#SBATCH --job-name=accumulation-ab
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --exclude=25a-hgpn001,25a-hgpn062
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
export PERFORMANCE_RUN_ID="$RUN_ID"
CONFIGURATION=${2:?configuration}
export ACCUMULATION_CONFIGURATION="$CONFIGURATION"
python3 "${ACCUMULATION_CONTROLLER:?}" check
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --num_cpu_threads_per_process 1 \
    "${PERFORMANCE_LAUNCHER:?}" "$REPO/slurm/training_entry.py" \
    --config "$CONFIGURATION" --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID"
