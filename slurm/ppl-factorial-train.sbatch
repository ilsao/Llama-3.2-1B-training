#!/bin/bash
#SBATCH --job-name=ppl-factorial
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=H200
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
export NCCL_SOCKET_IFNAME='=lo' NCCL_SOCKET_FAMILY=AF_INET
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
ARM=${2:?arm}
CONFIGURATION="$REPO/../configs/$ARM.json"
python "${FACTORIAL_CONTROLLER:?}" check
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1 "$REPO/slurm/training_entry.py" --config "$CONFIGURATION" --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID" --validate-export
python "$FACTORIAL_CONTROLLER" receipt "$ARM"
