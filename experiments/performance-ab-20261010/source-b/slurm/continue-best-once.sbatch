#!/bin/bash
#SBATCH --job-name=muon-resume-once
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --time=01:10:00
#SBATCH --signal=B:USR1@180
#SBATCH --exclude=25a-hgpn062
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
AUTHORIZATION=${2:?Supply the one-time authorization receipt}
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
    --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --num_cpu_threads_per_process 1 \
    /home/asciibase64/Llama-3.2-1B-training/slurm/continue_best_once.py worker "$AUTHORIZATION"
