#!/bin/bash
#SBATCH --job-name=lr-transfer
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus-per-node=8
#SBATCH --constraint=H200
#SBATCH --time=01:34:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
#SBATCH --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err
set -euo pipefail
export NCCL_SOCKET_IFNAME='=lo'
export NCCL_SOCKET_FAMILY=AF_INET
source "${TRAIN_SOURCE_ROOT:?}/slurm/training-common.sh"
export DATA_ROOT TRAIN_ENV_PREFIX JOB_RECORD
CONFIGURATION=${2:?configuration}
nvidia-smi --query-gpu=index,name,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit --format=csv > "$JOB_RECORD/gpu-startup-diagnostics.csv"
nvidia-smi topo -m > "$JOB_RECORD/gpu-topology.txt"
taskset -pc $$ > "$JOB_RECORD/launcher-affinity.txt"
python3 "${TRANSFER_CONTROLLER:?}" check
srun --ntasks=1 --kill-on-bad-exit=1 accelerate launch --multi_gpu \
 --main_process_port "$((15000 + SLURM_JOB_ID % 40000))" --num_processes 8 \
 --num_machines 1 --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1 \
 "$REPO/slurm/lr_transfer_entry.py" --config "$CONFIGURATION" \
 --prep-id dolma-raw-baseline-20261008 --run-id "$RUN_ID"
python3 "${TRANSFER_CONTROLLER:?}" receipt "$CONFIGURATION" "$RUN_ID"
